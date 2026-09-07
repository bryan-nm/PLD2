"""Prompt-conditioned generation: reweight PLD2's logits by a FILIP classifier.

ProteinGuide (Xiong et al. 2026) in the MLM / any-order case. A trained text<->protein
co-embedding is read as p(prompt | sequence), and at every decoding step the unconditional
distribution is tilted toward sequences the classifier believes match the prompt:

    log p(x~ | x, prompt)  =  log p(x~ | x)  +  gamma * log p(prompt | x~)
                              PLD2 prior        FILIP classifier

WHY THIS, AND WHY NOW. Five interventions on the denoiser -- span corruption, a 3Di structure
track, a corpus mix, foldseek's own substitution matrix, double the training -- left generated pTM
at 0.21-0.23 against 0.71 for natural proteins. The measurement that says why is the cold start:
on an all-MASK canvas the model's amino-acid distribution is 2.876 nats against an empirical
unigram of 2.875, i.e. EXACTLY composition. That is the correct answer -- there is nothing to
condition on -- but it means the first commits of every generation are near-random, and everything
after is built to be consistent with a random seed. An unconditional factorised model has no
mechanism to commit to a global plan before it starts emitting tokens. Guidance supplies one.

------------------------------------------------------------------------------------------------
TWO WAYS TO SCORE CANDIDATES, and the cost difference decides which is usable here.

DEG (exact). Enumerate the candidate residues at a position, build one sequence per candidate, and
score each with the classifier. This is what src/guide.py does in filip_guide, and it is exact.
Cost is one classifier forward per (position, candidate). PLD2's guidance hook is called once per
step with the WHOLE canvas, and a 512-step decode over 512 positions x 20 residues is ~5M AMPLIFY
forwards per sample. Unusable as the default; kept, restricted to the few positions actually in
contention, as the exact check that TAG is not lying.

TAG (first-order, the default). Run the classifier ONCE on the current canvas and take the gradient
of log p(prompt | x) with respect to the input embeddings. The score for putting residue c at
position i is then the inner product of that gradient with c's embedding:

    delta[i, c]  ~=  < d log p(prompt|x) / d e_i ,  E[c] - e_i >

The subtracted term depends only on i, so it vanishes under the softmax over c and is dropped. One
forward and one backward per step gives a score for EVERY (position, residue) at once, which is
exactly the shape PLD2's guidance hook wants, and the cost is independent of sequence length.

------------------------------------------------------------------------------------------------
p(prompt | x) has two forms, and the default is the cheap one for a reason. `softmax_bank`
normalises over a reference prompt bank and matches the InfoNCE training objective, but each
evaluation needs a [B, M, L_p, L_t] similarity tensor, and inside a 512-step decode that is paid
512 times. `sigmoid` needs only the target prompt. Since sigmoid is monotone in the FILIP
similarity, it induces the same ORDERING over candidates as the raw score at a given step, which is
all guidance uses; the bank changes the scale, not the direction. Default sigmoid, bank available.

MASKED POSITIONS ARE EXCLUDED from the FILIP score (filip_guide Decision #2): the classifier reads
only committed residues, so guidance means "given what is placed so far, does this look like the
prompt" rather than being diluted by positions that are not yet decided.

mini-embed-filip is IMPORTED, not vendored. Loading AMPLIFY on Aurora needs an xformers stub, a
RoPE-cache rematerialisation and two SDPA patches that were established the hard way there; a
second copy would rot. config.MINI_EMBED_REPO points at it.
"""
from __future__ import annotations

import json
import os
import sys
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from config import FILIP_CACHE, FILIP_CKPT, MINI_EMBED_REPO
from .blosum import AA


# ---------------------------------------------------------------------------
# mini-embed-filip import
# ---------------------------------------------------------------------------
# The four modules below import only stdlib, torch and numpy -- no intra-repo imports -- which is
# what makes loading them by PATH safe.
_MINI_EMBED_FILES = {"config": "config.py", "encoders": "src/encoders.py",
                     "model": "src/model.py", "losses": "src/losses.py", "data": "src/data.py"}


def _mini_embed(repo: str = None):
    """Load mini-embed-filip's config / encoders / model / losses / data.

    BY FILE PATH, UNDER PRIVATE NAMES, because both repos are laid out the same way and a plain
    import collides twice. PLD2 runs as `python -m src.sweep_sampler`, so `src` is already in
    sys.modules as PLD2's package and `import src.encoders` resolves against it -- "No module named
    'src.encoders'". `config` collides identically: PLD2 has its own, already imported. Putting the
    repo on sys.path cannot fix either, since sys.modules is consulted first.
    """
    import importlib.util
    repo = repo or MINI_EMBED_REPO
    if not os.path.isdir(repo):
        raise SystemExit(
            f"mini-embed-filip not found at {repo}. It provides the AMPLIFY loader (with the "
            f"Aurora xformers stub, RoPE rematerialisation and SDPA patches), the FILIP model and "
            f"the packed text cache reader. Set PLD2_MINI_EMBED_REPO.")
    mods = {}
    for name, rel in _MINI_EMBED_FILES.items():
        path = os.path.join(repo, rel)
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; is {repo} really mini-embed-filip?")
        key = f"_mini_embed_filip_{name}"
        if key in sys.modules:
            mods[name] = sys.modules[key]
            continue
        spec = importlib.util.spec_from_file_location(key, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as ex:                       # noqa: BLE001 - surface the real cause
            del sys.modules[key]
            raise SystemExit(
                f"loading {path} failed: {type(ex).__name__}: {ex}\n"
                f"These modules are loaded standalone, so an intra-repo import added to one of "
                f"them would surface here rather than resolving against PLD2's own src/ or "
                f"config.py -- which is the point.")
        mods[name] = mod
    return mods


# ---------------------------------------------------------------------------
# PLD2 canvas -> AMPLIFY ids
# ---------------------------------------------------------------------------
class CanvasBridge:
    """PLD2 token ids -> AMPLIFY ids, with alignment preserved.

    PLD2's canvas is (B, L) over 0..19 residues, EOS, PAD, MASK, and carries no CLS. AMPLIFY wants
    a framed sequence, so the bridge emits (B, L+2) as [cls] + canvas + [eos] and AMPLIFY position
    i+1 corresponds to canvas position i. That one-to-one alignment is what lets TAG map a gradient
    at an AMPLIFY position back onto a PLD2 position; without it the guidance would be applied to
    the wrong residues, silently.

    A PLD2 MASK becomes AMPLIFY's <mask> (which the FILIP score then excludes), and PAD becomes
    AMPLIFY's <pad> with attention 0.
    """

    def __init__(self, cfg, amp_tok, device):
        self.cfg = cfg
        self.device = device
        v = cfg.vocab_size
        reg = torch.full((v,), int(amp_tok.unk_token_id or 0), dtype=torch.long)
        for i, letter in enumerate(AA):
            j = amp_tok.convert_tokens_to_ids(letter)
            if j is not None and int(j) >= 0:
                reg[i] = int(j)
        reg[cfg.eos_token_id] = int(amp_tok.eos_token_id if amp_tok.eos_token_id is not None else 0)
        reg[cfg.pad_token_id] = int(amp_tok.pad_token_id if amp_tok.pad_token_id is not None else 0)
        reg[cfg.mask_token_id] = int(amp_tok.mask_token_id)
        self.register = reg.to(device)
        self.cls_id = int(amp_tok.cls_token_id if amp_tok.cls_token_id is not None
                          else amp_tok.bos_token_id or 0)
        self.eos_id = int(amp_tok.eos_token_id if amp_tok.eos_token_id is not None else 0)
        self.pad_id = int(amp_tok.pad_token_id if amp_tok.pad_token_id is not None else 0)
        self.mask_id = int(amp_tok.mask_token_id)
        # AMPLIFY ids of the 20 residues, in PLD2's own order, so a [.., 20] guidance tensor lines
        # up with logits[..., :20] without any reindexing.
        self.aa_ids = self.register[:20].clone()

    def to_amplify(self, canvas: torch.Tensor):
        """(B,L) PLD2 -> (ids (B,L+2), attn (B,L+2) 1/0, live (B,L) bool = a committed residue)."""
        B, L = canvas.shape
        cfg = self.cfg
        body = self.register[canvas]
        ids = torch.empty(B, L + 2, dtype=torch.long, device=canvas.device)
        ids[:, 0] = self.cls_id
        ids[:, 1:L + 1] = body
        ids[:, L + 1] = self.eos_id
        keep = (canvas != cfg.pad_token_id)                # PAD is outside the molecule
        attn = torch.ones(B, L + 2, dtype=torch.long, device=canvas.device)
        attn[:, 1:L + 1] = keep.long()
        live = keep & (canvas != cfg.mask_token_id) & (canvas != cfg.eos_token_id)
        return ids, attn, live


# ---------------------------------------------------------------------------
# prompts, straight out of the precomputed text cache
# ---------------------------------------------------------------------------
class PromptCache:
    """Per-token BioLinkBERT hidden states from mini-embed's packed cache.

    Guidance never runs the text encoder: the captions were encoded once by src/precompute.py and
    only the 768->16 projection head (which IS in the checkpoint) is needed to bring them into the
    shared space.
    """

    def __init__(self, cache_dir: str, text_dim: int, mods):
        self.cache = mods["data"].PackedPerTokenCache(cache_dir, "text", text_dim)
        self.ids = []
        for name in ("pair_ids.json", "text_ids.json"):
            p = os.path.join(cache_dir, name)
            if os.path.exists(p):
                self.ids = json.load(open(p))
                break

    def __len__(self):
        return len(self.cache)

    def rows_for(self, spec: Sequence) -> List[int]:
        """Accept row indices or accession strings."""
        out = []
        for s in spec:
            if isinstance(s, int) or (isinstance(s, str) and s.isdigit()):
                out.append(int(s))
            else:
                if s not in self.ids:
                    raise SystemExit(f"prompt id {s!r} is not in the cache "
                                     f"({len(self.ids)} ids; first few {self.ids[:3]})")
                out.append(self.ids.index(s))
        return out

    def encode(self, rows: Sequence[int], text_proj, device):
        """-> (z_t [M, Lt, D] projected & padded, mask_t [M, Lt] bool)."""
        hs, ms = [], []
        for r in rows:
            h, m = self.cache.get(int(r))
            hs.append(h)
            ms.append(m)
        Lt = max(h.shape[0] for h in hs)
        H = torch.zeros(len(hs), Lt, hs[0].shape[1], dtype=torch.float32)
        M = torch.zeros(len(hs), Lt, dtype=torch.bool)
        for i, (h, m) in enumerate(zip(hs, ms)):
            H[i, :h.shape[0]] = h.float()
            M[i, :m.shape[0]] = m
        with torch.no_grad():
            z = text_proj(H.to(device))
        return z, M.to(device)


# ---------------------------------------------------------------------------
# the guidance hook
# ---------------------------------------------------------------------------
class FilipGuidance:
    """A PLD2 `guidance_fn(canvas, logits) -> logits`.

    Holds the frozen AMPLIFY encoder, the FILIP projection heads, and the encoded target prompt.
    `mode="tag"` costs one AMPLIFY forward + backward per decoding step and scores every
    (position, residue) at once; `mode="deg"` is exact but is restricted to the `deg_positions`
    masked positions with the highest prior confidence, since it costs a classifier forward per
    candidate.
    """

    def __init__(self, cfg, device, *, ckpt=FILIP_CKPT, cache_dir=FILIP_CACHE,
                 repo=None, gamma: float = 1.0, mode: str = "tag",
                 likelihood: str = "sigmoid", tag_normalize: bool = True,
                 deg_positions: int = 4, deg_candidates: int = 8,
                 bank_rows: Optional[Sequence[int]] = None, verbose: bool = True):
        assert mode in ("tag", "deg"), mode
        assert likelihood in ("sigmoid", "softmax_bank"), likelihood
        mods = _mini_embed(repo)
        # default_cfg(), not a module-level CFG: mini-embed builds its config with a factory, and
        # build_retrieval falls back to `from config import default_cfg` when passed None -- which
        # from inside PLD2 would import PLD2's config. Always pass it explicitly.
        mcfg = mods["config"].default_cfg()
        self.mods, self.cfg, self.device = mods, cfg, device
        self.gamma, self.mode, self.likelihood = float(gamma), mode, likelihood
        self.tag_normalize = bool(tag_normalize)
        self.deg_positions, self.deg_candidates = int(deg_positions), int(deg_candidates)

        self.filip = mods["model"].load_retrieval(ckpt, device, mcfg, freeze=True)
        self.scale = float(self.filip.logit_scale.exp().item())
        self.amp, self.amp_tok = mods["encoders"].load_protein_encoder(
            mcfg.model.protein_encoder_path, device)
        for p in self.amp.parameters():
            p.requires_grad_(False)
        self.bridge = CanvasBridge(cfg, self.amp_tok, device)
        self.prompts = PromptCache(cache_dir, mcfg.model.text_hidden, mods)
        self._z_t = self._mask_t = None
        self._z_bank = self._mask_bank = None
        if bank_rows:
            self._z_bank, self._mask_bank = self.prompts.encode(
                bank_rows, self.filip.text_proj, device)

        # AMPLIFY's input embedding: TAG differentiates with respect to its OUTPUT, and the residue
        # rows of its weight are what a gradient is projected onto.
        emb = (self.amp.get_input_embeddings() if hasattr(self.amp, "get_input_embeddings")
               else next(m for m in self.amp.modules() if isinstance(m, torch.nn.Embedding)))
        if emb is None:
            raise SystemExit("could not locate AMPLIFY's input embedding; TAG needs it. "
                             "Use --guide-mode deg.")
        self.emb_module = emb
        self.E_aa = emb.weight.detach()[self.bridge.aa_ids]        # [20, d]
        self.calls = 0
        if verbose:
            print(f"[filip] ckpt={ckpt}\n[filip] encoder={mcfg.model.protein_encoder_path} "
                  f"| prompts cached: {len(self.prompts)} rows | mode={mode} "
                  f"likelihood={likelihood} gamma={gamma}", flush=True)

    # --- prompt ------------------------------------------------------------
    def set_target(self, row) -> str:
        r = self.prompts.rows_for([row])[0]
        self._z_t, self._mask_t = self.prompts.encode([r], self.filip.text_proj, self.device)
        name = self.prompts.ids[r] if r < len(self.prompts.ids) else str(r)
        return name

    # --- classifier --------------------------------------------------------
    def _log_prob(self, z_p, mask_p):
        """log p(target prompt | sequence) -> [B], differentiable."""
        fs = self.mods["losses"].filip_score_matrix
        s = fs(z_p, self._z_t, mask_p, self._mask_t).squeeze(1)          # [B]
        if self.likelihood == "sigmoid":
            return F.logsigmoid(self.scale * s)
        s_bank = fs(z_p, self._z_bank, mask_p, self._mask_bank)          # [B, M]
        logits = self.scale * torch.cat([s[:, None], s_bank], dim=1)
        return logits[:, 0] - torch.logsumexp(logits, dim=1)

    def _encode(self, ids, attn, live):
        additive = self.mods["encoders"]._amplify_additive_mask(attn)
        out = self.amp(input_ids=ids, attention_mask=additive, output_hidden_states=True)
        last = out.hidden_states[-1]
        if getattr(self.amp.config, "layer_norm_before_last_layer", False):
            last = self.amp.layer_norm_2(last)
        z_p = self.filip.protein_proj(last)
        # Only COMMITTED residues participate (filip_guide Decision #2): the score then means
        # "given what is placed so far", rather than being diluted by undecided positions.
        mask_p = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        mask_p[:, 1:live.shape[1] + 1] = live
        return z_p, mask_p

    # --- the hook ----------------------------------------------------------
    def __call__(self, canvas: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self._z_t is None:
            raise RuntimeError("call set_target(...) before generating")
        if self.gamma == 0.0:
            return logits
        self.calls += 1
        ids, attn, live = self.bridge.to_amplify(canvas)
        if not bool(live.any()):
            return logits            # nothing committed yet: the classifier has nothing to read
        delta = (self._tag(ids, attn, live) if self.mode == "tag"
                 else self._deg(canvas, logits, ids, attn, live))
        out = logits.clone()
        out[..., :20] = out[..., :20] + self.gamma * delta
        return out

    def _tag(self, ids, attn, live):
        """[B, L, 20] first-order guidance from one forward + backward."""
        L = live.shape[1]
        captured = {}

        def hook(_m, _i, out):
            out.requires_grad_(True)
            out.retain_grad()
            captured["e"] = out
            return out

        h = self.emb_module.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                z_p, mask_p = self._encode(ids, attn, live)
                self._log_prob(z_p, mask_p).sum().backward()
        finally:
            h.remove()
        g = captured["e"].grad
        if g is None:
            raise RuntimeError("no gradient reached AMPLIFY's embedding output; TAG cannot run")
        # <grad_i, E[c]>. The <grad_i, e_i> term of the Taylor expansion depends only on the
        # position, so it is constant across candidates and vanishes under the softmax -- dropped.
        delta = g[:, 1:L + 1, :].detach().float() @ self.E_aa.float().t()      # [B, L, 20]
        if self.tag_normalize:
            # The gradient's SCALE varies by orders of magnitude across a decode (it depends on how
            # many positions are committed and how peaked the max-sim is), so an unnormalised gamma
            # would mean something different at every step. Standardising across the 20 candidates
            # puts gamma in logit units, at the cost of discarding a magnitude that was never
            # calibrated anyway. Centring is free: a per-position constant cancels in the softmax.
            delta = (delta - delta.mean(-1, keepdim=True)) / delta.std(-1, keepdim=True).clamp_min(1e-6)
        return delta

    @torch.no_grad()
    def _deg(self, canvas, logits, ids, attn, live):
        """Exact enumeration, restricted to the positions actually in contention.

        Exact DEG would score every (position, residue); at 512 positions x 20 residues x 512 steps
        that is ~5M classifier forwards per sample. Only a handful of positions are candidates for
        commitment on any given step, so scoring the top `deg_positions` by prior confidence and the
        top `deg_candidates` residues at each recovers the part of the exact computation that can
        change the outcome, at deg_positions*deg_candidates forwards per step.
        """
        B, L = canvas.shape
        cfg = self.cfg
        delta = torch.zeros(B, L, 20, device=logits.device, dtype=torch.float32)
        masked = canvas == cfg.mask_token_id
        prior = torch.softmax(logits[..., :20].float(), dim=-1)
        conf = prior.max(-1).values.masked_fill(~masked, -1.0)
        P = min(self.deg_positions, int(masked.sum(1).max()))
        if P <= 0:
            return delta
        pos = conf.topk(P, dim=1).indices                                   # [B, P]
        C = min(self.deg_candidates, 20)
        for b in range(B):
            for p in pos[b].tolist():
                if not bool(masked[b, p]):
                    continue
                cand = prior[b, p].topk(C).indices                          # [C]
                rep = ids[b:b + 1].expand(C, -1).clone()
                rep[:, p + 1] = self.bridge.aa_ids[cand]
                lv = live[b:b + 1].expand(C, -1).clone()
                lv[:, p] = True
                z_p, mask_p = self._encode(rep, attn[b:b + 1].expand(C, -1), lv)
                delta[b, p, cand] = self._log_prob(z_p, mask_p).float()
        return delta
