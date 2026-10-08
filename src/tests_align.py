"""Invariants for the preference-tuning pipeline:  python -m src.tests_align

Six things, each of which has a specific way of being silently wrong:

  1. PROMPT FREEZING. Given positions come back bit-identical and the length is the prompt's. A
     prompt the decoder is free to edit is an initialisation, not a prompt -- and because
     substitution makes every committed residue a standing candidate, that is the DEFAULT failure
     here, not an exotic one.
  2. NO REGRESSION. generate() with prompt=None is bit-identical to the unprompted path. The last
     time this file's sibling caught something, a slice indexed the new track axis and blanked every
     logit at one position; nothing downstream noticed until the softmax produced NaN.
  3. SHARED MASKS. The same (pair_id, epoch) gives the same mask, a different epoch gives a
     different one, and only generated positions are ever scored. All four log-likelihoods in the
     contrastive term are estimates over masks; if the draws stop being shared the noise stops
     cancelling and the margins we are chasing are smaller than that noise.
  4. THE SURROGATE. surrogate_logp equals a hand-computed masked mean log-prob, and is a MEAN (so
     the IPO margin is per-token and length-invariant, which is the whole reason it is legible).
  5. IPO STOPS, DPO DOES NOT. The gradient of the IPO term vanishes at h = margin and the DPO term's
     never does. This is the property the default rests on.
  6. PAIRING. Winners beat losers on the reward; matched pairs are matched on pLDDT and split on TM;
     pass_at_k and selector_at_k agree with brute force.
"""
import glob as _glob
import itertools
import math
import random

from config import CFG as _CFG

CFG_ALIGN = _CFG.align

import numpy as np
import torch

import src.sampler as S
from src.model import LoopedDiffusionLM, Config
from src.objective import surrogate_logp, surrogate_mask
from src.preference import matched_pairs, pass_at_k, rank_pairs, selector_at_k
from src.prompts import _hex, _unhex

fails, checks = [], 0


def check(name, ok, extra=""):
    global checks
    checks += 1
    if not ok:
        fails.append(f"{name}{(' -- ' + extra) if extra else ''}")
        print(f"  FAIL {name} {extra}")


cfg = Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
             d_model=64, n_heads=4, d_ff=192, n_upstream=1, n_middle=2, n_downstream=1,
             n_recurrence=1, grad_checkpoint=False, n_tracks=2)
torch.manual_seed(0)
m = LoopedDiffusionLM(cfg).eval()
L, B, CANVAS = 40, 4, 64

# ---------------------------------------------------------------- 1. prompt freezing
prompt = torch.full((B, 2, CANVAS), cfg.pad_token_id, dtype=torch.long)
prompt[:, 0, :L] = torch.randint(0, 20, (B, L))
prompt[:, 0, L] = cfg.eos_token_id
prompt[:, 1, :L] = torch.randint(0, 20, (B, L))
given = torch.zeros((B, 2, CANVAS), dtype=torch.bool)
given[:, :, L:] = True                                    # EOS + PAD tail
rng = torch.Generator().manual_seed(7)
reveal = torch.rand((B, 1, L), generator=rng) < 0.5
given[:, :, :L] = reveal.expand(-1, 2, -1)

for subst, corr in itertools.product((0.0, 1.0), ((0, "remask"), (2, "remask"),
                                                  (2, "substitution"))):
    torch.manual_seed(3)
    cv, lens = S.generate(m, Lmax=CANVAS, batch_size=B, n_steps=CANVAS, device="cpu",
                          min_len=5, subst_per_residue=subst,
                          n_corrector=corr[0], corrector_type=corr[1],
                          rep_penalty=0.0, max_run=0,
                          prompt=prompt, prompt_mask=given)
    tag = f"subst={subst} corr={corr}"
    check(f"prompt preserved ({tag})", bool((cv[given] == prompt[given]).all()))
    check(f"prompt length ({tag})", lens == [L] * B, f"got {lens}")
    check(f"no MASK survives ({tag})", bool((cv != cfg.mask_token_id).all()))
    check(f"PAD tail intact ({tag})", bool((cv[:, :, L + 1:] == cfg.pad_token_id).all()))
    check(f"structure track padded from the boundary ({tag})",
          bool((cv[:, 1, L:] == cfg.pad_token_id).all()))

# a 1-track prompt on a 2-track model must not leak residues into the structure track
torch.manual_seed(3)
cv1, _ = S.generate(m, Lmax=CANVAS, batch_size=B, n_steps=CANVAS, device="cpu", min_len=5,
                    rep_penalty=0.0, max_run=0, prompt=prompt[:, 0], prompt_mask=given[:, 0])
check("1-track prompt leaves 3Di free",
      not bool((cv1[:, 1, :L] == prompt[:, 1, :L]).all()))

# ---------------------------------------------------------------- 2. no regression
torch.manual_seed(11)
a, la = S.generate(m, Lmax=CANVAS, batch_size=B, n_steps=CANVAS, device="cpu", min_len=5)
torch.manual_seed(11)
b, lb = S.generate(m, Lmax=CANVAS, batch_size=B, n_steps=CANVAS, device="cpu", min_len=5,
                   prompt=None, prompt_mask=None)
check("unprompted path unchanged", bool(torch.equal(a, b)) and la == lb)

# ---------------------------------------------------------------- 3. shared masks
gen_pos = torch.zeros(CANVAS, dtype=torch.bool)
gen_pos[:L] = reveal[0, 0] == False                       # the positions actually generated
m1 = surrogate_mask("p1:a:b:rank", gen_pos, CANVAS, torch.device("cpu"), epoch=0)
m2 = surrogate_mask("p1:a:b:rank", gen_pos, CANVAS, torch.device("cpu"), epoch=0)
m3 = surrogate_mask("p1:a:b:rank", gen_pos, CANVAS, torch.device("cpu"), epoch=1)
m4 = surrogate_mask("p2:a:b:rank", gen_pos, CANVAS, torch.device("cpu"), epoch=0)
check("mask is a pure function of (pair_id, epoch)", bool(torch.equal(m1, m2)))
check("mask changes with the epoch", not bool(torch.equal(m1, m3)))
check("mask changes with the pair", not bool(torch.equal(m1, m4)))
check("mask never scores context", bool((m1 & ~gen_pos).sum() == 0))
check("mask is never empty",
      all(bool(surrogate_mask(f"q{i}", gen_pos, CANVAS, torch.device("cpu")).any())
          for i in range(50)))

# ---------------------------------------------------------------- 4. the surrogate
y = prompt.clone()
y[:, 0, :L] = torch.randint(0, 20, (B, L))
y[:, 1, :L] = torch.randint(0, 20, (B, L))
mk = torch.zeros((B, CANVAS), dtype=torch.bool)
mk[:, :L] = torch.rand((B, L), generator=torch.Generator().manual_seed(5)) < 0.4
mk[:, 0] = True                                           # no empty row
with torch.no_grad():
    got = surrogate_logp(m, y, mk, cfg)
    xt = torch.where(mk.unsqueeze(1).expand(-1, 2, -1),
                     torch.full_like(y, cfg.mask_token_id), y)
    lg = m(xt[:, 0], struct=xt[:, 1])
    lp = torch.log_softmax(lg.float(), dim=-1).gather(-1, y.unsqueeze(-1)).squeeze(-1)[:, 0]
    want = (lp * mk).sum(1) / mk.sum(1)
check("surrogate_logp == masked mean log-prob", torch.allclose(got, want, atol=1e-5),
      f"max |diff| {float((got - want).abs().max()):.2e}")
check("surrogate_logp is negative", bool((got < 0).all()))
# A MEAN, not a sum: doubling the scored positions must not double the magnitude.
mk2 = mk.clone()
mk2[:, :L] = True
with torch.no_grad():
    g2 = surrogate_logp(m, y, mk2, cfg)
check("surrogate_logp is length-invariant (a mean)",
      float((g2 / got).abs().max()) < 3.0, f"ratio {float((g2 / got).max()):.2f}")

# ---------------------------------------------------------------- 5. IPO stops, DPO does not
class _A:
    ipo_margin, beta = 0.04, 0.05


from src.align import contrastive                                     # noqa: E402
for h0, near in ((0.04, True), (0.5, False), (2.0, False)):
    h = torch.tensor([h0], requires_grad=True)
    contrastive(h, _A, "ipo").sum().backward()
    g = float(h.grad)
    check(f"IPO gradient at h={h0} {'vanishes' if near else 'does not'}",
          (abs(g) < 1e-6) == near, f"grad {g:.3e}")
    h = torch.tensor([h0], requires_grad=True)
    contrastive(h, _A, "dpo").sum().backward()
    check(f"DPO gradient at h={h0} never vanishes", abs(float(h.grad)) > 1e-6)
check("IPO is minimised at the margin",
      float(contrastive(torch.tensor([0.04]), _A, "ipo")) <
      min(float(contrastive(torch.tensor([x]), _A, "ipo")) for x in (-0.5, 0.0, 0.5, 2.0)))

# ---------------------------------------------------------------- 6. pairing
class _P:
    reward_plddt = reward_tm = reward_struct = 1.0
    reward_blend = True
    ptm_success = 0.5
    reward_deg = 0.0                  # this section tests the rank/matched geometry alone
    deg_max_winner, deg_kmer_k = 1.0, 13
    plddt_success, tm_success = 0.7, 0.5
    top_k = bot_k = 2
    min_gap = 0.0
    matched_plddt_tol = 0.05


rnd = random.Random(0)
pool = [{"gid": f"g{i}", "plddt": rnd.random(), "tm": rnd.random()} for i in range(16)]
rp = rank_pairs(pool, _P)
check("rank pairs: winner beats loser",
      all(w["plddt"] + w["tm"] > l["plddt"] + l["tm"] for w, l, _, _ in rp))
check("rank pairs: top_k x bot_k", len(rp) == _P.top_k * _P.bot_k, f"got {len(rp)}")
mp = matched_pairs(pool, _P, limit=5)
check("matched pairs: matched on pLDDT",
      all(abs(w["plddt"] - l["plddt"]) <= _P.matched_plddt_tol for w, l, _, _ in mp))
check("matched pairs: split on TM", all(w["tm"] > l["tm"] for w, l, _, _ in mp))

# pass@k and selector@k against brute force over every k-subset
n, k = 7, 3
succ = [True, False, True, False, False, False, True]
idx = list(range(n))
brute_pass = np.mean([any(succ[i] for i in c) for c in itertools.combinations(idx, k)])
brute_sel = np.mean([succ[min(c)] for c in itertools.combinations(idx, k)])   # rank 0 = best
check("pass_at_k matches brute force",
      abs(pass_at_k(n, sum(succ), k) - brute_pass) < 1e-9,
      f"{pass_at_k(n, sum(succ), k):.6f} vs {brute_pass:.6f}")
check("selector_at_k matches brute force",
      abs(selector_at_k(succ, k) - brute_sel) < 1e-9,
      f"{selector_at_k(succ, k):.6f} vs {brute_sel:.6f}")
check("selector_at_k <= pass_at_k (a selector cannot beat an oracle)",
      all(selector_at_k(succ, kk) <= pass_at_k(n, sum(succ), kk) + 1e-9 for kk in range(1, n + 1)))

# hex round trip -- the manifest's only lossy-looking field
for L_ in (1, 7, 40, 512):
    bits = np.random.default_rng(L_).random(L_) < 0.5
    check(f"mask hex round-trips at L={L_}", bool((_unhex(_hex(bits), L_) == bits).all()))


# ------------------------------------------------- 7. the loss arithmetic at h = 0
# At step 0 the reference IS the policy, so h is identically zero and each loss collapses to a
# number that can be written down. If the four-likelihood bookkeeping is wrong -- a swapped winner
# and loser, a reference forward that saw a different mask, a sign error -- h is not zero at step 0
# and this is where it shows. It is the cheapest end-to-end check on the whole contrastive term.
_A.nll_weight = 1.0
h0 = torch.zeros(1)
# float32 tensors, so the tolerance is the dtype's, not the arithmetic's.
check("IPO at h=0 is (0 - margin)^2",
      abs(float(contrastive(h0, _A, "ipo")) - _A.ipo_margin ** 2) < 1e-9,
      f"{float(contrastive(h0, _A, 'ipo')):.12f}")
check("DPO at h=0 is log 2",
      abs(float(contrastive(h0, _A, "dpo")) - math.log(2)) < 1e-6,
      f"{float(contrastive(h0, _A, 'dpo')):.12f}")

# ------------------------------------------------- 8. the prompt pipeline, end to end
from src.prompts import build as build_prompts, materialize                       # noqa: E402

refs = []
_r = random.Random(4)
for i in range(12):
    n_ = _r.randint(30, 50)
    refs.append({"rid": f"r{i}", "row": 100 + i, "acc": f"P{i}",
                 "seq": "".join(_r.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n_)),
                 "di": "".join(_r.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(n_)),
                 "plddt": 0.8})
rows = build_prompts(6, refs, seed=1, canvas=CANVAS, min_len=20)
check("build is deterministic in its seed",
      [r["pid"] for r in rows] == [r["pid"] for r in build_prompts(6, refs, seed=1,
                                                                  canvas=CANVAS, min_len=20)])
check("every prompt carries its caption row", all(isinstance(r["caption"], int) for r in rows))
check("cold-start share is at least a third",
      sum(r["rate"] >= 1.0 for r in rows) / len(rows) >= 0.25,
      f"{sum(r['rate'] >= 1.0 for r in rows)}/{len(rows)}")

ok_ctx = ok_free = ok_pad = True
for r in rows:
    tok, given = materialize(r, cfg, CANVAS, n_tracks=2)
    L_, mk_ = r["L"], _unhex(r["masked"], r["L"])
    ok_ctx &= bool((given[0, :L_].numpy() == ~mk_).all())
    ok_free &= not bool(given[0, :L_][torch.from_numpy(mk_)].any())
    ok_pad &= bool(given[:, L_:].all()) and int(tok[0, L_]) == cfg.eos_token_id
check("materialize: given == the revealed residues", ok_ctx)
check("materialize: masked positions are free", ok_free)
check("materialize: EOS and the whole PAD tail are context", ok_pad)

# and the whole way through generate(), on prompts built by the real builder
torch.manual_seed(21)
r = rows[0]
tok, given = materialize(r, cfg, CANVAS, n_tracks=2)
cv, lens = S.generate(m, Lmax=CANVAS, batch_size=3, n_steps=CANVAS, device="cpu", min_len=5,
                      subst_per_residue=1.0, n_corrector=2, rep_penalty=0.0, max_run=0,
                      prompt=tok.unsqueeze(0).expand(3, -1, -1).contiguous(),
                      prompt_mask=given.unsqueeze(0).expand(3, -1, -1).contiguous())
g3 = given.unsqueeze(0).expand(3, -1, -1)
check("real prompt survives a full decode", bool((cv[g3] == tok.unsqueeze(0).expand(3, -1, -1)[g3]).all()))
check("real prompt fixes the length", lens == [r["L"]] * 3, f"got {lens} want {r['L']}")


# ------------------------------------------------- 9. reference_set join, on a faithful fixture
# This phase reads three files written by three different tools (fold_fasta's PDB index, its
# results JSONL, and foldseek's descriptor TSV) and has to agree with all of them. It failed in
# production on `di = parse_descriptor(...)` -- which returns a PAIR -- after phase 0 had already
# folded 1,200 references. A fixture is cheap; a queue slot is not.
import json as _json                                                              # noqa: E402
import os as _os                                                                  # noqa: E402
import tempfile                                                                   # noqa: E402
import types                                                                      # noqa: E402

from src.reference_set import cmd_join                                            # noqa: E402

_tmp = tempfile.mkdtemp(prefix="pld2refs_")
_os.makedirs(_os.path.join(_tmp, "refpdb"))
_rr = random.Random(7)
_meta, _idx, _folds, _tsv = [], [], [], []
for _i in range(10):
    _rid, _L = f"r{1000 + _i}", _rr.randint(40, 90)
    _seq = "".join(_rr.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(_L))
    _meta.append({"rid": _rid, "row": 1000 + _i, "acc": f"P{_i}", "seq": _seq})
    if _i == 9:                                   # folded never landed for this one
        continue
    _safe = f"refs_{_rid}"                        # fold_fasta._safe_name flattens '|' to '_'
    _idx.append({"file": _safe, "id": f"refs|{_rid}"})
    _folds.append({"id": f"refs|{_rid}", "length": _L, "seq": _seq,
                   "plddt": round(_rr.uniform(0.5, 0.95), 4), "ptm": 0.7})
    open(_os.path.join(_tmp, "refpdb", _safe + ".pdb"), "w").write("ATOM\n")
    _dl = _L if _i != 8 else _L - 3               # one 3Di that does not cover its sequence
    _tsv.append(f"{_safe}.pdb\t{_seq}\t"
                + "".join(_rr.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(_dl)) + "\tfeat")
_tsv.append("orphan_structure.pdb\tAAAA\tDDDD\tfeat")     # typed, but not in the index
for _name, _rows in (("refs.meta.jsonl", _meta), ("reffolds.jsonl", _folds)):
    open(_os.path.join(_tmp, _name), "w").write(
        "".join(_json.dumps(r) + "\n" for r in _rows))
open(_os.path.join(_tmp, "refpdb", "index.rank000.jsonl"), "w").write(
    "".join(_json.dumps(r) + "\n" for r in _idx))
open(_os.path.join(_tmp, "refs.3di.tsv"), "w").write("\n".join(_tsv) + "\n")

cmd_join(types.SimpleNamespace(dir=_tmp, pdb_dir=None, folds=None, foldseek="foldseek",
                               threads=0, refresh=False))
_out = [_json.loads(l) for l in open(_os.path.join(_tmp, "refs.jsonl"))]
check("reference join drops the unfolded and the length-mismatched",
      len(_out) == 8, f"got {len(_out)} want 8")
check("every reference's 3Di covers its sequence",
      all(len(r["di"]) == len(r["seq"]) for r in _out))
check("every reference carries its caption row and scores",
      all({"rid", "row", "acc", "seq", "di", "plddt", "ptm"} <= set(r) for r in _out))

# and the contract that broke: the callee returns a pair, so the call site must unpack one
from src.self_consistency import parse_descriptor                                 # noqa: E402
_r = parse_descriptor(_os.path.join(_tmp, "refs.3di.tsv"),
                      {r["file"]: r["id"] for r in _idx})
check("parse_descriptor returns (mapping, n_unmatched)",
      isinstance(_r, tuple) and len(_r) == 2 and isinstance(_r[0], dict) and _r[1] == 1,
      f"got {type(_r).__name__}")


# ------------------------------------------------- 10. tuner scale and loss scale
from src.align import loss_scale, recommended_ranks                               # noqa: E402
from src.align_compare import best_of, sign_test                                  # noqa: E402
from src.prompts import split_refs                                                # noqa: E402


class _S:
    nll_weight, ipo_margin, beta = 1.0, 0.04, 10.0
    alpha = 25.0


# The equilibrium derivation is the whole reason round 1 was an SFT run wearing an IPO label:
#     dL/dlp_w = -w_nll + 2*alpha*(h - m) = 0  =>  h* = m + w_nll/(2*alpha)
# At ESM3's alpha=0.8 that is 0.665 against a margin of 0.04, which is what the log showed.
check("IPO equilibrium is m + w/(2a)", abs(loss_scale(_S, "ipo")[1] - (0.04 + 1 / 50)) < 1e-12)
_S.alpha = 0.8
check("ESM3's alpha reproduces round 1's h*", abs(loss_scale(_S, "ipo")[1] - 0.665) < 1e-9,
      f"{loss_scale(_S, 'ipo')[1]:.6f}")
_S.alpha = 0.0
check("alpha=0 has no equilibrium (it is SFT)", loss_scale(_S, "ipo")[1] != loss_scale(_S, "ipo")[1])
check("DPO reports its saturation scale 1/beta", abs(loss_scale(_S, "dpo")[1] - 0.1) < 1e-12)
# and the IPO gradient really does vanish there, which is what "equilibrium" has to mean
_S.alpha = 25.0
_hq = torch.tensor([loss_scale(_S, "ipo")[1]], requires_grad=True)
(_S.nll_weight * (-_hq) + _S.alpha * contrastive(_hq, _S, "ipo")).sum().backward()
check("the two terms' gradients cancel at h*", abs(float(_hq.grad)) < 1e-5,
      f"net grad {float(_hq.grad):.3e}")

for n_pairs, steps, eps in ((5680, 500, 2.0), (170000, 500, 2.0), (83, 6, 2.0)):
    r = recommended_ranks(n_pairs, steps, eps)
    check(f"recommended_ranks({n_pairs},{steps}) hits ~{eps} epochs",
          abs(steps * r / n_pairs - eps) < max(0.5 * eps, 12 * steps / n_pairs),
          f"{r} ranks -> {steps * r / n_pairs:.2f} epochs")
    check(f"recommended_ranks({n_pairs}) is whole nodes", r % 12 == 0 and r >= 12, f"{r}")
# round 1, diagnosed: 192 ranks x 1000 steps over 5,680 pairs is 33.8 epochs
check("round 1's configuration reads as ~34 epochs",
      abs(1000 * 192 / 5680 - 33.8) < 0.1)

check("sign test is 1.0 on an even split", abs(sign_test(5, 5) - 1.0) < 1e-12)
check("sign test is symmetric", sign_test(9, 1) == sign_test(1, 9))
check("sign test matches the exact binomial",
      abs(sign_test(9, 1) - 2 * (math.comb(10, 0) + math.comb(10, 1)) / 2 ** 10) < 1e-12)
check("sign test is nan when nothing differs", sign_test(0, 0) != sign_test(0, 0))


class _R:
    reward_plddt = reward_tm = reward_struct = 1.0
    reward_blend = True
    ptm_success = 0.5
    reward_deg = 0.0


_pool = [{"plddt": v, "tm": 0.0, "loglik": -v} for v in (0.1, 0.4, 0.5, 0.9)]
check("best_of at k=1 is the mean reward",
      abs(best_of(_pool, _R, 1, "plddt") - 0.475) < 1e-12)
check("best_of at k=n is the max reward",
      abs(best_of(_pool, _R, 4, "plddt") - 0.9) < 1e-12)
check("best_of is monotone in k",
      all(best_of(_pool, _R, k, "plddt") <= best_of(_pool, _R, k + 1, "plddt") + 1e-12
          for k in range(1, 4)))
# a selector anti-correlated with reward must do WORSE than one draw -- the loglik pathology
check("an anti-correlated selector is worse than random",
      best_of(_pool, _R, 4, "loglik") < best_of(_pool, _R, 1, "loglik"),
      f"{best_of(_pool, _R, 4, 'loglik'):.3f} vs {best_of(_pool, _R, 1, 'loglik'):.3f}")

_refs = [{"rid": f"r{i}", "acc": f"A{i}"} for i in range(50)]
_tr, _ev = split_refs(_refs, 10, seed=3)
check("holdout is disjoint", not ({r["rid"] for r in _tr} & {r["rid"] for r in _ev}))
check("holdout is exhaustive", len(_tr) + len(_ev) == 50 and len(_ev) == 10)
check("holdout is deterministic",
      [r["rid"] for r in split_refs(_refs, 10, seed=3)[1]] == [r["rid"] for r in _ev])
check("holdout moves with the seed",
      [r["rid"] for r in split_refs(_refs, 10, seed=4)[1]] != [r["rid"] for r in _ev])


# ------------------------------------------------- 11. degeneracy detectors and the per-bin split
from src.align_compare import degeneracy, paired_delta, split_by_bin                 # noqa: E402

_rg = random.Random(0)
_rand = ["".join(_rg.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(250)) for _ in range(20)]
_poly = ["A" * 250 for _ in range(20)]
_ag = [("A" * 10 + "G" * 10) * 13 for _ in range(20)]
_20mer = ["".join(_rg.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(20)) * 13 for _ in range(20)]
_l_rand, _k_rand = degeneracy(_rand, (13,))
_l_poly, _k_poly = degeneracy(_poly, (13,))
_l_ag, _k_ag = degeneracy(_ag, (13,))
_l_20, _k_20 = degeneracy(_20mer, (13,))
check("LCR is ~0 on random sequence", _l_rand < 0.05, f"{_l_rand:.1%}")
check("LCR is 1.0 on poly-A", _l_poly > 0.99, f"{_l_poly:.1%}")
check("k13 is ~0 on random sequence", _k_rand[13] < 0.01, f"{_k_rand[13]:.1%}")
check("k13 is 1.0 on poly-A", _k_poly[13] > 0.99)
check("both fire on a 10+10 block repeat", _l_ag > 0.99 and _k_ag[13] > 0.99)
# THE REASON BOTH COLUMNS EXIST: a repeated 20-mer is longer than the SEG window, so LCR cannot
# see it at all, while k13 reads it at 100%. Either one alone would call this sequence clean.
check("a repeated 20-mer is INVISIBLE to LCR", _l_20 < 0.05, f"{_l_20:.1%}")
check("...and obvious to k13", _k_20[13] > 0.99, f"{_k_20[13]:.1%}")

_pool = {f"p{i}": [{"rate": (0.5, 1.0)[i % 2], "plddt": 0.5, "tm": 0.4, "loglik": -1.0,
                    "seq": "ACDE" * 10} for _ in range(4)] for i in range(10)}
_sp = split_by_bin(_pool)
check("per-bin split covers every prompt",
      sum(len(v) for v in _sp.values()) == len(_pool) and set(_sp) == {0.5, 1.0})
check("per-bin split assigns a prompt to exactly one bin",
      not (set(_sp[0.5]) & set(_sp[1.0])))

_A = {"_per_prompt": {"a": 1.0, "b": 2.0, "c": 3.0}}
_B = {"_per_prompt": {"a": 0.5, "b": 2.5, "c": 1.0}}
_d, _nb, _nw, _p = paired_delta(_A, _B)
check("paired_delta counts better/worse", (_nb, _nw) == (2, 1), f"{(_nb, _nw)}")
check("paired_delta mean is the mean difference", abs(_d - (0.5 - 0.5 + 2.0) / 3) < 1e-12)
check("paired_delta only uses shared prompts",
      paired_delta(_A, {"_per_prompt": {"a": 0.0}})[1:3] == (1, 0))


# ------------------------------------------------- 12. degeneracy on the loser side
# Round 1's pairs, at a mask rate of 1.0, promoted winners carrying +12.6 points MORE LCR than
# their losers -- 69% of pairs preferred the more repetitive side. The reward there was effectively
# pLDDT alone (TM's spread was sd 0.061 against pLDDT's 0.148) and r(LCR, pLDDT) = +0.277, so the
# ranking WAS the confound. These are the three things that had to change.
from src.preference import (build_pairs, clean_pairs, degeneracy_of, eligible_winner,   # noqa: E402
                            rank_pairs, score)


class _D:
    reward_plddt = reward_tm = reward_struct = 1.0
    reward_blend = True
    ptm_success = 0.5
    reward_deg = 0.5
    deg_max_winner = 0.15
    deg_kmer_k = 13
    top_k = bot_k = 2
    min_gap = 0.0
    matched_frac = 0.0
    clean_frac = 1.0
    clean_min_gap = 0.10
    matched_plddt_tol = 0.05
    plddt_success, tm_success = 0.7, 0.5
    success_weight = 1.0


# Hoist the generator: `random.Random(seed).choice(...)` inside a comprehension reseeds on every
# iteration and yields a homopolymer, which made this check fail against perfectly good code.
_dr = random.Random(2)
_real = "".join(_dr.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(250))
_rep20 = "".join(_dr.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(20)) * 13
check("degeneracy_of is ~0 on real-looking sequence", max(degeneracy_of(_real)) < 0.05,
      f"{degeneracy_of(_real)}")
check("degeneracy_of is 1.0 on poly-A", min(degeneracy_of("A" * 250)) > 0.99)
check("degeneracy_of catches a repeat LCR cannot see",
      degeneracy_of(_rep20)[1] > 0.99 and degeneracy_of(_rep20)[0] < 0.05,
      f"{degeneracy_of(_rep20)}")

# the exact shape of the round 1 failure: the degenerate sample has the best pLDDT
_deg = {"gid": "d", "plddt": 0.90, "tm": 0.20, "ptm": 0.30, "deg": 0.60, "seq": "A" * 200}
_ok = {"gid": "c", "plddt": 0.70, "tm": 0.30, "ptm": 0.60, "deg": 0.02, "seq": "ACDE" * 50}
_mid = {"gid": "m", "plddt": 0.68, "tm": 0.25, "ptm": 0.55, "deg": 0.05, "seq": "ACDF" * 50}
_bad = {"gid": "b", "plddt": 0.30, "tm": 0.10, "ptm": 0.20, "deg": 0.03, "seq": "ACDG" * 50}
check("the degeneracy gate refuses a repetitive winner", not eligible_winner(_deg, _D))
check("...and admits a clean one", eligible_winner(_ok, _D))
_D.reward_deg = 0.0
check("without the penalty the degenerate sample ranks FIRST",
      score(_deg, _D) > score(_ok, _D), "round 1's reward")
_D.reward_deg = 0.5
check("with it, the clean sample ranks first", score(_ok, _D) > score(_deg, _D))

_ss = [_deg, _ok, _mid, _bad]
_rp = rank_pairs(_ss, _D)
check("no rank pair promotes a gated sample",
      all(w["gid"] != "d" for w, _, _, _ in _rp), f"{[w['gid'] for w, _, _, _ in _rp]}")
check("the gated sample can still be a LOSER",
      any(l["gid"] == "d" for _, l, _, _ in _rp))

_cp = clean_pairs([_deg, {"gid": "x", "plddt": 0.88, "tm": 0.2, "deg": 0.01, "seq": "A"}], _D, 5)
check("clean pairs put the repetitive side second",
      len(_cp) == 1 and _cp[0][0]["gid"] == "x" and _cp[0][1]["gid"] == "d")
check("clean pairs are matched on pLDDT",
      all(abs(w["plddt"] - l["plddt"]) <= _D.matched_plddt_tol for w, l, _, _ in _cp))
check("clean pairs need a real degeneracy split",
      clean_pairs([_ok, _mid], _D, 5) == [])

_meta = {"ref_len": 200, "bin": 3, "rate": 1.0, "di": None}
_pool = {"p0": [dict(_deg, gid="d0", **_meta), dict(_ok, gid="c0", **_meta),
                dict(_mid, gid="m0", **_meta), dict(_bad, gid="b0", **_meta)]}
_pairs, _ns, _ng = build_pairs(_pool, _D)
check("build_pairs never promotes a gated sample",
      all(p["w"]["deg"] <= _D.deg_max_winner for p in _pairs))
check("build_pairs reports pairs, winner-successes and gated prompts",
      isinstance(_ns, int) and isinstance(_ng, int))
# The winner-success count drives success_weight, so it has to track the BAR and not just be an
# int. Under the old pLDDT-and-TM bar this was 0 for every cold-start pool, which is why the knob
# was never usable; a clean winner clearing pLDDT and pTM must now register.
_win = {"gid": "w", "plddt": 0.85, "tm": 0.05, "ptm": 0.70, "deg": 0.01, "seq": "ACDE" * 50}
_lose = {"gid": "l", "plddt": 0.40, "tm": 0.05, "ptm": 0.20, "deg": 0.01, "seq": "ACDF" * 50}
_, _ns_hi, _ = build_pairs({"q0": [dict(_win, **_meta), dict(_lose, **_meta)]}, _D)
check("a winner over the reference-free bar counts as a success", _ns_hi == 1, f"{_ns_hi}")
_, _ns_lo, _ = build_pairs(
    {"q1": [dict(_win, ptm=0.40, **_meta), dict(_lose, **_meta)]}, _D)
check("...and one under it does not, even with TM irrelevant", _ns_lo == 0, f"{_ns_lo}")
_allgated, _, _ng2 = build_pairs(
    {"p1": [dict(_deg, gid="z1", **_meta), dict(_deg, gid="z2", **_meta)]}, _D)
check("a prompt with nothing promotable is dropped, not forced",
      _allgated == [] and _ng2 == 1)


# ------------------------------------------------- 13. reference sizing
# A 1,500-prompt round drew 1,800 references on a flat 1.2x multiplier and died in phase 0d, AFTER
# folding every one of them: 267 were below ref_min_plddt and 200 went to the eval holdout, leaving
# 1,333. A reference yields at most one prompt, so the requirement is arithmetic, not a guess.
from src.reference_set import refs_needed, select                                 # noqa: E402

check("refs_needed covers the holdout and the drop rate",
      refs_needed(1500, n_eval=200, usable_frac=0.85, headroom=1.05) == 2100,
      f"{refs_needed(1500, 200, 0.85, 1.05)}")
check("refs_needed is enough for what broke",
      refs_needed(1500, 200, 0.85, 1.05) * 0.85 - 200 >= 1500)
check("the old 1.2x multiplier was NOT",
      int(1500 * 1.2) * 0.85 - 200 < 1500)
check("refs_needed grows with the holdout",
      refs_needed(1000, 400, 0.85, 1.0) > refs_needed(1000, 100, 0.85, 1.0))

# The recovery advice in the failure message is load-bearing -- raising the count must REUSE the
# folds already paid for, or the suggestion costs a second full fold campaign.
_rows = [{"acc": f"P{i}", "seq": "ACDEFGHIKLMNPQRSTVWY" * ((i % 20) + 3)} for i in range(3000)]
_a = [r["acc"] for r in select(_rows, 900, seed=1, min_len=40, max_len=511)]
_b = [r["acc"] for r in select(_rows, 1200, seed=1, min_len=40, max_len=511)]
check("raising --n at the same seed extends the draw", _b[:len(_a)] == _a)
check("...so only the new references cost anything", len(set(_b) - set(_a)) == 300)
check("a different seed does NOT extend it",
      [r["acc"] for r in select(_rows, 1200, seed=2, min_len=40, max_len=511)][:len(_a)] != _a)


# ------------------------------------------------- 14. merged reference dirs, and pass@k depth
# Both of these cost a whole job. align_test scored four variants against an EMPTY evalref/refpdb
# because align.pbs had started writing the eval holdout out of the MAIN reference set, and every
# TM row came back missing. And a single prompt whose folds mostly failed collapsed the pass@k
# table to k=1, because its depth was min(samples per prompt).
import tempfile as _tf                                                            # noqa: E402

from src.tm_align import pdb_paths                                                # noqa: E402

_d = _tf.mkdtemp(prefix="pld2ref_")
_os.makedirs(_os.path.join(_d, "a", "rank000"))
_os.makedirs(_os.path.join(_d, "b"))
# one directory sharded (as --pdb-shard writes it), one flat: both layouts, merged
open(_os.path.join(_d, "a", "rank000", "index.jsonl"), "w").write(
    _json.dumps({"file": "rank000/refs_r1", "id": "refs|r1"}) + "\n")
open(_os.path.join(_d, "a", "rank000", "refs_r1.pdb"), "w").write("ATOM\n")
open(_os.path.join(_d, "b", "index.rank000.jsonl"), "w").write(
    _json.dumps({"file": "refs_r2", "id": "refs|r2"}) + "\n")
open(_os.path.join(_d, "b", "refs_r2.pdb"), "w").write("ATOM\n")

_merged = pdb_paths(f"{_os.path.join(_d, 'a')}:{_os.path.join(_d, 'b')}", "test")
check("reference lookup merges several directories", set(_merged) == {"r1", "r2"}, f"{set(_merged)}")
check("...across the sharded and flat layouts alike",
      _merged["r1"].endswith("rank000/refs_r1.pdb") and _merged["r2"].endswith("b/refs_r2.pdb"))
check("a single directory still works", set(pdb_paths(_os.path.join(_d, "a"), "t")) == {"r1"})
try:
    pdb_paths(_os.path.join(_d, "b") + ":" + _os.path.join(_d, "nonexistent"), "t")
    _ok = True
except SystemExit:
    _ok = False
check("one empty directory in the list is not fatal", _ok)
try:
    pdb_paths(_os.path.join(_d, "nonexistent"), "t")
    check("an empty lookup raises", False)
except SystemExit as _e:
    check("an empty lookup raises and names the directories searched", "nonexistent" in str(_e))

# pass@k depth: one starved prompt must not collapse the table
from src.preference import report as _report                                      # noqa: E402
import io as _io, contextlib as _ctx                                              # noqa: E402

_rr = random.Random(5)
_pool = {}
for _i in range(60):
    _n = 1 if _i == 0 else 16                    # exactly the shape that broke: one prompt, 1 fold
    _pool[f"p{_i}"] = [{"plddt": _rr.uniform(.2, .95), "tm": _rr.uniform(0, .9), "ptm": 0.5,
                        "loglik": -_rr.uniform(1, 4), "deg": 0.01, "rate": 1.0,
                        "seq": "ACDEFGHIKL" * 12} for _ in range(_n)]
_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    _report(_pool, CFG_ALIGN)
_out = _buf.getvalue()
_ks = [int(l.split()[0]) for l in _out.splitlines()
       if l.strip() and l.split()[0].isdigit()]
check("one starved prompt does not collapse pass@k to k=1", max(_ks) > 1, f"max k = {max(_ks)}")
check("the table reports how many prompts support each row", "prompts" in _out)
check("...and says the coverage shortfall out loud", "folding coverage" in _out)


# ------------------------------------------------- 15. balanced work assignment
# Hashing ids into `world` buckets is a balls-in-bins draw, and a phase ends when its SLOWEST rank
# does. Measured on a 1,500-prompt generation pass at 192 ranks: rank 0 drew 16 against a mean of
# 7.8, so the phase cost twice what it had to. owns() is still right where the list shrinks
# underneath the ranks (--watch); partition() is for an immutable one.
from src.fold_fasta import owns as _owns, partition as _part                      # noqa: E402

_ids = [f"p{i:07d}" for i in range(1500)]
_W = 192
_parts = [_part(_ids, r, _W) for r in range(_W)]
_flat = [x for pp in _parts for x in pp]
check("partition is disjoint and complete", len(_flat) == len(set(_flat)) == len(_ids))
check("partition is deterministic", all(_part(_ids, r, _W) == _parts[r] for r in (0, 7, 191)))
_sz = [len(pp) for pp in _parts]
check("partition is balanced to +-1", max(_sz) - min(_sz) <= 1, f"{min(_sz)}-{max(_sz)}")
_hz = [sum(_owns(i, r, _W) for i in _ids) for r in range(_W)]
check("...where hashing was not", max(_hz) >= 2 * max(_sz), f"hash max {max(_hz)} vs {max(_sz)}")
check("world<=1 keeps everything", _part(_ids, 0, 1) == _ids and len(_part(_ids, 0, 0)) == len(_ids))
check("partition ignores the caller's order",
      _part(_ids, 3, _W) == _part(list(reversed(_ids)), 3, _W))

# fold_fasta hands it (id, sequence) pairs. It has to key on the ID: keying on the whole tuple
# would make the split depend on sequence text, so a rerun after regeneration -- which reuses ids
# for new content -- would reshuffle ownership for no reason.
_pairs = [(f"gen|p{i:05d}", "ACDE" * 10) for i in range(1000)]
_same = [(f"gen|p{i:05d}", "WWWW" * 10) for i in range(1000)]
_pp = [_part(_pairs, r, 12) for r in range(12)]
_pflat = [x for q in _pp for x in q]
check("partition handles (id, seq) pairs",
      len(_pflat) == len(set(_pflat)) == 1000 and max(map(len, _pp)) - min(map(len, _pp)) <= 1)
check("...keyed on the id, not the sequence",
      [x[0] for x in _part(_pairs, 3, 12)] == [x[0] for x in _part(_same, 3, 12)])

# and the fold phase's own imbalance, which is milder than the prompt phase's but not nothing
_seq = [f"gen|p{i // 16:07d}_{i % 16}" for i in range(16000)]
_fh = max(sum(_owns(i, r, 192) for i in _seq) for r in range(192))
_fp = max(len(_part(_seq, r, 192)) for r in range(192))
check("balancing the fold split is worth >15%", (_fh - _fp) / _fh > 0.15,
      f"hash max {_fh} vs stride max {_fp}")

# The hazard both call sites are shaped around: partitioning a list that CHANGES gives a late rank
# different work. align_sample splits the prompt manifest and tm_align splits the same manifest --
# never a directory listing, which --prune shrinks while the pass is running.
check("partitioning a mutable list really does break",
      set(_part(_ids[:1200], 7, _W)) != set(_parts[7]))

# resume at a different rank count: with a globally-read done set, nothing is redone or missed
_done = set(_parts[3] + _parts[9])
_todo = [p for r in range(96) for p in _part(_ids, r, 96) if p not in _done]
check("a resume at a new world size loses nothing",
      len(set(_todo)) == len(_todo) == len(_ids) - len(_done),
      f"{len(_todo)} todo, {len(_done)} done")


# ------------------------------------------------- 16. checkpoint retention
# A round left 31GB on disk after pruning every generated PDB: three align checkpoints at 10.8GB
# each (5.4GB weights + 5.4GB RMSProp state). Worse, the rolling keep_last=3 had already deleted
# the checkpoint the run then RECOMMENDED -- "ckpt_00000100.pt is the drift-minimal checkpoint",
# printed about a file removed several saves earlier.
import shutil as _sh                                                              # noqa: E402

from src.align import _keep_best                                                  # noqa: E402
from src.train import save_checkpoint as _save                                     # noqa: E402


class _Env:
    is_main = True


_ck = _tf.mkdtemp(prefix="pld2ck_")
_m = LoopedDiffusionLM(Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
                              d_model=32, n_heads=2, d_ff=64, n_upstream=1, n_middle=1,
                              n_downstream=1, n_recurrence=1, grad_checkpoint=False))
_o = torch.optim.RMSprop(_m.parameters(), lr=1e-5)
# RMSProp allocates square_avg only for parameters that carry a gradient, so a bare step() leaves
# the state empty and the two files come out the same size -- the test would pass for the wrong
# reason on a real model and fail here. Run a real backward first.
_m(torch.randint(0, 20, (2, 16))).sum().backward()
_o.step()
_ls = torch.optim.lr_scheduler.LambdaLR(_o, lambda s: 1.0)

_save(_m, _o, _ls, 1, _ck, _Env, keep_last=1, save_optimizer=True)
_with = _os.path.getsize(_os.path.join(_ck, "ckpt_00000001.pt"))
_save(_m, _o, _ls, 2, _ck, _Env, keep_last=1, save_optimizer=False)
_without = _os.path.getsize(_os.path.join(_ck, "ckpt_00000002.pt"))
check("dropping optimizer state shrinks the checkpoint", _without < _with * 0.7,
      f"{_without} vs {_with} bytes")
check("...and the weights are still there",
      set(torch.load(_os.path.join(_ck, "ckpt_00000002.pt"), map_location="cpu",
                     weights_only=False)) == {"model", "sched", "step"})
check("keep_last=1 leaves exactly one rolling checkpoint",
      len(_glob.glob(_os.path.join(_ck, "ckpt_*.pt"))) == 1)

# best.pt must survive that rotation -- it is the whole point
_keep_best(_m, _ls, 2, 2.36, _ck, _Env)
for _st in (3, 4, 5):
    _save(_m, _o, _ls, _st, _ck, _Env, keep_last=1, save_optimizer=False)
check("best.pt survives the rolling rotation", _os.path.exists(_os.path.join(_ck, "best.pt")))
_bj = _json.load(open(_os.path.join(_ck, "best.json")))
check("best.json records which step and why",
      _bj["step"] == 2 and abs(_bj["natural_nll"] - 2.36) < 1e-9 and "why" in _bj)
check("best.pt carries weights, not optimizer state",
      "opt" not in torch.load(_os.path.join(_ck, "best.pt"), map_location="cpu",
                              weights_only=False))
_sh.rmtree(_ck, ignore_errors=True)


# ------------------------------------------------- 17. the ipex shim
# Round 3 (job 8882115) lost a 16-node allocation two minutes in: Aurora's 2026.1 image dropped
# intel_extension_for_pytorch, and EsmFold's resolve_device() reads a failed ipex import as proof
# that no XPU exists. 1,872 tracebacks, zero structures. These checks pin the two halves of the
# fix: the stub satisfies the import, and it is installed ONLY when a GPU is genuinely there.
import sys as _sys
import types as _types

from src import ipex_shim as _ish

_saved = _sys.modules.pop(_ish.MODULE, None)
_real_have_xpu = _ish.have_xpu
try:
    _ish.have_xpu = lambda: False
    check("no XPU visible -> no shim (a missing GPU must stay a loud failure)",
          _ish.needed() is False and _ish.install() is False
          and _ish.MODULE not in _sys.modules)
    check("...and status() says so", "no XPU visible" in _ish.status())

    _ish.have_xpu = lambda: True
    check("XPU visible and ipex absent -> shim is needed", _ish.needed() is True)
    check("...status() warns before the job starts, not after",
          "WILL be installed" in _ish.status())
    check("install() registers it under the real module name", _ish.install() is True
          and _ish.MODULE in _sys.modules)
    _stub = __import__(_ish.MODULE)
    check("the stub is what a third-party `import intel_extension_for_pytorch` gets",
          getattr(_stub, "pld2_stub", False) is True)
    check("ipex.xpu IS torch.xpu, so ipex.xpu.is_available() answers correctly",
          _stub.xpu is torch.xpu)
    check("ipex.__version__ exists for callers that gate on it", bool(_stub.__version__))

    # THE REGRESSION THAT COST JOB 8884342. transformers probes every optional backend with
    # importlib.util.find_spec at import time, and find_spec RAISES ValueError on an imported
    # module whose __spec__ is None -- which is what types.ModuleType gives you. The stub took
    # down the first `import transformers` underneath esm, four frames below anything of ours.
    import importlib.metadata as _im
    import importlib.util as _iu
    _spec = _iu.find_spec(_ish.MODULE)
    check("find_spec on the stub returns a spec instead of raising",
          _spec is not None and _spec.name == _ish.MODULE)
    # And it must stay un-metadata'd: that is what makes transformers decide ipex is ABSENT and
    # keep off its ipex code paths, which is both true and what we want.
    try:
        _im.version(_ish.MODULE)
        _nometa = False
    except _im.PackageNotFoundError:
        _nometa = True
    check("the stub carries no dist metadata, so consumers read it as absent", _nometa)

    # optimize()'s two return shapes: both appear in the wild, and getting the arity wrong turns
    # a model into a tuple several frames away from here.
    _mm = torch.nn.Linear(2, 2)
    _oo = torch.optim.SGD(_mm.parameters(), lr=0.1)
    check("ipex.optimize(model) -> model", _stub.optimize(_mm) is _mm)
    check("ipex.optimize(model, optimizer=opt) -> (model, opt)",
          _stub.optimize(_mm, optimizer=_oo) == (_mm, _oo))
    check("ipex.optimize tolerates the dtype kwarg", _stub.optimize(_mm, dtype=torch.bfloat16)
          is _mm)

    # Anything we did NOT think about must fail loudly rather than return a plausible None.
    try:
        _stub.quantization
        _raised = False
    except AttributeError:
        _raised = True
    check("an unstubbed ipex attribute raises AttributeError", _raised)

    check("install() is idempotent once the stub is in place", _ish.install() is False)
    check("...and status() now reports the stub", "stub installed" in _ish.status())
    check("uninstall() removes our stub", _ish.uninstall() is True
          and _ish.MODULE not in _sys.modules)

    # THE SCOPE IS THE FIX. A stub left in sys.modules is a module that is not really installed,
    # and every library that introspects packages is a potential casualty -- transformers was the
    # first. only_for_import() narrows the window to the one import that needs it.
    with _ish.only_for_import() as _held:
        check("only_for_import installs for the duration", _held is True
              and _sys.modules.get(_ish.MODULE) is not None)
    check("...and removes it on exit, so nothing downstream sees it",
          _ish.MODULE not in _sys.modules)

    # It must also clean up when the import it is wrapping raises, or one bad fold rank poisons
    # every later import in that process.
    try:
        with _ish.only_for_import():
            raise RuntimeError("the wrapped import failed")
    except RuntimeError:
        pass
    check("...even when the wrapped import raises", _ish.MODULE not in _sys.modules)

    # And it must never remove a REAL ipex that happened to be installed.
    _sys.modules[_ish.MODULE] = _types.ModuleType(_ish.MODULE)      # no pld2_stub marker
    with _ish.only_for_import() as _held2:
        pass
    check("a real ipex is left alone by install/uninstall",
          _held2 is False and _ish.MODULE in _sys.modules)
    del _sys.modules[_ish.MODULE]
finally:
    _ish.have_xpu = _real_have_xpu
    _sys.modules.pop(_ish.MODULE, None)
    if _saved is not None:
        _sys.modules[_ish.MODULE] = _saved

# ------------------------------------------------- 18. the fold benchmark's own metric
# A benchmark that measures the wrong thing is worse than none. quartile_drift() is meant to
# detect allocator degradation across a batch, and the first version read 2.6x on a workload with
# no degradation at all -- it was picking up the length difference between the quartiles, because
# cost goes as ~L^2. These pin the ordering that makes it a real measurement.
from src.fold_bench import balanced_order as _bo, quartile_drift as _qd

def _qmeans(ordered):
    q = max(2, len(ordered) // 4)
    return [sum(map(len, ordered[i * q:(i + 1) * q])) / q for i in range(4)]


_pool = [("A" * n) for n in range(40, 352, 7)]            # a realistic spread, 45 sequences
_ord = _bo(_pool)
check("balanced_order keeps every sequence exactly once",
      sorted(map(len, _ord)) == sorted(map(len, _pool)))
# Exact when the count divides by four: the four buckets are then equal and the quartile windows
# line up with them. 44 of the 45 above, so the quartiles are within a couple of percent.
_means = _qmeans(_bo(_pool[:44]))
check("...and matches the quartile length profiles within 3% on a multiple of 4",
      max(_means) / min(_means) < 1.03, f"quartile means {[round(m) for m in _means]}")
# A ragged count leaves the quartile windows slightly out of step with the buckets, so the
# guarantee is weaker -- still far better than the spread of the set, which is what matters.
_ragged = _qmeans(_ord)
check("...and stays within 8% on a ragged count",
      max(_ragged) / min(_ragged) < 1.08, f"quartile means {[round(m) for m in _ragged]}")
# Plain round-robin is what fails this: it hands the last bucket the longest of every four.
_rrm = _qmeans([s for i in range(4) for s in sorted(_pool[:44], key=len)[i::4]])
check("...where plain round-robin does not, which is why snaking is there",
      max(_rrm) / min(_rrm) > max(_means) / min(_means),
      f"round-robin {[round(m) for m in _rrm]} vs snaked {[round(m) for m in _means]}")

check("drift reports nan for a variant that only has one averaged number",
      _qd([2.0] * 20) != _qd([2.0] * 20))                 # nan != nan
check("drift is ~1 on a flat series", abs(_qd([2.0 + (i % 3) * 0.01 for i in range(40)]) - 1) < 0.02)
check("drift exceeds 1 when the series degrades", _qd([1.0 + i * 0.1 for i in range(40)]) > 1.5)
check("drift needs enough samples to mean anything", _qd([1.0, 2.0, 3.0]) != _qd([1.0, 2.0, 3.0]))

check("fold_fasta scopes the shim to the esmfold_scorer import",
      (lambda src: 0 < src.index("ipex_shim.only_for_import")
       < src.index("from esmfold_scorer import"))(open("src/fold_fasta.py").read()))

# ------------------------------------------------- 19. per-mask-rate stats
# THE POOLED ROW CANNOT ANSWER THE QUESTION IT LOOKS LIKE IT ANSWERS. Prompts are stratified over
# mask rates 0.5/0.7/0.85/1.0, so a round's pLDDT averages four different tasks -- and a pooled
# gain is consistent with cold start (rate 1.0, no scaffold) standing still while the 50%-masked
# bin improves. That would be the alignment buying the easy half of the distribution. These pin
# that bin_stats separates the bins and that the flat-cold-start pattern is actually detectable.
from src.preference import bin_stats as _bs

_ACFG = CFG_ALIGN


def _samp(pid, b, rate, plddt, tm, deg=0.0):
    return {"pid": pid, "bin": b, "rate": rate, "plddt": plddt, "ptm": plddt - 0.05,
            "tm": tm, "deg": deg, "lcr": deg, "k13": 0.0, "L": 250}


# Two rounds: the three scaffolded bins gain, cold start does not.
def _round_pool(gain):
    pool = {}
    for b, rate in ((0, 0.5), (1, 0.7), (2, 0.85), (3, 1.0)):
        g = 0.0 if rate >= 1.0 else gain
        for i in range(10):
            pid = f"p{b}{i}"
            pool[pid] = [_samp(pid, b, rate, 0.70 + g, 0.60 + g) for _ in range(4)]
    return pool


_b0, _b1 = _bs(_round_pool(0.0), _ACFG), _bs(_round_pool(0.10), _ACFG)
check("bin_stats returns one row per bin, easiest first",
      [r["bin"] for r in _b0] == [0, 1, 2, 3] and [r["rate"] for r in _b0] == [0.5, 0.7, 0.85, 1.0])
check("every sample is counted exactly once",
      sum(r["n"] for r in _b0) == sum(len(v) for v in _round_pool(0.0).values()))
check("prompts are counted per bin, not pooled", all(r["prompts"] == 10 for r in _b0))
check("cold_start marks rate 1.0 and nothing else",
      [r["cold_start"] for r in _b0] == [False, False, False, True])
check("per-bin pLDDT is that bin's mean, not the pooled one",
      abs(_b0[0]["plddt"] - 0.70) < 1e-9 and abs(_b1[0]["plddt"] - 0.80) < 1e-9)

# The point of the whole exercise: scaffolded bins move, cold start does not, and the POOLED
# number rises anyway -- so only the split distinguishes the two explanations.
_d = [b["reward"] - a["reward"] for a, b in zip(_b0, _b1)]
check("the split sees the scaffolded bins gain", all(x > 0.15 for x in _d[:3]), f"{_d[:3]}")
check("...and sees cold start flat", abs(_d[3]) < 1e-9, f"{_d[3]:+.4f}")
_pooled = [sum(r["reward"] * r["n"] for r in bb) / sum(r["n"] for r in bb) for bb in (_b0, _b1)]
check("...while the POOLED reward rises, which is the trap",
      _pooled[1] - _pooled[0] > 0.10, f"pooled {_pooled[0]:.3f} -> {_pooled[1]:.3f}")

check("a bin with no samples is omitted rather than reported as zero",
      len(_bs({k: v for k, v in _round_pool(0.0).items() if not k.startswith("p3")}, _ACFG)) == 3)
# pTM is NOT the reward's TM. tm = foldseek against the prompt's reference (the right fold);
# ptm = ESMFold's own topology estimate (no reference). They diverge exactly at cold start, where
# the prompt reveals only the length, so pinning both into the row is what lets the cold-start
# bin be scored at all. These also pin the backfill, since rounds 1-10's report.json predates it.
from src.round_summary import pooled_ptm as _pp

_bb = _bs(_round_pool(0.0), _ACFG)
check("bin_stats reports pTM separately from TM",
      all(abs(r["ptm"] - (r["plddt"] - 0.05)) < 1e-9 for r in _bb)
      and all(r["tm"] != r["ptm"] for r in _bb))
check("...and the fraction of confident pTM", all("ptm_confident" in r for r in _bb))
check("pooled_ptm prefers report.json when it has ptm",
      _pp("round3", {"ptm": 0.321}, {"round3": _bb}) == 0.321)
_exp = sum(r["ptm"] * r["n"] for r in _bb) / sum(r["n"] for r in _bb)
check("...and backfills an n-weighted mean from the bins when it does not",
      abs(_pp("round3", {}, {"round3": _bb}) - _exp) < 1e-12)
check("...and reports nothing rather than guessing when neither is available",
      _pp("round3", {}, {}) is None)
check("report.json carries ptm, so future rounds need no backfill",
      '"ptm": float(np.mean([x["ptm"] for x in allg]))' in open("src/preference.py").read())

check("report.json carries by_bin so future rounds need no recompute",
      '"by_bin": bin_stats(pool, acfg)' in open("src/preference.py").read())
check("round_summary can split by bin without report.json",
      "not rows and not a.by_bin" in open("src/round_summary.py").read())

# ------------------------------------------------- 20. a fold table that can be audited
# TWELVE RANKS SHARE ONE STDOUT. A rank's "scored N sequence(s)" can land after rank 0's final
# table even when every record was already on disk, which reads as the summary having been printed
# before folding finished. coverage() makes the question answerable from the log instead.
import tempfile as _tf

from src.fold_fasta import coverage as _cov, mark_done as _md

_fd = _tf.mkdtemp(prefix="pld2cov")
_fb = _os.path.join(_fd, "folds.jsonl")
for _r in range(3):
    with open(f"{_fb[:-6]}.rank{_r:03d}.jsonl", "w") as _fh:
        for _i in range(4):
            _fh.write(_json.dumps({"id": f"g|s{_r}_{_i}", "length": 9, "seq": "A" * 9,
                                   "plddt": 0.8, "ptm": 0.7}) + "\n")
check("coverage counts records across every rank shard", "12 record(s) from 3 file(s)" in _cov(_fb))
check("...and flags a table whose ranks have not all reported",
      "0/3 rank(s) reported done" in _cov(_fb, world=3) and "INCOMPLETE" in _cov(_fb, world=3))
_md(_fb, 0); _md(_fb, 1)
check("...and counts partial completion honestly", "2/3" in _cov(_fb, world=3))
_md(_fb, 2)
_all = _cov(_fb, world=3)
check("...and stops warning once every rank is in", "3/3" in _all and "INCOMPLETE" not in _all)
check("single-rank runs say nothing about ranks", "rank(s) reported" not in _cov(_fb))
check("the record count is taken from the caller, not re-read",
      "n_recs=len(recs)" in open("src/fold_fasta.py").read())
_sh.rmtree(_fd, ignore_errors=True)

# A SWEEP AGAINST A NEW CHECKPOINT STILL PRINTS ROWS FOR EVERY CONFIGURATION IT DID NOT
# REGENERATE, because scoring is skipped on sequence content. Job 8913824 wrote 18 FASTAs and
# folded 45: the 27 carried-over ones -- every filip row -- described the pre-alignment model and
# read as a guidance study of the aligned one. The run stamp is what makes that visible.
_fd2 = _tf.mkdtemp(prefix="pld2run")
_fb2 = _os.path.join(_fd2, "folds.jsonl")
_os.environ["PBS_JOBID"] = "now.aurora"
with open(f"{_fb2[:-6]}.rank000.jsonl", "w") as _fh:
    for _grp, _run in (("fresh", "now.aurora"), ("old", "before.aurora"), ("legacy", None)):
        for _i in range(3):
            _rec = {"id": f"{_grp}|s{_i}", "length": 20, "seq": "ACDEFGHIKLMNPQRSTVWY",
                    "plddt": 0.6, "ptm": 0.4}
            if _run:
                _rec["run"] = _run
            _fh.write(_json.dumps(_rec) + "\n")

import io as _io
import contextlib as _ctx
from src.fold_fasta import summarize as _summ

_buf = _io.StringIO()
with _ctx.redirect_stdout(_buf):
    _summ(_fb2, _CFG.opt)
_txt = _buf.getvalue()
_line = lambda nm: next(l for l in _txt.splitlines() if l.startswith(nm))
check("a group folded by THIS job is not marked", "EARLIER RUN" not in _line("fresh"))
check("a group from another job IS marked", "EARLIER RUN" in _line("old"))
check("...as is one predating the run stamp entirely", "EARLIER RUN" in _line("legacy"))
check("...and the footer names them and says folding alone will not refresh them",
      "EARLIER RUN marks 2 group(s)" in _txt and "keyed on sequence content" in _txt)
check("fold records carry the run stamp", '"run": _run_id()' in open("src/fold_fasta.py").read())
_sh.rmtree(_fd2, ignore_errors=True)

# round_summary must NAME a round it is dropping. A round run with PHASES=012 has no report.json,
# and silently omitting it makes the table look identical to the previous run's.
_rs = open("src/round_summary.py").read()
check("round_summary explains an omitted round instead of dropping it",
      "no report.json --" in _rs and "PHASES=" in _rs)

# ------------------------------------------------- 21. the rate-weighted reward and bar
# TM is foldseek against the prompt's reference; at a mask rate of 1.0 the prompt reveals only the
# length, so across eleven rounds TM sat at 0.245-0.261 (sd 0.006) while pTM went 0.209 -> 0.447.
# A third of every round's pairs were ranked on pLDDT and degeneracy alone. These pin the split,
# the invariant that makes it safe, and the ONE case that rules out thresholding the blend.
from src.preference import reward_formula as _rf, score as _sc, struct_weights as _sw
from src.preference import succeeded as _ok

_A = CFG_ALIGN


def _s(rate, plddt=0.75, tm=0.8, ptm=0.6, deg=0.0):
    return {"plddt": plddt, "tm": tm, "ptm": ptm, "deg": deg, "rate": rate}


check("the structural weights are the mask rate, split",
      all(abs(_sw(_s(r), _A)[0] - _A.reward_struct * (1 - r)) < 1e-12
          and abs(_sw(_s(r), _A)[1] - _A.reward_struct * r) < 1e-12
          for r in (0.5, 0.7, 0.85, 1.0)))
check("...and always sum to reward_struct, so the reward stays on ONE scale across bins",
      all(abs(sum(_sw(_s(r), _A)) - _A.reward_struct) < 1e-12 for r in (0.5, 0.7, 0.85, 1.0)))
check("cold start puts the whole structural weight on pTM and none on TM",
      _sw(_s(1.0), _A) == (0.0, _A.reward_struct))
check("...so TM cannot move a cold-start score at all",
      _sc(_s(1.0, tm=0.0), _A) == _sc(_s(1.0, tm=1.0), _A))
check("...while pTM can", _sc(_s(1.0, ptm=0.9), _A) > _sc(_s(1.0, ptm=0.1), _A))
check("at rate 0.5 both still count",
      _sc(_s(0.5, tm=0.9), _A) > _sc(_s(0.5, tm=0.1), _A)
      and _sc(_s(0.5, ptm=0.9), _A) > _sc(_s(0.5, ptm=0.1), _A))
# A sample with no rate -- anything from before prompts carried one -- must score as it used to.
check("a sample with no rate falls back to the old all-TM reward",
      abs(_sc({"plddt": 0.7, "tm": 0.6, "ptm": 0.1, "deg": 0.0}, _A)
          - (_A.reward_plddt * 0.7 + _A.reward_tm * 0.6)) < 1e-12)


class _Legacy:
    reward_plddt = reward_tm = reward_struct = 1.0
    reward_blend = False
    reward_deg = 0.5
    deg_kmer_k = 13
    plddt_success, tm_success, ptm_success = 0.7, 0.5, 0.5


check("reward_blend=False restores the pre-run-2 reward at every rate",
      _sw(_s(1.0), _Legacy) == (1.0, 0.0) and _sw(_s(0.5), _Legacy) == (1.0, 0.0))
check("the formula string names which reward is in force",
      "rate*pTM" in _rf(_A) and "rate*pTM" not in _rf(_Legacy))
check("...and resolves per rate for a banner",
      "0.00*TM" in _rf(_A, 1.0) and "1.00*pTM" in _rf(_A, 1.0))

# THE SUCCESS BAR IS REFERENCE-FREE AT EVERY RATE, and that is a separate decision from the
# reward. A sequence that folds coherently into something other than the reference has done the
# job -- the reference is one sample from the folds compatible with that scaffold, not the only
# acceptable answer. So TM must not appear in the bar anywhere, including where it still carries
# reward weight. The bar and the reward asking different questions is the point, not an oversight.
check("success ignores TM at EVERY rate, however extreme",
      all(_ok(_s(r, tm=0.0, ptm=0.6), _A) is True for r in (0.5, 0.7, 0.85, 1.0))
      and all(_ok(_s(r, tm=1.0, ptm=0.4), _A) is False for r in (0.5, 0.7, 0.85, 1.0)))
check("...while the REWARD still weights TM where the scaffold makes it informative",
      _sc(_s(0.5, tm=0.9), _A) > _sc(_s(0.5, tm=0.1), _A))
check("pTM is the structural gate", _ok(_s(1.0, ptm=0.6), _A) is True
      and _ok(_s(1.0, ptm=0.4), _A) is False)
check("pLDDT remains a hard gate at every rate",
      all(_ok(_s(r, plddt=0.5), _A) is False for r in (0.5, 0.7, 0.85, 1.0)))
check("a missing pTM reads as failure rather than falling back to TM",
      _ok({"plddt": 0.9, "tm": 0.99, "deg": 0.0, "rate": 0.5}, _A) is False)
check("the bar does not depend on the reward's blend switch",
      _ok(_s(1.0, tm=0.0, ptm=0.6), _Legacy) is True)

# ------------------------------------------------- 22. the plateau rule
# Validated against run 1's ACTUAL pooled reward series, r3..r11. The best row is r8 (which scores
# round 7's policy); everything after it added drift and gave reward back.
from src.round_summary import plateau as _pl

_RUN1 = [1.2128, 1.2101, 1.2655, 1.2801, 1.2859, 1.3008, 1.2695, 1.2687, 1.2842]
check("a plateau cannot be called before patience+2 rounds",
      _pl(_RUN1[:3], 2)[0] is False and "need 4" in _pl(_RUN1[:3], 2)[1])
check("patience=2 fires on run 1 only after r11", _pl(_RUN1[:8], 2)[0] is False
      and _pl(_RUN1, 2)[0] is True)
check("patience=1 fires a round earlier, after r10",
      _pl(_RUN1[:7], 1)[0] is False and _pl(_RUN1[:8], 1)[0] is True)
check("a still-rising series never fires",
      _pl([1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 2)[0] is False)
check("the reason names the best round and the gap", "round(s) back" in _pl(_RUN1, 2)[1])
check("align.pbs consults the rule and can be told not to act on it",
      (lambda t: "--plateau" in t and "STOP_ON_PLATEAU" in t
       and "PHASES//5/" in t)(open("scripts/align.pbs").read()))
check("the exclusion window is read from config, not hardcoded to all history",
      "ref_exclude_window" in open("scripts/align.pbs").read()
      and "EXCL_FROM=$(( r - REF_WINDOW ))" in open("scripts/align.pbs").read())

print(f"\n{checks - len(fails)}/{checks} checks pass")
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
