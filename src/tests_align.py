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
import itertools
import math
import random

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
    reward_plddt = reward_tm = 1.0
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
    reward_plddt = reward_tm = 1.0
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
    reward_plddt = reward_tm = 1.0
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
_deg = {"gid": "d", "plddt": 0.90, "tm": 0.20, "deg": 0.60, "seq": "A" * 200}
_ok = {"gid": "c", "plddt": 0.70, "tm": 0.30, "deg": 0.02, "seq": "ACDE" * 50}
_mid = {"gid": "m", "plddt": 0.68, "tm": 0.25, "deg": 0.05, "seq": "ACDF" * 50}
_bad = {"gid": "b", "plddt": 0.30, "tm": 0.10, "deg": 0.03, "seq": "ACDG" * 50}
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
_allgated, _, _ng2 = build_pairs(
    {"p1": [dict(_deg, gid="z1", **_meta), dict(_deg, gid="z2", **_meta)]}, _D)
check("a prompt with nothing promotable is dropped, not forced",
      _allgated == [] and _ng2 == 1)

print(f"\n{checks - len(fails)}/{checks} checks pass")
if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    raise SystemExit(1)
