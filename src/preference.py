"""Turn folded, TM-scored generations into preference pairs -- and report what the pool contains.

    python -m src.preference --report          # the pass@k table; no pairs written
    python -m src.preference                   # write <round>/pairs.jsonl

WITHIN-PROMPT RANKING, NOT ABSOLUTE THRESHOLDS. ESM3 kept generations with pTM > 0.8 and
cRMSD < 1.5A and threw away every prompt that produced none (Appendix A.4.4). It could afford to:
its base model hit pTM 0.85 on ordinary prompted generation, so positives were abundant. Ours clear
an absolute bar about 1% of the time, so at n_gen=16 roughly 11% of prompts would contain a single
qualifying sample and the other 89% of the fold budget would be discarded. Ranking inside a prompt
yields a usable pair at every difficulty instead, and difficulty cancels exactly because both sides
of a pair saw the same context.

What that costs is the absolute quality of the winner, and align.success_weight is the knob that
buys some of it back: it upweights pairs whose winner clears the absolute bar outright. It is 1.0
by default -- OFF -- because the relative construction should be shown to work on its own first.

TWO PAIR CONSTRUCTIONS, AND THE SECOND IS THE ONE AIMED AT OUR ACTUAL FAILURE MODE.

  rank      top_k winners x bot_k losers by the scalar reward. The workhorse.
  matched   two generations with nearly the SAME pLDDT and different TM. Quality is held fixed
            across the pair, so the only thing the gradient can carry is prompt consistency --
            "fold into the right shape", not "fold". This is the construction that needs n_gen to
            be large: it is a matched-pair design on a second variable, and at n_gen=2 you take
            whatever you are given. It is also the one that pushes away from the degenerate
            solution, since a confident wrong fold is exactly what sits on the loser side.

REWARD. w_plddt * pLDDT + w_tm * TM, used for RANKING ONLY -- the loss never sees its magnitude,
because DPO and IPO both treat preference as binary. That is precisely why min_gap exists: a
mislabelled pair contributes a gradient of exactly the same size as a correct one, so pairs whose
two sides differ by less than the metric's own noise are worse than no pairs at all.
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from config import CFG
from .metrics import kmer_counts, lcr_counts
from .self_consistency import record_key


def _read_jsonl(paths):
    for p in paths:
        with open(p) as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:
                    continue                    # truncated final line after an abort


def load_pool(rdir, folds=None):
    """-> {pid: [sample dicts]} joined across generation, folding and TM.

    A sample is kept only if all three sides are present. They are produced by three separate
    passes that each crash and resume independently, so partial coverage is the normal state of
    this directory mid-campaign, not an error.
    """
    gen = {}
    for r in _read_jsonl(sorted(glob_.glob(os.path.join(rdir, "gen.rank*.jsonl")))):
        gen[r["gid"]] = r
    fold = {}
    for r in _read_jsonl(sorted(glob_.glob(folds or os.path.join(rdir, "folds*.jsonl")))):
        if "id" in r:
            fold[record_key(r["id"])] = r       # "gen|p0000042_7" -> "p0000042_7"
    tm = {}
    for r in _read_jsonl(sorted(glob_.glob(os.path.join(rdir, "tm.rank*.jsonl")))):
        tm[r["gid"]] = r

    pool = defaultdict(list)
    for gid, g in gen.items():
        f, t = fold.get(gid), tm.get(gid)
        if f is None or t is None:
            continue
        lcr, k13 = degeneracy_of(g.get("seq", ""))
        pool[g["pid"]].append({**g, "plddt": float(f["plddt"]), "ptm": float(f["ptm"]),
                               "tm": float(t.get(CFG.align.tm_field, 0.0)),
                               "alntm": float(t.get("alntmscore", 0.0)),
                               "lcr": lcr, "k13": k13, "deg": max(lcr, k13)})
    return dict(pool), len(gen), len(fold), len(tm)


def degeneracy_of(seq, k=None):
    """(LCR fraction, k-mer repeat coverage) for one sequence. ~0.12 ms, so ~2s a round.

    Two detectors because they see different things: SEG-style low complexity within a 12-residue
    window, and repetition at a k LONGER than that window, which LCR cannot see at all -- a
    sequence built from one repeated 20-mer reads LCR 0.0% and k13 100%.
    """
    if not seq:
        return 0.0, 0.0
    k = k or CFG.align.deg_kmer_k
    lcr, tot = lcr_counts([seq])
    c = kmer_counts([seq], (k,))
    return lcr / max(tot, 1), c[k]["rep_pos"] / max(c[k]["n_pos"], 1)


def score(s, acfg):
    """The RANKING reward. Its magnitude never reaches the loss -- DPO and IPO treat preference as
    binary -- so this only has to order samples within a prompt correctly."""
    return (acfg.reward_plddt * s["plddt"] + acfg.reward_tm * s["tm"]
            - acfg.reward_deg * s.get("deg", 0.0))


def eligible_winner(s, acfg):
    """Can this sample sit on the preferred side at all? A hard gate, not a penalty.

    Measured on round 1: at a mask rate of 1.0 the soft terms alone promoted winners carrying 12.6
    points MORE LCR than their losers, because pLDDT's spread there is 2.4x TM's and r(LCR, pLDDT)
    is +0.277. No weighting of a sum fixes a term that is itself the confound; the degenerate
    samples have to be taken off the winner side outright.
    """
    return s.get("deg", 0.0) <= acfg.deg_max_winner


def succeeded(s, acfg):
    return s["plddt"] > acfg.plddt_success and s["tm"] > acfg.tm_success


# --------------------------------------------------------------------------------------
# the pass@k table
# --------------------------------------------------------------------------------------
def pass_at_k(n, c, k):
    """Chen et al.'s unbiased estimator: P(at least one of k draws succeeds) given c of n do."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def selector_at_k(succ_sorted, k):
    """P(the top-1 BY SELECTOR of a random k-subset is a success), exactly.

    Sort the pool by the selector, descending. The top-1 of a random k-subset is the sample at rank
    r exactly when the other k-1 all come from the n-1-r samples ranked below it, so the weight on
    rank r is C(n-1-r, k-1) / C(n, k). This is the number the best-of-N curve reports, and it is
    NOT pass@k: pass@k is what an oracle would find, and the difference between them is how much
    the selector is leaving on the table.
    """
    n = len(succ_sorted)
    if k > n:
        return float("nan")
    tot = math.comb(n, k)
    return sum(math.comb(n - 1 - r, k - 1) for r, ok in enumerate(succ_sorted) if ok
               and n - 1 - r >= k - 1) / tot


def report(pool, acfg, max_k=None, coverage=0.25):
    pids = sorted(pool)
    ns = [len(pool[p]) for p in pids]
    all_s = [s for p in pids for s in pool[p]]
    if not all_s:
        print("[pref] nothing scored yet")
        return
    # THE TABLE'S DEPTH IS A QUANTILE, NOT THE MINIMUM. A single prompt whose folds mostly failed
    # used to collapse the whole thing to k=1 -- which is exactly what happened when folding got
    # through 66% of 24,000 generations and one prompt came back with a single sample. Every row
    # also carries the number of prompts that can support it, so a thinning tail is visible rather
    # than silently changing what the row means.
    n_min, n_max = min(ns), max(ns)
    max_k = max_k or int(np.quantile(ns, coverage))
    max_k = max(1, min(max_k, n_max))

    succ = [succeeded(s, acfg) for s in all_s]
    print(f"[pref] {len(pids):,} prompts, {len(all_s):,} scored generations "
          f"({np.mean(ns):.1f} per prompt, min {n_min}, median {int(np.median(ns))}, max {n_max})")
    if n_min < n_max:
        short = sum(1 for v in ns if v < n_max)
        print(f"[pref] {short:,} prompt(s) ({short / len(pids):.0%}) have fewer than {n_max} "
              f"generations, so the best-of-n ceiling this round aims at is really best-of-"
              f"{np.mean(ns):.1f}. That is folding coverage, not generation.")
    print(f"[pref] success = pLDDT > {acfg.plddt_success} AND {acfg.tm_field} > {acfg.tm_success}"
          f"  ->  per-draw rate {np.mean(succ):.3%}")
    print(f"[pref] reward = {acfg.reward_plddt}*pLDDT + {acfg.reward_tm}*TM "
          f"- {acfg.reward_deg}*max(LCR, k{acfg.deg_kmer_k}) | winner gate: degeneracy <= "
          f"{acfg.deg_max_winner:.0%} ({np.mean([s['deg'] > acfg.deg_max_winner for s in all_s]):.1%} "
          f"of generations are above it)")
    print(f"[pref] pLDDT {np.mean([s['plddt'] for s in all_s]):.3f}"
          f" +- {np.std([s['plddt'] for s in all_s]):.3f}   "
          f"TM {np.mean([s['tm'] for s in all_s]):.3f}"
          f" +- {np.std([s['tm'] for s in all_s]):.3f}   "
          f"pTM {np.mean([s['ptm'] for s in all_s]):.3f}")

    # WITHIN-PROMPT sigma is the quantity that sets the ceiling of this whole exercise: the winner
    # is a best-of-n order statistic, the supervised term teaches the model to imitate it, so
    # E[max of n] * sigma is how far the one-shot distribution can be pulled in one round.
    within = [np.std([score(s, acfg) for s in pool[p]]) for p in pids if len(pool[p]) > 1]
    if within:
        sig = float(np.mean(within))
        base = float(np.mean([score(s, acfg) for s in all_s]))
        rng = np.random.default_rng(0)
        print(f"[pref] reward {base:.3f}, within-prompt sigma {sig:.3f}, "
              f"between-prompt sigma {np.std([np.mean([score(s, acfg) for s in pool[p]]) for p in pids]):.3f}")
        print("[pref] implied one-shot ceiling (reward the tuned model is being aimed at):")
        for k in (1, 2, 4, 8, 16, 32):
            if k <= max_k or k <= 32:
                e = float(rng.standard_normal((200_000, k)).max(axis=1).mean())
                print(f"[pref]     n={k:<3} {base + e * sig:.3f}")

    row_last = None
    print(f"\n{'k':>4} {'prompts':>8} {'pass@k (oracle)':>16} {'best-of-k pLDDT':>17} "
          f"{'best-of-k loglik':>17} {'best-of-k reward':>17}")
    print("-" * 86)
    for k in range(1, max_k + 1):
        sup = [p for p in pids if len(pool[p]) >= k]
        row = [k, len(sup)]
        row.append(float(np.mean([pass_at_k(len(pool[p]), sum(succeeded(s, acfg) for s in pool[p]), k)
                                  for p in sup])))
        for key in ("plddt", "loglik", "_reward"):
            vals = []
            for p in pids:
                ss = pool[p]
                if len(ss) < k or any(s.get(key) is None for s in ss if key != "_reward"):
                    continue
                order = sorted(ss, key=(lambda s: score(s, acfg)) if key == "_reward"
                               else (lambda s: s.get(key, 0.0)), reverse=True)
                vals.append(selector_at_k([succeeded(s, acfg) for s in order], k))
            row.append(float(np.mean(vals)) if vals else float("nan"))
        print(f"{row[0]:>4} {row[1]:>8,} {row[2]:>16.4f} {row[3]:>17.4f} {row[4]:>17.4f} "
              f"{row[5]:>17.4f}")
        row_last = (row[0], row[2], row[3], row[4])
    if max_k < n_max:
        print(f"[pref] stopped at k={max_k}: beyond it fewer than {1 - coverage:.0%} of prompts "
              f"have the samples to support a row.")
    # READ OFF THIS TABLE, not off a remembered one. The earlier note asserted "pLDDT selection
    # already tracks the oracle", which was measured once and is not true in general -- on a pool
    # with uneven fold coverage it can be far from true, and a log that states it anyway is worse
    # than a log that says nothing.
    if row_last:
        k, oracle, sel_p, sel_l = row_last
        sel_loss, gap = oracle - sel_p, sel_p - sel_l
        print(f"\n[pref] at k={k}:  selection loss (oracle - pLDDT) {sel_loss:+.4f}   "
              f"loglik gap (pLDDT - loglik) {gap:+.4f}")
        if sel_loss > 0.02:
            print(f"[pref]   pLDDT leaves {sel_loss:.4f} on the table against a perfect selector. "
                  f"That is SELECTION\n[pref]   loss, and its fix is a better ranker, not a better "
                  f"model.")
        else:
            print(f"[pref]   pLDDT is within {sel_loss:.4f} of the oracle, so there is no "
                  f"selection loss left to\n[pref]   recover and a better ranker buys nothing.")
        if gap > 0.02:
            print(f"[pref]   The model's OWN likelihood is the weaker ranker by {gap:.4f}. That "
                  f"gap is what a\n[pref]   preference loss repairs, so it should narrow round "
                  f"over round; if it does not, the\n[pref]   tuning is not doing its job.")
        else:
            print(f"[pref]   The model's own likelihood ranks about as well as pLDDT here "
                  f"({gap:+.4f}); the\n[pref]   likelihood pathology is a COLD-START phenomenon "
                  f"and this pool is mostly not that.")


# --------------------------------------------------------------------------------------
# pair construction
# --------------------------------------------------------------------------------------
def rank_pairs(ss, acfg):
    """top_k winners x bot_k losers by the scalar reward, winners passing the degeneracy gate."""
    order = sorted(ss, key=lambda s: score(s, acfg), reverse=True)
    clean = [s for s in order if eligible_winner(s, acfg)]
    wins, losses = clean[:acfg.top_k], order[-acfg.bot_k:]
    out = []
    for w in wins:
        for l in losses:
            if w["gid"] == l["gid"]:
                continue
            gap = score(w, acfg) - score(l, acfg)
            if gap < acfg.min_gap:
                continue
            out.append((w, l, gap, "rank"))
    return out


def clean_pairs(ss, acfg, limit):
    """Matched on pLDDT, split on DEGENERACY: both sides fold equally well, one of them cheats.

    The direct expression of "fold without cheating". Quality is held fixed across the pair, so the
    gradient cannot carry anything else -- and unlike the reward penalty, which competes with pLDDT
    inside a sum, this construction cannot be outvoted. It is the same design as matched_pairs, one
    axis over.
    """
    out = []
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            if abs(a["plddt"] - b["plddt"]) > acfg.matched_plddt_tol:
                continue
            w, l = (a, b) if a.get("deg", 0) <= b.get("deg", 0) else (b, a)
            gap = l.get("deg", 0) - w.get("deg", 0)
            if gap < acfg.clean_min_gap or not eligible_winner(w, acfg):
                continue
            out.append((w, l, gap, "clean"))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:limit]


def matched_pairs(ss, acfg, limit):
    """Pairs matched on pLDDT and split on TM: same confidence, different correctness.

    Both sides fold; only one folds into the shape it was asked for. Nothing about overall quality
    distinguishes them, so nothing about overall quality can be what the gradient learns -- and
    measured on round 1, that alone was enough to make this construction ANTI-degenerate where the
    rank construction was strongly degenerate (-5.9 vs +12.6 points of winner LCR at cold start).
    """
    out = []
    for i, a in enumerate(ss):
        for b in ss[i + 1:]:
            if abs(a["plddt"] - b["plddt"]) > acfg.matched_plddt_tol:
                continue
            w, l = (a, b) if a["tm"] >= b["tm"] else (b, a)
            gap = w["tm"] - l["tm"]
            if gap < max(acfg.min_gap, 1e-9) or not eligible_winner(w, acfg):
                continue
            out.append((w, l, gap, "matched"))
    out.sort(key=lambda t: t[2], reverse=True)
    return out[:limit]


def build_pairs(pool, acfg):
    pairs, n_succ_w, n_gated = [], 0, 0
    for pid in sorted(pool):
        ss = pool[pid]
        if len(ss) < 2:
            continue
        if not any(eligible_winner(s, acfg) for s in ss):
            # Every sample this prompt produced is too degenerate to promote. ESM3 discarded
            # prompts with no valid pair for the same reason: a pair whose winner is bad is not a
            # weak training signal, it is a wrong one.
            n_gated += 1
            continue
        got = rank_pairs(ss, acfg)
        n_rank = max(len(got), 1)
        if acfg.matched_frac > 0:
            got += matched_pairs(ss, acfg, max(1, int(round(acfg.matched_frac * n_rank))))
        if acfg.clean_frac > 0:
            got += clean_pairs(ss, acfg, max(1, int(round(acfg.clean_frac * n_rank))))
        for w, l, gap, kind in got:
            ok = succeeded(w, acfg)
            n_succ_w += ok
            pairs.append({
                "pair_id": f"{pid}:{w['gid']}:{l['gid']}:{kind}",
                "pid": pid, "kind": kind, "gap": round(gap, 5),
                "L": int(w["ref_len"]), "bin": w["bin"], "rate": w["rate"],
                "winner_success": bool(ok),
                "weight": float(acfg.success_weight if ok else 1.0),
                "w": {"gid": w["gid"], "seq": w["seq"], "di": w["di"],
                      "plddt": w["plddt"], "tm": w["tm"], "deg": w.get("deg", 0.0)},
                "l": {"gid": l["gid"], "seq": l["seq"], "di": l["di"],
                      "plddt": l["plddt"], "tm": l["tm"], "deg": l.get("deg", 0.0)},
            })
    return pairs, n_succ_w, n_gated


def degeneracy_report(pairs, acfg):
    """Winner vs loser degeneracy, per mask rate and per construction.

    THE TABLE THAT WOULD HAVE CAUGHT ROUND 1. The pairs looked fine by every number that was
    printed; what they were actually teaching was only visible here. `d deg` must not be positive:
    a positive value means the preferred side of the average pair is the MORE repetitive one, and
    the tuning will faithfully learn that.
    """
    if not pairs:
        return
    by = defaultdict(list)
    for p in pairs:
        by[(p["rate"], p["kind"])].append(p)
    print(f"\nDEGENERACY ACROSS THE PAIR   (d deg > 0 means the winner is the MORE repetitive side)")
    print(f"{'rate':>5} {'kind':>8} {'n':>7} {'win deg':>8} {'lose deg':>9} {'d deg':>8} "
          f"{'d pLDDT':>8} {'d TM':>7} {'winner worse':>13}")
    print("-" * 80)
    bad = []
    for key in sorted(by):
        v = by[key]
        dd = [x["w"]["deg"] - x["l"]["deg"] for x in v]
        m = float(np.mean(dd))
        print(f"{key[0]:>5.2f} {key[1]:>8} {len(v):>7,} "
              f"{np.mean([x['w']['deg'] for x in v]):>7.1%} "
              f"{np.mean([x['l']['deg'] for x in v]):>8.1%} {m:>+7.1%} "
              f"{np.mean([x['w']['plddt'] - x['l']['plddt'] for x in v]):>+8.3f} "
              f"{np.mean([x['w']['tm'] - x['l']['tm'] for x in v]):>+7.3f} "
              f"{np.mean([d > 1e-9 for d in dd]):>12.1%}")
        if m > 0.01:
            bad.append((key, m))
    for key, m in bad:
        print(f"[pref] WARNING: rate {key[0]} '{key[1]}' pairs prefer the MORE repetitive side by "
              f"{m:+.1%} on average. That is what the tuning will learn. Raise align.reward_deg, "
              f"lower align.deg_max_winner, or drop this construction at this rate.")
    if not bad:
        print(f"[pref] no construction prefers the degenerate side at any mask rate.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg = CFG.align
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="round directory (default: config align.round_dir)")
    ap.add_argument("--folds", default=None, help="folds JSONL glob (default: <round>/folds*.jsonl)")
    ap.add_argument("--out", default=None, help="default: <round>/pairs.jsonl")
    ap.add_argument("--report", action="store_true", help="print the pass@k table and exit")
    ap.add_argument("--success-weight", type=float, default=None,
                    help="override align.success_weight (1.0 = off)")
    ap.add_argument("--min-gap", type=float, default=None)
    ap.add_argument("--reward-deg", type=float, default=None,
                    help="coefficient on the degeneracy penalty (0 reproduces round 1's reward)")
    ap.add_argument("--deg-max-winner", type=float, default=None,
                    help="a sample above this degeneracy can never be a winner (1.0 disables)")
    a = ap.parse_args()
    if a.reward_deg is not None:
        acfg.reward_deg = a.reward_deg
    if a.deg_max_winner is not None:
        acfg.deg_max_winner = a.deg_max_winner
    if a.success_weight is not None:
        acfg.success_weight = a.success_weight
    if a.min_gap is not None:
        acfg.min_gap = a.min_gap

    rdir = a.dir or acfg.round_dir
    pool, n_gen, n_fold, n_tm = load_pool(rdir, a.folds)
    print(f"[pref] {rdir}: {n_gen:,} generated | {n_fold:,} folded | {n_tm:,} TM-scored | "
          f"{sum(len(v) for v in pool.values()):,} joined across all three", flush=True)
    if not pool:
        raise SystemExit("nothing joined. Check that folding used --out <round>/folds.jsonl and "
                         "--pdb-dir <round>/pdb, and that src.tm_align has run.")
    report(pool, acfg)
    if a.report:
        return

    pairs, n_succ_w, n_gated = build_pairs(pool, acfg)
    out = a.out or os.path.join(rdir, "pairs.jsonl")
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)

    kinds = defaultdict(int)
    for p in pairs:
        kinds[p["kind"]] += 1
    n_prompts_with = len({p["pid"] for p in pairs})
    print(f"\n[pref] {len(pairs):,} pairs from {n_prompts_with:,} prompts "
          f"({dict(kinds)}), min_gap={acfg.min_gap}")
    if n_gated:
        print(f"[pref] {n_gated:,} prompt(s) dropped: every generation was above the degeneracy "
              f"gate ({acfg.deg_max_winner:.0%}), so there was nothing legitimate to promote.")
    degeneracy_report(pairs, acfg)
    print(f"[pref] winner clears the absolute bar in {n_succ_w:,} pairs "
          f"({n_succ_w / max(len(pairs), 1):.1%}); success_weight={acfg.success_weight}"
          f"{'  (OFF)' if acfg.success_weight == 1.0 else ''}")
    print(f"[pref] mean reward gap {np.mean([p['gap'] for p in pairs]):.3f}, "
          f"median {np.median([p['gap'] for p in pairs]):.3f}")
    for b in sorted({p["bin"] for p in pairs}):
        sel = [p for p in pairs if p["bin"] == b]
        print(f"[pref]   mask rate {sel[0]['rate']:<5} {len(sel):>7,} pairs "
              f"({len(sel) / len(pairs):5.1%})"
              + ("   <- cold start" if sel[0]["rate"] >= 1.0 else ""))
    print(f"[pref] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
