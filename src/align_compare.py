"""Put several tuned policies side by side on ONE held-out prompt set.

    python -m src.align_compare --eval-root <round>/eval --policy-root <round>

PAIRED, BECAUSE THE VARIANTS SHARE THE PROMPTS. Every variant generates from the same
prompts_eval.jsonl, so the interesting statistic is not "which mean is higher" -- prompt difficulty
dominates that and it is the same difficulty on both sides -- but the per-prompt difference against
the baseline. Comparing means across independently drawn prompt sets is how the first version of
the FILIP specificity test awarded a null control a +0.26 margin; this is the same lesson applied
before the fact.

THE COLUMN TO READ IS `loglik gap`. Measured on the base model, pLDDT selection tracks the oracle
exactly while best-of-N by the model's own log-likelihood runs 0.007 -> 0.000: the generator makes
good proteins and ranks them below the bad ones. Preference tuning exists to repair that ranking, so
    gap = best-of-8 by pLDDT  -  best-of-8 by the model's own likelihood
is the quantity it should shrink. A variant that improves pLDDT/TM while leaving the gap alone has
improved the generator, not the ranking, and will not compound over rounds.

The `nat dNLL` column is the drift monitor from each policy's metrics.json -- fixed sequences,
fixed masks, so it has no sampling noise at all. Rising means the policy left the natural manifold.
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import math
import os
import sys

import numpy as np

from config import CFG
from .preference import load_pool, pass_at_k, score, selector_at_k, succeeded


def sign_test(n_better, n_worse):
    """Two-sided exact binomial p at q=0.5, ties excluded. -> p, or nan when nothing differs."""
    n = n_better + n_worse
    if n == 0:
        return float("nan")
    k = min(n_better, n_worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def best_of(ss, acfg, k, key):
    """Mean reward of the top-1 of k by `key`, as an expectation over k-subsets.

    Exactly as selector_at_k weights ranks, but carrying the reward rather than a success bit, so
    small pools give a smooth number instead of a mostly-zero one.
    """
    n = len(ss)
    if k > n:
        return float("nan")
    order = sorted(ss, key=(lambda s: score(s, acfg)) if key == "_reward" else
                   (lambda s: s.get(key, 0.0)), reverse=True)
    tot = math.comb(n, k)
    return sum(math.comb(n - 1 - r, k - 1) * score(s, acfg)
               for r, s in enumerate(order) if n - 1 - r >= k - 1) / tot


def summarise(pool, acfg, k=8):
    pids = sorted(p for p in pool if len(pool[p]) >= k)
    allg = [s for p in pids for s in pool[p]]
    if not allg:
        return None
    row = {
        "prompts": len(pids), "n": len(allg),
        "success": float(np.mean([succeeded(s, acfg) for s in allg])),
        "plddt": float(np.mean([s["plddt"] for s in allg])),
        "tm": float(np.mean([s["tm"] for s in allg])),
        "reward": float(np.mean([score(s, acfg) for s in allg])),
        "oracle": float(np.mean([pass_at_k(len(pool[p]),
                                           sum(succeeded(s, acfg) for s in pool[p]), k)
                                 for p in pids])),
    }
    for key, name in (("plddt", "sel_plddt"), ("loglik", "sel_loglik")):
        vals = [selector_at_k([succeeded(s, acfg)
                               for s in sorted(pool[p], key=lambda s: s.get(key, 0.0),
                                               reverse=True)], k) for p in pids]
        row[name] = float(np.mean(vals))
    row["gap"] = row["sel_plddt"] - row["sel_loglik"]
    # per-prompt, for the paired comparison
    row["_per_prompt"] = {p: best_of(pool[p], acfg, k, "plddt") for p in pids}
    return row


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg = CFG.align
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-root", required=True, help="directory of per-variant eval dirs")
    ap.add_argument("--policy-root", default=None,
                    help="where policy_<variant>/metrics.json live (default: eval-root's parent)")
    ap.add_argument("--base", default="base", help="variant every other one is compared against")
    ap.add_argument("--k", type=int, default=8, help="best-of-k the table reports")
    ap.add_argument("--only", default=None, help="colon-separated subset of variant names")
    a = ap.parse_args()

    proot = a.policy_root or os.path.dirname(os.path.abspath(a.eval_root))
    names = sorted(os.path.basename(d.rstrip("/"))
                   for d in glob_.glob(os.path.join(a.eval_root, "*")) if os.path.isdir(d))
    if a.only:
        want = {t.strip() for t in a.only.split(":") if t.strip()}
        names = [n for n in names if n in want]
    if not names:
        raise SystemExit(f"no variant directories under {a.eval_root}")
    if a.base in names:                                   # baseline first, everything else after
        names = [a.base] + [n for n in names if n != a.base]

    rows, metrics = {}, {}
    for n in names:
        pool, ng, nf, nt = load_pool(os.path.join(a.eval_root, n))
        if not pool:
            print(f"[cmp] {n}: nothing joined ({ng} generated, {nf} folded, {nt} TM)", flush=True)
            continue
        r = summarise(pool, acfg, a.k)
        if r:
            rows[n] = r
        mp = os.path.join(proot, f"policy_{n}", "metrics.json")
        if os.path.exists(mp):
            metrics[n] = json.load(open(mp))

    if not rows:
        raise SystemExit("nothing to compare")
    k = a.k
    print(f"\nGENERATION on the held-out prompts   (best-of-{k}; success = pLDDT > "
          f"{acfg.plddt_success} AND {acfg.tm_field} > {acfg.tm_success})")
    print(f"{'variant':<10} {'prompts':>7} {'draw%':>7} {'pLDDT':>7} {'TM':>7} {'reward':>7} "
          f"{'oracle@'+str(k):>9} {'plddt@'+str(k):>9} {'logl@'+str(k):>9} {'loglik gap':>11}")
    print("-" * 94)
    for n in names:
        if n not in rows:
            continue
        r = rows[n]
        print(f"{n:<10} {r['prompts']:>7,} {r['success']:>6.2%} {r['plddt']:>7.3f} "
              f"{r['tm']:>7.3f} {r['reward']:>7.3f} {r['oracle']:>9.4f} {r['sel_plddt']:>9.4f} "
              f"{r['sel_loglik']:>9.4f} {r['gap']:>11.4f}")

    if a.base in rows:
        b = rows[a.base]
        print(f"\nPAIRED vs '{a.base}', per prompt (same prompts both sides, so difficulty cancels)")
        print(f"{'variant':<10} {'d reward':>9} {'better':>7} {'worse':>7} {'sign p':>8} "
              f"{'d draw%':>8} {'d gap':>8}")
        print("-" * 62)
        for n in names:
            if n == a.base or n not in rows:
                continue
            r = rows[n]
            shared = sorted(set(b["_per_prompt"]) & set(r["_per_prompt"]))
            d = [r["_per_prompt"][p] - b["_per_prompt"][p] for p in shared]
            nb = sum(1 for v in d if v > 1e-9)
            nw = sum(1 for v in d if v < -1e-9)
            print(f"{n:<10} {np.mean(d) if d else float('nan'):>+9.4f} {nb:>7} {nw:>7} "
                  f"{sign_test(nb, nw):>8.4f} {r['success'] - b['success']:>+7.2%} "
                  f"{r['gap'] - b['gap']:>+8.4f}")
        print("\n[cmp] 'd gap' is the one to want NEGATIVE: it is how much the model's own "
              "likelihood\n[cmp] caught up with pLDDT as a ranker, which is the thing preference "
              "tuning is for.")

    if metrics:
        print("\nTUNING   (from each policy's metrics.json; nat dNLL has NO sampling noise)")
        print(f"{'variant':<10} {'loss':>5} {'alpha':>7} {'beta':>6} {'h*':>7} {'epochs':>7} "
              f"{'h fin':>8} {'win':>6} {'nat dNLL':>9} {'nat min@':>9}")
        print("-" * 82)
        for n in names:
            m = metrics.get(n)
            if not m:
                continue
            d = (m.get("nat_nll_final") or float("nan")) - (m.get("nat_nll_baseline") or float("nan"))
            print(f"{n:<10} {m.get('loss', '-'):>5} {m.get('alpha', 0):>7.3g} "
                  f"{m.get('beta', 0):>6.3g} {m.get('equilibrium', float('nan')):>7.3g} "
                  f"{m.get('epochs', 0):>7.2f} {m.get('h_final', float('nan')):>+8.4f} "
                  f"{(m.get('win_final') or 0):>5.0%} {d:>+9.4f} "
                  f"{m.get('nat_nll_min_step', 0):>9}")
        print("\n[cmp] h fin should sit near h* for IPO. If it is far above, alpha is too small "
              "and the\n[cmp] margin is not binding -- the run is closer to SFT than to IPO "
              "(see config.AlignCfg).")


if __name__ == "__main__":
    main()
