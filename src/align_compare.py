"""Put several tuned policies side by side on ONE held-out prompt set.

    python -m src.align_compare --eval-root <round>/eval --policy-root <round>

PAIRED, BECAUSE THE VARIANTS SHARE THE PROMPTS. Every variant generates from the same
prompts_eval.jsonl, so the interesting statistic is not "which mean is higher" -- prompt difficulty
dominates that and it is the same difficulty on both sides -- but the per-prompt difference against
the baseline. Comparing means across independently drawn prompt sets is how the first version of
the FILIP specificity test awarded a null control a +0.26 margin; this is the same lesson applied
before the fact.

READ LCR AND k13 BEFORE BELIEVING A pLDDT GAIN. Repetitive sequences fold CONFIDENTLY, so pLDDT is
the gameable half of the reward and TM is the half that is not -- a poly-alanine helix scores well
on one and nowhere on the other. A variant whose pLDDT moves several times as far as its TM is the
shape of a reward being gamed, and the degeneracy columns are what settle it. The first version of
this table omitted them and a DPO variant won on pLDDT +0.069 against TM +0.014 with no way to tell
which it was.

READ THE PER-BIN SPLIT TOO. A pooled row mixes 50%-masked scaffold completion with cold start, and
they are different tasks with success rates an order of magnitude apart: measured, ~29% pooled
against ~0.7% on caption-conditioned cold start. The rate-1.0 bin is the deployment condition and
the only place the likelihood pathology below has ever been large, so a pooled gain that lives
entirely in the easy bins is not the gain anyone wants.

`loglik gap` = best-of-k by pLDDT minus best-of-k by the model's OWN log-likelihood. On cold-start
generation the model ranks good proteins below bad ones -- best-of-N by likelihood measured 0.007
falling to 0.000 where pLDDT rose to 0.040 -- and preference tuning exists to repair that. On
scaffold prompts the pathology is much milder, which is exactly why the per-bin split matters.

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
from .metrics import kmer_counts, lcr_counts
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


def degeneracy(seqs, ks=(13,)):
    """-> (LCR fraction, {k: within-sequence k-mer repeat coverage}).

    Both are the standard PLD2 detectors: SEG-style low complexity, and long-range k-mer repetition
    at k beyond the SEG window, which LCR cannot see at all.
    """
    lcr, tot = lcr_counts(seqs)
    c = kmer_counts(seqs, ks)
    return lcr / max(tot, 1), {k: c[k]["rep_pos"] / max(c[k]["n_pos"], 1) for k in ks}


def summarise(pool, acfg, k=8):
    pids = sorted(p for p in pool if len(pool[p]) >= k)
    allg = [s for p in pids for s in pool[p]]
    if not allg:
        return None
    lcr, rep = degeneracy([s["seq"] for s in allg if s.get("seq")])
    row = {
        "prompts": len(pids), "n": len(allg),
        "success": float(np.mean([succeeded(s, acfg) for s in allg])),
        "plddt": float(np.mean([s["plddt"] for s in allg])),
        "tm": float(np.mean([s["tm"] for s in allg])),
        "reward": float(np.mean([score(s, acfg) for s in allg])),
        "lcr": lcr, "k13": rep[13],
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


def split_by_bin(pool):
    """{mask rate: sub-pool}. A prompt has exactly one rate, so this splits prompts, not samples --
    which is what keeps the paired comparison inside a bin paired."""
    out = {}
    for pid, ss in pool.items():
        out.setdefault(float(ss[0].get("rate", -1.0)), {})[pid] = ss
    return out


def paired_delta(a, b):
    """(mean difference, n better, n worse, sign p) over prompts both rows scored."""
    shared = sorted(set(a["_per_prompt"]) & set(b["_per_prompt"]))
    d = [a["_per_prompt"][p] - b["_per_prompt"][p] for p in shared]
    nb = sum(1 for v in d if v > 1e-9)
    nw = sum(1 for v in d if v < -1e-9)
    return (float(np.mean(d)) if d else float("nan")), nb, nw, sign_test(nb, nw)


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

    rows, metrics, bins = {}, {}, {}
    for n in names:
        pool, ng, nf, nt = load_pool(os.path.join(a.eval_root, n))
        if not pool:
            print(f"[cmp] {n}: nothing joined ({ng} generated, {nf} folded, {nt} TM)", flush=True)
            continue
        r = summarise(pool, acfg, a.k)
        if r:
            rows[n] = r
            bins[n] = {rate: summarise(sub, acfg, a.k)
                       for rate, sub in split_by_bin(pool).items()}
        mp = os.path.join(proot, f"policy_{n}", "metrics.json")
        if os.path.exists(mp):
            metrics[n] = json.load(open(mp))

    if not rows:
        raise SystemExit("nothing to compare")
    k = a.k
    print(f"\nGENERATION on the held-out prompts   (best-of-{k}; success = pLDDT > "
          f"{acfg.plddt_success} AND {acfg.tm_field} > {acfg.tm_success})")
    print(f"{'variant':<10} {'prompts':>7} {'draw%':>7} {'pLDDT':>7} {'TM':>7} {'reward':>7} "
          f"{'LCR':>7} {'k13':>7} {'oracle@'+str(k):>9} {'plddt@'+str(k):>9} {'logl@'+str(k):>9} "
          f"{'gap':>8}")
    print("-" * 105)
    for n in names:
        if n not in rows:
            continue
        r = rows[n]
        print(f"{n:<10} {r['prompts']:>7,} {r['success']:>6.2%} {r['plddt']:>7.3f} "
              f"{r['tm']:>7.3f} {r['reward']:>7.3f} {r['lcr']:>6.1%} {r['k13']:>6.1%} "
              f"{r['oracle']:>9.4f} {r['sel_plddt']:>9.4f} {r['sel_loglik']:>9.4f} "
              f"{r['gap']:>8.4f}")
    if a.base in rows:
        b0 = rows[a.base]
        # The check the first version of this table could not make. pLDDT is the gameable half of
        # the reward and TM is not, so a gain that is mostly pLDDT with LCR rising is the metric
        # being played rather than the model improving.
        for n in names:
            if n == a.base or n not in rows:
                continue
            r = rows[n]
            dp, dt = r["plddt"] - b0["plddt"], r["tm"] - b0["tm"]
            if dp > 0.01 and (r["lcr"] > b0["lcr"] + 0.02 or r["k13"] > b0["k13"] + 0.01):
                print(f"[cmp] WARNING: '{n}' gains pLDDT {dp:+.3f} while LCR moves "
                      f"{r['lcr'] - b0['lcr']:+.1%} and k13 {r['k13'] - b0['k13']:+.1%}. "
                      f"Repetitive sequences fold confidently -- read the sequences before "
                      f"believing this.")
            elif dp > 0.01 and dt < 0.2 * dp:
                print(f"[cmp] note: '{n}' moves pLDDT {dp:+.3f} but TM only {dt:+.3f} "
                      f"({dp / max(dt, 1e-9):.0f}x). Degeneracy is flat, so this is not obviously "
                      f"gaming -- but TM is the half that cannot be gamed and it barely moved.")

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
        # PER-BIN, because a pooled row mixes 50%-masked completion with cold start and they are
        # different tasks with success rates an order of magnitude apart.
        rates = sorted({rt for n in bins for rt in bins[n] if bins[n][rt]})
        if len(rates) > 1:
            print("\nPER MASK-RATE BIN   (rate 1.00 is the cold start -- the deployment condition)")
            print(f"{'rate':>5} {'variant':<10} {'prompts':>7} {'draw%':>7} {'pLDDT':>7} "
                  f"{'TM':>7} {'reward':>7} {'d rew':>8} {'sign p':>7} {'LCR':>6} "
                  f"{'plddt@'+str(k):>9} {'logl@'+str(k):>9}")
            print("-" * 103)
            for rt in rates:
                for n in names:
                    r = bins.get(n, {}).get(rt)
                    if not r:
                        continue
                    bb = bins.get(a.base, {}).get(rt)
                    if bb and n != a.base:
                        dv, _, _, pv = paired_delta(r, bb)
                        dstr, pstr = f"{dv:+8.4f}", f"{pv:7.4f}"
                    else:
                        dstr, pstr = " " * 8, " " * 7
                    print(f"{rt:>5.2f} {n:<10} {r['prompts']:>7,} {r['success']:>6.2%} "
                          f"{r['plddt']:>7.3f} {r['tm']:>7.3f} {r['reward']:>7.3f} {dstr} {pstr} "
                          f"{r['lcr']:>5.1%} {r['sel_plddt']:>9.4f} {r['sel_loglik']:>9.4f}")
                print()
            print("[cmp] a pooled gain that lives only in the low-rate bins is scaffold completion\n"
                  "[cmp] getting better, which was never the failing capability. Read the 1.00 rows.")

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
