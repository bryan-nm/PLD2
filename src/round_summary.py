"""One row per alignment round, so a chain reads as a curve.  python -m src.round_summary

    python -m src.round_summary --dir <align dir>

THE QUESTION A CHAIN OF ROUNDS EXISTS TO ANSWER is where it plateaus, and that is not visible in
any single round's log. Each round writes report.json (generation quality, measured on prompts the
previous policy had never seen) and policy/metrics.json (what the tuning did). This joins them.

WHAT TO READ, in order:

  reward        what the tuning moves. Round over round this is the curve.
  d reward      the increment. When it stops being positive, the chain has plateaued.
  sigma         within-prompt spread -- the headroom the NEXT round has to move into, since the
                winner it will imitate is a best-of-n order statistic of this distribution. If
                sigma collapses the chain stops even while reward is still rising, so watch it
                ahead of the plateau rather than after.
  gate%         fraction of generations too degenerate to promote. Should fall; rising means the
                reward is being bought with repetition.
  nat dNLL      the drift monitor, which has no sampling noise. Departure from the natural manifold
                is partly the point, but it compounds across rounds in a way one round cannot show.

--by-bin SPLITS EVERY ROW BY MASK RATE, which the pooled table cannot do and which changes what
the curve means. A round's prompts are stratified over 0.5/0.7/0.85/1.0, so the pooled pLDDT
averages four different tasks. Rate 1.0 is cold start -- no scaffold at all -- and a pooled gain
is perfectly consistent with cold start standing still while the 50%-masked bin improves. That
would be the alignment buying the easy half of the distribution, and it is the one failure this
tool could not see. Rounds written before report.json carried by_bin are recomputed from
gen/folds/tm on disk, so the whole history is available without re-running anything.
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import os
import re
import sys

from config import CFG


def load(d):
    rep = os.path.join(d, "report.json")
    met = os.path.join(d, "policy", "metrics.json")
    out = {}
    for path, key in ((rep, "rep"), (met, "met")):
        if os.path.exists(path):
            try:
                out[key] = json.load(open(path))
            except Exception:
                pass
    return out


def bins_for(d):
    """Per-bin rows for one round: from report.json when it has them, else recomputed from disk.

    The recompute path is what makes the existing chain readable -- rounds 3-10 were written
    before report.json carried by_bin, and re-running them to get four numbers is not an option.
    load_pool() is the same join src.preference uses to build the pairs, so the figures are the
    ones the round actually trained on, not an approximation of them.
    """
    rep = os.path.join(d, "report.json")
    if os.path.exists(rep):
        try:
            by = json.load(open(rep)).get("by_bin")
            if by:
                return by, "report.json"
        except Exception:
            pass
    try:
        from src.preference import bin_stats, load_pool
        pool, _, _, _ = load_pool(d)
        if not pool:
            return [], "no pool"
        return bin_stats(pool, CFG.align), "recomputed"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"


def print_by_bin(dirs_rows, cache=None, cache_src=None):
    """One block per mask rate, each a round-over-round curve."""
    cache, cache_src = cache or {}, cache_src or {}
    per_round, srcs = [], {}
    for name, d in dirs_rows:
        by, src = ((cache[name], cache_src.get(name, "cached")) if name in cache
                   else bins_for(d))
        if by:
            per_round.append((name, by))
            srcs[name] = src
        else:
            print(f"[bins] {name}: no per-bin data ({src})")
    if not per_round:
        return
    rates = sorted({r["rate"] for _, by in per_round for r in by})
    for rate in rates:
        cold = rate >= 1.0
        print(f"\n  mask rate {rate}{'   <- COLD START (no scaffold)' if cold else ''}")
        print(f"  {'round':<8} {'n':>7} {'reward':>8} {'d rew':>8} {'pLDDT':>7} {'pTM':>7} "
              f"{'pTM>.5':>7} {'TM':>7} {'success':>8} {'gate%':>7} {'len':>6}")
        print("  " + "-" * 88)
        prev = None
        for name, by in per_round:
            r = next((x for x in by if x["rate"] == rate), None)
            if r is None:
                continue
            d = f"{r['reward'] - prev:+8.4f}" if prev is not None else " " * 8
            pc = r.get("ptm_confident")
            pcs = f"{pc:>6.1%}" if pc is not None else " " * 7
            print(f"  {name:<8} {r['n']:>7,} {r['reward']:>8.4f} {d} {r['plddt']:>7.3f} "
                  f"{r['ptm']:>7.3f} {pcs} {r['tm']:>7.3f} {r['success_rate']:>7.1%} "
                  f"{r['above_deg_gate']:>6.1%} {r['len']:>6.0f}")
            prev = r["reward"]
    print("\n[bins] Read the cold-start block against the others. If its reward curve is flat "
          "while the\n[bins] lower rates climb, the pooled gain is inpainting, not design.")
    if any(src == "recomputed" for src in srcs.values()):
        from src.preference import reward_formula
        rr = [k for k, v in srcs.items() if v == "recomputed"]
        print(f"[bins] RECOMPUTED rows use the CURRENT reward -- {reward_formula(CFG.align)} -- "
              f"which is\n[bins] not necessarily the reward those rounds trained on. "
              f"{len(rr)} row(s) here: {', '.join(sorted(rr)[:8])}"
              + (" ..." if len(rr) > 8 else "")
              + ".\n[bins] The pooled table above reads each round's own report.json and is "
                "therefore historical;\n[bins] the two will disagree for any round predating a "
                "reward change. That is not a bug in\n[bins] either -- it is the reward having "
                "changed, which only this line makes visible.")
    print("[bins] TM vs pTM: TM is foldseek against the prompt's reference -- the RIGHT fold. pTM "
          "is\n[bins] ESMFold's own topology estimate, no reference involved. At rate 1.0 the "
          "prompt gives\n[bins] only the length, so TM is pinned at the unrelated-fold floor by "
          "construction and pTM is\n[bins] the only one of the two that can move -- which is why "
          "the reward weights them by\n[bins] mask rate rather than using TM alone.")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=CFG.align.dir, help="directory holding round*/")
    ap.add_argument("--plateau", type=int, default=None, metavar="PATIENCE",
                    help="print 'STOP' or 'CONTINUE' for the chain loop and exit; the rule is "
                         "that the best round is more than PATIENCE rounds back")
    ap.add_argument("--by-bin", action="store_true",
                    help="split every round by mask rate; recomputes from gen/folds/tm when "
                         "report.json predates by_bin (slower, reads the whole round)")
    a = ap.parse_args()

    if a.plateau is not None:
        series = reward_series(a.dir)
        stop, why = plateau([v for _, v in series], a.plateau)
        names = ", ".join(f"{k.replace('round', 'r')}={v:.4f}" for k, v in series[-5:])
        print(f"[plateau] {'STOP' if stop else 'CONTINUE'} -- {why}")
        print(f"[plateau] last rounds: {names or '<none scored>'}")
        return

    dirs = sorted(glob_.glob(os.path.join(a.dir, "round*")),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    rows = [(os.path.basename(d), load(d)) for d in dirs]
    # SAY WHICH ROUNDS ARE BEING LEFT OUT, AND WHY. report.json is written by phase 4, so a round
    # run with PHASES=012 has generations and folds on disk and no row here -- and the table then
    # looks identical to the previous run's, which reads as "the summary ignored my new round".
    for name, r in rows:
        if r.get("rep"):
            continue
        d = os.path.join(a.dir, name)
        has_gen = bool(glob_.glob(os.path.join(d, "gen.rank*.jsonl")))
        has_tm = bool(glob_.glob(os.path.join(d, "tm.rank*.jsonl")))
        why = ("phase 4 (pairs) has not run" if has_gen and has_tm else
               "phase 3 (tm) has not run either" if has_gen else
               "no generations on disk")
        print(f"[rounds] {name}: no report.json -- {why}. "
              f"Re-run that round with PHASES={'4' if has_tm else '34'} to score it; "
              f"phases 0-2 are already on disk and will be skipped.")
    rows = [(n, r) for n, r in rows if r.get("rep")]
    if not rows and not a.by_bin:
        raise SystemExit(f"no round*/report.json under {a.dir}. Rounds written before "
                         f"src.preference started emitting it will not appear -- "
                         f"--by-bin reads gen/folds/tm directly and does not need it.")
    if not rows:
        print(f"[rounds] no round*/report.json under {a.dir}; the pooled table needs it. "
              f"Going straight to the per-bin split, which does not.")

    # --by-bin recomputes the whole round anyway, so when it runs it can also backfill pooled
    # pTM for rounds written before report.json carried it. Without it, those cells stay blank
    # rather than being silently filled with something that is not what the round measured.
    bins_by_round, bin_src = {}, {}
    if a.by_bin:
        for d in dirs:
            by, src = bins_for(d)
            if by:
                bins_by_round[os.path.basename(d)] = by
                bin_src[os.path.basename(d)] = src

    if rows:
        print_pooled(rows, bins_by_round)

    if a.by_bin:
        print_by_bin([(os.path.basename(d), d) for d in dirs], bins_by_round, bin_src)

    if len(rows) > 1:
        print_trend(rows)


def plateau(rewards, patience=2):
    """-> (stop, reason) from the reward series, oldest first.

    THE RULE IS "THE BEST ROUND IS MORE THAN `patience` ROUNDS BACK", not a threshold on the last
    increment. Run 1's increments were +0.055, +0.015, +0.006, +0.015, -0.031, -0.001: any
    two-round threshold test either fires on the +0.006 (noise) or waits for both of the negatives.
    Distance-from-the-best is monotone in evidence and needs no scale, which matters because the
    round-to-round noise here is 0.014-0.026 and the real increments are 0.01-0.05 -- a plateau
    simply cannot be called from two rounds. patience=2 is conservative; 1 is more aggressive.
    """
    if len(rewards) < patience + 2:
        return False, (f"only {len(rewards)} round(s); need {patience + 2} before a plateau can "
                       f"be called at patience={patience}")
    best = max(range(len(rewards)), key=lambda i: rewards[i])
    back = len(rewards) - 1 - best
    if back > patience:
        return True, (f"best reward {rewards[best]:.4f} was {back} round(s) back and nothing since "
                      f"has beaten it (latest {rewards[-1]:.4f}); patience={patience}")
    return False, (f"best reward {rewards[best]:.4f} is {back} round(s) back, within "
                   f"patience={patience} (latest {rewards[-1]:.4f})")


def reward_series(align_dir):
    """Each round's pooled reward from its report.json, in round order. Rounds without one are
    skipped: a round that has not been scored cannot argue either way."""
    out = []
    for d in sorted(glob_.glob(os.path.join(align_dir, "round*")),
                    key=lambda p_: int(re.sub(r"\D", "", os.path.basename(p_)) or 0)):
        rp = os.path.join(d, "report.json")
        if not os.path.exists(rp):
            continue
        try:
            out.append((os.path.basename(d), float(json.load(open(rp))["reward"])))
        except Exception:
            continue
    return out


def pooled_ptm(name, rep, bins_by_round):
    """Mean pTM for a round: from report.json, else weighted from the per-bin rows, else None.

    The two routes agree by construction -- bin_stats and the pooled block average the same
    samples -- so the weighted fallback is exact, not an approximation.
    """
    if rep.get("ptm") is not None:
        return rep["ptm"]
    by = bins_by_round.get(name)
    if not by:
        return None
    tot = sum(r["n"] for r in by)
    return sum(r["ptm"] * r["n"] for r in by) / tot if tot else None


def print_pooled(rows, bins_by_round=None):
    bins_by_round = bins_by_round or {}
    print(f"\n{'round':<8} {'reward':>8} {'d rew':>8} {'success':>8} {'pLDDT':>7} {'pTM':>7} "
          f"{'TM':>7} {'sigma':>7} {'gate%':>7} {'pairs':>7} {'nat dNLL':>9} {'h fin':>8}")
    print("-" * 100)
    prev = None
    for name, r in rows:
        rep, met = r["rep"], r.get("met", {})
        d = f"{rep['reward'] - prev:+8.4f}" if prev is not None else " " * 8
        nat = met.get("nat_nll_final")
        base = met.get("nat_nll_baseline")
        dn = f"{nat - base:+9.4f}" if (nat is not None and base is not None) else " " * 9
        hf = met.get("h_final")
        pt = pooled_ptm(name, rep, bins_by_round)
        pts = f"{pt:>7.3f}" if pt is not None else " " * 7
        print(f"{name:<8} {rep['reward']:>8.4f} {d} {rep['success_rate']:>7.2%} "
              f"{rep['plddt']:>7.3f} {pts} {rep['tm']:>7.3f} {rep['within_prompt_sigma']:>7.4f} "
              f"{rep['above_deg_gate']:>6.1%} {rep['pairs']:>7,} {dn} "
              f"{(f'{hf:+8.4f}' if hf is not None else ' ' * 8)}")
        prev = rep["reward"]


def print_trend(rows):
    first, last = rows[0][1]["rep"], rows[-1][1]["rep"]
    n = len(rows) - 1
    print(f"\n[rounds] over {n} transition(s): reward {first['reward']:.4f} -> "
          f"{last['reward']:.4f} ({last['reward'] - first['reward']:+.4f}, "
          f"{(last['reward'] - first['reward']) / n:+.4f}/round)")
    print(f"[rounds] success {first['success_rate']:.2%} -> {last['success_rate']:.2%} | "
          f"sigma {first['within_prompt_sigma']:.4f} -> {last['within_prompt_sigma']:.4f} "
          f"({last['within_prompt_sigma'] / first['within_prompt_sigma'] - 1:+.1%}) | "
          f"gate {first['above_deg_gate']:.1%} -> {last['above_deg_gate']:.1%}")
    deltas = [rows[i + 1][1]["rep"]["reward"] - rows[i][1]["rep"]["reward"]
              for i in range(len(rows) - 1)]
    if len(deltas) >= 2 and deltas[-1] < 0.25 * max(deltas):
        print(f"[rounds] the last increment ({deltas[-1]:+.4f}) is under a quarter of the "
              f"largest ({max(deltas):+.4f}).\n[rounds] That is what a plateau looks like -- "
              f"check sigma above before spending another round.")


if __name__ == "__main__":
    main()
