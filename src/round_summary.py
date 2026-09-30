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


def main():
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=CFG.align.dir, help="directory holding round*/")
    a = ap.parse_args()

    dirs = sorted(glob_.glob(os.path.join(a.dir, "round*")),
                  key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0))
    rows = [(os.path.basename(d), load(d)) for d in dirs]
    rows = [(n, r) for n, r in rows if r.get("rep")]
    if not rows:
        raise SystemExit(f"no round*/report.json under {a.dir}. Rounds written before "
                         f"src.preference started emitting it will not appear.")

    print(f"\n{'round':<8} {'reward':>8} {'d rew':>8} {'success':>8} {'pLDDT':>7} {'TM':>7} "
          f"{'sigma':>7} {'gate%':>7} {'pairs':>7} {'nat dNLL':>9} {'h fin':>8}")
    print("-" * 92)
    prev = None
    for name, r in rows:
        rep, met = r["rep"], r.get("met", {})
        d = f"{rep['reward'] - prev:+8.4f}" if prev is not None else " " * 8
        nat = met.get("nat_nll_final")
        base = met.get("nat_nll_baseline")
        dn = f"{nat - base:+9.4f}" if (nat is not None and base is not None) else " " * 9
        hf = met.get("h_final")
        print(f"{name:<8} {rep['reward']:>8.4f} {d} {rep['success_rate']:>7.2%} "
              f"{rep['plddt']:>7.3f} {rep['tm']:>7.3f} {rep['within_prompt_sigma']:>7.4f} "
              f"{rep['above_deg_gate']:>6.1%} {rep['pairs']:>7,} {dn} "
              f"{(f'{hf:+8.4f}' if hf is not None else ' ' * 8)}")
        prev = rep["reward"]

    if len(rows) > 1:
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
