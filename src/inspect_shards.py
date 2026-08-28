"""Inspect a shard directory's length distribution and integrity:  python -m src.inspect_shards

Reads only the .idx offset arrays and the .bin file SIZES, never the sequence data, so it runs in
seconds over a hundred shards.

WHY THIS EXISTS. A "natural" baseline drawn from the held-out split came back at 33.9 +- 2.3 aa from
a corpus filtered to 30-500 -- the shortest ~1% of the data, pinned against the floor, with a spread
far too tight for any random draw. The reader was fine; the SPLIT was. It reserved the last shard,
and shard order is FASTA order, so on a length-sorted corpus the last shard is the extreme tail of
the length distribution. The split is strided now (order-agnostic), and this tool is how you check
whether a corpus is ordered before trusting anything derived from its layout.

Three things it reports:

  PER-SHARD LENGTH STATS -- if the mean drifts monotonically across shards, the FASTA is sorted, and
  ANY by-position split of it (first N, last N, one shard) is biased.

  ORDERING CORRELATION -- Spearman-style rank correlation between shard index and mean length, so
  "is it sorted" is a number rather than an eyeball judgement.

  INTEGRITY -- each .idx's final offset against its .bin's size. A mismatched pair is the realistic
  silent corruption here: numpy slices a memmap past its end WITHOUT error, so truncated sequences
  would flow into training looking exactly like short ones.
"""
from __future__ import annotations
import argparse
import glob
import os

import numpy as np


def main():
    from config import UNIREF_SHARDS
    ap = argparse.ArgumentParser()
    ap.add_argument("shard_dir", nargs="?", default=UNIREF_SHARDS)
    ap.add_argument("--stride", type=int, default=100, help="holdout stride to simulate")
    ap.add_argument("--per-shard", type=int, default=12, help="how many shard rows to print")
    args = ap.parse_args()

    bins = sorted(glob.glob(os.path.join(args.shard_dir, "*.bin")))
    if not bins:
        raise SystemExit(f"no *.bin in {args.shard_dir}")
    print(f"{args.shard_dir}: {len(bins)} shards\n")

    lens, per_shard, bad = [], [], []
    for b in bins:
        idx = b[:-4] + ".idx"
        if not os.path.exists(idx):
            bad.append((os.path.basename(b), "missing .idx", 0, os.path.getsize(b)))
            continue
        off = np.fromfile(idx, dtype="int64")
        size = os.path.getsize(b)
        if len(off) < 2 or int(off[-1]) != size:
            bad.append((os.path.basename(b), "idx/bin mismatch",
                        int(off[-1]) if len(off) else 0, size))
        L = np.diff(off).astype(np.int64)
        per_shard.append((os.path.basename(b), len(L), L.mean(), L.std(), L.min(), L.max()))
        lens.append(L)

    all_len = np.concatenate(lens)
    n = len(all_len)
    print(f"{'shard':<20} {'n':>10} {'mean':>8} {'sd':>7} {'min':>6} {'max':>6}")
    print("-" * 62)
    k = args.per_shard // 2
    show = per_shard if len(per_shard) <= args.per_shard else per_shard[:k] + [None] + per_shard[-k:]
    for r in show:
        if r is None:
            print(f"{'...':<20} {'...':>10}")
            continue
        print(f"{r[0]:<20} {r[1]:>10,} {r[2]:>8.1f} {r[3]:>7.1f} {r[4]:>6} {r[5]:>6}")
    print("-" * 62)
    q = np.percentile(all_len, [0, 1, 25, 50, 75, 99, 100])
    print(f"{'CORPUS':<20} {n:>10,} {all_len.mean():>8.1f} {all_len.std():>7.1f} "
          f"{all_len.min():>6} {all_len.max():>6}")
    print(f"{'percentiles':<20} min {q[0]:.0f} | 1% {q[1]:.0f} | 25% {q[2]:.0f} | 50% {q[3]:.0f} "
          f"| 75% {q[4]:.0f} | 99% {q[5]:.0f} | max {q[6]:.0f}")

    # --- is the corpus ordered? rank correlation of shard index vs shard mean length ---
    if len(per_shard) >= 5:
        means = np.array([r[2] for r in per_shard])
        ri = np.argsort(np.argsort(np.arange(len(means)))).astype(float)
        rm = np.argsort(np.argsort(means)).astype(float)
        rho = float(np.corrcoef(ri, rm)[0, 1])
        # Direction alone is not enough: with a handful of shards ANY monotone jitter gives
        # |rho| = 1, and three shards whose means differ by 1 aa would be flagged as "sorted".
        # The effect size -- how far the shard means actually spread relative to the corpus's own
        # spread -- is what separates a sorted corpus from noise.
        spread = float(means.max() - means.min()) / max(float(all_len.std()), 1e-9)
        print(f"\nordering: rank correlation(shard index, mean length) = {rho:+.3f} | "
              f"shard means span {means.max() - means.min():.1f} aa = {spread:.2f} corpus sd")
        if abs(rho) > 0.7 and spread > 0.5:
            print(f"  -> THE CORPUS IS SORTED BY LENGTH ({'descending' if rho < 0 else 'ascending'}). "
                  f"Any by-position split of it is biased; only a strided/random split is safe.\n"
                  f"     first shard mean {means[0]:.1f} aa, last shard mean {means[-1]:.1f} aa.")
        elif abs(rho) > 0.7:
            print("  -> monotone but negligible (shard means barely differ); not a real ordering.")
        else:
            print("  -> no strong length ordering across shards.")
    else:
        print(f"\nordering: only {len(per_shard)} shard(s) -- too few to test for ordering.")

    # --- what the two candidate holdouts would actually give you ---
    print(f"\nholdout comparison (n={n:,} sequences):")
    last = lens[-1]
    print(f"  last shard (the OLD split) : n={len(last):,} mean {last.mean():.1f} +- {last.std():.1f}")
    strided = all_len[::args.stride]
    print(f"  every {args.stride}th (the NEW split): n={len(strided):,} mean {strided.mean():.1f} "
          f"+- {strided.std():.1f}")
    print(f"  whole corpus               : n={n:,} mean {all_len.mean():.1f} +- {all_len.std():.1f}")
    print("  the new split should match the corpus row; the old one is only unbiased if the "
          "ordering line above says the corpus is unordered.")

    if bad:
        print(f"\nINTEGRITY: {len(bad)} damaged shard(s) -- these would silently yield TRUNCATED "
              f"sequences, because numpy slices a memmap past its end without error:")
        for name, why, off_end, size in bad[:20]:
            print(f"  {name}: {why} (idx says {off_end:,} bytes, .bin is {size:,})")
        print("  Re-run src.preprocess_fasta. ProteinShards refuses to open these.")
    else:
        print(f"\nINTEGRITY: all {len(bins)} .idx/.bin pairs agree on size.")


if __name__ == "__main__":
    main()
