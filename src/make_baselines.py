"""Write the reference FASTAs the generated samples are read against.

Two references, both from the HELD-OUT shard (data.ProteinShards(split="holdout")), so they are
sequences the model has never been trained on:

  natural.fasta   real UniRef sequences -- the best case for every metric.
  shuffled.fasta  the SAME sequences, each independently permuted. This preserves every sequence's
                  exact amino-acid composition while destroying its order, which is the control that
                  separates "is the ORDER meaningful" from "is the COMPOSITION plausible". A model
                  whose samples fold no better than this has learned composition and nothing else --
                  which is exactly what ProLoopDiff's 181k checkpoint did at T=1.0 (pLDDT 33.9
                  against a 37.5 shuffled baseline).

They land in SAMPLES_DIR alongside the training-time samples, so the same fold_fasta watcher scores
them through the same code path and they appear as rows in the same summary table. Run once before
(or alongside) training; fold_fasta skips ids it has already scored, so a re-run is free.

    python -m src.make_baselines
    python -m src.make_baselines --n 500 --out-dir /path/to/samples
"""
from __future__ import annotations
import argparse
import os
import random

from config import CFG, SAMPLES_DIR, UNIREF_SHARDS
from .blosum import AA
from .data import ProteinShards
from .metrics import kmer_counts, kmer_line, lcr_counts, length_stats


def _to_seq(ids, eos_id):
    out = []
    for t in ids:
        if t == eos_id or t >= len(AA):
            break
        out.append(AA[t])
    return "".join(out)


def main():
    ocfg = CFG.opt
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=ocfg.n_baseline, help="sequences per baseline")
    ap.add_argument("--out-dir", default=SAMPLES_DIR)
    ap.add_argument("--shards", default=UNIREF_SHARDS)
    ap.add_argument("--max-len", type=int, default=None,
                    help="drop sequences longer than this (default: the eval canvas, so the "
                         "baseline covers the same length regime the model can generate)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mcfg = CFG.model_config()
    max_len = args.max_len or ocfg.eval_canvas
    shards = ProteinShards(args.shards, mcfg.eos_token_id, split="holdout")
    if len(shards) == 0:
        raise SystemExit(
            f"no held-out shard in {args.shards}. ProteinShards reserves the LAST shard file, so a "
            f"single-shard directory has none to give -- re-run src.preprocess_fasta with a smaller "
            f"--shard-size, or pass --shards at a directory that has more than one.")

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(shards)), min(args.n * 3, len(shards)))   # oversample, then filter
    natural = []
    for i in idx:
        s = _to_seq(shards.get(i), mcfg.eos_token_id)
        if ocfg.fold_min_len <= len(s) <= max_len:
            natural.append(s)
        if len(natural) >= args.n:
            break

    shuffled = []
    for s in natural:
        chars = list(s)
        rng.shuffle(chars)
        shuffled.append("".join(chars))

    os.makedirs(args.out_dir, exist_ok=True)
    for name, seqs in (("natural", natural), ("shuffled", shuffled)):
        path = os.path.join(args.out_dir, f"{name}.fasta")
        with open(path, "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">{name}_{i} len={len(s)}\n{s}\n")
        mean, sd = length_stats([len(s) for s in seqs])
        lcr, tot = lcr_counts(seqs)
        print(f"[baseline] {name:<9} n={len(seqs)} len {mean:.1f}+-{sd:.1f} "
              f"LCR {lcr / max(tot, 1):.1%} | {kmer_line(kmer_counts(seqs, ocfg.kmer_ks), ocfg.kmer_ks)}"
              f"\n           -> {path}", flush=True)
    print("[baseline] read every later number against these two rows: natural is the ceiling, "
          "shuffled is the composition-matched floor.", flush=True)


if __name__ == "__main__":
    main()
