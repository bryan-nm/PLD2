"""Write the reference FASTAs the generated samples are read against.

Two references, both from the held-out split (every Nth sequence globally; see data.ProteinShards),
so they are sequences the model has never been trained on:

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

from config import CFG, SAMPLES_DIR, UNIREF_SHARDS, AFDB_SHARDS
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
    # FOLLOWS n_tracks, like train.py. The natural/shuffled rows are the ceiling and floor every
    # checkpoint is read against, so they have to be drawn from the corpus the model was actually
    # trained on. Pinning them to UniRef while training on AFDB would compare generations against a
    # different length distribution (AFDB median 170 vs UniRef ~250) and a set selected for having
    # AlphaFold models -- a mismatch that shifts both baselines and is invisible in the table.
    ap.add_argument("--shards",
                    default=AFDB_SHARDS if CFG.data.n_tracks == 2 else UNIREF_SHARDS)
    ap.add_argument("--max-len", type=int, default=None,
                    help="drop sequences longer than this (default: the eval canvas, so the "
                         "baseline covers the same length regime the model can generate)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mcfg = CFG.model_config()
    max_len = args.max_len or ocfg.eval_canvas
    shards = ProteinShards(args.shards, mcfg.eos_token_id, split="holdout",
                           holdout_stride=max(CFG.data.holdout_stride, 2))
    if len(shards) == 0:
        raise SystemExit(f"no sequences found in {args.shards} -- run src.preprocess_fasta first.")

    rng = random.Random(args.seed)
    idx = rng.sample(range(len(shards)), min(args.n * 3, len(shards)))   # oversample, then filter
    # The corpus-wide length distribution, straight from the offset arrays. Printed next to the
    # drawn sample so a non-representative draw is visible IN THE LOG rather than having to be
    # noticed by eye -- which is how the last-shard holdout bug survived until someone spotted that
    # 33.9 +- 2.3 could not be a random draw from a 30-500 corpus.
    corpus_len = shards.all_lengths()
    c_mean, c_sd = float(corpus_len.mean()), float(corpus_len.std())
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
        if name == "natural" and abs(mean - c_mean) > 3 * max(c_sd, 1e-6) / max(len(seqs), 1) ** 0.5:
            print(f"[baseline] WARNING: the drawn sample (mean {mean:.1f}) does not match the "
                  f"corpus length distribution (mean {c_mean:.1f} +- {c_sd:.1f} over "
                  f"{len(corpus_len):,} sequences). The holdout is not representative -- run "
                  f"`python -m src.inspect_shards` to see whether the corpus is ordered.", flush=True)
    print(f"[baseline] corpus length distribution: {c_mean:.1f} +- {c_sd:.1f} over "
          f"{len(corpus_len):,} sequences (the baselines above should match it)", flush=True)
    print("[baseline] read every later number against these two rows: natural is the ceiling, "
          "shuffled is the composition-matched floor.", flush=True)


if __name__ == "__main__":
    main()
