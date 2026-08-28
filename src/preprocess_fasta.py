"""Pre-tokenise a UniRef FASTA into packed binary shards for fast, shardable training reads.

Streaming a multi-hundred-GB FASTA in the training loop is slow (parse every epoch) and awkward to
shard across thousands of ranks. Tokenise ONCE into packed uint8 shards of residue ids 0..19 (NO
EOS -- appended at read time by data.ProteinShards) plus an int64 offset index: random access is
O(1), rank-sharding is trivial, and there is no FASTA parsing on the hot path.

Filters: length in [min_len, max_len] and standard-mappable residues only (instruction 5 -- the
size-filtered UniRef sequences are the only corpus PLD2 sees). Optional exact dedup.

NOTE: data.ProteinShards reserves the LAST shard file as the held-out split for
src.make_baselines, so --shard-size must be small enough that the corpus produces more than one
shard. The 1M default gives ~100+ shards on UniRef90, i.e. a ~1% holdout.

    python -m src.preprocess_fasta in.fasta[.gz] out_shard_dir --min 30 --max 500
"""
from __future__ import annotations
import argparse
import gzip
import os

import numpy as np

from .data import _AA_ID, _RARE


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def encode(seq: str):
    out = []
    for c in seq.upper():
        c = _RARE.get(c, c)
        j = _AA_ID.get(c)
        if j is None:
            return None
        out.append(j)
    return out


def fasta_iter(path):
    name, buf = None, []
    with _open(path) as f:
        for line in f:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, "".join(buf)


class ShardWriter:
    def __init__(self, out_dir, shard_size):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.shard_size = out_dir, shard_size
        self.sid, self.buf, self.offs = 0, [], [0]

    def add(self, ids):
        self.buf.append(np.asarray(ids, dtype=np.uint8))
        self.offs.append(self.offs[-1] + len(ids))
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        base = os.path.join(self.out_dir, f"shard_{self.sid:05d}")
        np.concatenate(self.buf).tofile(base + ".bin")
        np.asarray(self.offs, dtype=np.int64).tofile(base + ".idx")
        self.sid, self.buf, self.offs = self.sid + 1, [], [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta")
    ap.add_argument("out_dir")
    ap.add_argument("--min", type=int, default=30)
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--dedup", action="store_true", help="drop exact duplicates within the FASTA")
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    a = ap.parse_args()

    seen = set()
    kept = skipped = 0
    w = ShardWriter(a.out_dir, a.shard_size)
    for _, seq in fasta_iter(a.fasta):
        seq = seq.strip().upper()
        if not (a.min <= len(seq) <= a.max):
            skipped += 1
            continue
        ids = encode(seq)
        if ids is None:
            skipped += 1
            continue
        if a.dedup:
            h = hash(seq)
            if h in seen:
                skipped += 1
                continue
            seen.add(h)
        w.add(ids)
        kept += 1
        if kept % 1_000_000 == 0:
            print(f"kept {kept:,} skipped {skipped:,}", flush=True)
    w.flush()
    n_shards = w.sid
    print(f"done: kept {kept:,} skipped {skipped:,} -> {n_shards} shards in {a.out_dir}", flush=True)
    if n_shards < 2:
        print("WARNING: only one shard, so there is no held-out split for src.make_baselines. "
              "Re-run with a smaller --shard-size.", flush=True)


if __name__ == "__main__":
    main()
