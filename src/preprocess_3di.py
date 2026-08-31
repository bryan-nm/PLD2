"""Pre-tokenise a PAIRED (amino acid, 3Di) corpus into two-track shards.

    python -m src.preprocess_3di aa.fasta 3di.fasta out_shard_dir --min 30 --max 500

THE PAIRING IS THE WHOLE POINT, so it is verified rather than assumed. The two FASTAs are streamed
in LOCKSTEP and a record is written only if the headers match and the two sequences are the same
length. A silently mis-paired corpus -- one extra record in one file and everything after it shifts
by one -- would train the model to map each sequence onto the WRONG structure, and nothing
downstream could detect it: the loss would still fall, the samples would still look like proteins,
and the structure track would be pure noise dressed as signal. Mismatches are counted, the first few
are printed, and --max-mismatch aborts the run rather than writing a corpus you cannot trust.

SHARD LAYOUT, chosen to leave the existing aa-only UniRef shards readable unchanged:

    shard_00000.bin    uint8 amino-acid ids 0..19          (exactly as preprocess_fasta.py writes)
    shard_00000.idx    int64 offsets, n+1 entries          (exactly as before -- SHARED by both)
    shard_00000.3di    uint8 3Di state ids 0..19           (NEW, same offsets, same total length)

One .idx serves both tracks because the pairing guarantees equal lengths, which also means the
offset check in data.ProteinShards validates the 3Di file for free. A shard directory with no .3di
files is an aa-only corpus and reads exactly as it did before; data.ProteinShards decides per shard,
so the two corpora can even be mixed in one directory.

THE 3Di ALPHABET IS NOT THE AMINO-ACID ALPHABET even though it is written with the same 20 letters.
'L' as a 3Di state has nothing to do with leucine. They are tokenised with separate tables here and
embedded with separate tables in the model, because sharing them would ask the model to hold one
vector that means both things.
"""
from __future__ import annotations
import argparse
import gzip
import os

import numpy as np

from .data import DI, _DI_ENCODE, _ENCODE_TABLE, _INVALID


def _open(path):
    return gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")


def fasta_iter(path):
    """(header, sequence) as BYTES -- no decode, no strip per line beyond the newline."""
    name, buf = None, []
    with _open(path) as f:
        for line in f:
            if line[:1] == b">":
                if name is not None:
                    yield name, b"".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line.strip())
    if name is not None:
        yield name, b"".join(buf)


def _encode(seq: bytes, table: np.ndarray):
    ids = table[np.frombuffer(seq, dtype=np.uint8)]
    return None if (ids == _INVALID).any() else ids


class PairedShardWriter:
    def __init__(self, out_dir, shard_size):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.shard_size = out_dir, shard_size
        self.sid, self.aa, self.di, self.offs = 0, [], [], [0]

    def add(self, aa_ids, di_ids):
        self.aa.append(aa_ids)
        self.di.append(di_ids)
        self.offs.append(self.offs[-1] + len(aa_ids))
        if len(self.aa) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.aa:
            return
        base = os.path.join(self.out_dir, f"shard_{self.sid:05d}")
        # .bin last: data.ProteinShards globs *.bin and validates the .idx against it, so a shard
        # only becomes visible to a reader once its index and structure track are already on disk.
        np.concatenate(self.di).tofile(base + ".3di")
        np.asarray(self.offs, dtype=np.int64).tofile(base + ".idx")
        np.concatenate(self.aa).tofile(base + ".bin")
        self.sid, self.aa, self.di, self.offs = self.sid + 1, [], [], [0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("aa_fasta")
    ap.add_argument("di_fasta")
    ap.add_argument("out_dir")
    ap.add_argument("--min", type=int, default=30)
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--shard-size", type=int, default=1_000_000)
    ap.add_argument("--max-mismatch", type=int, default=0,
                    help="abort if more than this many header/length mismatches are seen; the "
                         "default of 0 refuses any misalignment at all")
    a = ap.parse_args()

    w = PairedShardWriter(a.out_dir, a.shard_size)
    n = kept = bad_pair = bad_len = bad_char = 0
    shown = 0
    for (ha, sa), (hd, sd) in zip(fasta_iter(a.aa_fasta), fasta_iter(a.di_fasta)):
        n += 1
        if ha != hd or len(sa) != len(sd):
            bad_pair += 1
            if shown < 5:
                shown += 1
                print(f"  MISPAIR at record {n}: {ha[:40]!r} / {hd[:40]!r} "
                      f"len {len(sa)} vs {len(sd)}", flush=True)
            if bad_pair > a.max_mismatch:
                raise SystemExit(
                    f"\n{bad_pair} mispaired record(s) by record {n}. The two FASTAs are not in the "
                    f"same order, so every shard written past the first mismatch would attach the "
                    f"WRONG structure to each sequence -- a corruption that trains happily and is "
                    f"invisible downstream. Fix the inputs, or raise --max-mismatch if you are "
                    f"certain these are isolated and acceptable.")
            continue
        if not (a.min <= len(sa) <= a.max):
            bad_len += 1
            continue
        aa = _encode(sa, _ENCODE_TABLE)
        di = _encode(sd, _DI_ENCODE)
        if aa is None or di is None:
            bad_char += 1
            continue
        w.add(aa, di)
        kept += 1
        if kept % 1_000_000 == 0:
            print(f"kept {kept:,} / {n:,} read", flush=True)
    w.flush()
    print(f"\ndone: {n:,} records read | kept {kept:,} | dropped: {bad_len:,} out of length window, "
          f"{bad_char:,} unmappable, {bad_pair:,} mispaired", flush=True)
    print(f"      -> {w.sid} paired shards in {a.out_dir}  (3Di alphabet: {DI})", flush=True)
    if w.sid < 2:
        print("WARNING: fewer than 2 shards; make --shard-size smaller so the strided holdout has "
              "something to work with.", flush=True)


if __name__ == "__main__":
    main()
