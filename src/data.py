"""Data pipeline: pre-tokenised UniRef shards -> fixed-512-canvas batches.

Much smaller than ProLoopDiff's because PLD2 has one corpus and one shape:

  * ONE corpus. No SwissProt, no captions, no text embedder, no mixed/oversampled index -- PLD2 is
    unconditional and trains on the size-filtered UniRef shards only (instruction 5).

  * ONE canvas width, fixed at 512 (instruction 7). ProLoopDiff needed length buckets to keep the
    PAD tail short across variable widths; with a single fixed width there is nothing to bucket, so
    BucketedLengthSampler is gone. Every batch is exactly (B, 512) and B is constant, which is the
    static-shape property XPU wants anyway.

  * STATELESS, RESUMABLE SHUFFLING. StepBatchSampler derives each rank's batch from (seed, step,
    rank) alone -- there is no epoch counter and no giant permutation array. A run resumed at step S
    therefore draws EXACTLY the batches the original run would have, which ProLoopDiff's
    epoch-seeded permutation did not (its repro.py documents resumed jobs walking a batch sequence
    the original never saw). It also avoids materialising a ~1.2GB permutation of a 150M-row corpus
    on every rank.

  * HELD-OUT SPLIT. The LAST shard file is reserved: ProteinShards(split="train") skips it and
    split="holdout" reads only it. src/make_baselines.py draws the natural/shuffled fold references
    from the holdout, so the reference sequences a checkpoint is compared against are ones the model
    was never trained on. ProLoopDiff had no held-out split at all.
"""
from __future__ import annotations
import glob
import os
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .blosum import AA                                  # canonical 20-AA alphabet, id == index

# Map rare/ambiguous residues onto standard AAs; any other character rejects the sequence.
_RARE = {"B": "D", "Z": "E", "J": "L", "U": "C", "O": "K", "X": "A"}
_AA_ID = {c: i for i, c in enumerate(AA)}

# 256-entry byte->id table (255 = unmappable), so encode() is one C-level lookup rather than a
# per-character Python loop -- the difference that matters when tokenising ~150M sequences.
_INVALID = 255
_ENCODE_TABLE = np.full(256, _INVALID, dtype=np.uint8)
for _c, _i in _AA_ID.items():
    _ENCODE_TABLE[ord(_c)] = _i
for _r, _std in _RARE.items():
    _ENCODE_TABLE[ord(_r)] = _AA_ID[_std]


class ProteinTokenizer:
    def __init__(self, cfg):
        self.eos, self.pad, self.mask = cfg.eos_token_id, cfg.pad_token_id, cfg.mask_token_id

    def encode(self, seq: str) -> Optional[List[int]]:
        """Amino-acid string -> [ids..., EOS]. None if any character is unmappable."""
        b = seq.strip().upper().encode("ascii", "replace")   # non-ascii -> '?' -> _INVALID
        if not b:
            return None
        mapped = _ENCODE_TABLE[np.frombuffer(b, dtype=np.uint8)]
        if (mapped == _INVALID).any():
            return None
        return mapped.tolist() + [self.eos]

    def decode(self, ids) -> str:
        """Token ids -> amino-acid string, stopping at the first EOS/PAD/MASK."""
        out = []
        for t in ids:
            if t >= len(AA):
                break
            out.append(AA[t])
        return "".join(out)


class ProteinShards:
    """Reader for pre-tokenised shards: a uint8 .bin of residue ids 0..19 plus an int64 .idx of
    offsets (see preprocess_fasta.py). EOS is appended at read time; the shards store residues only.

    At ~100M+ sequences we must NOT build a per-sequence Python index. Only the per-shard offset
    arrays and a cumulative count are kept, and item i is located by searchsorted over the shards.

    THE TRAIN/HOLDOUT SPLIT IS STRIDED, NOT BY SHARD, and that correction matters. An earlier version
    reserved the LAST SHARD as the holdout, which is only unbiased if the corpus order is unbiased --
    an assumption there was never any basis for. Shard order is FASTA order, and a length-sorted
    FASTA (many distributions ship that way) puts the extreme tail of the length distribution in the
    last shard. Observed on the real UniRef shards: a 200-sequence "natural" baseline came out at
    33.9 +- 2.3 aa from a corpus filtered to 30-500, i.e. the shortest ~1% of the data, pinned
    against the floor. Nothing in the reader was wrong; the split was.

    Holding out every `holdout_stride`-th sequence GLOBALLY is order-agnostic -- it gives the same
    ~1% sample whatever the FASTA is sorted by -- and both directions of the map are closed-form, so
    the sampler still needs no materialised index:

        holdout   j -> j * S
        train     j -> (j // (S-1)) * S + (j % (S-1)) + 1

    Only whole blocks of S are used, so the map has no edge cases; the trailing N % S sequences
    (< 100 out of ~1e8) belong to neither split.

    Run `python -m src.inspect_shards` to see the per-shard length distribution and confirm whether a
    given corpus is ordered.
    """

    def __init__(self, shard_dir, eos_id, split: str = "train", holdout_stride: int = 100,
                 verify: bool = True):
        assert split in ("train", "holdout", "all")
        assert holdout_stride >= 2, "holdout_stride must be >= 2"
        self.eos_id = eos_id
        self.split = split
        self.stride = int(holdout_stride)
        bins = sorted(glob.glob(os.path.join(shard_dir, "*.bin"))) \
            if shard_dir and os.path.isdir(shard_dir) else []
        self.n_shards_total = len(bins)

        self.offsets, self.data = [], []
        for b in bins:
            off = np.fromfile(b[:-4] + ".idx", dtype="int64")
            if verify:
                # A .bin/.idx pair from an interrupted or re-run preprocess is the realistic silent
                # corruption here: numpy slices a memmap past its end WITHOUT error, returning a
                # short array, so truncated sequences would flow into training as if they were real.
                size = os.path.getsize(b)
                if len(off) < 2 or int(off[-1]) != size:
                    raise RuntimeError(
                        f"{b}: .idx says {int(off[-1]) if len(off) else 0} bytes but the .bin is "
                        f"{size}. The pair is mismatched (interrupted or re-run preprocess). "
                        f"Slicing past a memmap's end silently returns short sequences, so this is "
                        f"refused rather than trained on. Re-run src.preprocess_fasta.")
            self.offsets.append(off)
            self.data.append(np.memmap(b, dtype="uint8", mode="r"))
        counts = [len(off) - 1 for off in self.offsets]
        self.cum = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64) if counts \
            else np.array([0], np.int64)
        self.n_total = int(self.cum[-1])

        S = self.stride
        blocks = self.n_total // S
        if split == "all":
            self._n = self.n_total
        elif split == "holdout":
            self._n = blocks
        else:
            self._n = blocks * (S - 1)

    def __len__(self):
        return self._n

    def global_index(self, j: int) -> int:
        """Split-local index -> global corpus index. Closed form, no materialised list."""
        S = self.stride
        if self.split == "all":
            return int(j)
        if self.split == "holdout":
            return int(j) * S
        return (int(j) // (S - 1)) * S + (int(j) % (S - 1)) + 1

    def _locate(self, i):
        s = int(np.searchsorted(self.cum, i, side="right") - 1)
        return s, i - int(self.cum[s])

    def get(self, j) -> List[int]:
        s, k = self._locate(self.global_index(int(j)))
        a, b = int(self.offsets[s][k]), int(self.offsets[s][k + 1])
        return self.data[s][a:b].tolist() + [self.eos_id]

    def all_lengths(self) -> np.ndarray:
        """Every sequence's residue count, over ALL shards, straight from the offset arrays.
        Materialises one int32 per sequence -- fine for diagnostics, not for the hot path."""
        if not self.offsets:
            return np.zeros(0, dtype=np.int32)
        return np.concatenate([np.diff(off).astype(np.int32) for off in self.offsets])


class ShardDataset(Dataset):
    """Thin Dataset over ProteinShards; __getitem__ returns the token id list."""

    def __init__(self, shards: ProteinShards):
        self.shards = shards

    def __len__(self):
        return len(self.shards)

    def __getitem__(self, i):
        return self.shards.get(int(i))


class StepBatchSampler(torch.utils.data.Sampler):
    """Yields one batch of dataset indices per TRAINING STEP, derived from (seed, step, rank).

    Sampling is with replacement. Over a 50k-step run this rank draws
    batch_size * steps indices out of ~1e8, so within-step collisions are vanishingly rare and the
    with/without-replacement distinction is not measurable -- while statelessness buys exact
    resumability and removes the only large host allocation in the pipeline.
    """

    def __init__(self, n_items: int, batch_size: int, rank: int = 0, world: int = 1,
                 seed: int = 0, start_step: int = 0, total_steps: int = 1):
        if n_items <= 0:
            raise ValueError("empty corpus: no shards found (check UNIREF_SHARDS in config.py)")
        self.n_items, self.batch_size = int(n_items), int(batch_size)
        self.rank, self.world, self.seed = int(rank), int(world), int(seed)
        self.start_step, self.total_steps = int(start_step), int(total_steps)

    def indices_for_step(self, step: int) -> List[int]:
        # One global draw per step, sliced by rank -> ranks never see the same row in a step, and
        # the whole assignment is reproducible from the step number alone.
        rng = np.random.default_rng([self.seed, step])
        draw = rng.integers(0, self.n_items, size=self.batch_size * self.world)
        return draw[self.rank * self.batch_size:(self.rank + 1) * self.batch_size].tolist()

    def __iter__(self):
        for step in range(self.start_step, self.total_steps):
            yield self.indices_for_step(step)

    def __len__(self):
        return max(0, self.total_steps - self.start_step)


def make_collate(cfg, canvas: int = 512):
    """samples (lists of token ids) -> {"tokens": (B, canvas) long}, right-padded with PAD.

    A sequence longer than the canvas is truncated to canvas-1 residues plus EOS, so every row is
    well-formed [AA* EOS PAD*]. The corpus is size-filtered to <= 500 aa upstream, so this should
    never fire; it exists so a mis-built shard produces a valid batch rather than a silent
    off-by-one at the canvas edge.
    """
    pad, eos, vocab, mask = cfg.pad_token_id, cfg.eos_token_id, cfg.vocab_size, cfg.mask_token_id

    def collate(samples):
        B = len(samples)
        tokens = torch.full((B, canvas), pad, dtype=torch.long)
        for i, ids in enumerate(samples):
            if len(ids) > canvas:
                ids = list(ids[:canvas - 1]) + [eos]
            tokens[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        # Bounds check on the HOST, where it costs ~10us and raises something readable. nn.Embedding
        # and cross_entropy do NOT bounds-check on XPU: an id outside [0, vocab) there is an
        # unchecked out-of-range read that surfaces as an opaque "Segmentation fault from GPU ...
        # NotPresent" abort with no line number. A corrupt/truncated shard is how you get one.
        lo, hi = int(tokens.min()), int(tokens.max())
        if lo < 0 or hi >= vocab:
            bad = [(i, int(tokens[i].min()), int(tokens[i].max())) for i in range(B)
                   if int(tokens[i].min()) < 0 or int(tokens[i].max()) >= vocab]
            raise ValueError(f"token id out of range [0,{vocab}): batch min={lo} max={hi}; "
                             f"offending rows (idx, min, max)={bad[:8]}. Check the shard .bin/.idx.")
        # x0 must never contain MASK. The OADM branch puts MASK in on purpose; the D3PM branch
        # indexes its Q/Qbar stacks with x0 and those cover only the 22 non-MASK states, so a MASK
        # here would be an out-of-bounds gather -- unchecked on XPU, and an opaque GPU fault rather
        # than an exception. One host-side comparison per batch buys a readable error instead.
        if bool((tokens == mask).any()):
            raise ValueError(f"MASK (id {mask}) appears in the training targets. x0 must hold only "
                             f"amino acids, EOS and PAD; check the shard .bin for a stray byte.")
        return {"tokens": tokens}

    return collate
