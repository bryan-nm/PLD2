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

# The Foldseek 3Di structural alphabet. It is WRITTEN with the same 20 letters as the amino acids
# and means something entirely different -- a 3Di state describes the local tertiary environment of
# a residue, i.e. which residue it sits against in 3D. It is kept as its own name with its own
# encode table, and the model embeds it with its own table, because one vector cannot mean both
# "leucine" and "3Di state L". Measured on the paired AFDB corpus: the two tracks carry only
# 0.044 nats of mutual information at a position, so they are near-independent per position and
# genuinely complementary -- which is the entire reason for adding the track.
DI = "ACDEFGHIKLMNPQRSTVWY"
_DI_ID = {c: i for i, c in enumerate(DI)}

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

# Separate table for the structure track. No _RARE folding: Foldseek emits 'X' for positions it
# cannot type, and mapping that onto a real state would invent structure that was never observed.
# An unmappable character rejects the record in preprocess_3di.
_DI_ENCODE = np.full(256, _INVALID, dtype=np.uint8)
for _c, _i in _DI_ID.items():
    _DI_ENCODE[ord(_c)] = _i


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

        self.offsets, self.data, self.struct = [], [], []
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
            # Optional structure track. Decided PER SHARD, so an aa-only corpus (the UniRef shards)
            # and a paired one can sit in the same directory and both read correctly. The .idx was
            # already checked against the .bin above and the pairing guarantees equal lengths, so
            # checking the .3di against the same offsets validates it for free.
            d3 = b[:-4] + ".3di"
            if os.path.exists(d3):
                if verify and os.path.getsize(d3) != int(off[-1]):
                    raise RuntimeError(
                        f"{d3}: {os.path.getsize(d3)} bytes against an .idx of {int(off[-1])}. The "
                        f"structure track does not line up with the sequence track, which would "
                        f"attach the wrong structure to every record past the mismatch. Re-run "
                        f"src.preprocess_3di.")
                self.struct.append(np.memmap(d3, dtype="uint8", mode="r"))
            else:
                self.struct.append(None)
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
        """Amino-acid ids + EOS. Unchanged, so aa-only callers (ce_curve, make_baselines) still
        work exactly as before whether or not the shards carry a structure track."""
        s, k = self._locate(self.global_index(int(j)))
        a, b = int(self.offsets[s][k]), int(self.offsets[s][k + 1])
        return self.data[s][a:b].tolist() + [self.eos_id]

    def get_pair(self, j):
        """(aa ids + EOS, 3Di ids + EOS or None). None means this record has no structure label --
        the loss is masked off for it rather than being trained against invented structure."""
        s, k = self._locate(self.global_index(int(j)))
        a, b = int(self.offsets[s][k]), int(self.offsets[s][k + 1])
        aa = self.data[s][a:b].tolist() + [self.eos_id]
        st = self.struct[s]
        # NO EOS on the structure track. The two tracks describe one molecule, so it has ONE
        # boundary and the sequence track owns it; the structure track marks the same boundary by
        # where its PAD begins, which carries identical information without a second token that
        # could disagree. Giving it its own EOS made the sampler's tracks fall out of alignment
        # whenever the boundary moved (see src/sampler._enforce_eos).
        return aa, (None if st is None else st[a:b].tolist())

    @property
    def has_struct(self) -> bool:
        """True if ANY shard carries a structure track."""
        return any(s is not None for s in self.struct)

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
        return self.shards.get_pair(int(i))


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


def make_collate(cfg, canvas: int = 512, n_tracks: int = 1):
    """samples -> {"tokens": (B, canvas) long} and, at n_tracks=2, also "struct" and "has_struct".

    A sample is either a bare list of amino-acid ids (aa-only callers: ce_curve, make_baselines) or
    the (aa, 3Di-or-None) pair ShardDataset yields. Both are accepted, so nothing that predates the
    structure track needs to change.

    A sequence longer than the canvas is truncated to canvas-1 residues plus EOS, so every row is
    well-formed [AA* EOS PAD*]. The corpus is size-filtered to <= 500 aa upstream, so this should
    never fire; it exists so a mis-built shard produces a valid batch rather than a silent
    off-by-one at the canvas edge.

    THE TWO TRACKS ARE MIRRORED: the sequence track is [AA* EOS PAD*] and the structure track is
    [3Di* PAD*] with its PAD starting exactly at the sequence track's EOS, because the pairing
    guarantees equal lengths. Only the sequence track carries EOS -- one molecule, one boundary. A row with no structure label gets an all-PAD
    structure track and has_struct=False, and objective.training_step zeroes its structure loss. It
    is filled with PAD rather than MASK because x0 must never contain MASK -- the Q/Qbar stacks are
    indexed BY x0 and cover only the non-MASK states, so a MASK there is an out-of-bounds gather,
    unchecked on XPU and surfacing as an opaque GPU fault rather than an exception.
    """
    pad, eos, vocab, mask = cfg.pad_token_id, cfg.eos_token_id, cfg.vocab_size, cfg.mask_token_id

    def _fill(row, ids):
        if len(ids) > canvas:
            ids = list(ids[:canvas - 1]) + [eos]
        row[:len(ids)] = torch.tensor(ids, dtype=torch.long)

    def _check(t, name):
        # Bounds check on the HOST, where it costs ~10us and raises something readable. nn.Embedding
        # and cross_entropy do NOT bounds-check on XPU: an id outside [0, vocab) there is an
        # unchecked out-of-range read that surfaces as an opaque "Segmentation fault from GPU ...
        # NotPresent" abort with no line number. A corrupt/truncated shard is how you get one.
        lo, hi = int(t.min()), int(t.max())
        if lo < 0 or hi >= vocab:
            bad = [(i, int(t[i].min()), int(t[i].max())) for i in range(t.shape[0])
                   if int(t[i].min()) < 0 or int(t[i].max()) >= vocab]
            raise ValueError(f"{name}: token id out of range [0,{vocab}): min={lo} max={hi}; "
                             f"offending rows (idx, min, max)={bad[:8]}. Check the shard files.")
        if bool((t == mask).any()):
            raise ValueError(f"{name}: MASK (id {mask}) appears in the training targets. x0 must "
                             f"hold only real states, EOS and PAD; check the shard for a stray byte.")

    def collate(samples):
        B = len(samples)
        tokens = torch.full((B, canvas), pad, dtype=torch.long)
        struct = torch.full((B, canvas), pad, dtype=torch.long)
        has = torch.zeros(B, dtype=torch.bool)
        for i, smp in enumerate(samples):
            aa, di = smp if isinstance(smp, tuple) else (smp, None)
            _fill(tokens[i], aa)
            if di is not None:
                _fill(struct[i], di)
                has[i] = True
        _check(tokens, "tokens")
        if n_tracks == 1:
            return {"tokens": tokens}
        _check(struct, "struct")
        return {"tokens": tokens, "struct": struct, "has_struct": has}

    return collate
