"""Sequence-quality metrics: SEG-style low-complexity, and LONG-RANGE k-mer repetition.

WHY BOTH, AND WHY THE k FLOOR (instruction 8).

ProLoopDiff reported one repetition statistic -- a SEG-like sliding-window entropy with window=12
and a 2.2-bit trigger -- and its sampler penalised repeat periods 1..5 with a hard run cap of 5.
Both of those see only SHORT-RANGE structure:

  * a repeat of period p fits >= 2 copies inside a 12-residue window only when p <= 6, so that is
    roughly the longest period SEG reliably flags;
  * the sampler's penalty scores periods 1..5 explicitly and nothing beyond.

So a 20-residue motif repeated twice, 150 residues apart, is invisible to the guidance that is
supposed to prevent it AND to the metric that is supposed to detect it. Every window is
compositionally diverse and every local period is fine -- and the sequence is still degenerate.
That is exactly the regime a model falling into repetition would drift toward once the short-range
penalties are the only pressure.

kmer_counts() therefore measures repetition at k >= 13, one above the SEG window, which is the
shortest k provably outside what either short-range mechanism can act on. Defaults (13, 20, 30).

Three numbers per k, all returned as RAW COUNTS so they can be summed across ranks before any
division:

  rep_frac      fraction of residue positions covered by a k-mer that occurs >= 2 times in the SAME
                sequence. The direct "how much of this protein is a copy of another part of it".
  distinct      distinct k-mers / total k-mers, pooled per sequence. 1.0 means no k-mer ever repeats.
  shared        fraction of distinct k-mers that appear in >= 2 DIFFERENT generated sequences --
                mode collapse across samples rather than within one, which per-sequence statistics
                cannot see at all.

ALWAYS READ THESE AGAINST THE NATURAL BASELINE (src/make_baselines.py writes held-out UniRef and a
composition-matched shuffle through the same code path). Real proteins contain real repeats; the
question is never "is rep_frac > 0" but "is it far above what natural sequences of this length
show, and far above the shuffle".
"""
from __future__ import annotations
import math
from typing import Dict, Iterable, List, Sequence

from .blosum import AA

_AA_INDEX = {c: i for i, c in enumerate(AA)}

# SEG defaults for proteins. Max entropy over 20 residue types is log2(20) ~ 4.32 bits; 2.2 bits is
# ~4.6 effective types, which catches homopolymers and biased stretches like QQQNQQNQ.
SEG_WINDOW = 12
SEG_THRESHOLD = 2.2

# One above the SEG window: the shortest k that neither the SEG scan nor the sampler's period-1..5
# penalty can act on. See the module docstring.
KMER_KS = (13, 20, 30)


# ---------------------------------------------------------------------------
# SEG-style low-complexity regions
# ---------------------------------------------------------------------------
def lcr_scan(ids: Sequence[int], window: int = SEG_WINDOW, threshold: float = SEG_THRESHOLD):
    """(flagged_positions, total_positions) for one sequence of residue ids.

    A window whose composition entropy falls below `threshold` bits is flagged, and a position is
    low-complexity if ANY overlapping window is flagged. Kept as a single implementation so the
    string and token-id entry points can never drift -- this number is compared across runs.
    """
    n = len(ids)
    if n < window:
        return 0, n
    flagged = [False] * n
    counts: Dict[int, int] = {}
    for t in ids[:window]:
        counts[t] = counts.get(t, 0) + 1

    def entropy():
        e = 0.0
        for c in counts.values():
            p = c / window
            e -= p * math.log2(p)
        return e

    for i in range(n - window + 1):
        if i:                                          # roll the window instead of rebuilding it
            out = ids[i - 1]
            counts[out] -= 1
            if not counts[out]:
                del counts[out]
            inc = ids[i + window - 1]
            counts[inc] = counts.get(inc, 0) + 1
        if entropy() < threshold:
            for j in range(i, i + window):
                flagged[j] = True
    return sum(flagged), n


def lcr_counts(seqs: Iterable[str], window: int = SEG_WINDOW, threshold: float = SEG_THRESHOLD):
    """(low_complexity_positions, total_positions) over amino-acid strings."""
    lcr = tot = 0
    for s in seqs:
        a, b = lcr_scan([_AA_INDEX[c] for c in s if c in _AA_INDEX], window, threshold)
        lcr += a
        tot += b
    return lcr, tot


# ---------------------------------------------------------------------------
# Long-range k-mer repetition
# ---------------------------------------------------------------------------
def kmer_counts(seqs: Sequence[str], ks: Sequence[int] = KMER_KS) -> Dict[int, Dict[str, int]]:
    """Raw repetition counts per k. -> {k: {rep_pos, n_pos, n_distinct, n_kmers, n_shared, n_uniq}}

    rep_pos / n_pos            -> within-sequence repeat coverage
    n_distinct / n_kmers       -> distinct-k-mer ratio (1.0 = nothing ever repeats)
    n_shared / n_uniq          -> fraction of distinct k-mers seen in >= 2 of THESE sequences

    Counts, not fractions, so a caller can all-reduce them and divide once. The cross-sequence pair
    is computed over the `seqs` handed in, so summing it across ranks yields the mean WITHIN-rank
    sharing rate, not a global one -- enough to detect collapse, and it needs no collective.
    """
    out: Dict[int, Dict[str, int]] = {}
    for k in ks:
        rep_pos = n_pos = n_distinct = n_kmers = 0
        seen: Dict[str, int] = {}                      # kmer -> index of the first sequence holding it
        n_shared = 0
        for si, s in enumerate(seqs):
            n_pos += len(s)
            if len(s) < k:
                continue
            positions: Dict[str, List[int]] = {}
            for i in range(len(s) - k + 1):
                positions.setdefault(s[i:i + k], []).append(i)
            n_kmers += len(s) - k + 1
            n_distinct += len(positions)

            covered = bytearray(len(s))
            for km, pos in positions.items():
                if len(pos) > 1:
                    for p in pos:
                        for j in range(p, p + k):
                            covered[j] = 1
                first = seen.get(km)
                if first is None:
                    seen[km] = si
                elif first != si and first >= 0:
                    n_shared += 1
                    seen[km] = -1                      # counted once; -1 marks "already shared"
            rep_pos += sum(covered)
        out[k] = {"rep_pos": rep_pos, "n_pos": n_pos, "n_distinct": n_distinct,
                  "n_kmers": n_kmers, "n_shared": n_shared, "n_uniq": len(seen)}
    return out


def kmer_fractions(counts: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, float]]:
    """Turn kmer_counts (or a cross-rank sum of them) into the three reported fractions."""
    return {k: {"rep_frac": c["rep_pos"] / max(c["n_pos"], 1),
                "distinct": c["n_distinct"] / max(c["n_kmers"], 1),
                "shared": c["n_shared"] / max(c["n_uniq"], 1)}
            for k, c in counts.items()}


# ---------------------------------------------------------------------------
# Flat packing, so a whole metrics round rides in ONE fixed-size all-reduce
# ---------------------------------------------------------------------------
def flat_len(ks: Sequence[int] = KMER_KS) -> int:
    return 6 * len(ks)


def flatten_kmer(counts: Dict[int, Dict[str, int]], ks: Sequence[int] = KMER_KS) -> List[float]:
    keys = ("rep_pos", "n_pos", "n_distinct", "n_kmers", "n_shared", "n_uniq")
    return [float(counts[k][key]) for k in ks for key in keys]


def unflatten_kmer(flat: Sequence[float], ks: Sequence[int] = KMER_KS) -> Dict[int, Dict[str, int]]:
    keys = ("rep_pos", "n_pos", "n_distinct", "n_kmers", "n_shared", "n_uniq")
    return {k: {key: int(flat[i * 6 + j]) for j, key in enumerate(keys)} for i, k in enumerate(ks)}


def kmer_line(counts: Dict[int, Dict[str, int]], ks: Sequence[int] = KMER_KS) -> str:
    """One-line rendering: `k13 rep 4.2% dist .991 shar .3% | k20 ...`"""
    f = kmer_fractions(counts)
    return " | ".join(f"k{k} rep {f[k]['rep_frac']:.1%} dist {f[k]['distinct']:.3f} "
                      f"shar {f[k]['shared']:.1%}" for k in ks if k in f)


def length_stats(lengths: Sequence[int]):
    n = len(lengths)
    if not n:
        return 0.0, 0.0
    mean = sum(lengths) / n
    var = sum(v * v for v in lengths) / n - mean * mean
    return mean, (var ** 0.5 if var > 0 else 0.0)


if __name__ == "__main__":
    import random
    random.seed(0)
    rnd = "".join(random.choice(AA) for _ in range(400))

    # A 20-mer repeated twice, 150 residues apart: invisible to SEG (every window is diverse) and
    # to the sampler's period-1..5 penalty, but exactly what kmer_counts is for.
    motif = rnd[:20]
    hidden = rnd[:200] + motif + rnd[220:]
    poly = rnd[:150] + "Q" * 40 + rnd[190:]                 # the SHORT-range case SEG does catch

    for name, s in (("random", rnd), ("hidden 20-mer repeat", hidden), ("40x Q homopolymer", poly)):
        lcr, tot = lcr_counts([s])
        print(f"{name:<22} len={len(s)} LCR={lcr/max(tot,1):6.1%}  {kmer_line(kmer_counts([s]))}")
    print("\nExpect: the homopolymer lights up LCR; the hidden 20-mer repeat leaves LCR at the "
          "random level and shows up only in k13/k20 rep_frac. That gap is the whole point.")
