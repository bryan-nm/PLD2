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
from typing import Dict, Iterable, List, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Structure-track (3Di) statistics
# ---------------------------------------------------------------------------
# THE AMINO-ACID THRESHOLDS DO NOT TRANSFER, and using them here would be actively misleading.
# Measured over 107,404 natural AlphaFold-DB 3Di strings:
#
#                        3Di (natural)      amino acid (natural)
#   k13 repeat coverage      9.52%                  0.5%
#   k20                      6.34%                  0.3%
#   k30                      4.05%                  0.2%
#   longest run per seq   mean 16, p90 34, max 350       --
#
# 3Di is ~20x more repetitive than protein sequence at k13, and a 34-residue run of one state sits
# at the 90th percentile of NORMAL -- a helix is exactly that. The fold table's DEGENERATE rule
# (k13 > 2.3%) would flag essentially every real structure string. So nothing here is a threshold:
# every statistic is reported against a reference computed from the corpus's own holdout, and the
# question is always "does this look like the natural distribution", never "is this above a line".
#
# The run-length SHAPE is what discriminates. Natural 3Di is mostly short runs with occasional long
# ones (mean 1.48, median 1) while still reaching a per-sequence max of 16 on average. A collapsed
# head has a large mean AND a large median; a head that learned the marginal but no grammar has the
# right composition and the wrong conditional entropy. Reporting mean, median and p90 separately
# separates those cases, which any single summary number would blur together.
def struct_bigram(seqs: Sequence[str], alphabet: str = None):
    """(20,20) adjacent-pair counts for a 3Di track. Summable across ranks, for the same reason
    cross_track_table is: composition and especially H(x_i | x_(i-1)) are taken over 400 cells, and
    the plug-in conditional entropy is biased DOWNWARD at small N -- so a rank-local estimate on
    ~1,000 positions would make a grammar-free track look like it had grammar. Pool first."""
    import numpy as np
    from .data import DI
    ab = alphabet or DI
    idx = {c: i for i, c in enumerate(ab)}
    T = np.zeros((len(ab), len(ab)), dtype=np.float64)
    for s in seqs:
        a = [idx[c] for c in s if c in idx]
        for i in range(1, len(a)):
            T[a[i - 1], a[i]] += 1.0
    return T


def struct_entropies(T):
    """(H(x), H(x_i | x_(i-1))) in nats from a pooled bigram table."""
    import numpy as np
    T = np.asarray(T, dtype=np.float64)
    n = T.sum()
    if n <= 0:
        return 0.0, 0.0
    J = T / n
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return H(J.sum(1)), H(J) - H(J.sum(1))


def struct_stats(seqs: Sequence[str], ks: Sequence[int] = KMER_KS, alphabet: str = None,
                 bigram=None) -> Dict:
    """Distributional description of a 3Di track: composition, run shape, repetition, grammar.

    `bigram` overrides the local adjacency table with a pooled one (see struct_bigram). Run-length
    quantiles stay local -- a few hundred sequences give tens of thousands of runs, so they are
    already well estimated where the entropies are not.
    """
    from .data import DI
    ab = alphabet or DI
    idx = {c: i for i, c in enumerate(ab)}
    seqs = [s for s in seqs if len(s) >= 2]
    if not seqs:
        return {}
    runs, maxruns = [], []
    for s in seqs:
        a = [idx.get(c, -1) for c in s]
        here, cur = [], 1
        for i in range(1, len(a)):
            if a[i] == a[i - 1]:
                cur += 1
            else:
                here.append(cur)
                cur = 1
        here.append(cur)
        runs.extend(here)
        maxruns.append(max(here))          # THIS sequence's runs, not a tail of the global list
    T = struct_bigram(seqs, ab) if bigram is None else bigram
    ent, cond = struct_entropies(T)
    import numpy as np
    uni = np.asarray(T, dtype=np.float64).sum(1)
    runs.sort()
    ms = sorted(maxruns)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]
    km = kmer_counts(seqs, ks)
    return {"n": len(seqs), "entropy": ent, "cond_entropy": cond,
            "run_mean": sum(runs) / len(runs), "run_p50": q(runs, 0.5), "run_p90": q(runs, 0.9),
            "maxrun_mean": sum(maxruns) / len(maxruns), "maxrun_p90": q(ms, 0.9),
            "top": "".join(ab[i] for i in sorted(range(len(ab)), key=lambda j: -uni[j])[:4]),
            "kmer": {k: km[k]["rep_pos"] / max(km[k]["n_pos"], 1) for k in ks}}


def cross_track_table(aa_seqs: Sequence[str], di_seqs: Sequence[str]):
    """(20,20) float count table of paired (amino acid, 3Di) positions. Summable across ranks --
    which is the point: at eval_n=4 per rank a single rank has ~1,000 paired positions, and the MI
    estimator's variance at 400 cells is far too large to read at that size. All-reducing the table
    pools every rank's samples before any entropy is taken."""
    import numpy as np
    from .blosum import AA
    from .data import DI
    ai = {c: i for i, c in enumerate(AA)}
    di = {c: i for i, c in enumerate(DI)}
    T = np.zeros((20, 20), dtype=np.float64)
    for a, d in zip(aa_seqs, di_seqs):
        for ca, cd in zip(a, d):                      # zip stops at the shorter; they should match
            if ca in ai and cd in di:
                T[ai[ca], di[cd]] += 1.0
    return T


def mi_from_table(T, seed: int = 0, n_null: int = 3):
    """(observed MI, null, excess, n_paired) in nats from a pooled contingency table.

    n_paired is returned so a caller can tell "the tracks are uncoupled" from "there were not enough
    paired positions to say" -- both look like an excess of ~0 and they mean opposite things.

    THE MOST INFORMATIVE THING MEASURABLE WITHOUT FOLDING ANYTHING, because it asks whether the two
    tracks are coupled AT ALL in the generations. The model can emit a perfectly natural-looking
    sequence and a perfectly natural-looking structure string that have nothing to do with each
    other; every other statistic here would pass and the structure track would be decoration.

    Reported against a NULL rather than against zero. The plug-in MI estimator is biased upward by
    roughly (r-1)(c-1)/2N -- at 20x20 over 12,000 paired positions that is ~0.015 nats, a third of
    the natural signal of 0.044. The null resamples the SAME number of points from the product of
    the observed marginals, so it carries the same bias at the same N and the difference is clean.
    Natural corpus excess: +0.044 nats. An excess near 0 means the tracks were generated
    independently, whatever each looks like on its own.
    """
    import numpy as np
    T = np.asarray(T, dtype=np.float64)
    N = T.sum()
    if N < 400:
        return 0.0, 0.0, 0.0, int(N)
    H = lambda p: float(-(p[p > 0] * np.log(p[p > 0])).sum())
    mi = lambda J: H(J.sum(1)) + H(J.sum(0)) - H(J)
    J = T / N
    obs = mi(J)
    px, py = J.sum(1), J.sum(0)
    rng = np.random.default_rng(seed)
    nulls = [mi(rng.multinomial(int(N), np.outer(px, py).ravel()).reshape(20, 20) / N)
             for _ in range(n_null)]
    null = float(np.mean(nulls))
    return obs, null, obs - null, int(N)


def cross_track_mi(aa_seqs: Sequence[str], di_seqs: Sequence[str], seed: int = 0):
    """Single-process convenience wrapper over cross_track_table + mi_from_table."""
    return mi_from_table(cross_track_table(aa_seqs, di_seqs), seed)


def struct_line(gen: Dict, ref: Optional[Dict] = None, mi=None) -> str:
    """One line: the generated structure track against the natural reference, never a threshold."""
    if not gen:
        return "[3di] no structure track decoded"
    ks = sorted(gen["kmer"])
    f = lambda d: (f"H {d['entropy']:.2f} H(x|x-1) {d['cond_entropy']:.2f} | run mean "
                   f"{d['run_mean']:.2f} med {d['run_p50']} p90 {d['run_p90']} | longest/seq mean "
                   f"{d['maxrun_mean']:.0f} p90 {d['maxrun_p90']} | "
                   + " ".join(f"k{k} {d['kmer'][k]:.1%}" for k in ks) + f" | top {d['top']}")
    out = f"[3di] gen  n={gen['n']:<4} {f(gen)}"
    if ref:
        out += f"\n[3di] nat  n={ref['n']:<4} {f(ref)}"
    if mi is not None:
        obs, null, exc, npos = mi
        if npos < 400:
            out += (f"\n[3di] aa<->3Di coupling: only {npos:,} paired positions -- too few to "
                    f"estimate (needs >=400; a real eval pools ~11k across ranks)")
        else:
            out += (f"\n[3di] aa<->3Di coupling: MI {obs:.4f} - null {null:.4f} = {exc:+.4f} nats "
                    f"over {npos:,} positions  (natural corpus +0.044; ~0 means the two tracks "
                    f"were generated independently of each other)")
    return out
