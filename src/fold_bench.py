"""Why is PLD2's folding slower than EsmFold's own benchmark?  python -m src.fold_bench

THE QUESTION. EsmFold's README measures 1.10 s/seq at 20 steps / 1 loop on one Aurora tile.
PLD2's round-2 log reports 1693 structures/node-hour, which reads as 25 s/seq per tile. That gap
is 20x and would decide how far prompts can scale, so it is worth one debug-queue job.

WHAT THE LOGS ALREADY SETTLE, so this job does not have to:

  * Steady-state folding is 2.26 s/seq, not 25. Measured over 8,600 sequence-steps from the
    `[fold] rNN i/N (Ts)` progress lines in pld2_align.o8869221, model load excluded. The
    structures/node-hour figure is an END-TO-END number: it divides by the whole phase, including
    model load, retries and -- decisively -- the tail.
  * The tail is the cost. Ranks reach their last progress checkpoint at p50 257s / max 301s, yet
    fold attempt 1 ran 1946s and still finished only 14,841 of 16,000 records. ~9 of 192 ranks
    never printed a progress line at all. So roughly 300s of work is followed by ~1,650s of
    waiting for a handful of ranks that died or hung, until the supervisor's budget expires.
  * It is NOT the linalg guard. xpu_linalg_guard.report() named exactly one op that ran on an XPU
    tensor -- `det`. norm and cross, the two that would have been hot, never fired.
  * Model load is 81s at p50 under 12-way contention (max 116s), not the README's ~30s on an idle
    node. Real, amortised over ~83 sequences, and not where the time goes.

So the remaining 2x on the per-sequence rate, and the question of WHY ranks die, is what this
measures. The suspects, each isolated as a variant below:

  A. empty_cache never runs on our path. StructureScorer.score() honours empty_cache_every;
     _infer() does not -- and fold_all() calls _infer() whenever --pdb-dir is set, which align.pbs
     always does. So fold_empty_cache_every=1 is INERT in every production fold we have run. The
     EsmFold README is explicit that not clearing "lets a long, length-varied batch fragment HBM,
     which on Aurora XPU manifests as a GPU page fault rather than a clean OOM" -- which is the
     tail we are paying for. If this is right, s/seq RISES across a batch without it and stays
     flat with it, and peak memory climbs.
  B. inference_mode is missing. score() is decorated; _infer() is not, so our path runs under
     infer_protein's own no_grad, which still tracks versions and views.
  C. Our bookkeeping: a PDB write plus TWO fsyncs per sequence, on Lustre.
  D. Sequence length. Ours average 258 aa against the benchmark's 180, and pair tensors go as L^2.

Run it on ONE tile so nothing contends, which also makes the numbers directly comparable to the
README's. --ranks-per-node 12 in the PBS script repeats it under production contention.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import shutil
import statistics
import tempfile
import time

import torch

from config import CFG, ESMFOLD_WEIGHTS
from .dist import init_distributed
from . import ipex_shim, xpu_linalg_guard
from .fold_fasta import fold_all, read_fasta

OCFG = CFG.opt


def quartile_drift(times):
    """Mean of the last quarter over the mean of the first quarter.

    THE FRAGMENTATION SIGNATURE. A rate that is flat across a batch is not an allocator problem,
    whatever the mean is. One that climbs is, and that is what empty_cache exists to stop.

    Only meaningful because balanced_order() below makes the quartiles length-matched. The first
    version of this read 2.6x on a stand-in with no leak at all: with sequences in random order
    the quartiles have different mean lengths, and at ~L^2 that swamps any real drift. Returns
    nan for a variant that could only report one averaged number, rather than a flattering 1.00.
    """
    n = len(times)
    if n < 8 or len(set(times)) == 1:
        return float("nan")
    q = max(2, n // 4)
    return statistics.fmean(times[-q:]) / statistics.fmean(times[:q])


def peak_gb(dev):
    try:
        if dev.type == "xpu":
            return torch.xpu.max_memory_allocated() / 2**30
        if dev.type == "cuda":
            return torch.cuda.max_memory_allocated() / 2**30
    except Exception:
        pass
    return float("nan")


def reset_peak(dev):
    try:
        if dev.type == "xpu":
            torch.xpu.reset_peak_memory_stats()
        elif dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# --- the variants ------------------------------------------------------------------------------
# Each takes (scorer, seqs, workdir) and returns a list of per-sequence seconds. They differ ONLY
# in the thing being tested, so a difference between two adjacent rows has one cause.

def v_score_batch(scorer, seqs, _wd):
    """EsmFold's own benchmark path: score() over the whole list. inference_mode + empty_cache."""
    out = []
    for s in seqs:                      # timed per sequence so drift is visible; score() is still
        t = time.perf_counter()         # the thing under test, one sequence per call
        scorer.score([s], num_sampling_steps=OCFG.fold_steps, num_loops=OCFG.fold_loops)
        out.append(time.perf_counter() - t)
    return out


def v_score_list(scorer, seqs, _wd):
    """score() ONCE over the list -- the literal README measurement, for a like-for-like number.

    Reports score()'s own elapsed_seconds, which is the number the README table prints, rather
    than re-timing it here: the point of this row is to be comparable to that table."""
    r = scorer.score(list(seqs), num_sampling_steps=OCFG.fold_steps, num_loops=OCFG.fold_loops)
    per = r.elapsed_seconds / max(r.num_sequences, 1)
    return [per] * r.num_sequences


def v_infer_raw(scorer, seqs, _wd):
    """PLD2's path, stripped to the inference call: _infer, no inference_mode, no empty_cache."""
    out = []
    for s in seqs:
        t = time.perf_counter()
        o = scorer._infer(s, loops=OCFG.fold_loops, steps=OCFG.fold_steps)
        float(o["plddt"].mean())        # force the device sync score() also pays for
        del o
        out.append(time.perf_counter() - t)
    return out


def v_infer_cache(scorer, seqs, _wd):
    """_infer + the empty_cache that score() does and our path skips.  Suspect A, alone."""
    from esmfold_scorer.scorer import empty_cache
    out = []
    for s in seqs:
        t = time.perf_counter()
        o = scorer._infer(s, loops=OCFG.fold_loops, steps=OCFG.fold_steps)
        float(o["plddt"].mean())
        del o
        empty_cache(scorer.device)
        out.append(time.perf_counter() - t)
    return out


def v_infer_inference_mode(scorer, seqs, _wd):
    """_infer under inference_mode, no empty_cache.  Suspect B, alone."""
    out = []
    for s in seqs:
        t = time.perf_counter()
        with torch.inference_mode():
            o = scorer._infer(s, loops=OCFG.fold_loops, steps=OCFG.fold_steps)
            float(o["plddt"].mean())
            del o
        out.append(time.perf_counter() - t)
    return out


def v_infer_both(scorer, seqs, _wd):
    """_infer with BOTH fixes -- what the production path should become."""
    from esmfold_scorer.scorer import empty_cache
    out = []
    for s in seqs:
        t = time.perf_counter()
        with torch.inference_mode():
            o = scorer._infer(s, loops=OCFG.fold_loops, steps=OCFG.fold_steps)
            float(o["plddt"].mean())
            del o
        empty_cache(scorer.device)
        out.append(time.perf_counter() - t)
    return out


def v_fold_all(scorer, seqs, wd):
    """THE REAL PRODUCTION PATH: src.fold_fasta.fold_all with --pdb-dir, exactly as align.pbs runs
    it. PDB extraction, the file write and both fsyncs included.  Suspect C is this minus v_infer_raw."""
    pdb = os.path.join(wd, "pdb")
    out_jsonl = os.path.join(wd, "bench.jsonl")
    todo = [(f"b{i}", s) for i, s in enumerate(seqs)]
    t = time.perf_counter()
    n = fold_all(todo, scorer, out_jsonl, OCFG, time.perf_counter(), tag=" bench", pdb_dir=pdb)
    total = time.perf_counter() - t
    shutil.rmtree(pdb, ignore_errors=True)
    os.remove(out_jsonl)
    return [total / max(n, 1)] * max(n, 1)


VARIANTS = [
    ("score(list)      README path", v_score_list),
    ("score([s]) x N   per-call", v_score_batch),
    ("_infer           ours, bare", v_infer_raw),
    ("_infer +cache    suspect A", v_infer_cache),
    ("_infer +infmode  suspect B", v_infer_inference_mode),
    ("_infer +both", v_infer_both),
    ("fold_all +pdb    production", v_fold_all),
]


def pick(seqs, n, lo, hi, seed):
    pool = [s for s in seqs if lo <= len(s) <= hi]
    random.Random(seed).shuffle(pool)
    return balanced_order(pool[:n])


def balanced_order(seqs):
    """Reorder so every quarter of the batch has the same length profile.

    Sort by length, then SNAKE-deal into four buckets -- 0,1,2,3 then 3,2,1,0 then 0,1,2,3 --
    and concatenate. Plain round-robin (by_len[i::4]) is not good enough: it hands bucket 3 the
    longest of every group of four, so with a skewed length distribution the last quartile stays
    systematically longer and a flat workload still reads as 1.4x drift. Snaking pairs each
    bucket's long picks with short ones, which is what makes quartile_drift() a measurement of
    the allocator rather than of which sequences happened to land last.
    """
    by_len = sorted(seqs, key=len)
    buckets = [[] for _ in range(4)]
    for g, start in enumerate(range(0, len(by_len), 4)):
        group = by_len[start:start + 4]
        order = range(4) if g % 2 == 0 else reversed(range(4))
        for b, item in zip(order, group):
            buckets[b].append(item)
    return [s for b in buckets for s in b]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fasta", action="append", default=[],
                    help="FASTA to draw from; repeatable. Default: EsmFold's test_sequences.fasta")
    ap.add_argument("-n", "--n-seq", type=int, default=24,
                    help="sequences per variant (24 x 7 variants x ~2.5s = ~7 min)")
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=CFG.opt.fold_max_len)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--no-linalg-guard", action="store_true",
                    help="skip xpu_linalg_guard.patch() -- run once each way to price the guard")
    ap.add_argument("--only", default="", help="substring filter on variant names")
    ap.add_argument("--json", default="", help="also write the table here")
    a = ap.parse_args()

    env = init_distributed(a.device, no_dist=True)
    dev = env.device

    fastas = a.fasta or [os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                                      os.pardir, "EsmFold", "test_sequences.fasta")]
    seqs = []
    for f in fastas:
        if not os.path.exists(f):
            raise SystemExit(f"no such FASTA: {f}  (pass --fasta)")
        seqs += [s for _, s in read_fasta(f)]
    use = pick(seqs, a.n_seq, a.min_len, a.max_len, a.seed)
    if not use:
        raise SystemExit(f"no sequences in [{a.min_len}, {a.max_len}] across {fastas}")
    lens = [len(s) for s in use]
    q = max(2, len(use) // 4)
    print(f"[bench] {len(use)} sequences, {min(lens)}-{max(lens)} aa, mean {statistics.fmean(lens):.0f}"
          f" | {OCFG.fold_steps} steps, {OCFG.fold_loops} loop(s) | device {dev}", flush=True)
    # Print the balance so a drift number can be trusted or discounted on sight.
    print(f"[bench] quartile mean length: first {statistics.fmean(lens[:q]):.0f} aa, "
          f"last {statistics.fmean(lens[-q:]):.0f} aa (balanced_order keeps these close)",
          flush=True)
    print(f"[bench] from {', '.join(fastas)}", flush=True)

    # WHICH STACK IS THIS, ACTUALLY? EsmFold's speed_test.pbs puts the venv's site-packages FIRST
    # on PYTHONPATH because "the frameworks module prepends its site-packages, shadowing
    # venv-installed packages". scripts/pbs_common.sh does not, so a job can silently import a
    # different transformers than a login-node preflight saw -- and esm pins transformers<5 while
    # the frameworks module carries 5.x. Print the resolved paths so the benchmark says what it
    # measured instead of leaving it to be inferred.
    for mod in ("esm", "transformers"):
        try:
            m = __import__(mod)
            print(f"[bench] {mod:<13} {getattr(m, '__version__', '?'):<14} "
                  f"{os.path.dirname(getattr(m, '__file__', '') or '')}", flush=True)
        except Exception as e:
            print(f"[bench] {mod:<13} import failed: {type(e).__name__}: {e}", flush=True)

    with ipex_shim.only_for_import(verbose=True):
        from esmfold_scorer import StructureScorer
    t0 = time.perf_counter()
    scorer = StructureScorer(ESMFOLD_WEIGHTS, device=dev.type,
                             num_sampling_steps=OCFG.fold_steps, num_loops=OCFG.fold_loops,
                             num_diffusion_samples=1,
                             empty_cache_every=OCFG.fold_empty_cache_every)
    print(f"[bench] model loaded in {time.perf_counter() - t0:.0f}s", flush=True)
    if dev.type == "xpu" and not a.no_linalg_guard:
        xpu_linalg_guard.patch(verbose=True)

    wd = tempfile.mkdtemp(prefix="pld2bench")
    rows = []
    try:
        for name, fn in VARIANTS:
            if a.only and a.only not in name:
                continue
            reset_peak(dev)
            try:
                times = fn(scorer, use, wd)
            except Exception as e:                     # one broken variant must not lose the rest
                print(f"[bench] {name}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            rows.append({"variant": name, "n": len(times),
                         "mean_s": statistics.fmean(times),
                         "median_s": statistics.median(times),
                         "drift": quartile_drift(times), "peak_gb": peak_gb(dev)})
            r = rows[-1]
            print(f"[bench] {name:<30} {r['mean_s']:6.2f} s/seq  median {r['median_s']:6.2f}"
                  f"  drift {r['drift']:5.2f}x  peak {r['peak_gb']:5.1f} GiB", flush=True)
    finally:
        shutil.rmtree(wd, ignore_errors=True)

    print(f"\n{'variant':<30} {'s/seq':>7} {'median':>7} {'drift':>7} {'peak GiB':>9}")
    print("-" * 65)
    for r in rows:
        print(f"{r['variant']:<30} {r['mean_s']:>7.2f} {r['median_s']:>7.2f} "
              f"{r['drift']:>6.2f}x {r['peak_gb']:>9.1f}")
    base = next((r for r in rows if "README path" in r["variant"]), None)
    if base and base["mean_s"] > 0:
        print(f"\nAgainst the README path ({base['mean_s']:.2f} s/seq):")
        for r in rows:
            if r is not base:
                print(f"  {r['variant']:<30} {r['mean_s'] / base['mean_s']:5.2f}x")
    print("\nREAD IT LIKE THIS. 'drift' is the last quarter over the first: >1.15 means the rate "
          "degrades across the batch, which is the allocator, not the model. A variant that "
          "removes drift is the fix for the rank deaths, whatever it does to the mean.")
    if dev.type == "xpu":
        print(xpu_linalg_guard.report("[bench]"))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"rows": rows, "n_seq": len(use), "lengths": lens,
                       "steps": OCFG.fold_steps, "loops": OCFG.fold_loops,
                       "linalg_guard": not a.no_linalg_guard}, fh, indent=2)
        print(f"[bench] wrote {a.json}")


if __name__ == "__main__":
    main()
