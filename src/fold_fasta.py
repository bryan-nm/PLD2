"""Fold FASTA files with ESMFold in a process that does NOTHING else, and survive crashes.

WHY THIS IS A SEPARATE PROCESS (inherited from ProLoopDiff, where it was established the hard way).
Folding on Aurora dies with "Segmentation fault from GPU ... NotPresent" at a rate that survived
four hypotheses and four refutations: oneCCL (a 1-rank run with no process group crashed
identically), sequence content (scrambled naturals fold fine), unpatched aten XPU->CPU fallbacks
(PYTORCH_DEBUG_XPU_FALLBACK printed nothing beyond det/svd), and memory pressure (14.2 GiB peak of
64 GiB, flat, when it died on the FIRST row). So this stops trying to prevent the crash and makes it
cost almost nothing:

  * SEPARATION. Generation and folding never share a process, so ESMFold's 12.4 GiB and its
    process-global monkey-patches (torch.linalg.svd/det, F.linear/F.layer_norm) never coexist with
    the ipex-optimised trainer. Whatever the interaction is, there isn't one to have.

  * RESUMABILITY. Every scored sequence is appended to a JSONL and fsynced immediately, and a rerun
    skips ids already present. A crash 60 sequences into 100 costs 40, not 100 -- which turns a
    fatal bug into a slow one. scripts/fold.pbs just relaunches until it exits clean.

Scoring is one sequence per score() call rather than handing the whole list over, purely so a crash
cannot take unrecorded work with it. The scorer loops internally anyway, so it costs nothing.

PLD2 adds --watch: poll SAMPLES_DIR for FASTAs the trainer drops every eval_every steps and fold
each as it appears, so the structural metric tracks training live instead of being a post-hoc pass.
The summary groups by step (all ranks of one step are one row) and reports the long-range k-mer
repetition statistic next to pLDDT/pTM -- the two numbers that together say whether the model is
producing foldable, non-degenerate sequences.

    python -m src.fold_fasta --watch                     # follow a training run
    python -m src.fold_fasta --device xpu samples/*.fasta --out folds.jsonl
    python -m src.fold_fasta --summarize --out folds.jsonl        # no GPU needed
"""
from __future__ import annotations
import argparse
import glob as _glob
import json
import os
import re
import time

import torch

from config import CFG, ESMFOLD_WEIGHTS, FOLDS_JSONL, SAMPLES_DIR
from .metrics import kmer_counts, kmer_fractions, lcr_counts
from . import xpu_linalg_guard

try:
    import intel_extension_for_pytorch as ipex
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)
except Exception:
    ipex = None


def group_of(path: str) -> str:
    """Summary row a FASTA belongs to: its filename minus the .rankNNN suffix.

    The trainer writes one file per rank per eval (step_00001000.rank003.fasta), so stripping the
    rank makes all of a step's ranks a single row -- which is the unit anyone actually reads.
    """
    return re.sub(r"\.rank\d+$", "", os.path.splitext(os.path.basename(path))[0])


def read_fasta(path):
    """-> list of (id, sequence). Ids are prefixed with the group so several FASTAs can share one
    JSONL without colliding, and so summarize() can group by the part before the '|'."""
    g = group_of(path)
    out, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    out.append((f"{g}|{name}", "".join(buf)))
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        out.append((f"{g}|{name}", "".join(buf)))
    return out


def done_ids(out_path):
    """Ids already scored. Tolerates a truncated final line -- a crash mid-write leaves one."""
    ids = set()
    if not os.path.exists(out_path):
        return ids
    with open(out_path) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                continue
    return ids


def summarize(out_path, ocfg):
    recs = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    if not recs:
        print(f"[fold] {out_path}: nothing scored yet")
        return
    groups = {}
    for r in recs:
        groups.setdefault(r["id"].split("|", 1)[0], []).append(r)

    ks = list(ocfg.kmer_ks)
    kh = "".join(f"{'k' + str(k):>8}" for k in ks)
    print(f"\n{'source':<26} {'n':>5} {'pLDDT':>7} {'>70':>6} {'pTM':>7} {'>.5':>6} "
          f"{'LCR':>6} {'len':>6}{kh}   <- kmer rep_frac")
    print("-" * (76 + 8 * len(ks)))
    for name in sorted(groups):
        g = groups[name]
        n = len(g)
        seqs = [r["seq"] for r in g if "seq" in r]
        lcr, tot = lcr_counts(seqs)
        kf = kmer_fractions(kmer_counts(seqs, ks))
        row = "".join(f"{kf[k]['rep_frac']:>7.1%} " for k in ks)
        print(f"{name:<26} {n:>5} {100.0 * sum(r['plddt'] for r in g) / n:>7.1f} "
              f"{sum(1 for r in g if r['plddt'] > ocfg.plddt_confident) / n:>5.0%} "
              f"{sum(r['ptm'] for r in g) / n:>7.3f} "
              f"{sum(1 for r in g if r['ptm'] > ocfg.ptm_confident) / n:>5.0%} "
              f"{lcr / max(tot, 1):>5.1%} {sum(r['length'] for r in g) / n:>6.1f} {row}")
    print("-" * (76 + 8 * len(ks)))
    print("Read every step row against the 'natural' and 'shuffled' rows (src.make_baselines): "
          "natural is the ceiling, shuffled the composition-matched floor. A step whose pLDDT sits "
          "at or below shuffled has learned composition and not structure.")


def collect(paths, out_path, lo, hi, limit=0):
    """-> (todo, n_already, n_out_of_range) for the given FASTA paths."""
    todo, skipped = [], 0
    already = done_ids(out_path)
    for p in paths:
        for sid, seq in read_fasta(p):
            if sid in already:
                continue
            if len(seq) < lo or len(seq) > hi:
                skipped += 1
                continue
            todo.append((sid, seq))
    if limit:
        todo = todo[:limit]
    return todo, len(already), skipped


def fold_all(todo, scorer, out_path, ocfg, t0):
    n_ok = 0
    with open(out_path, "a") as fh:
        for i, (sid, seq) in enumerate(todo):
            r = scorer.score([seq], num_sampling_steps=ocfg.fold_steps, num_loops=ocfg.fold_loops)
            fh.write(json.dumps({"id": sid, "length": len(seq), "seq": seq,
                                 "plddt": r.per_sequence_plddt[0],
                                 "ptm": r.per_sequence_ptm[0]}) + "\n")
            # Flush AND fsync every record. A GPU fault aborts the process outright -- no atexit, no
            # buffer drain -- so anything still in userspace or the page cache is simply lost.
            fh.flush()
            os.fsync(fh.fileno())
            n_ok += 1
            if (i + 1) % 25 == 0:
                print(f"[fold] {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)", flush=True)
    return n_ok


def main():
    ocfg = CFG.opt
    ap = argparse.ArgumentParser()
    ap.add_argument("fasta", nargs="*", help="FASTA files (globs ok); default: SAMPLES_DIR/*.fasta")
    ap.add_argument("--out", default=FOLDS_JSONL, help="JSONL results, appended to and resumed from")
    ap.add_argument("--watch", action="store_true",
                    help="keep polling for new FASTAs (follow a training run) instead of exiting")
    ap.add_argument("--dir", default=SAMPLES_DIR, help="directory watched in --watch mode")
    ap.add_argument("--poll", type=int, default=60, help="seconds between polls in --watch mode")
    ap.add_argument("--max-idle", type=int, default=0,
                    help="exit after this many seconds with nothing new (0 = never; use it so the "
                         "watcher stops on its own when training finishes)")
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--esmfold-weights", default=ESMFOLD_WEIGHTS)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0, help="stop after N sequences per pass (0 = all)")
    ap.add_argument("--summarize", action="store_true", help="print the table and exit; no GPU")
    ap.add_argument("--no-ipex", action="store_true")
    args = ap.parse_args()

    if args.summarize:
        summarize(args.out, ocfg)
        return

    lo = ocfg.fold_min_len if args.min_len is None else args.min_len
    hi = ocfg.fold_max_len if args.max_len is None else args.max_len
    patterns = args.fasta or [os.path.join(args.dir, "*.fasta")]

    def current_paths():
        return [p for pat in patterns for p in sorted(_glob.glob(pat))]

    paths = current_paths()
    todo, n_done, n_skip = collect(paths, args.out, lo, hi, args.limit)
    print(f"[fold] {len(paths)} file(s) | {n_done} already scored | {len(todo)} to do "
          f"| {n_skip} outside [{lo},{hi}] | watch={args.watch}", flush=True)
    if not todo and not args.watch:
        summarize(args.out, ocfg)
        return

    from esmfold_scorer import StructureScorer
    dev = torch.device(args.device if args.device != "auto" else
                       ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"))
    t0 = time.perf_counter()
    scorer = StructureScorer(args.esmfold_weights, device=dev.type,
                             num_sampling_steps=ocfg.fold_steps, num_loops=ocfg.fold_loops,
                             num_diffusion_samples=1,
                             empty_cache_every=ocfg.fold_empty_cache_every)
    if dev.type == "xpu":
        # Extends EsmFold's own svd/det CPU round-trip to every other torch.linalg op, without
        # double-wrapping those two. An unpatched aten XPU->CPU fallback corrupts GPU memory on
        # Aurora's compute runtime; xpu_linalg_guard.report() names whichever ops actually fired.
        xpu_linalg_guard.patch()
    print(f"[fold] ESMFold loaded in {time.perf_counter() - t0:.0f}s on {dev}", flush=True)

    total = 0
    last_progress = time.perf_counter()
    while True:
        if todo:
            total += fold_all(todo, scorer, args.out, ocfg, t0)
            last_progress = time.perf_counter()
            summarize(args.out, ocfg)
        if not args.watch:
            break
        if args.max_idle and time.perf_counter() - last_progress > args.max_idle:
            print(f"[fold] idle for {args.max_idle}s; exiting.", flush=True)
            break
        time.sleep(args.poll)
        todo, _, _ = collect(current_paths(), args.out, lo, hi, args.limit)

    print(f"[fold] scored {total} sequence(s) this process", flush=True)
    if dev.type == "xpu":
        print(xpu_linalg_guard.report("[fold]"), flush=True)


if __name__ == "__main__":
    main()
