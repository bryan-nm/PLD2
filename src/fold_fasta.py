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

MULTI-RANK. Folding is one sequence at a time, so per-rank parallelism is the only parallelism
available, and a whole eval node was running on one of its twelve tiles. This now runs N independent
processes, each pinned to its own tile, splitting `todo` by a deterministic stride. It NEVER calls
init_process_group -- dist.init_distributed(no_dist=True) reads the MPI topology and pins the device
but stops short of oneCCL -- because oneCCL's node-local Level-Zero IPC peer mappings are what made
multi-rank folding fault on Aurora (12 ranks WITH a process group died 3 for 3, always reading an
unmapped page in the 0xff03ffff... IPC range, while 1 rank with no process group folded cleanly).
Each rank appends to its OWN <out>.rankNNN.jsonl rather than sharing one file: concurrent appends to
a single file on Lustre can interleave mid-line, and per-rank files cost nothing since every reader
here already globs.

THE RESUME KEY IS (id, sequence), NOT id. An id is "<file stem>|<fasta header>", which is derived
entirely from POSITION -- so regenerating natural.fasta with completely different sequences produces
the identical ids natural|natural_0 .. natural|natural_199, and the stale records silently suppress
every one of the new ones. Observed exactly that after the holdout fix changed which sequences the
baselines draw: "400 already scored | 0 to do" for a pair of files whose contents had entirely
changed. Keying on content as well means a re-scored id is re-folded, while an unchanged rerun still
skips instantly.

    python -m src.fold_fasta --watch                     # follow a training run
    mpiexec -n 12 -ppn 12 python -m src.fold_fasta --watch --device xpu
    python -m src.fold_fasta --summarize --out folds.jsonl        # no GPU needed
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import zlib
import os
import re
import sys
import time

import torch

from config import CFG, ESMFOLD_WEIGHTS, FOLDS_JSONL, SAMPLES_DIR
from .dist import init_distributed
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


def result_paths(base):
    """Every results file for `base`, legacy single file first then per-rank shards in rank order.

    The ordering matters for summarize(): it keeps the LAST record seen for an id, so a re-scored
    sequence supersedes the stale one it replaced.
    """
    stem = base[:-6] if base.endswith(".jsonl") else base
    return ([base] if os.path.exists(base) else []) + sorted(glob_.glob(stem + ".rank*.jsonl"))


def rank_path(base, rank):
    stem = base[:-6] if base.endswith(".jsonl") else base
    return f"{stem}.rank{rank:03d}.jsonl"


def done_path(base, rank):
    stem = base[:-6] if base.endswith(".jsonl") else base
    return f"{stem}.rank{rank:03d}.done"


def _run_id():
    """Scopes sentinels to THIS job, so a previous run's leftovers cannot satisfy the wait."""
    return os.environ.get("PBS_JOBID", "local")


def clear_done(base, rank):
    try:
        os.remove(done_path(base, rank))
    except OSError:
        pass


def mark_done(base, rank):
    tmp = done_path(base, rank) + ".tmp"
    with open(tmp, "w") as f:
        f.write(_run_id())
    os.replace(tmp, done_path(base, rank))


def wait_for_ranks(base, world, timeout=3600, poll=5.0):
    """Block until every rank has finished folding. -> True if all reported.

    Rank 0 used to summarise the moment IT finished its own share, while other ranks were still
    folding, so the table was silently partial and had to be regenerated by hand. There is no
    process group to barrier on (deliberately -- oneCCL is what makes multi-rank folding fault), so
    the ranks agree through the filesystem instead. Sentinels carry the job id, so a previous run's
    files cannot satisfy this.
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        n = 0
        for r in range(world):
            try:
                with open(done_path(base, r)) as f:
                    n += int(f.read().strip() == _run_id())
            except OSError:
                pass
        if n >= world:
            return True
        time.sleep(poll)
    return False


def read_records(base):
    """All scored records across every results file, in `result_paths` order. Tolerates a truncated
    final line -- a GPU fault aborts the process mid-write and leaves one."""
    recs = []
    for path in result_paths(base):
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    return recs


def done_pairs(base):
    """{(id, sequence)} already scored -- keyed on CONTENT as well as id. See the module docstring:
    ids come from file position, so a regenerated FASTA reuses them for entirely different
    sequences, and an id-only key would skip work that has never actually been done."""
    return {(r["id"], r.get("seq", "")) for r in read_records(base) if "id" in r}


def summarize(out_path, ocfg, partial=False):
    recs = read_records(out_path)
    if not recs:
        print(f"[fold] {out_path}: nothing scored yet", flush=True)
        return
    # LAST record per id wins. Two things produce duplicate ids and both want the newest: a
    # regenerated FASTA reusing position-derived ids for new sequences, and two ranks briefly
    # disagreeing about the file list in --watch mode and folding the same entry.
    by_id = {}
    for r in recs:
        by_id[r["id"]] = r
    groups = {}
    for r in by_id.values():
        groups.setdefault(r["id"].split("|", 1)[0], []).append(r)

    ks = list(ocfg.kmer_ks)
    kh = "".join(f"{'k' + str(k):>8}" for k in ks)
    if partial:
        print("\n[fold] PARTIAL -- other ranks are still folding; the final table follows when "
              "they finish.", flush=True)
    # Degeneracy reference. A repetitive sequence folds CONFIDENTLY -- ESMFold is happy with simple
    # repeats -- so the highest-pLDDT row can be the worst one. Measured: a configuration scored 69.0
    # pLDDT with 66% of samples over 70, and 98.7% of its residues inside a repeated 13-mer. On
    # pLDDT alone that reads as a 34-point win. Rows are flagged against the `natural` row's own
    # k-mer rate where it exists, so the bar is what real proteins do rather than a guess.
    nat = groups.get("natural")
    ref = (kmer_fractions(kmer_counts([r["seq"] for r in nat if "seq" in r], ks))[ks[0]]["rep_frac"]
           if nat else 0.01)
    limit = max(5.0 * ref, 0.02)
    print(f"\n{'source':<26} {'n':>5} {'pLDDT':>7} {'>70':>6} {'pTM':>7} {'>.5':>6} "
          f"{'LCR':>6} {'len':>6}{kh}   <- kmer rep_frac", flush=True)
    print("-" * (76 + 8 * len(ks)), flush=True)
    for name in sorted(groups):
        g = groups[name]
        n = len(g)
        seqs = [r["seq"] for r in g if "seq" in r]
        lcr, tot = lcr_counts(seqs)
        kf = kmer_fractions(kmer_counts(seqs, ks))
        row = "".join(f"{kf[k]['rep_frac']:>7.1%} " for k in ks)
        flag = "  <-- DEGENERATE" if kf[ks[0]]['rep_frac'] > limit else ""
        print(f"{name:<26} {n:>5} {100.0 * sum(r['plddt'] for r in g) / n:>7.1f} "
              f"{sum(1 for r in g if r['plddt'] > ocfg.plddt_confident) / n:>5.0%} "
              f"{sum(r['ptm'] for r in g) / n:>7.3f} "
              f"{sum(1 for r in g if r['ptm'] > ocfg.ptm_confident) / n:>5.0%} "
              f"{lcr / max(tot, 1):>5.1%} {sum(r['length'] for r in g) / n:>6.1f} {row}{flag}",
              flush=True)
    print("-" * (76 + 8 * len(ks)), flush=True)
    print("Read every step row against the 'natural' and 'shuffled' rows (src.make_baselines): "
          "natural is the ceiling, shuffled the composition-matched floor. A step whose pLDDT sits "
          "at or below shuffled has learned composition and not structure.", flush=True)
    print(f"DEGENERATE marks k{ks[0]} repeat coverage above {limit:.1%} "
          f"({'5x the natural row' if nat else 'absolute fallback'}): repetitive sequences fold "
          f"CONFIDENTLY, so a high pLDDT there is the metric being gamed, not a better model. Also "
          f"compare the len column before believing a pLDDT gain -- short chains score higher.",
          flush=True)


def owns(sid: str, rank: int, world: int) -> bool:
    """Does this rank own this sequence? A stable hash of the id, NOT a stride over the todo list.

    Position-based striding looks equivalent and is not. In --watch mode each rank re-collects on its
    own poll, and the list shrinks as ANY rank writes records, so two ranks polling seconds apart
    partition different lists: `todo[rank::world]` then maps to different items, which both
    duplicates work and leaves items owned by nobody that round. Hashing the id makes ownership a
    property of the sequence alone -- stable no matter what else is in the list, or when.

    zlib.crc32, not hash(): Python randomises str hashing per process unless PYTHONHASHSEED is set,
    so hash() would give every rank a different partition of the same ids.
    """
    return world <= 1 or zlib.crc32(sid.encode()) % world == rank


def collect(paths, out_path, lo, hi, limit=0, pdb_dir=None):
    """-> (todo, n_already, n_out_of_range, n_superseded) for the given FASTA paths.

    `todo` is in a deterministic order (sorted paths, then file order), but ranks partition it by
    `owns()` rather than by position -- see there.

    WITH --pdb-dir, "already scored" ALSO requires an INDEXED structure. pLDDT and pTM are the only
    things the JSONL records, so every sequence folded before --pdb-dir existed counts as done and
    would be skipped -- leaving the self-consistency check with structures for nothing but whatever
    happened to be new. Observed exactly that: "1520 already scored | 200 to do", and the 200 were
    the reference pairs written moments earlier.

    The test is INDEXED, not "the .pdb file exists", and the difference is not academic. A GPU abort
    leaves structures on disk that the index never recorded -- the first crashed run left ~200 of
    them. Keyed on file existence those are skipped forever while being unusable, since nothing can
    say which record each belongs to. Keyed on the index they are simply re-folded.
    """
    todo, skipped, superseded = [], 0, 0
    already = done_pairs(out_path)
    done_id_only = {i for i, _ in already}
    indexed = set(load_pdb_index(pdb_dir).values()) if pdb_dir else set()
    need_pdb = 0
    for p in paths:
        for sid, seq in read_fasta(p):
            if (sid, seq) in already:
                if not pdb_dir or sid in indexed:
                    continue
                need_pdb += 1                    # scored, but has no indexed structure yet
            if len(seq) < lo or len(seq) > hi:
                skipped += 1
                continue
            if sid in done_id_only:
                superseded += 1        # same id, different sequence: the FASTA was regenerated
            todo.append((sid, seq))
    if limit:
        todo = todo[:limit]
    if need_pdb:
        print(f"[fold] {need_pdb} sequence(s) already scored but with no INDEXED structure -- "
              f"re-folding them so src.self_consistency has something it can join", flush=True)
    return todo, len(already), skipped, superseded


# --- optional PDB output, for the 3Di self-consistency check (src/self_consistency.py) ---
# ATOM37 ordering is the AlphaFold convention; foldseek's 3Di encoder needs CA, N, C and CB, and
# rebuilds the backbone with pulchra if it finds CA only -- so writing the four is enough and
# writing only CA still works, at some cost in fidelity.
_ATOM37 = [(0, "N", "N"), (1, "CA", "C"), (2, "C", "C"), (3, "CB", "C"), (4, "O", "O")]
_AA3 = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS",
        "I": "ILE", "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN",
        "R": "ARG", "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"}


def _np(v):
    import numpy as np
    if v is None:
        return None
    return v.detach().float().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)


def _squeeze_lead(v, keep_ndim):
    """Drop leading batch / diffusion-sample axes until `keep_ndim` remain, taking index 0."""
    while v is not None and v.ndim > keep_ndim:
        v = v[0]
    return v


def _to_atoms(v, n_atoms):
    """Align an array's FIRST axis to the atom axis by dropping leading batch/sample axes.

    Anchoring on the known atom count rather than on the number of dimensions is what makes this
    robust: `atom_to_token` arrives as either (B, n_atoms) integer indices or (B, n_atoms, n_tokens)
    one-hot, and a rule based on ndim alone reads the first of those as one-hot and argmaxes over
    the atom axis -- silently collapsing every atom into one.
    """
    if v is None:
        return None
    while v.ndim > 1 and v.shape[0] != n_atoms:
        v = v[0]
    return v if v.shape[0] == n_atoms else None


def _atom_names(output, n_atoms):
    """Decode ref_atom_name_chars -> ['N', 'CA', ...], or None.

    Boltz/AF3-style models carry each atom name as 4 characters stored as ord(c) - 32, either as
    integer codes (n_atoms, 4) or one-hot over a 64-symbol alphabet (n_atoms, 4, 64)."""
    v = _to_atoms(_np(output.get("ref_atom_name_chars")), n_atoms)
    if v is None or v.ndim < 2:
        return None
    if v.ndim == 3:                      # one-hot over the symbol alphabet
        v = v.argmax(-1)
    out = []
    for row in v:
        nm = "".join(chr(int(c) + 32) for c in row).strip()
        out.append("".join(ch for ch in nm if ch.isalnum() or ch == "'") or "X")
    return out


def structure_from(output, seq):
    """ESMFold output -> [(residue_index, atom_name, x, y, z)], or None.

    THE MODEL IS ALL-ATOM AND FLAT, not the (L, 37, 3) of classic ESMFold. Its output carries
    `sample_atom_coords` over a packed atom list, `atom_to_token` mapping each atom to its residue,
    `atom_pad_mask` marking the real ones, and `ref_atom_name_chars` naming them. Residue identity
    comes from the INPUT SEQUENCE indexed by token rather than from `res_type`, which avoids having
    to know the model's residue alphabet ordering.
    """
    import numpy as np
    if not hasattr(output, "keys"):
        return None
    xyz = _np(output.get("sample_atom_coords"))
    if xyz is None:                              # older/classic layouts, kept as a fallback
        for k in ("positions", "final_atom_positions", "atom37_positions", "coords"):
            if k in output:
                v = _squeeze_lead(_np(output[k]), 3)
                if v is not None and v.ndim == 3 and v.shape[-1] == 3 and v.shape[1] >= 5:
                    return [(i, nm, *v[i, j]) for i in range(min(len(seq), v.shape[0]))
                            for j, nm, _ in _ATOM37 if j < v.shape[1]
                            and np.all(np.isfinite(v[i, j])) and not (nm == "CB" and seq[i] == "G")]
        return None
    xyz = _squeeze_lead(xyz, 2)                  # -> (n_atoms, 3)
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        return None
    n_atoms = xyz.shape[0]
    tok = _to_atoms(_np(output.get("atom_to_token")), n_atoms)
    if tok is not None:
        if tok.ndim == 2:                        # one-hot over tokens
            tok = tok.argmax(-1)
        tok = tok.reshape(-1).astype(int)
    pad = _to_atoms(_np(output.get("atom_pad_mask")), n_atoms)
    if pad is not None:
        pad = pad.reshape(-1)
    names = _atom_names(output, n_atoms)
    rows = []
    for a in range(n_atoms):
        if pad is not None and not pad[a]:
            continue
        if not np.all(np.isfinite(xyz[a])):
            continue
        r = int(tok[a]) if tok is not None else a // 5
        if r >= len(seq):
            continue
        nm = names[a] if names else "CA"
        rows.append((r, nm, float(xyz[a, 0]), float(xyz[a, 1]), float(xyz[a, 2])))
    return rows or None


def write_pdb(path, seq, rows):
    """rows: [(residue_index, atom_name, x, y, z)] -> a PDB file. Returns the atom count."""
    lines, serial = [], 1
    for i, name, x, y, z in rows:
        res = _AA3.get(seq[i], "GLY") if i < len(seq) else "GLY"
        elem = next((c for c in name if c.isalpha()), "C")
        # Column-exact PDB. The fixed-width layout is: 1-6 record, 7-11 serial, 13-16 atom name,
        # 17 altLoc, 18-20 resName, 22 chain, 23-26 resSeq, 31-38/39-46/47-54 x/y/z. Gemmi (which
        # foldseek parses with) reads by column, so a single missing space shifts resName into
        # altLoc and every residue silently becomes unknown.
        an = f" {name}" if len(name) < 4 else name        # 1-2 char elements are space-padded
        lines.append(f"ATOM  {serial:>5} {an:<4} {res:>3} A{i + 1:>4}    "
                     f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00          {elem:>2}")
        serial += 1
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return serial - 1


def _safe_name(sid: str) -> str:
    """Record id -> a basename foldseek can carry through as its header."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(sid))[:180]


def pdb_index_path(pdb_dir, rank):
    return os.path.join(pdb_dir, f"index.rank{rank:03d}.jsonl")


def load_pdb_index(pdb_dir):
    """{pdb basename: record id}, merged across every rank's append-only index."""
    idx = {}
    legacy = os.path.join(pdb_dir, "index.json")
    if os.path.exists(legacy):                       # written by the pre-crash-safe version
        try:
            idx.update(json.load(open(legacy)))
        except Exception:
            pass
    for p in sorted(glob_.glob(os.path.join(pdb_dir, "index.rank*.jsonl"))):
        with open(p) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue                         # a truncated final line after an abort
                idx[r["file"]] = r["id"]
    return idx


def fold_all(todo, scorer, out_path, ocfg, t0, tag="", pdb_dir=None, rank=0):
    n_ok, n_pdb = 0, 0
    idx_fh = None
    if pdb_dir:
        os.makedirs(pdb_dir, exist_ok=True)
        # APPEND-ONLY AND FSYNCED PER RECORD, exactly like the results JSONL and for exactly the
        # same reason: a GPU fault aborts the process outright, with no atexit and no buffer drain.
        # An index written once at the end is lost every time, which is how a crashed run left 200
        # structures on disk and nothing able to say which record each one belonged to. Per-rank
        # files because twelve ranks read-modify-writing one JSON would race and lose entries.
        idx_fh = open(pdb_index_path(pdb_dir, rank), "a")
    with open(out_path, "a") as fh:
        for i, (sid, seq) in enumerate(todo):
            if pdb_dir:
                # score() discards the coordinates, so go through the same private entry point it
                # uses and keep them. Falls back to score() if this build's output has no usable
                # coordinate array, so the pLDDT/pTM table never depends on the PDB path working.
                out = scorer._infer(seq, loops=ocfg.fold_loops, steps=ocfg.fold_steps)
                rows = structure_from(out, seq)
                if rows:
                    base = _safe_name(sid)
                    if write_pdb(os.path.join(pdb_dir, base + ".pdb"), seq, rows):
                        idx_fh.write(json.dumps({"file": base, "id": sid}) + "\n")
                        idx_fh.flush()
                        os.fsync(idx_fh.fileno())
                        n_pdb += 1
                if i == 0:
                    # Report the layout on the FIRST sequence whether extraction worked or not. The
                    # coordinate keys are a property of the model build, and a failed job that only
                    # says "no coordinates" costs another full round trip to diagnose.
                    shp = {k: tuple(getattr(v, "shape", ())) for k, v in out.items()
                           if k in ("sample_atom_coords", "atom_to_token", "atom_pad_mask",
                                    "ref_atom_name_chars", "res_type")} if hasattr(out, "keys") \
                        else {}
                    if rows:
                        nres = len({r[0] for r in rows})
                        nm = sorted({r[1] for r in rows})[:8]
                        print(f"[fold]{tag} PDB layout OK: {len(rows)} atoms over {nres} residues "
                              f"(sequence is {len(seq)}), atom names {nm} | shapes {shp}", flush=True)
                    else:
                        keys = sorted(out.keys()) if hasattr(out, "keys") else type(out).__name__
                        print(f"[fold]{tag} WARNING: --pdb-dir given but no usable coordinates. "
                              f"shapes {shp} | all keys {keys}. Folding continues WITHOUT PDBs.",
                              flush=True)
                r = type("R", (), {"per_sequence_plddt": [float(out["plddt"].mean())],
                                   "per_sequence_ptm": [float(out["ptm"].mean())]})()
                del out
            else:
                r = scorer.score([seq], num_sampling_steps=ocfg.fold_steps,
                                 num_loops=ocfg.fold_loops)
            fh.write(json.dumps({"id": sid, "length": len(seq), "seq": seq,
                                 "plddt": r.per_sequence_plddt[0],
                                 "ptm": r.per_sequence_ptm[0]}) + "\n")
            # Flush AND fsync every record. A GPU fault aborts the process outright -- no atexit, no
            # buffer drain -- so anything still in userspace or the page cache is simply lost.
            fh.flush()
            os.fsync(fh.fileno())
            n_ok += 1
            if (i + 1) % 25 == 0:
                print(f"[fold]{tag} {i + 1}/{len(todo)} ({time.perf_counter() - t0:.0f}s)",
                      flush=True)

    if idx_fh is not None:
        idx_fh.close()
        print(f"[fold]{tag} wrote {n_pdb} PDBs to {pdb_dir}", flush=True)
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
    # Opt-in, and off by default, so the pLDDT/pTM path this job exists for is untouched. Writing
    # PDBs routes folding through scorer._infer to keep the coordinates score() throws away; the
    # 3Di self-consistency check (src/self_consistency.py) reads them.
    ap.add_argument("--pdb-dir", default=None,
                    help="also write one backbone PDB per folded sequence here, for "
                         "src.self_consistency")
    ap.add_argument("--summarize", action="store_true", help="print the table and exit; no GPU")
    ap.add_argument("--no-ipex", action="store_true")
    args = ap.parse_args()

    # Line-buffer stdout. Under PBS the job's output is a FILE, so Python block-buffers it, and
    # every print without flush=True sits in a 8KB buffer until something else fills it. The
    # progress lines passed flush=True and appeared promptly; the summary TABLE did not, so a
    # finished fold round looked like it had produced nothing. One line fixes the whole module.
    sys.stdout.reconfigure(line_buffering=True)

    if args.summarize:
        summarize(args.out, ocfg)
        return

    lo = ocfg.fold_min_len if args.min_len is None else args.min_len
    hi = ocfg.fold_max_len if args.max_len is None else args.max_len
    patterns = args.fasta or [os.path.join(args.dir, "*.fasta")]
    # Never fold a structure track. The 3Di alphabet reuses the amino-acid letters, so a .3di file
    # that reached ESMFold would parse cleanly and produce confident-looking nonsense rather than an
    # error. The trainer already names them ".3di.fa" to miss the glob above; this is the backstop
    # for an explicit argument or a widened pattern.
    _is_struct = lambda q: ".3di." in os.path.basename(q)

    def current_paths():
        return [p for pat in patterns for p in sorted(glob_.glob(pat)) if not _is_struct(p)]

    # Topology WITHOUT a process group: this reads the MPI rank/size and pins this rank's tile, but
    # never calls init_process_group, so oneCCL is never initialised in this process. That is the
    # whole reason multi-rank folding is safe here -- see the module docstring.
    env = init_distributed(args.device, no_dist=True)
    rank, world, dev = env.rank, env.world_size, env.device
    tag = f" r{rank}" if world > 1 else ""
    my_out = rank_path(args.out, rank) if world > 1 else args.out

    paths = current_paths()
    todo_all, n_done, n_skip, n_super = collect(paths, args.out, lo, hi, args.limit,
                                                pdb_dir=args.pdb_dir)
    todo = [x for x in todo_all if owns(x[0], rank, world)]      # no coordination needed
    if rank == 0:
        print(f"[fold] {len(paths)} file(s) | {n_done} already scored | {len(todo_all)} to do "
              f"| {n_skip} outside [{lo},{hi}] | watch={args.watch} | {world} rank(s)", flush=True)
        if n_super:
            print(f"[fold] {n_super} entr(y/ies) have a stale record under the same id with a "
                  f"DIFFERENT sequence -- a regenerated FASTA. Re-folding them; the summary keeps "
                  f"the newest record per id.", flush=True)
    if not todo_all and not args.watch:
        if rank == 0:
            summarize(args.out, ocfg)
        return

    from esmfold_scorer import StructureScorer
    t0 = time.perf_counter()
    scorer = StructureScorer(args.esmfold_weights, device=dev.type,
                             num_sampling_steps=ocfg.fold_steps, num_loops=ocfg.fold_loops,
                             num_diffusion_samples=1,
                             empty_cache_every=ocfg.fold_empty_cache_every)
    if dev.type == "xpu":
        # Extends EsmFold's own svd/det CPU round-trip to every other torch.linalg op, without
        # double-wrapping those two. An unpatched aten XPU->CPU fallback corrupts GPU memory on
        # Aurora's compute runtime; xpu_linalg_guard.report() names whichever ops actually fired.
        xpu_linalg_guard.patch(verbose=(rank == 0))
    print(f"[fold]{tag} ESMFold loaded in {time.perf_counter() - t0:.0f}s on {dev} "
          f"| {len(todo)} of {len(todo_all)} sequences", flush=True)

    clear_done(args.out, rank)
    total = 0
    last_progress = time.perf_counter()
    while True:
        if todo:
            total += fold_all(todo, scorer, my_out, ocfg, t0, tag, pdb_dir=args.pdb_dir,
                              rank=env.rank)
            last_progress = time.perf_counter()
            # Only rank 0 prints the table; every rank's records are in it, because summarize()
            # globs all the per-rank files. Twelve copies of the same table would bury the log.
            if rank == 0:
                summarize(args.out, ocfg, partial=(world > 1))
        if not args.watch:
            break
        if args.max_idle and time.perf_counter() - last_progress > args.max_idle:
            print(f"[fold]{tag} idle for {args.max_idle}s; exiting.", flush=True)
            break
        time.sleep(args.poll)
        todo = [x for x in collect(current_paths(), args.out, lo, hi, args.limit,
                                   pdb_dir=args.pdb_dir)[0]
                if owns(x[0], rank, world)]

    print(f"[fold]{tag} scored {total} sequence(s) this process", flush=True)
    mark_done(args.out, rank)
    if rank == 0:
        if world > 1:
            print(f"[fold] waiting for {world} rank(s) to finish before the final table...",
                  flush=True)
            if not wait_for_ranks(args.out, world):
                print(f"[fold] WARNING: not all ranks reported; the table below is INCOMPLETE.",
                      flush=True)
        summarize(args.out, ocfg)
    if dev.type == "xpu" and rank == 0:
        print(xpu_linalg_guard.report("[fold]"), flush=True)


if __name__ == "__main__":
    main()
