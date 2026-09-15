"""TM-score every generated structure against the query its prompt was carved from.

    mpiexec -n 12 python -m src.tm_align --pdb-dir <round>/pdb

THIS IS THE PROMPT-CONSISTENCY HALF OF THE REWARD, and the half that makes the other half safe to
optimise. pLDDT alone says "this folds into something confident", which a poly-alanine helix
satisfies; TM to the query says "it folds into the RIGHT something", which it does not. Pairs built
on pLDDT alone would have a degenerate optimum sitting directly on the reward's frontier -- and we
have measured the model's density already leaning that way, since best-of-N by its own
log-likelihood is worse than a random draw. TM closes that door.

It is also the direct analogue of the cRMSD term ESM3 aligned against (Appendix A.4.3). Their
prompts were backbone coordinates, so consistency was RMSD to the prompted atoms; ours are partial
scaffolds carved out of a natural protein, so consistency is TM to that protein. Both ask the same
question: did the completion respect the structure it was handed?

ONE PREDICTOR, END TO END. The target is the reference PDB that src/reference_set.py folded --
the same structure the prompt's 3Di scaffold was read off. So the scaffold the model is handed and
the structure it is graded against come from one fold of one protein by one predictor. Mixing them
(an AlphaFold-derived scaffold against an ESMFold target, which is what carving prompts out of the
AFDB shards would give) puts a predictor disagreement inside the reward, and that disagreement is
not a constant offset: it is largest exactly where prediction is hardest, which is where the
interesting samples are.

ONE FOLDSEEK INVOCATION PER CHUNK OF PROMPTS, not per prompt. Startup dominates a small alignment
job, so prompts are batched: a chunk's generations are the query set, that chunk's natural proteins
are the target set, --exhaustive-search forces every pair to be aligned rather than prefiltered, and
rows are then kept only where the target is the generation's OWN query. Work per rank is constant in
the number of nodes -- 16 nodes and 200 nodes both give a rank a few dozen prompts -- so this scales
by doing nothing that depends on the world size.

RESUMABLE per chunk, per rank, appended and fsynced, like every other pass in this pipeline.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from config import CFG
from .dist import init_distributed
from .fold_fasta import load_pdb_index, owns
from .prompts import read_manifest
from .self_consistency import record_key

FIELDS = ("query", "target", "alntmscore", "qtmscore", "ttmscore", "lddt", "alnlen")


def pid_of(gid: str) -> str:
    """'p0000042_7' -> 'p0000042'."""
    return gid.rsplit("_", 1)[0]


def pdb_paths(pdb_dir, what):
    """-> {record key: pdb path} for one folding directory.

    The index is the authority, not the filenames: fold_fasta._safe_name flattens '|' to '_', so
    'gen|p0000042_7' lands on disk as 'gen_p0000042_7.pdb' and reading the id back off the basename
    would be guesswork. load_pdb_index maps file -> the id that was actually folded.
    """
    idx = load_pdb_index(pdb_dir)
    if not idx:
        raise SystemExit(
            f"no PDB index under {pdb_dir} ({what}). Fold with `--pdb-dir {pdb_dir}` first -- "
            f"fold_fasta writes structures only when it is given one, and only if the ESMFold "
            f"build exposes coordinates (its log says so on the first sequence).")
    out = {}
    for fname, sid in idx.items():
        path = os.path.join(pdb_dir, fname + ".pdb")
        if os.path.exists(path):           # indexed but cut short by an abort -> skip
            out[record_key(sid)] = path
    return out


def run_chunk(gen_paths, qry_paths, work, binary, threads, alignment_type=1):
    """-> {(query id, target id): {field: value}} for one foldseek easy-search."""
    qdir, tdir = os.path.join(work, "q"), os.path.join(work, "t")
    for d in (qdir, tdir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
    # Copies, not symlinks: foldseek resolves and names structures by the path it is handed, and a
    # dangling or relative link inside a scratch dir is a silent empty database rather than an error.
    for gid, p in gen_paths.items():
        shutil.copyfile(p, os.path.join(qdir, gid + ".pdb"))
    for pid, p in qry_paths.items():
        shutil.copyfile(p, os.path.join(tdir, pid + ".pdb"))

    tsv = os.path.join(work, "aln.tsv")
    tmp = os.path.join(work, "fstmp")
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    cmd = [binary, "easy-search", qdir, tdir, tsv, tmp,
           "--alignment-type", str(alignment_type),      # 1 = TMalign
           "--exhaustive-search", "1",                   # force every pair; no prefilter
           "--format-output", ",".join(FIELDS),
           "-e", "inf", "--max-seqs", str(max(10, 4 * len(qry_paths)))]
    if threads:
        cmd += ["--threads", str(threads)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"foldseek exited {r.returncode}\ncmd: {' '.join(cmd)}\n"
                           f"{r.stderr[-2000:]}")
    out = {}
    with open(tsv) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(FIELDS):
                continue
            rec = dict(zip(FIELDS, parts))
            # foldseek echoes the structure name, sometimes with a chain suffix appended.
            q, t = record_key(rec["query"]), record_key(rec["target"])
            q, t = q.removesuffix(".pdb"), t.removesuffix(".pdb")
            row = {}
            for f in FIELDS[2:]:
                try:
                    row[f] = float(rec[f])
                except ValueError:
                    row[f] = 0.0
            prev = out.get((q, t))
            # Keep the best alignment when foldseek reports more than one for a pair.
            if prev is None or row.get("alntmscore", 0) > prev.get("alntmscore", 0):
                out[(q, t)] = row
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg = CFG.align
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="round directory (default: config align.round_dir)")
    ap.add_argument("--pdb-dir", default=None, help="generations; default: <round>/pdb")
    ap.add_argument("--ref-pdb-dir", default=None, help="references; default: <round>/refpdb")
    ap.add_argument("--manifest", default=None, help="default: <round>/prompts.jsonl")
    ap.add_argument("--out", default=None, help="default: <round>/tm.rankNNN.jsonl")
    ap.add_argument("--foldseek", default=os.environ.get("PLD2_FOLDSEEK", "foldseek"))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", "8")))
    ap.add_argument("--chunk", type=int, default=32, help="prompts per foldseek invocation")
    ap.add_argument("--work", default=None, help="scratch dir (default: $TMPDIR)")
    a = ap.parse_args()

    env = init_distributed("cpu", no_dist=True)
    rank, world = env.rank, env.world_size
    rdir = a.dir or acfg.round_dir
    pdb_dir = a.pdb_dir or os.path.join(rdir, "pdb")
    ref_dir = a.ref_pdb_dir or os.path.join(rdir, "refpdb")
    out_path = a.out or os.path.join(rdir, f"tm.rank{rank:03d}.jsonl")

    # pid -> rid. A prompt names its reference; the reference names the PDB. Going through the
    # manifest rather than parsing ids keeps that one mapping in one place.
    ref_of = {r["pid"]: r["rid"] for r in read_manifest(a.manifest
                                                       or os.path.join(rdir, "prompts.jsonl"))}
    gen = pdb_paths(pdb_dir, "generations")
    ref = pdb_paths(ref_dir, "references")
    by_pid = {}
    for gid, p in gen.items():
        by_pid.setdefault(pid_of(gid), {})[gid] = p
    # A prompt whose reference has no structure has nothing to score against; report it rather than
    # dropping it silently, because it means the reference fold pass is incomplete.
    mine = sorted(pid for pid in by_pid if owns(pid, rank, world))
    missing = [pid for pid in mine if ref_of.get(pid) not in ref]
    mine = [pid for pid in mine if ref_of.get(pid) in ref]

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["pid"])
                except Exception:
                    continue
    todo = [p for p in mine if p not in done]
    if rank == 0:
        print(f"[tm] {len(gen):,} generated structures in {pdb_dir}, {len(ref):,} references in "
              f"{ref_dir}\n[tm] rank 0 owns {len(mine):,} prompts, {len(done):,} done, "
              f"{len(todo):,} to do"
              + (f" | {len(missing)} prompt(s) have no folded reference yet" if missing else ""),
              flush=True)
    if not todo:
        print(f"[tm] rank {rank}: nothing to do", flush=True)
        return

    work_root = a.work or os.environ.get("TMPDIR", "/tmp")
    work = tempfile.mkdtemp(prefix=f"pld2tm{rank:03d}_", dir=work_root)
    t0, n_rows, n_fail = time.perf_counter(), 0, 0
    try:
        with open(out_path, "a") as fh:
            for start in range(0, len(todo), a.chunk):
                chunk = todo[start:start + a.chunk]
                gpaths = {g: p for pid in chunk for g, p in by_pid[pid].items()}
                qpaths = {ref_of[pid]: ref[ref_of[pid]] for pid in chunk}
                try:
                    aln = run_chunk(gpaths, qpaths, work, a.foldseek, a.threads)
                except FileNotFoundError:
                    raise SystemExit(
                        f"'{a.foldseek}' not found. Foldseek ships a static binary:\n"
                        f"    wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz\n"
                        f"    tar xzf foldseek-linux-avx2.tar.gz\n"
                        f"    export PATH=$PWD/foldseek/bin:$PATH\n"
                        f"or pass --foldseek /path/to/foldseek.")
                except Exception as ex:
                    # One bad chunk must not cost the rest of the rank's work. Structures written by
                    # a folding process that was killed mid-file are the realistic cause.
                    n_fail += len(chunk)
                    print(f"[tm] rank {rank}: chunk of {len(chunk)} failed ({ex}); continuing",
                          flush=True)
                    continue
                for pid in chunk:
                    rid = ref_of[pid]
                    for gid in by_pid[pid]:
                        row = aln.get((gid, rid))
                        rec = {"gid": gid, "pid": pid, "rid": rid, "aligned": row is not None}
                        # No alignment at all is a real, informative outcome -- two structures with
                        # no superposable core -- and it is a TM of 0, not missing data. Dropping
                        # these rows would silently restrict the reward to the successes.
                        for f in FIELDS[2:]:
                            rec[f] = float(row[f]) if row else 0.0
                        fh.write(json.dumps(rec) + "\n")
                        n_rows += 1
                fh.flush()
                os.fsync(fh.fileno())
                print(f"[tm] rank {rank}: {min(start + a.chunk, len(todo))}/{len(todo)} prompts "
                      f"({n_rows:,} rows, {time.perf_counter() - t0:.0f}s)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"[tm] rank {rank}: wrote {n_rows:,} rows to {out_path}"
          + (f" | {n_fail} prompt(s) in failed chunks" if n_fail else ""), flush=True)


if __name__ == "__main__":
    main()
