"""Did the model build the fold it said it was building?

    python -m src.self_consistency --emit-reference 200      # once, to get a ceiling row
    python -m src.self_consistency                            # after folding with --pdb-dir

THE ONLY REMAINING QUESTION THE CHEAP METRICS CANNOT ANSWER. A two-track model emits an amino-acid
sequence and a 3Di structure string together. Every statistic available without folding -- 3Di
composition, run-length shape, local grammar, even the per-position aa<->3Di mutual information --
measures whether the two tracks look right MARGINALLY. Measured on the first two-track run, the
coupling hit the natural corpus level (+0.044 nats) at the FIRST eval, step 500, and then sat flat
for 17,500 more: it is a local, compositional statistic and it saturates immediately. It cannot
distinguish "the model knows glycine sits in coils" from "the model built a coherent fold".

This can. Fold the generated sequence with ESMFold, 3Di-encode the resulting structure with the
same tool that produced the training labels, and compare it position-by-position to the 3Di the
model generated ALONGSIDE that sequence. The model stated a structural intent; ESMFold reports what
the sequence actually specifies.

WHAT THE NUMBER MEANS -- it needs both ends, because neither is obvious:

  ~12-14%   FLOOR. Two unrelated 3Di strings agree this often by composition alone (measured:
            sum(p_i^2) = 0.122 over the AFDB corpus, 13.9% +- 6.5% for random real pairs). A model
            generating its two tracks independently lands here.
  ceiling   NOT 100%. It is the agreement between ESMFold's structure and AlphaFold's for the SAME
            natural sequence, and nothing about the generated samples can exceed it. --emit-reference
            writes natural AFDB sequences alongside their true 3Di so this run appears as its own
            row, measured rather than assumed.

The metric is unforgiving, which is the point: a one-residue register shift between two otherwise
identical structures already drops identity to 32%. There is no partial credit for approximately
the right fold in approximately the right place.
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import os
import random
import subprocess
import sys

from config import CFG, AFDB_SHARDS, SAMPLES_DIR
from .data import DI, ProteinShards
from .blosum import AA

FLOOR = 0.122                      # sum(p_i^2) over natural AFDB 3Di composition


# ---------------------------------------------------------------------------
# reading what the model generated
# ---------------------------------------------------------------------------
def read_fasta(path):
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    out[name] = "".join(buf)
                name, buf = line[1:].strip().split()[0], []
            else:
                buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def load_generated(samples_dir):
    """{record id: generated 3Di} across every *.3di.fa the trainer wrote."""
    gen = {}
    for p in sorted(glob_.glob(os.path.join(samples_dir, "*.3di.fa"))):
        gen.update(read_fasta(p))
    return gen


def group_of(sid: str) -> str:
    """Record id -> the checkpoint it came from. Ids are written as s<step>r<rank>_<i> by the
    trainer's eval, and sc_natural_<i> by --emit-reference."""
    if sid.startswith("sc_natural"):
        return "natural (ceiling)"
    if sid.startswith("s") and "r" in sid:
        head = sid.split("_")[0]
        step = head[1:].split("r")[0]
        if step.isdigit():
            return f"step_{int(step):08d}"
    return "other"


# ---------------------------------------------------------------------------
# foldseek
# ---------------------------------------------------------------------------
def run_foldseek(pdb_dir, tsv_path, binary="foldseek", threads=0):
    """structureto3didescriptor over a directory of PDBs -> TSV of
    `header \\t amino acids \\t 3Di \\t features`."""
    if not os.path.isdir(pdb_dir):
        raise SystemExit(f"no PDB directory at {pdb_dir}. Run the folding job with --pdb-dir first.")
    n_pdb = len(glob_.glob(os.path.join(pdb_dir, "*.pdb")))
    if n_pdb == 0:
        raise SystemExit(f"{pdb_dir} holds no .pdb files. src/fold_fasta.py writes them only when "
                         f"--pdb-dir is given, and only if the ESMFold build exposes coordinates -- "
                         f"check its log for the 'no coordinate array' warning.")
    cmd = [binary, "structureto3didescriptor", pdb_dir, tsv_path]
    if threads:
        cmd += ["--threads", str(threads)]
    print(f"[sc] {' '.join(cmd)}   ({n_pdb:,} structures)", flush=True)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(
            f"'{binary}' not found. Foldseek ships a static binary that needs no compiler and no "
            f"conda packages:\n"
            f"    wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz\n"
            f"    tar xzf foldseek-linux-avx2.tar.gz\n"
            f"    export PATH=$PWD/foldseek/bin:$PATH\n"
            f"or pass --foldseek /path/to/foldseek.")
    if r.returncode != 0:
        raise SystemExit(f"foldseek exited {r.returncode}:\n{r.stderr[-3000:]}")
    return tsv_path


def parse_descriptor(tsv_path, index):
    """TSV -> {record id: refolded 3Di}. `index` maps PDB basename -> record id."""
    out, unmatched = {}, 0
    with open(tsv_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            head = parts[0].split()[0]
            # foldseek's header is the structure name, optionally with a _<chain> suffix appended.
            for key in (head, head.rsplit("_", 1)[0], os.path.splitext(head)[0]):
                if key in index:
                    out[index[key]] = parts[2]
                    break
            else:
                unmatched += 1
    return out, unmatched


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------
def identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n if n else 0.0


def summarise(rows, seed=0):
    """rows: [(id, generated 3Di, refolded 3Di)] -> per-group statistics with a shuffled control.

    The control re-pairs each generated track with ANOTHER record's refolded structure inside the
    same group. That holds length, composition and checkpoint fixed and destroys only the
    correspondence, so the gap between the two columns is the whole claim."""
    groups = {}
    for sid, g, r in rows:
        groups.setdefault(group_of(sid), []).append((sid, g, r))
    rng = random.Random(seed)
    out = []
    for name, items in sorted(groups.items()):
        idn = [identity(g, r) for _, g, r in items]
        shuf = list(range(len(items)))
        if len(items) > 1:
            for _ in range(20):                      # derange: nobody keeps their own structure
                rng.shuffle(shuf)
                if all(i != j for i, j in enumerate(shuf)):
                    break
        ctrl = [identity(items[i][1], items[j][2]) for i, j in enumerate(shuf)]
        lens = [(len(g), len(r)) for _, g, r in items]
        mism = sum(1 for a, b in lens if a != b)
        out.append({"group": name, "n": len(items),
                    "identity": sum(idn) / len(idn), "control": sum(ctrl) / len(ctrl),
                    "best": max(idn), "len_mismatch": mism,
                    "mean_len": sum(a for a, _ in lens) / len(lens)})
    return out


def print_table(stats):
    print(f"\n{'source':<24}{'n':>5}{'3Di identity':>14}{'shuffled ctrl':>15}"
          f"{'excess':>9}{'best':>8}{'len':>7}  <- vs the {FLOOR:.0%} composition floor")
    print("-" * 90)
    for s in stats:
        print(f"{s['group']:<24}{s['n']:>5}{s['identity']:>13.1%} {s['control']:>14.1%}"
              f"{s['identity'] - s['control']:>+9.1%}{s['best']:>8.1%}{s['mean_len']:>7.0f}"
              + ("   len mismatch!" if s["len_mismatch"] else ""))
    print("-" * 90)
    print(f"IDENTITY is the fraction of positions where the 3Di the model GENERATED matches the 3Di "
          f"of the\nstructure ESMFold actually folds that sequence into. SHUFFLED CTRL re-pairs each "
          f"generated track\nwith another record's structure from the same checkpoint -- same lengths, "
          f"same composition, no\ncorrespondence. An excess near zero means the model's structural "
          f"claim is uninformative about\nthe sequence it emitted, however natural each track looks "
          f"on its own. The 'natural (ceiling)'\nrow, if present, is ESMFold-vs-AlphaFold agreement "
          f"on REAL sequences: nothing can exceed it.")


# ---------------------------------------------------------------------------
# the ceiling row
# ---------------------------------------------------------------------------
def emit_reference(n, samples_dir, stride):
    """Natural AFDB holdout sequences + their TRUE 3Di, written in the same shape as a generated
    sample pair so they flow through folding and foldseek by exactly the same path."""
    mcfg = CFG.model_config()
    sh = ProteinShards(AFDB_SHARDS, mcfg.eos_token_id, split="holdout",
                       holdout_stride=max(stride, 2))
    if len(sh) == 0 or not sh.has_struct:
        raise SystemExit(f"no paired shards in {AFDB_SHARDS}; run src.preprocess_3di first.")
    os.makedirs(samples_dir, exist_ok=True)
    fa = os.path.join(samples_dir, "sc_natural.fasta")
    di = os.path.join(samples_dir, "sc_natural.3di.fa")
    k = 0
    with open(fa, "w") as f1, open(di, "w") as f2:
        for j in range(min(len(sh), n * 4)):
            aa, d = sh.get_pair(j)
            if not d:
                continue
            s_aa = "".join(AA[t] for t in aa if t < len(AA))
            s_di = "".join(DI[t] for t in d if t < len(DI))
            if len(s_aa) != len(s_di) or len(s_aa) < 30:
                continue
            f1.write(f">sc_natural_{k}\n{s_aa}\n")
            f2.write(f">sc_natural_{k}\n{s_di}\n")
            k += 1
            if k >= n:
                break
    print(f"[sc] wrote {k} reference pairs to {fa} and {di}\n"
          f"     Fold them with the same job that folds the samples:\n"
          f"       python -m src.fold_fasta {fa} --pdb-dir <PDB_DIR>\n"
          f"     They then appear as the 'natural (ceiling)' row -- ESMFold-vs-AlphaFold agreement "
          f"on real\n     sequences, which is the highest score any generated row could reach.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=SAMPLES_DIR)
    ap.add_argument("--pdb-dir", default=os.path.join(SAMPLES_DIR, "pdb"))
    ap.add_argument("--foldseek", default=os.environ.get("PLD2_FOLDSEEK", "foldseek"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tsv", default=None, help="reuse an existing descriptor TSV instead of "
                                                "re-running foldseek")
    ap.add_argument("--out", default=None, help="JSONL of per-record identities")
    ap.add_argument("--emit-reference", type=int, default=0,
                    help="write N natural AFDB pairs for the ceiling row, then exit")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if a.emit_reference:
        return emit_reference(a.emit_reference, a.samples_dir, CFG.data.holdout_stride)

    gen = load_generated(a.samples_dir)
    if not gen:
        raise SystemExit(f"no *.3di.fa in {a.samples_dir}. Those are written by the training eval "
                         f"at n_tracks=2; a one-track run has no structure track to check.")
    ipath = os.path.join(a.pdb_dir, "index.json")
    if not os.path.exists(ipath):
        raise SystemExit(f"no {ipath}. It is written by src/fold_fasta.py --pdb-dir alongside the "
                         f"PDBs, and maps each structure file back to its record id.")
    index = json.load(open(ipath))

    tsv = a.tsv or os.path.join(a.pdb_dir, "descriptor.tsv")
    if not a.tsv:
        run_foldseek(a.pdb_dir, tsv, a.foldseek, a.threads)
    refold, unmatched = parse_descriptor(tsv, index)
    rows = [(sid, gen[sid], refold[sid]) for sid in gen if sid in refold]
    print(f"[sc] {len(gen):,} generated 3Di tracks | {len(refold):,} refolded | "
          f"{len(rows):,} joined"
          + (f" | {unmatched} descriptor rows matched no index entry" if unmatched else ""))
    if not rows:
        raise SystemExit("nothing joined. The PDB index and the .3di.fa record ids disagree -- "
                         "check that both came from the same run.")
    stats = summarise(rows)
    print_table(stats)
    if a.out:
        with open(a.out, "w") as fh:
            for sid, g, r in rows:
                fh.write(json.dumps({"id": sid, "group": group_of(sid), "identity": identity(g, r),
                                     "len_gen": len(g), "len_refold": len(r)}) + "\n")
        print(f"\n[sc] per-record identities -> {a.out}")


if __name__ == "__main__":
    main()
