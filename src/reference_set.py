"""The SwissProt reference set: captioned proteins, ESMFolded, 3Di-encoded from THOSE structures.

    python -m src.reference_set fasta --n 1200 --dir <round>      # -> refs.fasta + refs.meta.jsonl
    # ... fold refs.fasta with --pdb-dir <round>/refpdb ...
    python -m src.reference_set join --dir <round>                # -> refs.jsonl

WHY THE REFERENCE IS FOLDED RATHER THAN LOOKED UP. An earlier version of this pipeline carved
prompts out of the AFDB shards, whose 3Di track is derived from AlphaFold structures, and then
scored the completions by TM against an ESMFolded query. That puts a predictor change in the middle
of the measurement: the scaffold the model is handed and the structure it is graded against come
from two different models, and their disagreement is NOT a constant offset -- it is largest exactly
where prediction is hardest, which is where the interesting samples are. The reward would then
contain a term that has nothing to do with the sample.

So the reference set is built end to end from one predictor. A caption's protein is folded with
ESMFold; foldseek reads 3Di off THAT structure; the prompt's structure track is that 3Di; and the
TM target is that same PDB file. Every structural quantity in a round traces back to one fold.

WHAT THIS COSTS, STATED PLAINLY. PLD2 was TRAINED on AlphaFold-derived 3Di, so an ESMFold-derived
scaffold is slightly out of distribution for it. That shift is symmetric and bounded -- it is
exactly the "natural (ceiling)" row src/self_consistency.py measures, the agreement between the two
predictors' 3Di for the same natural sequence -- whereas the predictor mismatch it replaces was
systematic and pointed one way. Read that ceiling row before scaling the round up; if it is low,
the scaffolds are noisier than intended and the mask rates should come down.

AND IT IS WHAT MAKES GUIDANCE POSSIBLE. These proteins have captions, and the caption's row in the
FILIP cache travels with the prompt all the way to sampling. Prompts carved out of AFDB had no text
side at all, so caption guidance was a silent no-op.

THE SPLIT DEFAULTS TO `test`. Those rows are ones the FILIP co-embedding never trained on, so the
guidance signal here is the one it would give on genuinely new captions rather than a memorised
association. Note separately that PLD2's own corpus may overlap SwissProt -- config.py records that
the UniRef shards had exact SwissProt matches dropped, and says nothing about the AFDB ones -- so
this is a clean split for the GUIDANCE, not necessarily for the generator.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys

from config import CFG, FILIP_CACHE, MINI_EMBED_REPO
from .fold_fasta import load_pdb_index, read_records
from .self_consistency import parse_descriptor, record_key, run_foldseek

# Column names ship with the SwissProt-full CSV and are mirrored from mini-embed-filip's
# config.DataCfg. Read with the csv module, never by splitting on commas: captions contain them.
ID_COL, PROT_COL, TEXT_COL = "primary_Accession", "protein_sequence", "[final]text_caption"
DEFAULT_CSV = os.environ.get(
    "PLD2_SWISSPROT_CSV",
    os.path.join(os.path.dirname(MINI_EMBED_REPO.rstrip("/")), "datasets", "SwissProt_full",
                 "fully_annotated_swiss_prot_080326.csv"))


def load_rows(csv_path, cache_dir, split=None, splits_path=None):
    """-> list of {row, acc, seq}. `row` is the index into the FILIP cache, which is what
    FilipGuidance.set_target takes and therefore what has to travel with a prompt.

    pair_ids.json is 'one accession per CSV row' in CSV order (mini-embed's precompute docstring),
    so the cache row IS the CSV row. Verified rather than assumed: a silent off-by-one here would
    condition every generation on somebody else's caption and still look completely healthy.
    """
    if not os.path.exists(csv_path):
        raise SystemExit(f"no SwissProt CSV at {csv_path} (set PLD2_SWISSPROT_CSV)")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in (ID_COL, PROT_COL, TEXT_COL) if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{csv_path} is missing {missing}; found {reader.fieldnames}")
        for i, r in enumerate(reader):
            rows.append({"row": i, "acc": r[ID_COL], "seq": (r[PROT_COL] or "").strip().upper()})

    pair_ids_path = os.path.join(cache_dir, "pair_ids.json")
    if os.path.exists(pair_ids_path):
        with open(pair_ids_path) as f:
            pair_ids = json.load(f)
        if len(pair_ids) != len(rows):
            raise SystemExit(
                f"{pair_ids_path} has {len(pair_ids):,} rows but {csv_path} has {len(rows):,}. The "
                f"cache was built from a different CSV, so cache row indices do not name these "
                f"proteins -- every prompt would carry somebody else's caption.")
        bad = [i for i in range(0, len(rows), max(1, len(rows) // 500))
               if pair_ids[i] != rows[i]["acc"]]
        if bad:
            raise SystemExit(
                f"accession mismatch between {pair_ids_path} and the CSV at rows {bad[:5]} "
                f"(e.g. {pair_ids[bad[0]]!r} vs {rows[bad[0]]['acc']!r}). Same length, different "
                f"order or content.")
    else:
        print(f"[refs] WARNING: no {pair_ids_path}; assuming cache row == CSV row UNVERIFIED. "
              f"Caption guidance is only as correct as that assumption.", flush=True)

    if split:
        sp = splits_path or os.path.join(cache_dir, "splits.json")
        if not os.path.exists(sp):
            raise SystemExit(f"--split {split} needs {sp}; build it with mini-embed-filip, or pass "
                             f"--split '' to draw from the whole corpus.")
        with open(sp) as f:
            keep = set(json.load(f)[split])
        rows = [r for r in rows if r["row"] in keep]
    return rows


def select(rows, n, seed, min_len, max_len):
    ok = [r for r in rows if min_len <= len(r["seq"]) <= max_len
          and set(r["seq"]) <= set("ACDEFGHIKLMNPQRSTVWY")]
    # Shuffled, not strided. The corpus is in accession order, so consecutive rows are frequently
    # paralogs or isoforms whose captions read almost the same -- and a prompt set full of near
    # duplicates makes caption guidance untestable whether or not it works.
    random.Random(seed).shuffle(ok)
    if len(ok) < n:
        raise SystemExit(f"only {len(ok):,} usable proteins ({len(rows):,} before the "
                         f"{min_len}-{max_len} length filter and the alphabet check); lower --n")
    return ok[:n]


def cmd_fasta(a):
    rows = load_rows(a.csv, a.cache, a.split or None, a.splits)
    print(f"[refs] {len(rows):,} captioned proteins"
          + (f" in split '{a.split}'" if a.split else " (whole corpus)"), flush=True)
    sel = select(rows, a.n, a.seed, a.min_len, a.max_len)
    os.makedirs(a.dir, exist_ok=True)
    fa, meta = os.path.join(a.dir, "refs.fasta"), os.path.join(a.dir, "refs.meta.jsonl")
    with open(fa, "w") as f, open(meta, "w") as m:
        for r in sel:
            # The header IS the cache row, so the caption survives folding, foldseek, prompt
            # construction and sampling without a side table at any step.
            f.write(f">r{r['row']}\n{r['seq']}\n")
            m.write(json.dumps({"rid": f"r{r['row']}", "row": r["row"], "acc": r["acc"],
                                "seq": r["seq"]}) + "\n")
    L = [len(r["seq"]) for r in sel]
    print(f"[refs] wrote {len(sel):,} proteins to {fa} (length {min(L)}-{max(L)}, "
          f"mean {sum(L) / len(L):.0f})\n[refs] wrote {meta}\n"
          f"[refs] now FOLD it: --pdb-dir {os.path.join(a.dir, 'refpdb')} "
          f"--out {os.path.join(a.dir, 'reffolds.jsonl')}", flush=True)


def cmd_join(a):
    meta_path = os.path.join(a.dir, "refs.meta.jsonl")
    if not os.path.exists(meta_path):
        raise SystemExit(f"no {meta_path}; run `reference_set fasta` first")
    meta = {}
    with open(meta_path) as f:
        for line in f:
            r = json.loads(line)
            meta[r["rid"]] = r

    pdb_dir = a.pdb_dir or os.path.join(a.dir, "refpdb")
    tsv = os.path.join(a.dir, "refs.3di.tsv")
    if a.refresh or not os.path.exists(tsv):
        run_foldseek(pdb_dir, tsv, a.foldseek, a.threads)
    di = parse_descriptor(tsv, load_pdb_index(pdb_dir))          # {record_key: 3Di}

    folds = {}
    for r in read_records(a.folds or os.path.join(a.dir, "reffolds.jsonl")):
        if "id" in r:
            folds[record_key(r["id"])] = r

    out, n_no3di, n_nofold, n_len = [], 0, 0, 0
    for rid, r in meta.items():
        f, d = folds.get(rid), di.get(rid)
        if f is None:
            n_nofold += 1
            continue
        if d is None:
            n_no3di += 1
            continue
        # A 3Di string that does not cover the sequence cannot be a position-aligned structure
        # track, and a prompt built from one would silently shift the two tracks against each other.
        if len(d) != len(r["seq"]):
            n_len += 1
            continue
        out.append({**r, "di": d, "plddt": float(f["plddt"]), "ptm": float(f["ptm"])})

    path = os.path.join(a.dir, "refs.jsonl")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    mean_p = sum(r["plddt"] for r in out) / max(len(out), 1)
    conf = sum(r["plddt"] > CFG.opt.plddt_confident for r in out)
    print(f"[refs] {len(out):,}/{len(meta):,} references complete "
          f"(dropped {n_nofold} unfolded, {n_no3di} without 3Di, {n_len} length-mismatched)")
    print(f"[refs] reference pLDDT {mean_p:.3f}, {conf / max(len(out), 1):.0%} above "
          f"{CFG.opt.plddt_confident}")
    print("[refs] A LOW-CONFIDENCE REFERENCE IS A BAD PROMPT AND A BAD TM TARGET -- its 3Di is "
          "whatever ESMFold guessed. src.prompts drops them below align.ref_min_plddt.")
    print(f"[refs] wrote {path}", flush=True)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg = CFG.align
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fasta", help="select captioned proteins -> refs.fasta")
    f.add_argument("--dir", default=acfg.round_dir)
    f.add_argument("--n", type=int, default=int(acfg.n_prompts * 1.2),
                   help="draw more than n_prompts: folding and foldseek both lose some")
    f.add_argument("--csv", default=DEFAULT_CSV)
    f.add_argument("--cache", default=FILIP_CACHE)
    f.add_argument("--split", default="test", help="'' for the whole corpus")
    f.add_argument("--splits", default=None)
    f.add_argument("--seed", type=int, default=acfg.prompt_seed)
    f.add_argument("--min-len", type=int, default=40)
    f.add_argument("--max-len", type=int, default=CFG.data.canvas - 1)
    f.set_defaults(fn=cmd_fasta)

    j = sub.add_parser("join", help="foldseek 3Di + fold scores -> refs.jsonl")
    j.add_argument("--dir", default=acfg.round_dir)
    j.add_argument("--pdb-dir", default=None)
    j.add_argument("--folds", default=None)
    j.add_argument("--foldseek", default=os.environ.get("PLD2_FOLDSEEK", "foldseek"))
    j.add_argument("--threads", type=int, default=0)
    j.add_argument("--refresh", action="store_true", help="re-run foldseek even if the TSV exists")
    j.set_defaults(fn=cmd_join)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
