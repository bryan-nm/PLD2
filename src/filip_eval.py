"""Did conditioning actually condition?  python -m src.filip_eval --fasta 'samples/sweep_filip_*.fasta'

pLDDT and pTM cannot answer this. Guidance that made every sample a little more protein-like would
raise both while steering nothing, and would look like a success. The question conditioning has to
answer is SPECIFIC: do the sequences generated for prompt A score higher against A than against
prompts they were not conditioned on?

So this scores every generated set against every prompt in the sweep and reports the matrix. The
diagonal is each set against its own prompt. What matters is not that the diagonal is high -- FILIP
scores have no absolute meaning -- but that it beats the off-diagonal for the SAME set, which
controls for "these samples just score well against everything".

    top-1     fraction of sets whose own prompt is their highest-scoring one (chance = 1/n_prompts)
    margin    own-prompt score minus the mean over the other prompts, in FILIP units
    z         that margin in units of the set's own spread across prompts

An unconditional row (gamma=0) is the null: it was generated without seeing any prompt, so its
margin should be ~0 whatever its pLDDT says. A guided row whose margin is also ~0 is not
conditioning, however good its fold metrics look.
"""
from __future__ import annotations
import argparse
import glob as glob_
import os
import sys

import torch

from config import FILIP_CACHE, FILIP_CKPT
from .filip_guidance import PromptCache, _mini_embed


def read_fasta(path):
    seqs, buf = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                buf = []
            else:
                buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", nargs="+", required=True, help="FASTA paths or globs")
    ap.add_argument("--prompts", required=True,
                    help="comma-separated cache rows / accessions -- the prompts to score against")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--ckpt", default=FILIP_CKPT)
    ap.add_argument("--cache", default=FILIP_CACHE)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--max-seqs", type=int, default=64, help="sequences scored per FASTA")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    paths = sorted({p for pat in a.fasta for p in glob_.glob(pat)})
    if not paths:
        raise SystemExit(f"no FASTA matched {a.fasta}")
    dev = torch.device(a.device if a.device != "xpu" or torch.xpu.is_available() else "cpu")

    mods = _mini_embed(a.repo)
    mcfg = mods["config"].default_cfg()
    filip = mods["model"].load_retrieval(a.ckpt, dev, mcfg, freeze=True)
    pmodel, ptok = mods["encoders"].load_protein_encoder(mcfg.model.protein_encoder_path, dev)
    cache = PromptCache(a.cache, mcfg.model.text_hidden, mods)
    rows = cache.rows_for([t.strip() for t in a.prompts.split(",")])
    z_t, mask_t = cache.encode(rows, filip.text_proj, dev)
    names = [cache.ids[r] if r < len(cache.ids) else str(r) for r in rows]
    print(f"[eval] {len(paths)} FASTA(s) vs {len(rows)} prompt(s): "
          f"{list(zip(rows, names))}", flush=True)

    fs = mods["losses"].filip_score_matrix
    out = {}
    for path in paths:
        seqs = [s for s in read_fasta(path) if s][:a.max_seqs]
        if not seqs:
            continue
        with torch.no_grad():
            h, valid = mods["encoders"].encode_protein_batch(
                pmodel, ptok, seqs, dev, mcfg.model.max_protein_tokens)
            z_p = filip.protein_proj(h)
            S = fs(z_p, z_t, valid, mask_t)                 # [n_seq, n_prompt]
        out[os.path.basename(path).replace(".fasta", "")] = S.float().cpu()

    w = max(len(k) for k in out) + 2
    print(f"\n{'sample set':<{w}}" + "".join(f"{n[:11]:>13}" for n in names)
          + f"{'own':>8}{'margin':>9}{'z':>7}{'top1':>7}")
    print("-" * (w + 13 * len(names) + 31))
    for name, S in out.items():
        mean = S.mean(0)
        # Which prompt was this set generated for? The filename carries it (filip_<prompt>_<tag>).
        own = next((j for j, r in enumerate(rows)
                    if f"filip_{_tag(str(r))}_" in name or f"filip_{_tag(names[j])}_" in name), None)
        cells = "".join(f"{v:>13.4f}" for v in mean.tolist())
        if own is None or len(rows) < 2:
            print(f"{name:<{w}}{cells}{'--':>8}{'--':>9}{'--':>7}{'--':>7}")
            continue
        others = [j for j in range(len(rows)) if j != own]
        margin = float(mean[own] - mean[others].mean())
        z = margin / float(mean.std().clamp_min(1e-6))
        top1 = float((S.argmax(1) == own).float().mean())
        print(f"{name:<{w}}{cells}{mean[own]:>8.4f}{margin:>+9.4f}{z:>7.2f}{top1:>6.0%}")
    print("-" * (w + 13 * len(names) + 31))
    print(f"MARGIN is the own-prompt score minus the mean over the others, for the SAME samples --\n"
          f"so it is immune to a set that simply scores high against everything. TOP1 is the "
          f"fraction of\nindividual sequences whose own prompt ranks first; chance is "
          f"{1 / max(len(rows), 1):.0%}. The gamma=0 rows are the\nnull: generated without seeing "
          f"any prompt, they should sit at margin ~0 however they fold.")


def _tag(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:24]


if __name__ == "__main__":
    main()
