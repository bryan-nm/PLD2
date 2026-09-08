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
            # max_protein_tokens lives on DataCfg, not ModelCfg. getattr across both rather than
            # hard-coding one, since this reads another repo's config and a rename there should
            # degrade to the default rather than crash after the expensive phases have run.
            max_len = getattr(getattr(mcfg, "data", None), "max_protein_tokens",
                              getattr(mcfg.model, "max_protein_tokens", 512))
            h, valid = mods["encoders"].encode_protein_batch(
                pmodel, ptok, seqs, dev, max_len)
            z_p = filip.protein_proj(h)
            S = fs(z_p, z_t, valid, mask_t)                 # [n_seq, n_prompt]
        out[os.path.basename(path).replace(".fasta", "")] = S.float().cpu()

    _report(out, rows, names)


def _parse(name):
    """'sweep_filip_<prompt>_<uncond|gN>' -> (prompt tag, gamma) or (None, None)."""
    import re
    m = re.match(r"^sweep_filip_(.+?)_(uncond|g[\d.]+)$", name)
    if not m:
        return None, None
    return m.group(1), (0.0 if m.group(2) == "uncond" else float(m.group(2)[1:]))


def _report(out, rows, names):
    """Difference-in-differences, because the raw score is dominated by the PROMPT.

    The first version of this compared, within a sample set, its own prompt against the others --
    and that is confounded, badly. Measured on the first real run, every set scored ~0.90 against
    one caption, ~0.70 against another and ~0.56 against a third, whatever it had been conditioned
    on: the spread ACROSS prompts (0.34) dwarfed the spread across sample sets for a fixed prompt
    (0.04-0.10). So a row-wise margin just reports whether a set's own prompt happens to be the one
    everything scores highly against, and it duly gave the gamma=0 null -- generated without ever
    seeing a prompt -- a +0.26 margin and 94% top-1.

    The contrast that isolates guidance holds the PROMPT fixed and varies the sample set:

        lift_own    = s(guided_j, prompt_j) - s(uncond_j, prompt_j)
        lift_other  = mean over k != j of the same difference on prompt k
        DiD         = lift_own - lift_other

    lift_own alone would still credit guidance for making samples score higher against everything;
    subtracting lift_other removes exactly that. DiD > 0 is steering. Each prompt's own gamma=0 run
    is the baseline, so per-prompt offsets cancel by construction.
    """
    import torch
    # sample set -> (prompt index it was conditioned on, gamma)
    tagged = {}
    for name, S in out.items():
        ptag, gamma = _parse(name)
        if ptag is None:
            continue
        j = next((i for i, (r, nm) in enumerate(zip(rows, names))
                  if ptag == str(r) or ptag == _tag(str(r)) or ptag == _tag(nm)), None)
        if j is not None:
            tagged[name] = (j, gamma, S.mean(0))
    skipped = len(out) - len(tagged)
    if skipped:
        print(f"\n[eval] {skipped} FASTA(s) skipped: conditioned on a prompt outside --prompts "
              f"(left over from an earlier run with a different prompt set)")
    if not tagged:
        raise SystemExit("no sample set matched the given --prompts")

    base = {j: v for (jj, g, v) in tagged.values() for j in [jj] if g == 0.0}
    missing = {j for j, _, _ in tagged.values()} - set(base)
    if missing:
        print(f"[eval] no gamma=0 control for prompt(s) {[names[j] for j in sorted(missing)]}; "
              f"their rows cannot be differenced and are omitted.")

    print(f"\n{'prompt':<16}{'gamma':>7}{'own score':>11}{'lift_own':>10}"
          f"{'lift_other':>12}{'DiD':>9}   steering?")
    print("-" * 78)
    for j in sorted({j for j, _, _ in tagged.values()}):
        if j not in base:
            continue
        b = base[j]
        rowset = sorted(((g, v) for (jj, g, v) in tagged.values() if jj == j and g > 0),
                        key=lambda t: t[0])
        others = [k for k in range(len(rows)) if k != j]
        for g, v in rowset:
            d = v - b
            lo, lt = float(d[j]), float(d[others].mean()) if others else 0.0
            did = lo - lt
            print(f"{names[j][:15]:<16}{g:>7.1f}{float(v[j]):>11.4f}{lo:>+10.4f}"
                  f"{lt:>+12.4f}{did:>+9.4f}   {'yes' if did > 0.01 else 'no'}")
    print("-" * 78)
    print("Every number is differenced against THAT PROMPT'S OWN gamma=0 run, so the per-prompt\n"
          "offset cancels. lift_own is the gain on the prompt the samples were conditioned on;\n"
          "lift_other is the gain on prompts they were not. DiD is the difference, and only DiD\n"
          "distinguishes steering from guidance that raises every score at once.")


def _tag(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))[:24]


if __name__ == "__main__":
    main()
