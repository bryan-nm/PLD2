"""Partial-scaffold prompts for preference tuning: the `x` in (y_w, y_l, x).

    python -m src.prompts --n 1000 --dir <round>      # needs <round>/refs.jsonl

WHY PROMPTS AT ALL, RATHER THAN UNCONDITIONAL PAIRS. Two reasons, one statistical and one about
what the pairs can teach.

The statistical one: the positive side of a pair is only as good as the best of N draws, and
unconditional draws from this model clear an absolute foldability bar 0.2% of the time against 0.7%
conditional -- measured, at n=100 queries. Completing a scaffold is an easier problem, so the
winners are genuinely good rather than merely least-bad, and the supervision has somewhere to point.

The other: a shared prompt is what makes a pair a pair. Both completions see the same context, so
the difference between them is the completion and nothing else -- prompt difficulty cancels
exactly. Contrast pairing across prompts, where the gradient would mostly learn "40%-masked prompts
produce better proteins than 100%-masked prompts", which is true, useless, and not a property of
any sequence.

ESM3 built its alignment prompts the same way (Appendix A.4.4): half synthetic active sites, half
structure coordinates masked on a cosine schedule, and of those, half masked "according to an
autocorrelation mechanism that prefers sequentially masked positions" -- span corruption under
another name. We already have that in src/corruption.span_mask_field and train with it, so prompts
are drawn from the distribution the model was trained on.

EVERYTHING COMES FROM THE REFERENCE SET, which is one ESMFold pass over captioned SwissProt
proteins (src/reference_set.py). The scaffold's 3Di, the TM target and the caption all trace back to
the same fold of the same protein, so no predictor change sits inside the measurement -- and the
caption travels with the prompt, which is what makes guidance at generation time possible at all.

STRATIFIED BY MASK RATE, DELIBERATELY. The bins are sampled on fixed weights rather than uniformly,
because the supervised half of the IRPO loss is NOT contrastive -- it just raises the likelihood of
the winners. Let the winner pool fill with easy low-mask completions and the anchor trains scaffold
completion, which was never the failing capability. A third of prompts sit at rate 1.0, the cold
start, which is the condition we deploy in.

THE PROMPT OWNS THE LENGTH. Every prompt gives EOS and the whole PAD tail, at rate 1.0 too, so all
N completions of a prompt are the same length. That makes their rewards comparable (TM is much
better behaved at matched length) and it makes the shared surrogate mask in src/align.py EXACT
rather than merely identically distributed.

SIZE. A manifest row carries its sequence and 3Di, so it is ~1KB: 1MB at 1k prompts, 100MB at 100k.
At the top of that range the one-manifest-per-rank read is the first thing in this pipeline that
would need sharding (see scripts/align.pbs, which names the other one).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from config import CFG
from .corruption import span_mask_field


def _hex(bits: np.ndarray) -> str:
    """Bool array -> hex string, LSB-first within each byte. Round-trips through _unhex."""
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes().hex()


def _unhex(h: str, n: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8),
                         bitorder="little")[:n].astype(bool)


def read_refs(path, min_plddt=0.0):
    if not os.path.exists(path):
        raise SystemExit(f"no reference set at {path}. Build it with:\n"
                         f"    python -m src.reference_set fasta --dir <round>\n"
                         f"    <fold refs.fasta with --pdb-dir <round>/refpdb>\n"
                         f"    python -m src.reference_set join --dir <round>")
    out, dropped = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # A low-confidence reference is a bad prompt AND a bad TM target: its 3Di is whatever
            # ESMFold guessed, so the scaffold is noise and the thing being scored against is too.
            if r.get("plddt", 1.0) < min_plddt:
                dropped += 1
                continue
            out.append(r)
    return out, dropped


def build(n, refs, *, seed=0, canvas=512, bins=(0.5, 0.7, 0.85, 1.0),
          weights=(1.0, 1.0, 1.0, 1.5), span_widths=(8, 32, 128), min_len=40):
    """-> list of manifest rows. Deterministic in (seed, n, the reference set)."""
    if len(bins) != len(weights):
        raise SystemExit(f"{len(bins)} bins against {len(weights)} weights")
    g = torch.Generator().manual_seed(seed)
    rs = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()

    # Drawn per prompt up front, so a prompt's identity is a pure function of its ordinal and the
    # seed -- rebuilding with a larger --n leaves the first n rows untouched.
    bin_idx = rs.choice(len(bins), size=n, p=w)
    span_idx = rs.integers(0, len(span_widths), size=n)
    order = rs.permutation(len(refs))

    rows, cursor, skipped = [], 0, 0
    while len(rows) < n and cursor < len(order):
        ref = refs[int(order[cursor])]
        cursor += 1
        seq, di = ref["seq"], ref["di"]
        L = len(seq)
        if L < min_len or L + 1 > canvas or len(di) != L:
            skipped += 1
            continue
        k = len(rows)
        rate = float(bins[int(bin_idx[k])])
        width = int(span_widths[int(span_idx[k])])
        if rate >= 1.0:
            masked = np.ones(L, dtype=bool)
        else:
            # The same correlated field the objective corrupts with, so a prompt's revealed
            # stretches look like the ones training exposed rather than a scatter of residues.
            p = torch.full((1,), rate, dtype=torch.float32)
            masked = span_mask_field(p, L, (width,), torch.zeros(1, dtype=torch.long),
                                     generator=g)[0].numpy()
            if masked.all() or not masked.any():     # nothing to condition on, or nothing to fill
                skipped += 1
                continue
        rows.append({"pid": f"p{k:07d}", "rid": ref["rid"], "caption": int(ref["row"]),
                     "acc": ref["acc"], "L": L, "rate": rate, "bin": int(bin_idx[k]),
                     "span": width, "ref_plddt": round(float(ref.get("plddt", 0.0)), 4),
                     "seq": seq, "di": di, "masked": _hex(masked)})
    if len(rows) < n:
        raise SystemExit(f"only {len(rows)} usable prompts from {len(refs)} references "
                         f"({skipped} skipped for length or a 3Di mismatch); lower --n or draw a "
                         f"bigger reference set")
    return rows


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_manifest(path):
    if not os.path.exists(path):
        raise SystemExit(f"no prompt manifest at {path}. Run `python -m src.prompts` first.")
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def materialize(row, mcfg, canvas: int, n_tracks: int = 2):
    """Manifest row -> (tokens (K,canvas) long, given (K,canvas) bool) for sampler.generate.

    GIVEN COVERS THE WHOLE CANVAS OUTSIDE THE MASKED RESIDUES, not just the revealed ones: EOS and
    the PAD tail are context too. The sampler freezes exactly what `given` marks, and a PAD tail
    left unfrozen is a tail the decoder is free to fill with residues.
    """
    from .data import AA, DI
    L = int(row["L"])
    K = max(1, int(n_tracks))
    aa_id = {c: i for i, c in enumerate(AA)}
    di_id = {c: i for i, c in enumerate(DI)}
    tok = torch.full((K, canvas), mcfg.pad_token_id, dtype=torch.long)
    tok[0, :L] = torch.tensor([aa_id[c] for c in row["seq"]], dtype=torch.long)
    tok[0, L] = mcfg.eos_token_id
    if K > 1:
        # No EOS on the structure track, and PAD from the boundary INCLUSIVE -- the training layout
        # (data.ProteinShards.get_pair) and what sampler._enforce_eos maintains.
        tok[1, :L] = torch.tensor([di_id.get(c, 0) for c in row["di"]], dtype=torch.long)

    masked = torch.from_numpy(_unhex(row["masked"], L))
    given = torch.zeros((K, canvas), dtype=torch.bool)
    given[:, :L] = ~masked                                   # revealed residues, both tracks
    given[:, L:] = True                                      # EOS + the PAD tail
    return tok, given


def summarize(rows):
    import collections
    by_bin = collections.Counter(r["bin"] for r in rows)
    n = len(rows)
    print(f"[prompts] {n:,} prompts over {len({r['rid'] for r in rows}):,} distinct references")
    for b in sorted(by_bin):
        rate = next(r["rate"] for r in rows if r["bin"] == b)
        sel = [r for r in rows if r["bin"] == b]
        print(f"[prompts]   rate {rate:<5} {by_bin[b]:>7,} ({by_bin[b] / n:5.1%})  "
              f"len {np.mean([r['L'] for r in sel]):5.0f}  to generate "
              f"{np.mean([r['rate'] * r['L'] for r in sel]):5.0f}"
              + ("   <- cold start" if rate >= 1.0 else ""))
    print(f"[prompts] mean length {np.mean([r['L'] for r in rows]):.0f}, "
          f"reference pLDDT {np.mean([r['ref_plddt'] for r in rows]):.3f}")


def main():
    acfg = CFG.align
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=acfg.round_dir, help="round directory")
    ap.add_argument("--n", type=int, default=acfg.n_prompts)
    ap.add_argument("--refs", default=None, help="default: <round>/refs.jsonl")
    ap.add_argument("--out", default=None, help="default: <round>/prompts.jsonl")
    ap.add_argument("--seed", type=int, default=acfg.prompt_seed)
    ap.add_argument("--canvas", type=int, default=CFG.data.canvas)
    ap.add_argument("--min-len", type=int, default=40)
    ap.add_argument("--min-plddt", type=float, default=acfg.ref_min_plddt,
                    help="drop references ESMFold was not confident about; their 3Di is a guess")
    a = ap.parse_args()

    refs, dropped = read_refs(a.refs or os.path.join(a.dir, "refs.jsonl"), a.min_plddt)
    print(f"[prompts] {len(refs):,} references usable"
          + (f" ({dropped:,} below pLDDT {a.min_plddt})" if dropped else ""), flush=True)
    rows = build(a.n, refs, seed=a.seed, canvas=a.canvas, bins=acfg.mask_bins,
                 weights=acfg.bin_weights, span_widths=acfg.span_widths, min_len=a.min_len)
    out = a.out or os.path.join(a.dir, "prompts.jsonl")
    write_manifest(out, rows)
    summarize(rows)
    print(f"[prompts] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
