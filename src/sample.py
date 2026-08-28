"""Sample protein sequences unconditionally from a trained checkpoint, to FASTA.

Every default comes from CFG.opt, the same object src.train's in-loop eval reads, so this and the
training-time eval sample IDENTICALLY unless a flag is passed. In ProLoopDiff they agreed only
because hardcoded literals in eval happened to equal this file's CLI defaults -- so tuning either of
the cheapest levers against repetition would have silently left the eval measuring something nobody
generates.

    python -m src.sample --n 64 --out samples.fasta
    python -m src.sample --ckpt /path/ckpt_00050000.pt --n 32 --temperature 0.7
"""
from __future__ import annotations
import argparse
import os

import torch

from config import CFG, CKPT_DIR
from .dist import init_distributed
from .metrics import kmer_counts, kmer_line, lcr_counts, length_stats
from .model import LoopedDiffusionLM
from .sampler import decode_seqs, generate, write_fasta
from .train import find_latest_ckpt


def main():
    ocfg = CFG.opt
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (default: latest in CKPT_DIR)")
    ap.add_argument("--n", type=int, default=32, help="number of sequences to sample")
    ap.add_argument("--canvas", type=int, default=ocfg.eval_canvas, help="max-length canvas")
    ap.add_argument("--steps", type=int, default=None,
                    help="decoding steps (default: canvas; fewer makes the repetition penalty inert)")
    ap.add_argument("--temperature", type=float, default=ocfg.sample_temperature)
    ap.add_argument("--gumbel", type=float, default=ocfg.sample_gumbel_temp,
                    help="noise on the COMMIT ORDER (>0 adds diversity; raising it breaks the "
                         "commit-the-most-predictable-first loop that drives repeats)")
    ap.add_argument("--corrector", type=int, default=ocfg.sample_corrector,
                    help="post-decode corrector sweeps")
    ap.add_argument("--corrector-type", default=ocfg.sample_corrector_type,
                    choices=("remask", "substitution"),
                    help="'substitution' resamples residues directly; it is what the D3PM half of "
                         "the training objective exists to support")
    ap.add_argument("--no-eos-first", action="store_true",
                    help="decode residues and EOS together (lets a late EOS force residues into "
                         "what should be PAD -- the old repetitive-tail behaviour)")
    ap.add_argument("--min-len", type=int, default=ocfg.sample_min_len)
    ap.add_argument("--eos-temp", type=float, default=ocfg.sample_eos_temp,
                    help="length-draw temperature: <1 sharpens toward the mode, >1 widens")
    ap.add_argument("--rep-penalty", type=float, default=ocfg.sample_rep_penalty)
    ap.add_argument("--max-run", type=int, default=ocfg.sample_max_run,
                    help="hard cap on identical consecutive residues (0 disables). Needs "
                         "--steps ~= --canvas to bite; see the sampler's n_steps warning")
    ap.add_argument("--recurrence", type=int, default=None,
                    help="override the number of loop passes (adaptive compute at inference)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="FASTA output path (default: stdout)")
    args = ap.parse_args()

    env = init_distributed(args.device)              # single process -> world=1, picks xpu/cpu
    dev = env.device
    mcfg = CFG.model_config()
    model = LoopedDiffusionLM(mcfg).to(dev).eval()

    ckpt = args.ckpt or find_latest_ckpt(CKPT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint found (looked at: {args.ckpt or CKPT_DIR})")
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(state["model"])
    print(f"[sample] loaded {ckpt} (step {state.get('step', '?')}) on {dev}; "
          f"{args.n} sequences, canvas={args.canvas}, T={args.temperature}", flush=True)

    # Seed HERE, immediately before decoding. Seeding before the model is built would make --seed
    # control model-init-plus-sampling: the ~55M random draws load_state_dict throws away still
    # advance the RNG, so any architecture change would silently shift the sample stream.
    torch.manual_seed(args.seed)

    use_amp = dev.type in ("xpu", "cuda")
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        canvas, lengths = generate(
            model, Lmax=args.canvas, batch_size=args.n, n_steps=args.steps, device=str(dev),
            temperature=args.temperature, gumbel_temp=args.gumbel,
            n_corrector=args.corrector, corrector_type=args.corrector_type,
            eos_first=not args.no_eos_first, min_len=args.min_len, eos_temp=args.eos_temp,
            rep_penalty=args.rep_penalty, max_run=args.max_run,
            rep_periods=ocfg.sample_rep_periods, n_recurrence=args.recurrence)

    seqs, _ = decode_seqs(canvas, mcfg)
    mean, sd = length_stats([len(s) for s in seqs])
    lcr, tot = lcr_counts(seqs)
    print(f"[sample] len {mean:.1f}+-{sd:.1f} | no-EOS {sum(1 for v in lengths if v >= args.canvas)}"
          f"/{len(lengths)} | LCR {lcr / max(tot, 1):.1%} | "
          f"{kmer_line(kmer_counts(seqs, ocfg.kmer_ks), ocfg.kmer_ks)}", flush=True)

    if args.out:
        write_fasta(seqs, args.out, prefix="sample")
        print(f"[sample] wrote {len(seqs)} sequences -> {args.out}", flush=True)
    else:
        for i, s in enumerate(seqs):
            print(f">sample_{i} len={len(s)}\n{s}")


if __name__ == "__main__":
    main()
