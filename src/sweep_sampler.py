"""Sample one FASTA per SAMPLER CONFIGURATION, so the fold pipeline scores them side by side.

    python -m src.sweep_sampler --device xpu --n 64
    qsub scripts/fold.pbs                 # then read the summary table

Each configuration writes SAMPLES_DIR/sweep_<name>.fasta, which means src/fold_fasta.py picks them
up with no changes and they appear as their own rows in the same table as `natural` and `shuffled`.
Nothing here needs a new metric or a new comparison -- the existing one already has the right
reference lines.

WHY THIS EXISTS. A flat pLDDT at the ESMFold floor has two very different causes, and they need
different fixes: either the model has nothing to say at cold start (see src/ce_curve.py), or the
sampler is destroying what it does say. This sweeps the sampler side.

The first configuration to look at is `no_reppen`. The anti-repetition machinery -- a 1.5 logit
penalty per matching residue at periods 1..5 in BOTH directions, plus a hard 5-residue run cap -- was
tuned for ProLoopDiff, whose samples contained homopolymer runs of 42. It is not free: a run at
50k steps produced samples with 0.0% SEG low-complexity against 4.7% for a RANDOM SHUFFLE of real
proteins and 7.9% for the naturals themselves. Being under the shuffle is the tell -- local
compositional bias at that scale is what chance alone produces, so suppressing it below chance means
the penalty is removing structure that real proteins have, not just degeneracy they do not.
"""
from __future__ import annotations
import argparse
import os

import torch

from config import CFG, CKPT_DIR, SAMPLES_DIR
from .dist import init_distributed
from .metrics import kmer_counts, kmer_line, lcr_counts, length_stats
from .model import LoopedDiffusionLM
from .sampler import decode_seqs, generate, write_fasta
from .train import find_latest_ckpt

try:
    import intel_extension_for_pytorch as ipex
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)
except Exception:
    ipex = None


def configurations(ocfg):
    """name -> generate() overrides. The first is the shipped default, as the reference row."""
    base = dict(temperature=ocfg.sample_temperature, gumbel_temp=ocfg.sample_gumbel_temp,
                rep_penalty=ocfg.sample_rep_penalty, max_run=ocfg.sample_max_run,
                rep_periods=ocfg.sample_rep_periods,
                subst_per_residue=ocfg.sample_subst_per_residue,
                eos_first=ocfg.sample_eos_first)
    def v(**kw):
        d = dict(base)
        d.update(kw)
        return d
    return {
        "default":        v(),
        # --- the anti-repetition machinery, which measurably pushes LCR below chance ---
        "no_reppen":      v(rep_penalty=0.0, max_run=0),
        "reppen_soft":    v(rep_penalty=0.3),
        "periods_1_2":    v(rep_periods=(1, 2)),          # leave helical 3/4 periodicity alone
        "no_maxrun":      v(max_run=0),
        # --- the other decode knobs ---
        "no_subst":       v(subst_per_residue=0.0),
        "subst_only":     v(rep_penalty=0.0, max_run=0, subst_per_residue=4.0),
        "t0.5":           v(temperature=0.5),
        "t0.8":           v(temperature=0.8),
        "no_gumbel":      v(gumbel_temp=0.0),
        "no_eos_first":   v(eos_first=False),
        # --- everything off: the plainest possible confidence-ordered decode ---
        "bare":           v(rep_penalty=0.0, max_run=0, subst_per_residue=0.0, gumbel_temp=0.0),
    }


def main():
    ocfg = CFG.opt
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=64, help="sequences per configuration")
    ap.add_argument("--canvas", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-dir", default=SAMPLES_DIR)
    ap.add_argument("--only", default=None, help="comma-separated subset of configuration names")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = init_distributed(args.device, no_dist=True)
    dev = env.device
    mcfg = CFG.model_config()
    model = LoopedDiffusionLM(mcfg).to(dev).eval()
    ckpt = args.ckpt or find_latest_ckpt(CKPT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint (looked at {args.ckpt or CKPT_DIR})")
    st = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(st["model"])
    if ipex is not None and dev.type == "xpu":
        model = ipex.optimize(model, dtype=torch.bfloat16)
    canvas = args.canvas or ocfg.eval_canvas
    steps = args.steps or canvas
    print(f"[sweep] {ckpt} (step {st.get('step', '?')}) on {dev} | {args.n} seqs x canvas {canvas} "
          f"x {steps} steps per configuration", flush=True)

    cfgs = configurations(ocfg)
    if args.only:
        want = {t.strip() for t in args.only.split(",")}
        cfgs = {k: v for k, v in cfgs.items() if k in want}
    os.makedirs(args.out_dir, exist_ok=True)
    use_amp = dev.type in ("xpu", "cuda")

    print(f"\n{'configuration':<16} {'len':>13} {'no-EOS':>8} {'LCR':>7}   k-mer repeat coverage")
    print("-" * 88)
    for name, kw in cfgs.items():
        torch.manual_seed(args.seed)                      # same noise draw for every configuration
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            cv, lengths = generate(model, Lmax=canvas, batch_size=args.n, n_steps=steps,
                                   device=str(dev), min_len=ocfg.sample_min_len,
                                   eos_temp=ocfg.sample_eos_temp, **kw)
        seqs, _ = decode_seqs(cv, mcfg)
        path = os.path.join(args.out_dir, f"sweep_{name}.fasta")
        tmp = path + ".tmp"
        write_fasta(seqs, tmp, prefix=f"sweep_{name}")
        os.replace(tmp, path)
        mean, sd = length_stats([len(s) for s in seqs])
        lcr, tot = lcr_counts(seqs)
        print(f"{name:<16} {mean:>6.1f}+-{sd:<5.1f} "
              f"{sum(1 for v in lengths if v >= canvas):>4}/{len(lengths):<3} "
              f"{lcr / max(tot, 1):>6.1%}   {kmer_line(kmer_counts(seqs, ocfg.kmer_ks), ocfg.kmer_ks)}",
              flush=True)
    print("-" * 88)
    print(f"wrote {len(cfgs)} FASTA(s) to {args.out_dir}. Fold them with `qsub scripts/fold.pbs` "
          f"(or `python -m src.fold_fasta --device xpu`); each becomes its own row in the summary "
          f"next to natural (81.8) and shuffled (39.0).")
    print("Read LCR against the shuffled baseline, not against zero: a configuration whose LCR sits "
          "BELOW the shuffle is suppressing local composition that chance alone would produce.")


if __name__ == "__main__":
    main()
