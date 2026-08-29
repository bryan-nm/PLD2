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
import sys

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
    """name -> generate() overrides.

    Centred on `no_reppen`, not on the shipped default, because the default is already known bad:
    at the 50k checkpoint it gave pLDDT 35.0 with LCR 0.0%, while rep_penalty=0 gave 43.4 with LCR
    7.4% against natural's 7.9% -- four statistics moving to natural at once. So `default` is kept
    only as the reference row, `nr_*` build on rep_penalty=0, and two configurations disentangle
    WHICH half of the anti-repetition machinery did the damage (the periodic logit penalty, or the
    hard run cap) since the earlier test removed both together.
    """
    base = dict(temperature=ocfg.sample_temperature, gumbel_temp=ocfg.sample_gumbel_temp,
                rep_penalty=ocfg.sample_rep_penalty, max_run=ocfg.sample_max_run,
                rep_periods=ocfg.sample_rep_periods,
                subst_per_residue=ocfg.sample_subst_per_residue,
                eos_first=ocfg.sample_eos_first)
    def v(**kw):
        d = dict(base)
        d.update(kw)
        return d
    off = dict(rep_penalty=0.0, max_run=0)
    return {
        "default":        v(),                                   # reference: the shipped config
        # --- which half of the anti-repetition machinery mattered? ---
        "no_reppen":      v(**off),                              # both off: the known-good point
        "penalty_off":    v(rep_penalty=0.0),                    # ...but keep the hard run cap
        "maxrun_off":     v(max_run=0),                          # ...but keep the periodic penalty
        "reppen_soft":    v(rep_penalty=0.3),                    # is a light touch harmless?
        "periods_1_2":    v(rep_periods=(1, 2)),                 # leave helical 3/4 periodicity alone
        # --- everything else, built ON TOP of rep_penalty=0 ---
        "nr_t0.8":        v(**off, temperature=0.8),
        "nr_t1.2":        v(**off, temperature=1.2),
        "nr_no_subst":    v(**off, subst_per_residue=0.0),
        "nr_subst4":      v(**off, subst_per_residue=4.0),
        "nr_no_gumbel":   v(**off, gumbel_temp=0.0),
        "nr_no_eos_1st":  v(**off, eos_first=False),
        # Adaptive compute at inference: the looped trunk takes an n_recurrence override, so this
        # costs nothing but a flag and is the one knob that adds model capacity at decode time.
        "nr_recur6":      v(**off, n_recurrence=6),
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
    sys.stdout.reconfigure(line_buffering=True)

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

    # One rank per configuration where there are ranks to spare. No process group is ever created
    # (init_distributed(no_dist=True) above), configurations write disjoint filenames, and nothing
    # needs aggregating -- the fold summary does that later from the FASTAs themselves.
    rank, world = env.rank, env.world_size
    mine = {k: v for i, (k, v) in enumerate(cfgs.items()) if i % world == rank}
    if rank == 0:
        print(f"\n{'configuration':<16} {'len':>13} {'no-EOS':>8} {'LCR':>7}   k-mer repeat coverage")
        print("-" * 88, flush=True)
    for name, kw in mine.items():
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
    if rank != 0:
        return
    print("-" * 88)
    print(f"wrote {len(cfgs)} FASTA(s) to {args.out_dir}. Fold them with `qsub scripts/fold.pbs` "
          f"(or `python -m src.fold_fasta --device xpu`); each becomes its own row in the summary "
          f"next to natural (81.8) and shuffled (39.0).")
    print("Read LCR against the shuffled baseline, not against zero: a configuration whose LCR sits "
          "BELOW the shuffle is suppressing local composition that chance alone would produce.")


if __name__ == "__main__":
    main()
