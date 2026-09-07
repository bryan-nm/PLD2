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
from .sampler import decode_seqs, generate, lengths_of, write_fasta
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
        "default":        v(),                                   # the shipped config, now the good one
        # --- anti-repetition, stated ABSOLUTELY rather than relative to the base. The base has
        #     since moved to rep_penalty=0 / max_run=5, and configurations written as deltas from it
        #     silently collapsed onto each other (default==penalty_off, no_reppen==maxrun_off) --
        #     two of twelve tiles doing duplicate work.
        "reppen_1.5":     v(rep_penalty=1.5, max_run=5),         # the OLD default: negative control
        "reppen_soft":    v(rep_penalty=0.3, max_run=5),         # is a light touch harmless?
        "no_maxrun":      v(rep_penalty=0.0, max_run=0),         # is the run cap doing anything?
        "periods_1_2":    v(rep_penalty=1.5, max_run=5, rep_periods=(1, 2)),  # spare helical 3/4
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
        # --- WHERE IS THE JOINT STRUCTURE LOST? -------------------------------------------------
        # The cosine schedule already commits 0.92 slots per step on average (71% of commits happen
        # one at a time, 29% in pairs, never three), so "positions sampled independently from their
        # own marginals" has very little room to be true. These bound it rather than assume it.
        "nr_steps2x":     v(**off, n_steps=2 * ocfg.eval_steps),   # 1 slot/step: factorisation exact
        "nr_steps4x":     v(**off, n_steps=4 * ocfg.eval_steps),   # does even that help?
        # STRUCTURE FIRST. Built when the 3Di track went in and never once tested on a trained
        # model -- sample_struct_first has been 0.0 in all three two-track runs. It is the one lever
        # that lets the decoder lay down a fold BEFORE filling sequence into it, which is exactly
        # the "commit to a global plan first" the cold-start measurement says is missing.
        "nr_struct1st.3": v(**off, struct_first=0.3),
        "nr_struct1st.7": v(**off, struct_first=0.7),
        # BEST-OF-N over whole samples, ranked by the model's own held-out-style likelihood. The
        # first commits are drawn from an all-MASK canvas where the model's distribution measures
        # 2.876 nats against a 2.875 unigram -- pure composition -- so every generation is built to
        # be consistent with a near-random seed. Search is the cheapest way to ask how much of the
        # gap is that seed rather than the model.
        "nr_best4":       v(**off, n_best=4),
        "nr_best8":       v(**off, n_best=8),
    }


@torch.no_grad()
def model_nll(model, canvas, mcfg, sched_free_levels=(0.9, 0.7, 0.5, 0.3, 0.1), seed=0):
    """Mean masked-position NLL of each row under the model -> [B], lower is better.

    By the ARDM identity the mean of this curve over corruption levels IS the model's per-token
    generative NLL, so it is the model's own opinion of the sequence it just produced -- which is
    what best-of-n needs to rank candidates without a second network. Corruption is i.i.d. here
    rather than spanned: the ranking only has to be consistent across candidates, and i.i.d. gives
    the lower-variance estimate at a fixed number of draws.
    """
    aa = canvas[:, 0] if canvas.dim() == 3 else canvas
    B, L = aa.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot = torch.zeros(B, device=aa.device)
    for frac in sched_free_levels:
        keep = (torch.rand(aa.shape, generator=g).to(aa.device) >= frac)
        x = torch.where(keep, aa, torch.full_like(aa, mcfg.mask_token_id))
        st = None
        if mcfg.n_tracks == 2:
            st = canvas[:, 1] if canvas.dim() == 3 else torch.full_like(aa, mcfg.mask_token_id)
        lg = model(x, struct=st) if mcfg.n_tracks == 2 else model(x)
        lg = lg[:, 0] if lg.dim() == 4 else lg
        lp = torch.log_softmax(lg[..., :20].float(), dim=-1)
        scored = (~keep) & (aa < 20)
        nll = -lp.gather(-1, aa.clamp(max=19).unsqueeze(-1)).squeeze(-1)
        tot += (nll * scored).sum(1) / scored.sum(1).clamp_min(1)
    return tot / len(sched_free_levels)


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
    # --- FILIP prompt conditioning (src/filip_guidance.py). Giving --guide-prompt REPLACES the
    #     sweep with one conditional configuration per gamma, so the unconditional rows and the
    #     guided ones can be folded side by side in the same table.
    ap.add_argument("--guide-prompt", default=None,
                    help="cache row index or accession to condition on; enables FILIP guidance")
    ap.add_argument("--guide-gammas", default="0,1,3,10",
                    help="comma-separated guidance strengths; 0 is the unconditional control")
    ap.add_argument("--guide-mode", default="tag", choices=("tag", "deg"))
    ap.add_argument("--guide-likelihood", default="sigmoid", choices=("sigmoid", "softmax_bank"))
    ap.add_argument("--guide-bank", default=None,
                    help="comma-separated cache rows for softmax_bank normalisation")
    ap.add_argument("--filip-ckpt", default=None)
    ap.add_argument("--filip-cache", default=None)
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
    guide = None
    if args.guide_prompt is not None:
        from .filip_guidance import FilipGuidance
        from config import FILIP_CACHE, FILIP_CKPT
        bank = [int(x) for x in args.guide_bank.split(",")] if args.guide_bank else None
        guide = FilipGuidance(mcfg, dev, ckpt=args.filip_ckpt or FILIP_CKPT,
                              cache_dir=args.filip_cache or FILIP_CACHE,
                              mode=args.guide_mode, likelihood=args.guide_likelihood,
                              bank_rows=bank, verbose=(env.rank == 0))
        name = guide.set_target(args.guide_prompt)
        if env.rank == 0:
            print(f"[filip] conditioning on prompt row {args.guide_prompt} ({name})", flush=True)
        base = dict(rep_penalty=0.0, max_run=0, temperature=ocfg.sample_temperature,
                    gumbel_temp=ocfg.sample_gumbel_temp,
                    subst_per_residue=ocfg.sample_subst_per_residue,
                    eos_first=ocfg.sample_eos_first)
        cfgs = {}
        for gstr in args.guide_gammas.split(","):
            g = float(gstr)
            tag = "uncond" if g == 0 else f"g{gstr}"
            cfgs[f"filip_{tag}"] = dict(base, _gamma=g)

    rank, world = env.rank, env.world_size
    mine = {k: v for i, (k, v) in enumerate(cfgs.items()) if i % world == rank}
    if rank == 0:
        print(f"\n{'configuration':<16} {'len':>13} {'no-EOS':>8} {'LCR':>7}   k-mer repeat coverage")
        print("-" * 88, flush=True)
    for name, kw in mine.items():
        kw = dict(kw)
        n_steps = kw.pop("n_steps", steps)
        n_best = int(kw.pop("n_best", 1))
        gamma = kw.pop("_gamma", None)
        if gamma is not None:
            guide.gamma = float(gamma)
            kw["guidance_fn"] = guide if gamma > 0 else None
        torch.manual_seed(args.seed)                      # same noise draw for every configuration
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            if n_best <= 1:
                cv, lengths = generate(model, Lmax=canvas, batch_size=args.n, n_steps=n_steps,
                                       device=str(dev), min_len=ocfg.sample_min_len,
                                       eos_temp=ocfg.sample_eos_temp, **kw)
            else:
                # Draw n_best independent candidate sets and keep, per slot, the one the model
                # itself scores best. Different seeds per round: the point is to search over the
                # near-random opening commits, and a shared seed would replay the same ones.
                best_cv = best_nll = None
                for r in range(n_best):
                    torch.manual_seed(args.seed + 7919 * (r + 1))
                    c, _ = generate(model, Lmax=canvas, batch_size=args.n, n_steps=n_steps,
                                    device=str(dev), min_len=ocfg.sample_min_len,
                                    eos_temp=ocfg.sample_eos_temp, **kw)
                    nll = model_nll(model, c, mcfg, seed=args.seed)
                    if best_cv is None:
                        best_cv, best_nll = c, nll
                    else:
                        take = nll < best_nll
                        sel = take.view(-1, *([1] * (c.dim() - 1)))
                        best_cv = torch.where(sel, c, best_cv)
                        best_nll = torch.where(take, nll, best_nll)
                cv, lengths = best_cv, lengths_of(best_cv, mcfg)
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
