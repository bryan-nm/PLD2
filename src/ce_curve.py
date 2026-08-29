"""Cross-entropy as a function of CORRUPTION LEVEL -- the number the training loss hides.

    python -m src.ce_curve --device xpu

WHY. The training loss draws t ~ U(1, T), so the single reported figure averages the whole corruption
range and hides the shape. The shape is the thing: it says how much the model gains from CONTEXT, at
every level of context.

WHAT THE CURVE MEANS, AND WHAT IT DOES NOT.

  CE at 100% corruption is NOT a defect signal. There is no context on an all-MASK canvas, so the
  per-position marginal is the correct answer, and the marginal is approximately the unigram. Worse,
  the unigram line is a nearly meaningless bar for proteins: uniform over 20 residues is 3.00 nats
  and the empirical unigram is 2.89, so the entire span between "knows nothing" and "knows
  composition" is 0.11 nats. An earlier version of this file tested exactly that and it was wrong.

  THE SLOPE IS THE SIGNAL. A model that has learned protein sequence constraints should get steadily
  better as the rest of the sequence is revealed. The reported CONTEXT GAIN column is
  CE(100%) - CE(frac): how many nats the model buys from seeing the rest. A curve that stays flat
  near its own 100% value means context is not being used at any level -- which is a far stronger
  and more general statement than anything about cold start.

  THE AREA IS THE MODEL. By the ARDM identity the OADM/any-order bound on log p(x) is D times the
  MEAN of this curve, so mean CE across the corruption range IS the model's per-token generative
  NLL. Compare it to the 2.89 unigram ceiling: a mean within ~0.1 of it describes a sampler that
  draws from composition, whatever it does at any individual level.

WHAT HEALTHY LOOKS LIKE (a 55M any-order model on UniRef90). Absolute values matter less than shape,
and protein sequence is genuinely high-entropy, so the whole usable dynamic range is about a nat:

    corruption   100%    90%    50%    20%    10%     5%
    CE          ~2.9   ~2.8   ~2.4   ~2.1   ~2.0   ~1.9      healthy: falls steadily
    gain         0.0   ~0.1   ~0.5   ~0.8   ~0.9   ~1.0
    CE          ~2.9   ~2.9   ~2.8   ~2.8   ~2.8   ~2.8      broken: flat, context unused
    gain         0.0   ~0.0   ~0.1   ~0.1   ~0.1   ~0.1

The single number to look at is the context gain at 5-10% corruption -- given ~90% of a real protein,
how much better than no context at all. Below ~0.2 nats the model is not using sequence context;
above ~0.5 it clearly is, and a floor-level pLDDT then points at the sampler instead.

IT ALSO CHECKS THE SAMPLER'S ACTUAL COLD-START CANVAS, which is not the same shape as anything
training produces. After eos_first commits the boundary, the canvas is [MASK x n, EOS, PAD x ...] --
every residue unknown but the length REVEALED. Training corrupts positions uniformly, so at ~100%
corruption EOS and PAD are masked too; a canvas with all residues masked and the boundary intact is
vanishingly rare in training. If the model is much worse on the sampler's canvas than on the
training-shaped one at the same corruption level, eos_first is handing it an off-distribution input.

NOTE this is a HELD-OUT measurement (data.ProteinShards split="holdout"), unlike ProLoopDiff's
version which had no held-out split to use.
"""
from __future__ import annotations
import argparse
import math
import os

import torch
import torch.nn.functional as F

from config import CFG, CKPT_DIR, UNIREF_SHARDS
from .data import ProteinShards, make_collate
from .dist import init_distributed
from .model import LoopedDiffusionLM
from .train import find_latest_ckpt

try:
    import intel_extension_for_pytorch as ipex
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)
except Exception:
    ipex = None


def masked_ce(model, tokens, mask_pos, n_aa=20):
    """Mean CE over the masked AMINO-ACID positions only. EOS/PAD are excluded: they are trivially
    predictable from the canvas geometry and would dilute exactly the number we are asking about."""
    corrupted = torch.where(mask_pos, torch.full_like(tokens, model.cfg.mask_token_id), tokens)
    with torch.no_grad():
        lg = model(corrupted)
    ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tokens.reshape(-1),
                         reduction="none").reshape(tokens.shape)
    score = mask_pos & (tokens < n_aa)
    return float((ce * score).sum() / score.sum().clamp_min(1)), int(score.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = init_distributed(args.device, no_dist=True)
    dev = env.device
    mcfg, dcfg = CFG.model_config(), CFG.data
    model = LoopedDiffusionLM(mcfg).to(dev).eval()
    ckpt = args.ckpt or find_latest_ckpt(CKPT_DIR)
    if not ckpt or not os.path.exists(ckpt):
        raise SystemExit(f"no checkpoint (looked at {args.ckpt or CKPT_DIR})")
    st = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(st["model"])
    print(f"[ce] {ckpt} (step {st.get('step', '?')}) on {dev}", flush=True)
    if ipex is not None and dev.type == "xpu":
        model = ipex.optimize(model, dtype=torch.bfloat16)

    shards = ProteinShards(UNIREF_SHARDS, mcfg.eos_token_id, split="holdout",
                           holdout_stride=max(dcfg.holdout_stride, 2))
    collate = make_collate(mcfg, dcfg.canvas)
    g = torch.Generator().manual_seed(args.seed)
    batches = [collate([shards.get(int(i)) for i in
                        torch.randint(0, len(shards), (args.batch_size,), generator=g)])["tokens"].to(dev)
               for _ in range(args.batches)]

    # --- composition baselines from the SAME data ---
    counts = torch.zeros(20, device=dev)
    for t in batches:
        counts += torch.bincount(t[t < 20].reshape(-1), minlength=20).float()
    p = counts / counts.sum()
    unigram = float(-(p * p.clamp_min(1e-12).log()).sum())
    uniform = math.log(20)

    use_amp = dev.type in ("xpu", "cuda")
    torch.manual_seed(args.seed)
    fracs = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02)
    ce_by_frac = {}
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for frac in fracs:
            tot, n = 0.0, 0
            for t in batches:
                # Training-shaped corruption: every canvas position is a candidate, so at frac=1.0
                # EOS and PAD are masked too -- exactly what q_sample does at t=T.
                mp = torch.rand(t.shape, device=dev) < frac
                ce, k = masked_ce(model, t, mp)
                tot += ce * k
                n += k
            ce_by_frac[frac] = (tot / max(n, 1), n)

        # --- the canvas the sampler actually cold-starts from ---
        tot, n = 0.0, 0
        for t in batches:
            mp = t < 20                       # every residue masked; EOS and PAD left REVEALED
            ce, k = masked_ce(model, t, mp)
            tot += ce * k
            n += k
        sampler_ce = tot / max(n, 1)

    base = ce_by_frac[1.0][0]                 # the model's OWN no-context level
    print(f"\n{'corruption':>11} {'CE (nats)':>11} {'perplexity':>11} {'context gain':>13} "
          f"{'n masked':>10}")
    print("-" * 60)
    for frac in fracs:
        ce, n = ce_by_frac[frac]
        print(f"{frac:>10.0%} {ce:>11.3f} {math.exp(ce):>11.2f} {base - ce:>+13.3f} {n:>10,}")
    print("-" * 60)
    mean_ce = sum(ce_by_frac[f][0] for f in fracs) / len(fracs)
    print(f"{'unigram':>10}  {unigram:>10.3f} {math.exp(unigram):>11.2f}"
          f"{'':>14}  <- composition ceiling")
    print(f"{'uniform':>10}  {uniform:>10.3f} {math.exp(uniform):>11.2f}"
          f"{'':>14}  <- knows nothing (only {uniform - unigram:.2f} nats above unigram)")
    print(f"\nmean CE across the curve = {mean_ce:.3f} nats (ppl {math.exp(mean_ce):.1f})")
    edge = unigram - mean_ce
    print(f"  By the ARDM identity this IS the model's per-token generative NLL. Composition "
          f"ceiling {unigram:.3f} -> the model is "
          f"{abs(edge):.3f} nats {'BETTER than' if edge > 0 else 'WORSE than'} composition."
          + ("  A generative model within ~0.1 of the ceiling samples from composition."
             if edge < 0.1 else ""))
    print(f"\nsampler cold-start canvas (all residues MASK, EOS/PAD revealed): "
          f"CE {sampler_ce:.3f} (ppl {math.exp(sampler_ce):.2f})")
    print(f"  vs training-shaped corruption at 100%:                        "
          f"CE {base:.3f} (ppl {math.exp(base):.2f})")

    gain = base - ce_by_frac[0.05][0]
    print("\nREADING (the slope, not the intercept -- CE ~ unigram at 100% is CORRECT, there is no")
    print("         context to use there, and uniform is only 0.11 nats above unigram anyway):")
    if gain < 0.2:
        print(f"  * Context gain at 5% corruption is only {gain:+.3f} nats. Given ~95% of a real "
              f"protein the model is barely better than with nothing at all -- it is not using "
              f"sequence context AT ANY LEVEL. That is a model/objective problem, and no sampler "
              f"change will fix it.")
    elif gain < 0.5:
        print(f"  * Context gain at 5% corruption is {gain:+.3f} nats -- weak. The model uses "
              f"context but has learned much less than it should. Expect generations near "
              f"composition; treat the sampler as a secondary effect.")
    else:
        print(f"  * Context gain at 5% corruption is {gain:+.3f} nats, so the model HAS learned "
              f"real sequence constraints. If generations still fold at the floor, the sampler is "
              f"losing that information -- run src/sweep_sampler.py.")
    d = sampler_ce - base
    if d > 0.05:
        print(f"  * The sampler's cold-start canvas is {d:.3f} nats WORSE than training-shaped "
              f"corruption at the same level, despite revealing strictly MORE (the length). That is "
              f"an off-distribution input: eos_first builds a canvas training almost never shows.")
    else:
        print(f"  * The sampler's cold-start canvas is fine ({d:+.3f} nats vs training-shaped), and "
              f"a healthy model should be slightly BETTER here since the length is revealed.")


if __name__ == "__main__":
    main()
