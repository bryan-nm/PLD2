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
    CE          ~2.9   ~2.9   ~2.8   ~2.8   ~2.8   ~2.8      broken: flat, context unused
    gain         0.0   ~0.0   ~0.1   ~0.1   ~0.1   ~0.1
    CE         2.876  2.697  2.268  2.201  2.188  2.209      MEASURED, PLD2 50k (aa-only)
    gain       0.000  0.179  0.609  0.676  0.689  0.668      mean CE 2.390, ppl 10.9

The measured row is the calibration point, not an ideal -- it is what a 54.6M-parameter model looks
like after 50k steps on UniRef90. Note two things about it. Cold start lands on 2.876 against a
unigram of 2.875, i.e. EXACTLY composition, which is the correct answer where there is no context
and confirms the model is calibrated once PAD mass is removed. And the curve is nearly flat below
50% corruption (0.609 -> 0.689), so almost all of the learnable signal this model captures is
already available from half the sequence; it is not using long-range context.

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
from .corruption import span_mask_field
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


def _corrupt_mask(shape, frac: float, width: int, dev) -> torch.Tensor:
    """(B,L) bool corruption indicator at the given fraction. width=1 is independent coin flips;
    beyond that the same fraction is drawn as contiguous runs (src/corruption.span_mask_field), with
    the per-position marginal held exactly at `frac` either way."""
    if width <= 1:
        return torch.rand(shape, device=dev) < frac
    p = torch.full((shape[0],), float(frac), device=dev)
    return span_mask_field(p, shape[1], (width,))


def masked_ce(model, tokens, mask_pos, n_aa=20):
    """-> (full-vocab CE, AA-only CE, n) over the masked AMINO-ACID positions.

    BOTH numbers are needed and they answer different questions.

      full   CE under the model's whole 23-way distribution. On a canvas where the length is
             unknown this is dominated by PAD: the canvas is 512 wide and the mean protein 246 aa,
             so hedging most of the mass onto PAD is CORRECT for the modelled distribution -- and
             scoring only true-AA positions then charges the model for being right. Measured 4.93
             nats at 100% corruption, which is 1.9 nats WORSE than uniform-over-20 and impossible
             for a calibrated 20-way choice; that gap IS the PAD mass, ~87% of it.

      AA     CE after renormalising over the 20 residues, which is EXACTLY what the sampler sees:
             _step_logits sets MASK and PAD to -inf, so decoding never sees that mass at all. This
             is the number that says whether cold-start generation is informed. Comparing it to the
             unigram ceiling is the honest version of the test.
    """
    corrupted = torch.where(mask_pos, torch.full_like(tokens, model.cfg.mask_token_id), tokens)
    with torch.no_grad():
        lg = model(corrupted).float()
    score = mask_pos & (tokens < n_aa)
    n = score.sum().clamp_min(1)
    ce_full = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tokens.reshape(-1),
                              reduction="none").reshape(tokens.shape)
    ce_aa = F.cross_entropy(lg[..., :n_aa].reshape(-1, n_aa), tokens.clamp(max=n_aa - 1).reshape(-1),
                            reduction="none").reshape(tokens.shape)
    return (float((ce_full * score).sum() / n), float((ce_aa * score).sum() / n), int(score.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # SHAPE of the corruption, not its amount. 1 = i.i.d. per position, which is what the two
    # earlier runs' curves were measured under and therefore the setting to use for a like-for-like
    # comparison. A larger width masks in contiguous runs, which is the regime a span-trained model
    # is supposed to have learned: measure both and the difference between them is the whole claim.
    ap.add_argument("--span-width", type=int, default=1,
                    help="target mean masked-run length in residues (1 = i.i.d., the comparable "
                         "setting; 8/32/128 are the training ladder)")
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
    if args.span_width > 1:
        print(f"[ce] masking in SPANS (box width {args.span_width}); the corruption AMOUNT at each "
              f"row of the table is identical to the i.i.d. curve, only its shape differs. Run "
              f"once at --span-width 1 for the number comparable to earlier runs.", flush=True)
    fracs = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02)
    ce_by_frac = {}
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for frac in fracs:
            tf, ta, n = 0.0, 0.0, 0
            for t in batches:
                # Training-shaped corruption: every canvas position is a candidate, so at frac=1.0
                # EOS and PAD are masked too -- exactly what q_sample does at t=T.
                mp = _corrupt_mask(t.shape, frac, args.span_width, dev)
                cf, ca, k = masked_ce(model, t, mp)
                tf += cf * k
                ta += ca * k
                n += k
            ce_by_frac[frac] = (tf / max(n, 1), ta / max(n, 1), n)

        # --- the canvas the sampler actually cold-starts from ---
        tf, ta, n = 0.0, 0.0, 0
        for t in batches:
            mp = t < 20                       # every residue masked; EOS and PAD left REVEALED
            cf, ca, k = masked_ce(model, t, mp)
            tf += cf * k
            ta += ca * k
            n += k
        sampler_ce, sampler_ce_aa = tf / max(n, 1), ta / max(n, 1)

    base = ce_by_frac[1.0][1]                 # AA-only no-context level: what the sampler sees
    print(f"\n{'corruption':>11} {'CE full':>9} {'CE aa-only':>11} {'ppl (aa)':>9} "
          f"{'gain (aa)':>10} {'PAD mass':>9} {'n masked':>10}")
    print("-" * 74)
    for frac in fracs:
        cf, ca, n = ce_by_frac[frac]
        pad = 1 - math.exp(-(cf - ca))        # the mass the sampler discards by banning MASK/PAD
        print(f"{frac:>10.0%} {cf:>9.3f} {ca:>11.3f} {math.exp(ca):>9.2f} {base - ca:>+10.3f} "
              f"{pad:>8.1%} {n:>10,}")
    print("-" * 74)
    mean_ce = sum(ce_by_frac[f][1] for f in fracs) / len(fracs)
    print(f"{'unigram':>10} {'':>9} {unigram:>11.3f} {math.exp(unigram):>9.2f}"
          f"{'':>10}{'':>9}  <- composition ceiling")
    print(f"{'uniform':>10} {'':>9} {uniform:>11.3f} {math.exp(uniform):>9.2f}"
          f"{'':>10}{'':>9}  <- knows nothing ({uniform - unigram:.2f} above unigram)")
    print(f"\nAll gains and the mean below use the AA-ONLY column, because that is the distribution "
          f"the sampler decodes from.")
    print(f"mean CE across the curve = {mean_ce:.3f} nats (ppl {math.exp(mean_ce):.1f})")
    edge = unigram - mean_ce
    print(f"  By the ARDM identity this IS the model's per-token generative NLL. Composition "
          f"ceiling {unigram:.3f} -> the model is "
          f"{abs(edge):.3f} nats {'BETTER than' if edge > 0 else 'WORSE than'} composition."
          + ("  A generative model within ~0.1 of the ceiling samples from composition."
             if edge < 0.1 else ""))
    print(f"\nsampler cold-start canvas (all residues MASK, EOS/PAD revealed):")
    print(f"  full {sampler_ce:.3f} | aa-only {sampler_ce_aa:.3f} (ppl {math.exp(sampler_ce_aa):.2f})"
          f" | vs unigram {unigram - sampler_ce_aa:+.3f}")
    print(f"  training-shaped 100%:  full {ce_by_frac[1.0][0]:.3f} | aa-only {base:.3f}")
    print(f"  The aa-only pair is what matters: the length is revealed in the first and not the "
          f"second, so a model that USES the boundary should be better on the first.")

    gain = base - ce_by_frac[0.05][1]
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
    d = sampler_ce_aa - base          # BOTH aa-only; mixing scales here was a real bug
    if d > 0.05:
        print(f"  * The sampler's cold-start canvas is {d:.3f} nats WORSE (aa-only) than "
              f"training-shaped corruption at the same level, despite revealing strictly MORE (the "
              f"length). That is an off-distribution input: eos_first builds a canvas training "
              f"almost never shows.")
    else:
        print(f"  * The sampler's cold-start canvas is fine ({d:+.3f} nats aa-only vs "
              f"training-shaped), so eos_first is not handing the model an off-distribution input. "
              f"A model that USED the revealed length would be meaningfully better here, not merely "
              f"equal, so this also says the boundary is barely informing residue choice.")


if __name__ == "__main__":
    main()
