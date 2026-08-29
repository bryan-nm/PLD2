"""Cross-entropy as a function of CORRUPTION LEVEL -- the number the training loss hides.

    python -m src.ce_curve --device xpu

WHY. The training loss draws t ~ U(1, T), so the single reported figure averages the whole corruption
range, and those ends are not equally relevant. Sampling COLD-STARTS from an all-MASK canvas and
commits its first -- and most structurally decisive -- positions there, so generation quality is
governed almost entirely by the ~100%-corrupted end, while the average is dominated by the easy low
end where most of the sequence is visible and the task is near-copying. A model that infills well and
cold-starts at composition posts a healthy-looking loss and still generates sequences that fold no
better than a shuffle. ProLoopDiff had exactly this diagnostic; PLD2 shipped without it.

Read the LAST rows. Two reference points are computed from the SAME data:

    uniform   ln(20) = 3.00 nats -- a model that knows nothing
    unigram   the empirical residue distribution -- a model that knows only composition

If CE at 90-100% corruption sits at the unigram line, the model is drawing its opening commitments
from composition alone. No amount of sampler work fixes that, and it is the signature of samples that
have the right amino-acid frequencies and no structure.

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
    print(f"\n{'corruption':>12} {'CE (nats)':>11} {'perplexity':>11} {'vs unigram':>11}  "
          f"{'n masked':>10}")
    print("-" * 62)
    rows = []
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        for frac in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
            tot, n = 0.0, 0
            for t in batches:
                # Training-shaped corruption: every canvas position is a candidate, so at frac=1.0
                # EOS and PAD are masked too -- exactly what q_sample does at t=T.
                mp = torch.rand(t.shape, device=dev) < frac
                ce, k = masked_ce(model, t, mp)
                tot += ce * k
                n += k
            ce = tot / max(n, 1)
            rows.append((frac, ce))
            print(f"{frac:>11.0%} {ce:>11.3f} {math.exp(ce):>11.2f} "
                  f"{ce - unigram:>+11.3f}  {n:>10,}")

        # --- the canvas the sampler actually cold-starts from ---
        tot, n = 0.0, 0
        for t in batches:
            mp = t < 20                       # every residue masked; EOS and PAD left REVEALED
            ce, k = masked_ce(model, t, mp)
            tot += ce * k
            n += k
        sampler_ce = tot / max(n, 1)

    print("-" * 62)
    print(f"{'unigram':>11}  {unigram:>10.3f} {math.exp(unigram):>11.2f} {0.0:>+11.3f}   "
          f"<- knows only composition")
    print(f"{'uniform':>11}  {uniform:>10.3f} {math.exp(uniform):>11.2f} "
          f"{uniform - unigram:>+11.3f}   <- knows nothing")
    print(f"\nsampler cold-start canvas (all residues MASK, EOS/PAD revealed): "
          f"CE {sampler_ce:.3f} (ppl {math.exp(sampler_ce):.2f})")
    print(f"  vs training-shaped corruption at 100%:                        "
          f"CE {rows[-1][1]:.3f} (ppl {math.exp(rows[-1][1]):.2f})")

    top = rows[-1][1]
    print("\nREADING:")
    if top > unigram - 0.05:
        print(f"  * At full corruption the model is at or above the unigram line "
              f"({top:.3f} vs {unigram:.3f}). It opens generation from COMPOSITION ONLY, so its "
              f"first and most structurally decisive commitments carry no information. Samples will "
              f"fold like a shuffle no matter what the sampler does.")
    else:
        print(f"  * At full corruption the model beats unigram by {unigram - top:.3f} nats, so it "
              f"DOES know something at cold start. If generations still fold at the floor, the "
              f"sampler is losing that information -- see src/sweep_sampler.py.")
    d = sampler_ce - rows[-1][1]
    if d > 0.05:
        print(f"  * The sampler's cold-start canvas is {d:.3f} nats WORSE than training-shaped "
              f"corruption at the same level, despite revealing strictly more (the length). That is "
              f"an off-distribution input: eos_first builds a canvas training almost never shows.")
    else:
        print(f"  * The sampler's cold-start canvas is fine ({d:+.3f} nats vs training-shaped), so "
              f"eos_first is not handing the model an off-distribution input.")


if __name__ == "__main__":
    main()
