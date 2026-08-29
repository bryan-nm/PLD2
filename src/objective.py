"""PLD2 training objective: ONE process, ONE ELBO, beta as the slider.

This replaces a two-branch design (pure-OADM rows scored by a reweighted cross-entropy, pure-
substitution D3PM rows scored by a KL-ELBO). That design had the two frameworks as disjoint
alternatives, and its D3PM half reproduced EvoDiff's uniform-stationary chain -- which cold-starts
from uniform-random residues, has no "this position is unknown" signal, and is the configuration
that loses to OADM. Blending them at generation time then needed a bolted-on second channel with
its own schedule.

Now there is one corruption process with MASK as an absorbing state (see src/corruption.py), and
beta is a parameter OF that process: the fraction of corruption events that are maskings rather
than BLOSUM-weighted substitutions. beta=1 is exactly OADM's corruption; beta<1 enriches the same
trajectories with substitution moves. Every beta shares the all-MASK stationary distribution, so
every beta shares OADM's cold-start entry point.

    L = L_vb + ce_weight * L_ce

  L_vb   T * E_t KL( q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t) ) -- the D3PM variational bound over
         the unified Q. It covers BOTH move types in one term (unmasking an absorbed position and
         substituting a placed one), and at t=1 reduces to L_0 = -log p_theta(x_0|x_1) with no
         special case (Qbar_0 = I). L_T is dropped, and here that is free rather than an
         approximation: P(x_T = MASK) measures 0.999-1.000 at every beta.

         THE FACTOR OF T IS NOT COSMETIC. L_vb is a SUM over t = 1..T; sampling one t ~ U(1,T) and
         taking the KL estimates the MEAN, so the unbiased estimator of the sum multiplies by T.
         The first PLD2 run omitted it and the consequence was severe: the reported vb sat at 0.005
         against a ce of ~1.3, so the variational term -- the entire reason this is a diffusion
         model rather than a masked LM -- contributed 0.4% of the gradient and the run was, in
         effect, trained on the auxiliary cross-entropy alone. At T=500 the scaled term lands near
         2.5, comparable to L_ce, which is what the ELBO says it should be.

  L_ce   -log p~_theta(x_0 | x_t) over the CORRUPTED positions, EvoDiff's lambda term. They set
         lambda=0; PLD2 keeps it at 1 because at beta=1 it IS the OADM cross-entropy, so the
         objective that demonstrably works stays present as a component.

         SCORED ON CORRUPTED POSITIONS ONLY, which the first run did not do. Mask fraction is
         U(0,1), so about half of every row is a position whose answer is VISIBLE in the input --
         a copy, solved within the first few hundred steps, and worth no gradient thereafter.
         Scoring it spent roughly half the objective's weight on nothing and made the reported
         number uninterpretable (it mixes ~0-cost copies with real predictions, which is why
         estimating the true masked-position CE from the log took three assumptions). A position
         counts as corrupted when x_t != x_0, which is exactly "the model cannot copy the answer":
         MASK always qualifies, and a substitution that happened to land on the original token
         correctly does not. `ce_uncorrupted_weight` restores the old behaviour at 1.0.

         Note the "keep this token" signal is not lost by restricting L_ce -- it lives in L_vb,
         whose posterior at an uncorrupted position is peaked on the current token and which the
         factor of T now makes count.

A DISTRIBUTION OVER BETA IS NOW COHERENT, and is the default. Under the old design mixing beta
would have blurred two different bounds together -- the mistake ProLoopDiff made. Here beta only
selects which transition matrix a row is corrupted by, and one ELBO scores them all, so drawing a
different beta per row is just training over a family of corruption processes. Rows are assigned
round-robin (`arange(B) % n_beta`) rather than randomly: the batch order is already shuffled, so
that is unbiased, and it makes the mix exactly balanced every step instead of only in expectation.

----------------------------------------------------------------------------------------------
POSITION WEIGHTS (upweight EOS).

A fixed 512 canvas holds about 350 residues, 160 PAD and exactly ONE EOS. Unweighted, the token that
decides sequence LENGTH carries 1/512 = 0.2% of the loss, and ProLoopDiff duly learned everything
except where to stop: at 181k steps, 53% of unconditional samples never placed EOS at all. At
eos_loss_weight=20 against pad_loss_weight=0.1 the split is roughly 350 : 16 : 20, i.e. EOS is ~5%
of the per-sequence loss -- ~25x its unweighted share, and still far from dominating.
"""

from __future__ import annotations
from typing import Optional

import torch

from .corruption import CorruptionSchedule, kl_categorical, x0_probs
from .model import LoopedDiffusionLM, Config, count_params


# --------------------------------------------------------------------------------------
# Position weights
# --------------------------------------------------------------------------------------
def position_weights(x0: torch.Tensor, eos_id: int, pad_id: int,
                     eos_weight: float = 1.0, pad_weight: float = 1.0) -> torch.Tensor:
    """(B,L) float loss weight per position, keyed on the TARGET token."""
    w = torch.ones_like(x0, dtype=torch.float32)
    if pad_weight != 1.0:
        w = torch.where(x0 == pad_id, w.new_full((), pad_weight), w)
    if eos_weight != 1.0:
        w = torch.where(x0 == eos_id, w.new_full((), eos_weight), w)
    return w


def _weighted_row_mean(per_pos: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean over positions for each row, then mean over rows. -> scalar."""
    per_seq = (per_pos * w).sum(dim=1) / w.sum(dim=1).clamp_min(1e-6)
    return per_seq.mean()


# --------------------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------------------
def diffusion_loss(logits, x0, xt, flat_idx, flat_prev, sched: CorruptionSchedule,
                   weights, ce_weight: float = 1.0, vb_weight: float = 1.0,
                   ce_uncorrupted_weight: float = 0.0):
    """vb_weight * T * E_t[KL] + ce_weight * L_ce over the unified process. See the module docstring
    for why the T and the corrupted-only restriction both matter.

    AUTOCAST OFF for the whole posterior computation. The trainer runs under bf16 autocast and
    torch.bmm is on autocast's lower-precision list, so `p_tilde @ Qbar_{t-1}` would be done in bf16
    however fp32 its operands are. Qbar_{t-1} is near-identity for small t with off-diagonal entries
    ~1e-3 that an 8-bit mantissa cannot hold beside a diagonal near 1 -- and a KL between two nearly
    equal categoricals is exactly where that turns into noise.
    """
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p_tilde = x0_probs(logits, sched.n)                     # (B,L,n) over non-MASK states
        q_post, p_post = sched.posteriors(x0, xt, flat_idx, flat_prev, p_tilde)
        # Per-step KL, then scaled to the SUM over t that the ELBO actually is.
        vb_step = _weighted_row_mean(kl_categorical(q_post, p_post), weights)
        vb = sched.T * vb_step
        # Corrupted == the model cannot copy the answer out of its own input.
        corrupted = (xt != x0).float()
        ce_w = weights * (corrupted + ce_uncorrupted_weight * (1.0 - corrupted))
        ce = _weighted_row_mean(
            -p_tilde.gather(-1, x0.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log(), ce_w)
    return (vb_weight * vb + ce_weight * ce, vb.detach(), vb_step.detach(), ce.detach())


def training_step(model: LoopedDiffusionLM, batch: dict, sched: CorruptionSchedule,
                  eos_loss_weight: float = 1.0, pad_loss_weight: float = 1.0,
                  ce_weight: float = 1.0, vb_weight: float = 1.0,
                  ce_uncorrupted_weight: float = 0.0):
    """batch: {"tokens": (B, L) long}. Returns (loss, metrics).

    Metrics are on-device 0-d tensors: converting them here would force a device->host sync every
    step for numbers the trainer only prints every log_every steps.
    """
    cfg = model.cfg
    x0 = batch["tokens"]
    canvas_mask = batch.get("canvas_mask")            # None -> the whole 512 canvas is modelled
    B = x0.shape[0]

    beta_idx = torch.arange(B, device=x0.device) % sched.n_beta      # balanced, unbiased, static
    t = sched.sample_t(B, x0.device)
    flat_idx, flat_prev = sched.flat(beta_idx, t), sched.flat(beta_idx, t - 1)

    xt = sched.q_sample(x0, flat_idx)
    logits = model(xt, canvas_mask=canvas_mask)

    w = position_weights(x0, cfg.eos_token_id, cfg.pad_token_id, eos_loss_weight, pad_loss_weight)
    loss, vb, vb_step, ce = diffusion_loss(logits, x0, xt, flat_idx, flat_prev, sched, w,
                                           ce_weight, vb_weight, ce_uncorrupted_weight)

    return loss, {"loss": loss.detach(), "vb": vb, "vb_step": vb_step, "ce": ce,
                  # What the corruption actually did this step -- the cheapest guard against a
                  # schedule that silently stops masking or stops substituting.
                  "masked": (xt == cfg.mask_token_id).float().mean(),
                  "subst": ((xt != x0) & (xt != cfg.mask_token_id)).float().mean()}


# --------------------------------------------------------------------------------------
# Overfit demo
# --------------------------------------------------------------------------------------
def _eval_oadm(model, tokens, cfg, frac=0.5):
    """Deterministically mask each row's C-terminal `frac` and report the masked-position NLL.
    Low-variance progress signal: unlike the training loss it does not average over noise levels."""
    import torch.nn.functional as F
    mask_pos = torch.zeros_like(tokens, dtype=torch.bool)
    k = max(1, int(tokens.shape[1] * frac))
    mask_pos[:, -k:] = True
    corrupted = torch.where(mask_pos, torch.full_like(tokens, cfg.mask_token_id), tokens)
    model.eval()
    with torch.no_grad():
        lg = model(corrupted)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tokens.reshape(-1),
                             reduction="none").reshape(tokens.shape)
        out = float((ce * mask_pos).sum() / mask_pos.sum())
    model.train()
    return out


if __name__ == "__main__":
    from .blosum import uniform_substitution_kernel
    torch.manual_seed(0)

    cfg = Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
                 d_model=128, n_heads=4, d_ff=384,
                 n_upstream=2, n_middle=4, n_downstream=2, n_recurrence=2)
    betas = (1.0, 0.9, 0.75, 0.5)
    sched = CorruptionSchedule(uniform_substitution_kernel(22), 23, 22, betas=betas, T=100)
    model = LoopedDiffusionLM(cfg)
    print(f"demo params={count_params(model)/1e6:.2f}M | betas={betas} T={sched.T}")
    for bi, b in enumerate(betas):
        print(f"  beta={b:<5} P(x_T=MASK)={sched.terminal_mask_fraction(bi):.4f} "
              f"| mask fraction {sched.mask_fraction(bi, (0, 25, 50, 75, 100))} "
              f"| substituted among survivors at T/2: {sched.substitution_fraction(bi):.1%}")

    B, L = 8, 24
    lengths = [18, 15, 12, 9, 20, 7, 16, 11]
    tokens = torch.full((B, L), cfg.pad_token_id, dtype=torch.long)
    for i, n in enumerate(lengths):
        tokens[i, :n] = torch.randint(0, 20, (n,))
        tokens[i, n] = cfg.eos_token_id

    print(f"\ninit: masked-NLL {_eval_oadm(model, tokens, cfg):.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, 1201):
        opt.zero_grad()
        loss, m = training_step(model, {"tokens": tokens}, sched,
                                eos_loss_weight=20.0, pad_loss_weight=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"step {step:4d} | loss {float(m['loss']):.3f} (vb {float(m['vb']):.3f} "
                  f"ce {float(m['ce']):.3f}) | corruption: {float(m['masked']):.0%} masked "
                  f"{float(m['subst']):.0%} substituted | masked-NLL {_eval_oadm(model, tokens, cfg):.3f}")

    model.eval()
    with torch.no_grad():
        probe = tokens.clone()
        for i, n in enumerate(lengths):
            probe[i, n:] = cfg.mask_token_id
        p_eos = torch.softmax(model(probe).float(), dim=-1)[..., cfg.eos_token_id]
    pred = p_eos.argmax(-1).tolist()
    mae = sum(abs(pred[i] - lengths[i]) for i in range(B)) / B
    print(f"\npredicted boundary {pred}\n    vs true lengths {lengths}  (MAE {mae:.2f} positions)")
    print("\nWhat this shows: one loss trains BOTH move types -- the corruption line reports real "
          "masking AND real substitution every step, and the masked-NLL falls, which is the "
          "unmasking half. It does NOT measure eos_loss_weight: 8 uniformly random sequences make "
          "their lengths 8 arbitrary facts to memorise rather than a rule to learn. The real EOS "
          "check is the `no-EOS` column of the [eval] line during training.")
