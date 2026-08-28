"""PLD2 training objective: OADM and D3PM, both, in one forward pass.

THE SLIDER. `p_d3pm` is the single knob that slides between the two frameworks:

    p_d3pm = 0.0   pure OADM  -- absorbing (MASK) corruption, reweighted x0 cross-entropy
    p_d3pm = 1.0   pure D3PM  -- substitution corruption, true KL-ELBO (+ optional x0-CE)
    0 < p < 1      both, trained jointly (the PLD2 default, p_d3pm = 0.5)

It replaces ProLoopDiff's `beta`, and is strictly more than a rename. `beta` mixed the CORRUPTION
per position while always scoring with the absorbing-state surrogate loss, so the D3PM half of the
model was trained against the wrong likelihood. Here the row's mode picks the corruption AND the
loss together, which is what "train on both objectives" has to mean.

HOW BOTH FIT IN ONE FORWARD. The batch is split by ROW at a fixed index: rows [0, n_d3pm) are D3PM,
rows [n_d3pm, B) are OADM. n_d3pm = round(B * p_d3pm) is a Python int fixed for the whole run, so
every tensor here has a shape known ahead of time -- one static graph for XPU, no boolean gathers,
no host syncs. The rows arrive in random order from the sampler, so a fixed split is unbiased.

----------------------------------------------------------------------------------------------
OADM loss (unchanged from ProLoopDiff, which had it right).

  BioM3 / ARDM, per sequence of length D:
        L = D * E_{t~U(1..D)} E_{sigma} [ 1/(D-t+1) * sum_{k in sigma(>=t)} -log p(x_k | x_sigma(<t)) ]
  With n := D-t+1 the number of masked positions the inner term is the MEAN NLL over those n
  positions, and t ~ U(1..D) <=> n ~ U(1..D). The leading D is constant and dropped. So: draw
  n ~ U(1,D), mask n positions uniformly, take the mean NLL over them.

  D here is the FULL 512 canvas, because PAD is modelled. Generation cold-starts from an all-MASK
  canvas and has to resolve the PAD tail as well as the residues.

----------------------------------------------------------------------------------------------
POSITION WEIGHTS (instruction 6: upweight EOS).

  A fixed 512 canvas holds about 350 residues, 160 PAD and exactly ONE EOS. Unweighted, the single
  token that decides the sequence LENGTH carries 1/512 = 0.2% of the loss, and ProLoopDiff duly
  learned everything except where to stop: at 181k steps, 53% of unconditional samples never placed
  EOS at all. With pad_loss_weight=0.1 and eos_loss_weight=20 the split becomes roughly
  350 : 16 : 20, i.e. EOS is ~5% of the per-sequence loss -- ~25x its unweighted share, and still
  far from dominating. Both weights apply identically to the OADM and D3PM branches.
"""

from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F

from .model import LoopedDiffusionLM, Config, count_params
from .d3pm import D3PMSchedule, kl_categorical, x0_probs


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
# OADM: absorbing corruption + reweighted x0 cross-entropy
# --------------------------------------------------------------------------------------
def oadm_corrupt(x0: torch.Tensor, mask_id: int, target_mask: Optional[torch.Tensor] = None):
    """n ~ U(1, D) positions per row -> MASK. Returns (x_t, corrupt_pos).

    Every tensor has a shape fixed by (B, L): the positions are chosen by ranking a uniform random
    score, not by a boolean gather, so there is no data-dependent allocation and no host sync.
    """
    B, L = x0.shape
    device = x0.device
    if target_mask is None:
        target_mask = torch.ones_like(x0, dtype=torch.bool)
    n_targets = target_mask.sum(dim=1).clamp(min=1)
    u = torch.rand(B, device=device)
    n_corrupt = torch.minimum((u * n_targets.float()).floor().long() + 1, n_targets)  # U(1, D)
    scores = torch.rand(B, L, device=device).masked_fill(~target_mask, -1.0)
    ranks = scores.argsort(dim=1, descending=True).argsort(dim=1)
    corrupt_pos = (ranks < n_corrupt[:, None]) & target_mask
    return torch.where(corrupt_pos, x0.new_full((), mask_id), x0), corrupt_pos


def oadm_loss(logits: torch.Tensor, x0: torch.Tensor, corrupt_pos: torch.Tensor,
              weights: torch.Tensor) -> torch.Tensor:
    """Weighted mean NLL over the masked positions of each row, then mean over rows."""
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V).float(), x0.reshape(-1),
                         reduction="none").reshape(B, L)
    return _weighted_row_mean(ce, corrupt_pos.float() * weights)


# --------------------------------------------------------------------------------------
# D3PM: substitution corruption + KL-ELBO (+ optional x0 cross-entropy)
# --------------------------------------------------------------------------------------
def d3pm_loss(logits: torch.Tensor, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
              sched: D3PMSchedule, weights: torch.Tensor, ce_weight: float = 1.0):
    """L = L_vb + ce_weight * L_ce, both weighted per position. Returns (loss, vb, ce).

    L_vb is the per-position KL( q(x_{t-1}|x_t,x_0) || p_theta(x_{t-1}|x_t) ); at t=1 that is
    exactly D3PM's L_0 = -log p_theta(x_0|x_1), with no special case (Qbar_0 = I). The
    theta-independent L_T term is dropped, as in EvoDiff.

    L_ce is -log p~_theta(x_0 | x_t): EvoDiff's lambda term, which they set to 0. PLD2 defaults it
    to 1 because the D3PM branch here shares a trunk with an OADM branch whose loss is on the
    cross-entropy scale, while the KL term is numerically tiny at small t; a CE-scaled companion
    keeps the two objectives contributing comparably instead of letting OADM silently own the
    gradient. Set d3pm_ce_weight=0 to recover EvoDiff exactly.
    """
    # AUTOCAST OFF for the whole posterior computation. The trainer runs under bf16 autocast, and
    # torch.bmm is on autocast's lower-precision list -- so `p_tilde @ Qbar_{t-1}` would be done in
    # bf16 no matter that both operands are fp32 tensors. Qbar_{t-1} is near-identity for small t
    # and its off-diagonal entries are ~1e-3; bf16's 8-bit mantissa cannot hold them next to a
    # diagonal near 1, and the KL between two nearly-equal categoricals is exactly where that loss
    # of precision turns into noise. Disabling autocast here costs one 22x22 matmul per position.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        p_tilde = x0_probs(logits, sched.K)                    # (B,L,K) fp32
        q_post, p_post = sched.posteriors(x0, xt, t, p_tilde)
        vb = _weighted_row_mean(kl_categorical(q_post, p_post), weights)
        ce = _weighted_row_mean(
            -p_tilde.gather(-1, x0.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log(), weights)
    return vb + ce_weight * ce, vb.detach(), ce.detach()


# --------------------------------------------------------------------------------------
# Training step
# --------------------------------------------------------------------------------------
def split_sizes(batch_size: int, p_d3pm: float) -> tuple:
    """(n_d3pm, n_oadm) for a batch. Fixed for the run, so every shape here is static."""
    n_d3pm = int(round(batch_size * float(p_d3pm)))
    return n_d3pm, batch_size - n_d3pm


def training_step(model: LoopedDiffusionLM, batch: dict, sched: Optional[D3PMSchedule] = None,
                  p_d3pm: float = 0.5, eos_loss_weight: float = 1.0, pad_loss_weight: float = 1.0,
                  d3pm_ce_weight: float = 1.0, d3pm_vb_weight: float = 1.0,
                  oadm_weight: float = 1.0):
    """batch: {"tokens": (B, L) long}. Returns (loss, metrics) with metrics as on-device 0-d
    tensors -- converting them here would force a device->host sync every step for numbers the
    trainer only prints every log_every steps."""
    cfg = model.cfg
    x0 = batch["tokens"]
    canvas_mask = batch.get("canvas_mask")              # None -> the whole 512 canvas is modelled
    B, L = x0.shape

    n_d, n_o = split_sizes(B, p_d3pm)
    if n_d and sched is None:
        raise ValueError("p_d3pm > 0 needs a D3PMSchedule")

    w = position_weights(x0, cfg.eos_token_id, cfg.pad_token_id, eos_loss_weight, pad_loss_weight)

    parts, t_d, cpos_o = [], None, None
    if n_d:
        t_d = sched.sample_t(n_d, x0.device)
        parts.append(sched.q_sample(x0[:n_d], t_d))
    if n_o:
        tm = canvas_mask[n_d:] if canvas_mask is not None else None
        x_o, cpos_o = oadm_corrupt(x0[n_d:], cfg.mask_token_id, tm)
        parts.append(x_o)
    corrupted = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]

    logits = model(corrupted, canvas_mask=canvas_mask)

    zero = x0.new_zeros((), dtype=torch.float32)
    loss = zero
    m = {"d3pm_vb": zero, "d3pm_ce": zero, "oadm": zero}
    if n_d:
        l_d, vb, ce = d3pm_loss(logits[:n_d], x0[:n_d], corrupted[:n_d], t_d, sched, w[:n_d],
                                ce_weight=d3pm_ce_weight)
        loss = loss + d3pm_vb_weight * l_d
        m["d3pm_vb"], m["d3pm_ce"] = vb, ce
    if n_o:
        l_o = oadm_loss(logits[n_d:], x0[n_d:], cpos_o, w[n_d:])
        loss = loss + oadm_weight * l_o
        m["oadm"] = l_o.detach()
    m["loss"] = loss.detach()
    return loss, m


# --------------------------------------------------------------------------------------
# Overfit demo: both objectives on a tiny fixed batch, on CPU.
# --------------------------------------------------------------------------------------
def _eval_oadm(model, tokens, cfg, frac=0.5):
    """Deterministically mask each row's C-terminal `frac` and report the OADM loss. Low-variance
    progress signal -- unlike the training loss it does not average over corruption levels."""
    mask_pos = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(tokens.shape[0]):
        k = max(1, int(tokens.shape[1] * frac))
        mask_pos[i, -k:] = True
    corrupted = torch.where(mask_pos, torch.full_like(tokens, cfg.mask_token_id), tokens)
    model.eval()
    with torch.no_grad():
        lg = model(corrupted)
        out = oadm_loss(lg, tokens, mask_pos, torch.ones_like(tokens, dtype=torch.float32)).item()
    model.train()
    return out


if __name__ == "__main__":
    from .blosum import uniform_transition
    torch.manual_seed(0)

    cfg = Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
                 d_model=128, n_heads=4, d_ff=384,
                 n_upstream=2, n_middle=4, n_downstream=2, n_recurrence=2)
    model = LoopedDiffusionLM(cfg)
    sched = D3PMSchedule(uniform_transition(cfg.d3pm_vocab), T=100)
    print(f"demo params={count_params(model)/1e6:.2f}M | D3PM K={sched.K} T={sched.T} "
          f"| TV(Qbar_T, uniform)={sched.stationary_tv():.2e} "
          f"| unchanged-fraction {sched.corruption_curve((1, 25, 50, 100))}")

    B, L = 8, 24
    lengths = [18, 15, 12, 9, 20, 7, 16, 11]
    tokens = torch.full((B, L), cfg.pad_token_id, dtype=torch.long)
    for i, n in enumerate(lengths):
        tokens[i, :n] = torch.randint(0, 20, (n,))
        tokens[i, n] = cfg.eos_token_id
    batch = {"tokens": tokens}

    print(f"init: eval oadm={_eval_oadm(model, tokens, cfg):.3f}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for step in range(1, 1501):
        opt.zero_grad()
        loss, m = training_step(model, batch, sched, p_d3pm=0.5,
                                eos_loss_weight=20.0, pad_loss_weight=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            print(f"step {step:3d} | loss {float(m['loss']):.3f} oadm {float(m['oadm']):.3f} "
                  f"d3pm_vb {float(m['d3pm_vb']):.4f} d3pm_ce {float(m['d3pm_ce']):.3f} "
                  f"| eval oadm {_eval_oadm(model, tokens, cfg):.3f}")

    # Where does the model think the boundary is? Reveal the residues, mask from the boundary on,
    # and take argmax_i P(EOS at i) -- the exact quantity sampler._place_eos_first draws from.
    model.eval()
    with torch.no_grad():
        probe = tokens.clone()
        for i, n in enumerate(lengths):
            probe[i, n:] = cfg.mask_token_id
        p_eos = torch.softmax(model(probe).float(), dim=-1)[..., cfg.eos_token_id]
    pred = p_eos.argmax(-1).tolist()
    mae = sum(abs(pred[i] - lengths[i]) for i in range(B)) / B
    print(f"predicted boundary {pred}\n    vs true lengths {lengths}  (MAE {mae:.2f} positions)")
    print("\nWhat this demo does and does not show. It SHOWS that both branches are wired up and "
          "optimise: oadm, d3pm_vb and d3pm_ce all fall, and the eval OADM loss falls with them. "
          "It does NOT measure eos_loss_weight -- these are 8 uniformly random sequences, so their "
          "lengths are 8 arbitrary facts to memorise rather than a rule to learn, and at 1.7M "
          "parameters the boundary MAE is dominated by that. The real EOS check is the `no-EOS` "
          "column of the [eval] line during training, which should collapse toward zero; the case "
          "for the weight itself is the loss-share arithmetic in config.OptCfg.")
