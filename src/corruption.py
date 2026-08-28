"""The PLD2 corruption process: ONE diffusion with MASK as an absorbing state.

Replaces the earlier two-process design (a pure-OADM branch and a pure-substitution D3PM branch on
disjoint rows). That design reproduced EvoDiff's D3PM exactly -- including the property that makes
it lose: its transition matrix was doubly stochastic over an alphabet WITHOUT a mask state, so its
stationary distribution was uniform over amino acids and generation would have to cold-start from
uniform-random residues. No "this position is definitely unknown" signal, no implicit known-count
signal, and no shared entry point with the OADM branch.

Here there is one process. Per step, a non-MASK position either

    stays          with probability 1 - c_t
    -> MASK        with probability beta * c_t          (absorbing)
    -> substitute  with probability (1 - beta) * c_t    (BLOSUM-weighted, zero diagonal)

and MASK is absorbing: Q_t[MASK, MASK] = 1. So beta is literally "of the corruption events that
happen, what fraction are maskings", which is what makes it the OADM <-> substitution slider:

    beta = 1     no substitution channel at all -> the corruption IS OADM's
    beta -> 0    substitution dominates, but masking still proceeds on its own schedule

Because MASK is absorbing and reachable from every state at any beta > 0, the stationary
distribution is all-MASK for the WHOLE family. That is the point:

  * generation always cold-starts from a fully-masked canvas (OADM's clean entry point), at every
    beta -- so beta does not change what the sampler starts from;
  * the "how many positions are known" signal that makes OADM scale is preserved;
  * L_T is genuinely ~0 rather than dropped on credit (measured below via mask_fraction);
  * and the substitution channel is enriched into the SAME trajectories, so the model learns
    token->token repair as part of one objective rather than as a bolt-on.

THE SCHEDULE IS CLOSED-FORM, NOT CALIBRATED. The absorbing channel's cumulative mask fraction is
pinned to the linear target t/T, which is what makes beta=1 exactly the absorbing-state form of
OADM's n ~ U(1, D) masking. Survival through step t is prod(1 - beta*c_s), so setting
beta*c_t = 1/(T - t + 1) gives prod = (T-t)/T, i.e. mask fraction exactly t/T. Hence

    c_t = min(1, 1 / (beta * (T - t + 1)))

The clamp only binds over the last ceil(1/beta) steps, where the survival probability is already
~1e-3, so the mask fraction still reaches ~1 at T (mask_fraction() reports the achieved curve).
The previous file's numerical Sinkhorn/bisection calibration is gone with the uniform-stationary
requirement that motivated it.

THE TWO CHANNELS FACTOR, which is worth knowing because it makes the whole thing verifiable. Write
the non-MASK block of Q_t as P_t; then P_t = (1 - beta*c_t) * M_t with M_t row-stochastic, so

    Qbar_t[i, MASK] = 1 - survival_t          (the same for every i)
    Qbar_t[i, j]    = survival_t * Mbar_t[i, j]

Everything below builds Qbar by honest matrix products and the self-test checks it against that
factorisation.
"""
from __future__ import annotations
from typing import Sequence

import torch


class CorruptionSchedule:
    """Precomputed Q_t and Qbar_t over the FULL vocabulary, MASK included and absorbing.

    Holds one chain per beta in `betas`, so a training step can draw a different corruption process
    per row (see objective.training_step) without any dynamic shapes. Stacks are
    (n_beta, T+1, V, V) with t=0 the identity; at n_beta=4, T=500, V=23 that is ~4.2MB each.
    """

    def __init__(self, sub_kernel: torch.Tensor, vocab_size: int, mask_id: int,
                 betas: Sequence[float] = (1.0,), T: int = 500, device=None,
                 dtype=torch.float32):
        assert mask_id == vocab_size - 1, "MASK must be the last id (see model.Config.assert_vocab)"
        n = vocab_size - 1                                   # non-MASK states
        assert sub_kernel.shape == (n, n), f"sub_kernel must be ({n},{n})"
        betas = tuple(float(b) for b in betas)
        assert all(0.0 < b <= 1.0 for b in betas), (
            "beta must be in (0, 1]. beta=0 removes the absorbing channel entirely, which throws "
            "away the all-MASK stationary distribution and therefore the cold-start entry point -- "
            "the whole reason this process has a mask state.")

        V = vocab_size
        B = sub_kernel.double()
        eyeN = torch.eye(n, dtype=torch.float64)

        Q = torch.zeros(len(betas), T + 1, V, V, dtype=torch.float64)
        Qbar = torch.zeros(len(betas), T + 1, V, V, dtype=torch.float64)
        cs = torch.zeros(len(betas), T + 1, dtype=torch.float64)
        for bi, beta in enumerate(betas):
            Q[bi, 0] = torch.eye(V, dtype=torch.float64)
            Qbar[bi, 0] = torch.eye(V, dtype=torch.float64)
            for t in range(1, T + 1):
                c = min(1.0, 1.0 / (beta * (T - t + 1)))     # total corruption rate this step
                cs[bi, t] = c
                P = (1.0 - c) * eyeN + (1.0 - beta) * c * B  # non-MASK block; row sums 1 - beta*c
                Q[bi, t, :n, :n] = P
                Q[bi, t, :n, mask_id] = beta * c             # the absorbing move
                Q[bi, t, mask_id, mask_id] = 1.0             # MASK never leaves
                Qbar[bi, t] = Qbar[bi, t - 1] @ Q[bi, t]

        self.T, self.V, self.n, self.mask_id = T, V, n, mask_id
        self.betas = betas
        self.n_beta = len(betas)
        self.sub_kernel = sub_kernel.to(dtype)
        self.c = cs.to(dtype)
        # Flattened to (n_beta*(T+1), V, V) so one gather serves a per-row (beta, t) pair.
        self.Q = Q.reshape(-1, V, V).to(dtype)
        self.QT = Q.transpose(-1, -2).reshape(-1, V, V).to(dtype)
        self.Qbar = Qbar.reshape(-1, V, V).to(dtype)
        self.Qbar_cdf = self.Qbar.cumsum(dim=-1)
        if device is not None:
            self.to(device)

    # -- indexing ------------------------------------------------------------
    def flat(self, beta_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(beta_idx, t) -> the flat index into the (n_beta*(T+1)) stacks."""
        return beta_idx * (self.T + 1) + t

    def to(self, device):
        for name in ("sub_kernel", "c", "Q", "QT", "Qbar", "Qbar_cdf"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def device(self):
        return self.Q.device

    # -- diagnostics ---------------------------------------------------------
    def mask_fraction(self, beta_idx: int = 0, steps=(0, 100, 250, 400, 500)):
        """Achieved P(x_t = MASK | x_0 = a residue) at each t. Should track t/T."""
        return {int(t): round(float(self.Qbar[self.flat(torch.tensor(beta_idx),
                                                        torch.tensor(min(int(t), self.T)))][0,
                                                                                            self.mask_id]), 4)
                for t in steps if int(t) <= self.T}

    def terminal_mask_fraction(self, beta_idx: int = 0) -> float:
        """P(x_T = MASK). This is L_T's whole story: at 1.0 the prior is exactly the all-MASK
        canvas the sampler starts from, so dropping L_T costs nothing."""
        i = self.flat(torch.tensor(beta_idx), torch.tensor(self.T))
        return float(self.Qbar[i][:self.n, self.mask_id].min())

    def substitution_fraction(self, beta_idx: int = 0) -> float:
        """Of the positions still UNMASKED at t=T/2, what fraction have been substituted away from
        x_0? The substitution channel's actual footprint, which beta only indirectly controls."""
        i = self.flat(torch.tensor(beta_idx), torch.tensor(self.T // 2))
        row = self.Qbar[i][:self.n, :self.n]                 # survival * Mbar
        surv = row.sum(dim=-1)
        return float(1.0 - (row.diagonal() / surv.clamp_min(1e-12)).mean())

    # -- forward process -----------------------------------------------------
    def sample_t(self, B: int, device) -> torch.Tensor:
        return torch.randint(1, self.T + 1, (B,), device=device)

    def q_sample(self, x0: torch.Tensor, flat_idx: torch.Tensor) -> torch.Tensor:
        """x_t ~ Cat(x_0 Qbar_t), per row. x0 (B,L), flat_idx (B,) -> (B,L)."""
        cdf = _rows(self.Qbar_cdf, flat_idx, x0)
        r = torch.rand(x0.shape[0], x0.shape[1], 1, device=x0.device, dtype=cdf.dtype)
        return (r > cdf).sum(dim=-1).clamp_(max=self.V - 1)

    # -- reverse process -----------------------------------------------------
    def posteriors(self, x0, xt, flat_idx, flat_idx_prev, p_tilde):
        """-> (q_post, p_post), both (B,L,V) normalised.

        The standard D3PM identities, which hold unchanged for an absorbing Q:
            q(x_{t-1}|x_t,x_0) prop (x_t Q_t^T) * (x_0 Qbar_{t-1})
            p_theta(x_{t-1}|x_t) prop (x_t Q_t^T) * (p~ Qbar_{t-1})
        and they behave correctly at the absorbing state for free: if x_t != MASK then
        Q_t[MASK, x_t] = 0, so x_{t-1} cannot have been MASK -- an unmasking is irreversible in the
        forward direction, exactly as it should be.

        p_tilde is over the n = V-1 NON-MASK states, because x_0 is never MASK. Contracting it
        against Qbar[:n] rather than the full matrix is what keeps that true.
        """
        a = _rows(self.QT, flat_idx, xt)                     # a[j] = Q_t[j, x_t]
        b = _rows(self.Qbar, flat_idx_prev, x0)              # b[j] = Qbar_{t-1}[x_0, j]
        c = torch.bmm(p_tilde, self.Qbar[flat_idx_prev][:, :self.n, :])
        return _normalize(a * b), _normalize(a * c)

    def p_reverse(self, xt: torch.Tensor, flat_i: int, prev_i: int,
                  p_tilde: torch.Tensor) -> torch.Tensor:
        """p_theta(x_{t-1}|x_t) for a SCALAR (beta, t) shared by the batch -- the sampler's path,
        where the (V,V) matrices index once and the bmm collapses to a broadcast matmul."""
        a = self.QT[flat_i][xt]
        c = p_tilde @ self.Qbar[prev_i][:self.n, :]
        return _normalize(a * c)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rows(stack: torch.Tensor, idx: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
    """stack (N,V,V); idx (B,); tok (B,L) -> (B,L,V) with out[b,l] = stack[idx[b]][tok[b,l]]."""
    M = stack[idx]
    return M.gather(1, tok.unsqueeze(-1).expand(-1, -1, M.shape[-1]))


def _normalize(p: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)


def kl_categorical(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(q || p) over the last dim -> (B, L)."""
    q = q.clamp_min(eps)
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw over the last dim. No multinomial: static shapes, no host sync, and it
    cannot return an out-of-range index on a float-rounding edge."""
    cdf = probs.cumsum(dim=-1)
    r = torch.rand(probs.shape[:-1] + (1,), device=probs.device, dtype=probs.dtype)
    return (r > cdf).sum(dim=-1).clamp_(max=probs.shape[-1] - 1)


def x0_probs(logits: torch.Tensor, n: int) -> torch.Tensor:
    """p~_theta(x_0 | x_t) over the n = V-1 non-MASK states. x_0 is never MASK, so dropping that
    logit rather than letting it absorb probability mass is what keeps p_tilde a distribution over
    the states x_0 can actually take. fp32: the posterior multiplies three near-degenerate factors."""
    return torch.softmax(logits[..., :n].float(), dim=-1)
