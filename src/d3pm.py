"""D3PM: discrete (substitution) corruption and its true KL-ELBO loss.

ProLoopDiff only ever had the OADM surrogate: a mean cross-entropy over corrupted positions, applied
whether the corruption was absorbing or substitutive. That is exact for absorbing corruption and a
stand-in otherwise -- its own README listed "tight D3PM KL-ELBO" as the missing piece. This module
is that piece, so PLD2 can train on BOTH objectives (see objective.training_step).

Following Austin et al. (D3PM) and EvoDiff's Methods:

  forward     q(x_t | x_{t-1}) = Cat(x_t ; p = x_{t-1} Q_t)
              Q_t = (1 - b_t) I + b_t B     (b_t from calibrate_betas, below)
              Qbar_t = Q_1 Q_2 ... Q_t, so x_t is drawn in one shot from Cat(x_0 Qbar_t).
  posterior   q(x_{t-1} | x_t, x_0)  = norm[ (x_t Q_t^T)  *  (x_0 Qbar_{t-1}) ]
  model       p(x_{t-1} | x_t)       = norm[ (x_t Q_t^T)  *  (p~ Qbar_{t-1}) ]
              where p~ = p~_theta(x~_0 | x_t) is the network's x0 prediction (element-wise *).
  loss        L_vb term at t = KL( q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t) )

B is the doubly-stochastic base matrix from blosum.py (uniform or BLOSUM-derived). Doubly stochastic
makes uniform the stationary distribution, which is what lets L_T be dropped as theta-independent.

THE BETA SCHEDULE IS CALIBRATED, NOT COPIED. Sohl-Dickstein's b_t = 1/(T-t+1) makes corruption
exactly linear for a UNIFORM base, and EvoDiff-D3PM-Uniform uses it as-is. It does NOT transfer to a
BLOSUM base: BLOSUM62 is strongly self-preferring (mean diagonal 0.79 after Sinkhorn at temp=1, with
sub-dominant eigenvalues at 0.994), so borrowing that schedule leaves 44% of tokens UNCHANGED at
t=T -- nowhere near the uniform stationary distribution the ELBO assumes when it drops L_T. EvoDiff
notes the same problem and answers it with "an empirical schedule that corrupts half the sequence
content by half of T ... chosen to approximate the linear rate of mutations observed over 500
timesteps in the uniform transition matrix case".

calibrate_betas() does that numerically for whatever base matrix it is given: it solves for the b_t
that puts mean diag(Qbar_t) exactly on the uniform case's linear curve, one step at a time. For a
uniform base it reproduces 1/(T-t+1) to 1e-6 (verified), so there is one code path and no separate
"uniform" special case. What it cannot fix is a base that mixes too slowly to reach uniform at all
within T; stationary_tv() measures the residual and D3PMSchedule warns when it is large. Measured
at T=500, K=22: blosum_temp 1.0 -> TV 0.56 (bad), 2.0 -> 0.071, 3.0 -> 0.019, 4.0 -> 0.007, with the
biochemical ordering (A->S, I->V, K->R, W->Y) preserved at every temperature -- which is why
config defaults d3pm_blosum_temp to 3.0. Sharpness of the base and speed of the chain are separate
knobs precisely because the calibration owns the corruption RATE.

Two implementation notes worth keeping:

  * t=1 NEEDS NO SPECIAL CASE. Qbar_0 = I, so the posterior collapses to onehot(x_0) and the KL
    above becomes exactly -log p_theta(x_0 | x_1) -- D3PM's L_0 term. EvoDiff describes it as a
    separate branch; it falls out of the same formula, so there is one code path and no chance of
    the two drifting apart.

  * THE D3PM STATE SPACE EXCLUDES MASK. A substitution process has no absorbing state, so K =
    vocab_size - 1 and the model's x0 prediction is renormalised over logits[..., :K]. EOS and PAD
    ARE in the state space: with a fixed 512 canvas, a D3PM row has its EOS and its PAD tail
    corrupted like any other token, so this branch trains the model to REPAIR a misplaced boundary.
    That is directly on the failure ProLoopDiff had (53% of samples never placed EOS at all).

Everything is allocation-stable and free of host syncs: no torch.multinomial (inverse-CDF instead),
no boolean gathers, no data-dependent shapes. That matters on XPU.
"""
from __future__ import annotations

import torch


def uniform_unchanged_curve(T: int, K: int) -> torch.Tensor:
    """Mean diag(Qbar_t) for a UNIFORM base under Sohl-Dickstein's b_t = 1/(T-t+1). -> (T+1,)

    For that base the product telescopes: Qbar_t = a_t I + (1 - a_t) 11^T/K with a_t = (T-t)/T, so
    the expected fraction of tokens still holding their original value falls linearly from 1 to 1/K.
    This curve is the calibration TARGET for every other base -- it is what "corrupt at the uniform
    case's linear rate" means quantitatively.
    """
    a = (T - torch.arange(0, T + 1, dtype=torch.float64)) / T
    return a + (1.0 - a) / K


def calibrate_betas(base: torch.Tensor, T: int, iters: int = 60):
    """-> (betas (T+1,), Q (T+1,K,K), Qbar (T+1,K,K)) in float64, index 0 = identity.

    Solves, for each t in turn, the b_t that lands mean diag(Qbar_{t-1} Q_t) on the target curve.
    mean diag is monotonically decreasing in b_t, so a plain bisection on [0, 1] is exact and
    converges in `iters` halvings. When even b_t = 1 cannot corrupt fast enough the bisection
    saturates at 1, which is the right degradation: the chain then simply applies the base matrix as
    hard as it can, and stationary_tv() reports how far short it ended up.

    Cost is T * iters tiny (K x K) matmuls -- well under a second at T=500, K=22, and done once.
    """
    K = base.shape[0]
    eye = torch.eye(K, dtype=torch.float64)
    target = uniform_unchanged_curve(T, K)
    betas = torch.zeros(T + 1, dtype=torch.float64)
    Q = torch.empty(T + 1, K, K, dtype=torch.float64)
    Qbar = torch.empty(T + 1, K, K, dtype=torch.float64)
    Q[0], Qbar[0] = eye, eye
    for t in range(1, T + 1):
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            d = float((Qbar[t - 1] @ ((1.0 - mid) * eye + mid * base)).diagonal().mean())
            if d > float(target[t]):                   # too little corruption -> need a bigger beta
                lo = mid
            else:
                hi = mid
        betas[t] = hi
        Q[t] = (1.0 - hi) * eye + hi * base
        Qbar[t] = Qbar[t - 1] @ Q[t]
    return betas, Q, Qbar


class D3PMSchedule:
    """Precomputed Q_t, Q_t^T and Qbar_t for t = 0..T over a K-token alphabet.

    Stacks are (T+1, K, K) with index 0 = identity (Qbar_0 = I is used by the posterior at t=1).
    At the defaults (T=500, K=22) the whole thing is 3 * 501 * 22 * 22 * 4B ~ 2.9MB on device.
    """

    def __init__(self, base: torch.Tensor, T: int = 500, device=None, dtype=torch.float32,
                 warn_tv: float = 0.15):
        assert base.dim() == 2 and base.shape[0] == base.shape[1]
        K = base.shape[0]
        # Built in float64: Qbar_t is a product of T near-identity matrices, and in float32 the
        # accumulated error is comparable to the small off-diagonal entries the posterior divides by.
        b64 = base.double()
        eye = torch.eye(K, dtype=torch.float64)
        betas, Q64, Qbar64 = calibrate_betas(b64, T)

        Q = Q64.to(dtype)
        Qbar = Qbar64.to(dtype)
        self.T, self.K = T, K
        self.betas = betas.to(dtype)
        self.base = base.to(dtype)
        self.Q = Q
        self.QT = Q.transpose(1, 2).contiguous()       # QT[t][i] = column i of Q_t
        self.Qbar = Qbar
        self.Qbar_cdf = Qbar.cumsum(dim=-1)            # inverse-CDF sampling of q(x_t | x_0)
        tv = self.stationary_tv()
        if tv > warn_tv:
            print(f"[d3pm] WARNING: TV(Qbar_T, uniform) = {tv:.3f}. The corruption has NOT reached "
                  f"its stationary distribution by t={T}, so the ELBO's dropped L_T term is not "
                  f"negligible and the reverse process starts from a prior it was never trained "
                  f"against. Raise T, or flatten the base matrix (raise d3pm_blosum_temp).",
                  flush=True)
        if device is not None:
            self.to(device)

    def to(self, device):
        for name in ("base", "betas", "Q", "QT", "Qbar", "Qbar_cdf"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def device(self):
        return self.Q.device

    # -- diagnostics ---------------------------------------------------------
    def stationary_tv(self) -> float:
        """Max total-variation distance between a row of Qbar_T and the uniform distribution.

        The ELBO drops L_T on the grounds that q(x_T|x_0) has forgotten x_0. If this is not small,
        that assumption is being taken on credit -- raise T or use a more diffuse base matrix.
        """
        u = 1.0 / self.K
        return float(0.5 * (self.Qbar[self.T] - u).abs().sum(dim=-1).max())

    def corruption_curve(self, steps=(1, 50, 100, 250, 500)):
        """Expected fraction of tokens left UNCHANGED at each t (the diagonal of Qbar_t)."""
        return {int(t): float(self.Qbar[min(int(t), self.T)].diagonal().mean())
                for t in steps if int(t) <= self.T}

    # -- forward process -----------------------------------------------------
    def sample_t(self, B: int, device, generator=None) -> torch.Tensor:
        """t ~ U(1, ..., T), one per row."""
        return torch.randint(1, self.T + 1, (B,), device=device, generator=generator)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Draw x_t ~ Cat(x_0 Qbar_t) for every position. x0: (B,L) long, t: (B,) long -> (B,L).

        Inverse-CDF rather than torch.multinomial: multinomial makes the shapes data-dependent and
        can return an out-of-range index on a float-rounding edge. Here the draw is
        #{j : cdf[x0, j] <= r}, which is exact and cannot leave [0, K).
        """
        cdf = _rows(self.Qbar_cdf, t, x0)                       # (B,L,K)
        r = torch.rand(x0.shape[0], x0.shape[1], 1, device=x0.device, dtype=cdf.dtype)
        return (r > cdf).sum(dim=-1).clamp_(max=self.K - 1)

    # -- reverse process -----------------------------------------------------
    def p_reverse(self, xt: torch.Tensor, t: int, p_tilde: torch.Tensor) -> torch.Tensor:
        """p_theta(x_{t-1} | x_t) for a SCALAR t shared by the whole batch. -> (B, L, K)

        The training path (`posteriors`, below) takes a per-row t and pays for a bmm against a
        (B,K,K) gather. The SAMPLER path is different: every row of the canvas is at the same point
        in the same annealing schedule, so the matrices index once as (K,K) and the bmm collapses
        into a broadcast matmul -- 64x512x22x22 ~ 16 MFLOPs, i.e. free next to the model forward
        that produced p_tilde.

        Only the model half of the pair is computable here. q(x_{t-1}|x_t,x_0) needs x_0, which at
        sampling time is exactly what we do not have; that asymmetry is why this returns one
        distribution and `posteriors` returns two.
        """
        a = self.QT[t][xt]                                  # (B,L,K)  a[j] = Q_t[j, x_t]
        c = p_tilde @ self.Qbar[t - 1]                      # (B,L,K)  c[j] = sum_i p~[i] Qbar[i,j]
        return _normalize(a * c)

    def posteriors(self, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
                   p_tilde: torch.Tensor):
        """-> (q_post, p_post), both (B, L, K) and normalised.

        q_post = q(x_{t-1} | x_t, x_0)      -- the true posterior, the KL target
        p_post = p_theta(x_{t-1} | x_t)     -- the model's, built from its x0 prediction p_tilde
        """
        a = _rows(self.QT, t, xt)                               # a[j] = Q_t[j, x_t]
        b = _rows(self.Qbar, t - 1, x0)                         # b[j] = Qbar_{t-1}[x_0, j]
        c = torch.bmm(p_tilde, self.Qbar[t - 1])                # c[j] = sum_i p~[i] Qbar_{t-1}[i,j]
        return _normalize(a * b), _normalize(a * c)


def _rows(stack: torch.Tensor, t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """stack: (T+1,K,K); t: (B,); idx: (B,L) -> (B,L,K) where out[b,l] = stack[t[b]][idx[b,l]]."""
    M = stack[t]                                                # (B,K,K)
    K = M.shape[-1]
    return M.gather(1, idx.unsqueeze(-1).expand(-1, -1, K))


def _normalize(p: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    return p / p.sum(dim=-1, keepdim=True).clamp_min(eps)


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw over the last dim -> (..., ) long.

    Same reasoning as q_sample: no torch.multinomial, which needs a 2-D reshape, makes the shape
    data-dependent, and can return an out-of-range index on a float-rounding edge. This is exact,
    allocation-stable, and cannot leave [0, K).
    """
    cdf = probs.cumsum(dim=-1)
    r = torch.rand(probs.shape[:-1] + (1,), device=probs.device, dtype=probs.dtype)
    return (r > cdf).sum(dim=-1).clamp_(max=probs.shape[-1] - 1)


def kl_categorical(q: torch.Tensor, p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """KL(q || p) over the last dim, elementwise-stable. -> (B, L)."""
    q = q.clamp_min(eps)
    p = p.clamp_min(eps)
    return (q * (q.log() - p.log())).sum(dim=-1)


def x0_probs(logits: torch.Tensor, K: int) -> torch.Tensor:
    """p~_theta(x~_0 | x_t) over the D3PM alphabet: softmax over the first K logits.

    Dropping the MASK logit (the last one) rather than letting it absorb probability mass is what
    keeps p_tilde a distribution over the actual D3PM state space. Computed in fp32 -- the
    posterior arithmetic multiplies three near-degenerate factors and bf16 loses the small entries.
    """
    return torch.softmax(logits[..., :K].float(), dim=-1)
