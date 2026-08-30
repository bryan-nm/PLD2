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

SPAN CORRUPTION (`span_width`) SHAPES THE MASK, NOT ITS AMOUNT. Two runs -- 55M/50k and 1.35B/18k --
produced generations whose pTM sat at the shuffled baseline (0.184 and 0.176 vs a shuffle's 0.169,
against 0.677 for naturals), and the held-out CE curve was nearly flat below 50% corruption: going
from 40% of the sequence visible to 95% visible bought 0.18 nats. Under i.i.d. per-position masking
that is what you would predict. At 40% corruption the expected distance from a masked position to
its nearest VISIBLE neighbour is under two residues, so local context is already saturated there and
every further reveal is long-range information the gradient never has to use. The model duly learned
a local infiller -- competent (ppl 7.6 at 5% corruption, in the range published protein MLMs report)
and unable to place a fold.

Span corruption removes the near neighbours. The mask indicator is drawn as a spatially correlated
field instead of independent coin flips, so masked positions arrive in RUNS: to fill the middle of a
64-residue hole the model has to reach past it. The amount of corruption is untouched.

`span_width` IS THE TARGET MEAN MASKED-RUN LENGTH IN RESIDUES at 50% corruption, not an opaque
kernel size -- the field is smoothed with sigma = span_width / 4.44 (the mean zero-crossing interval
of a Gaussian-smoothed Gaussian field) and run_length() reports what was actually achieved:

    span_width  |    1        8       32      128        <- requested
    measured    |  2.0      7.9     28.2     82.6        <- mean masked run at 50% corruption
    nearest visible residue from a masked position, at 50% corruption:
      mean      |  1.3      3.7     12.7     39.6
      median    |  1.0      3.0      9.0     30.0

That first column is the diagnosis restated as a number: under i.i.d. masking the median masked
position has a revealed residue DIRECTLY ADJACENT to it. 128 undershoots its target because a 512
canvas at 50% corruption cannot hold many 128-residue runs; that is a real constraint, not a bug.

`span_width` is a TUPLE assigned per row, for the same reason `betas` is. Keeping w=1 in the mix is
not conservatism: mid-decode the sampler's canvas is a SCATTERED set of confidence-committed
positions among masks, which is the i.i.d. regime, so a model trained only on wide spans would be
off-distribution exactly where it is used. The mixture covers both.

WHAT THIS COSTS, STATED PLAINLY: the exact ELBO. D3PM's q(x_{t-1}|x_t,x_0) factorises over positions
only because the forward corruption does, and a correlated field does not. Two things survive and
one does not:

  * PER-POSITION MARGINALS ARE EXACT. P(x_t = MASK | x_0) is still exactly Qbar_t[x_0, MASK] at every
    position (see _span_field: a stationary field plus a rank threshold with randomised rounding
    gives the marginal for free). So L_ce -- a per-position cross-entropy -- is unchanged, and the
    stationary distribution, the cold start, and the CE curve stay directly comparable to both
    previous runs.
  * L_vb BECOMES A MEAN-FIELD SURROGATE. We still sum per-position KLs between the true per-position
    posterior marginals and the model's factorised reverse. That is a sensible denoising objective
    and it is the same computation as before, but it is no longer a bound on -log p(x_0): the true
    joint posterior now has cross-position correlation that neither side represents. Reported vb
    numbers remain comparable ACROSS span-corruption runs and should not be read as an ELBO.
  * THE SAMPLER IS UNAFFECTED. It runs the model's reverse process, which is factorised by
    construction either way; nothing in src/sampler.py needs to know about this.

The two channels are split apart to do it, which the factorisation above makes exact: the mask event
is Bernoulli(1 - survival_t) INDEPENDENT of x_0 (the mask column is constant across rows), so it can
be drawn as a correlated field, and the residue outcome is then drawn from Qbar_t's non-MASK block
renormalised. Substitutions stay i.i.d.: a substituted position leaves a token visible, so it is not
a hole in context and it is not what the locality argument is about.
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
                 betas: Sequence[float] = (1.0,), T: int = 500,
                 span_width: Sequence[int] = (1,), device=None,
                 dtype=torch.float32):
        assert mask_id == vocab_size - 1, "MASK must be the last id (see model.Config.assert_vocab)"
        n = vocab_size - 1                                   # non-MASK states
        assert sub_kernel.shape == (n, n), f"sub_kernel must be ({n},{n})"
        betas = tuple(float(b) for b in betas)
        assert all(0.0 < b <= 1.0 for b in betas), (
            "beta must be in (0, 1]. beta=0 removes the absorbing channel entirely, which throws "
            "away the all-MASK stationary distribution and therefore the cold-start entry point -- "
            "the whole reason this process has a mask state.")
        span_width = tuple(int(w) for w in
                           (span_width if isinstance(span_width, (tuple, list)) else (span_width,)))
        assert all(w >= 1 for w in span_width), "span_width entries are run lengths in positions, >= 1"

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

        # Qbar with the absorbing column removed and renormalised: the residue outcome CONDITIONAL
        # on not having been masked. Exact by the survival*Mbar factorisation. Where survival is 0
        # (t=T) the row is all zeros; fall back to "stay" so the draw is defined -- the field masks
        # 100% of positions there, so the fallback is never actually read.
        sub = Qbar.clone()
        sub[..., mask_id] = 0.0
        deg = sub.sum(dim=-1, keepdim=True) < 1e-12
        sub = torch.where(deg, torch.eye(V, dtype=torch.float64).expand_as(sub), sub)
        sub = sub / sub.sum(dim=-1, keepdim=True)

        self.T, self.V, self.n, self.mask_id = T, V, n, mask_id
        self.betas = betas
        self.span_width = span_width
        self.n_span = len(span_width)
        self.n_beta = len(betas)
        self.sub_kernel = sub_kernel.to(dtype)
        self.c = cs.to(dtype)
        # Flattened to (n_beta*(T+1), V, V) so one gather serves a per-row (beta, t) pair.
        self.Q = Q.reshape(-1, V, V).to(dtype)
        self.QT = Q.transpose(-1, -2).reshape(-1, V, V).to(dtype)
        self.Qbar = Qbar.reshape(-1, V, V).to(dtype)
        self.Qbar_cdf = self.Qbar.cumsum(dim=-1)
        self.Qbar_sub_cdf = sub.reshape(-1, V, V).to(dtype).cumsum(dim=-1)
        if device is not None:
            self.to(device)

    # -- indexing ------------------------------------------------------------
    def flat(self, beta_idx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(beta_idx, t) -> the flat index into the (n_beta*(T+1)) stacks."""
        return beta_idx * (self.T + 1) + t

    def to(self, device):
        for name in ("sub_kernel", "c", "Q", "QT", "Qbar", "Qbar_cdf", "Qbar_sub_cdf"):
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

    def run_length(self, frac: float = 0.5, L: int = 512, n: int = 512) -> dict:
        """MEASURED mean run length of masked and of visible stretches, per span width, at a given
        mask fraction -- the achieved figure against the requested `span_width`, which is a target
        rather than a guarantee (a 512 canvas cannot hold many 128-residue runs at 50% corruption).
        Sampled rather than derived, because the relation depends on the threshold as well as the
        smoothing. Runs that wrap the circular boundary count as two; at L=512 the bias is small and
        identical across widths."""
        out = {}
        for i, w in enumerate(self.span_width):
            f = self.span_field(torch.full((n,), float(frac)), L,
                                torch.full((n,), i, dtype=torch.long)).int()
            starts = int((f[:, 1:] > f[:, :-1]).sum() + f[:, 0].sum())
            gaps = int(((1 - f)[:, 1:] > (1 - f)[:, :-1]).sum() + (1 - f)[:, 0].sum())
            out[w] = (round(float(f.sum()) / max(starts, 1), 1),
                      round(float((1 - f).sum()) / max(gaps, 1), 1))
        return out

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

    def q_sample(self, x0: torch.Tensor, flat_idx: torch.Tensor,
                 span_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x_t ~ Cat(x_0 Qbar_t), per row. x0 (B,L), flat_idx (B,) -> (B,L).

        With span_width == (1,) this is one inverse-CDF draw over the whole Qbar row and the code
        below is bit-for-bit the original path. Otherwise the row is split into its two exactly
        factorising channels -- a mask event that does not depend on x_0, and a residue outcome that
        does -- so the mask half can be drawn as a spatially correlated field while every
        per-position marginal stays exactly Qbar_t[x_0]. span_idx (B,) picks each row's span width.
        """
        if self.n_span == 1 and self.span_width[0] <= 1:
            cdf = _rows(self.Qbar_cdf, flat_idx, x0)
            r = torch.rand(x0.shape[0], x0.shape[1], 1, device=x0.device, dtype=cdf.dtype)
            return (r > cdf).sum(dim=-1).clamp_(max=self.V - 1)

        # P(masked) is the SAME for every source token (the absorbing column is constant across
        # rows -- tests_corruption checks this to 1e-16), so row 0 reads it for all of them.
        p_mask = self.Qbar[flat_idx][:, 0, self.mask_id]
        field = self.span_field(p_mask, x0.shape[1], span_idx)
        cdf = _rows(self.Qbar_sub_cdf, flat_idx, x0)
        r = torch.rand(x0.shape[0], x0.shape[1], 1, device=x0.device, dtype=cdf.dtype)
        subst = (r > cdf).sum(dim=-1).clamp_(max=self.n - 1)
        return torch.where(field, torch.full_like(x0, self.mask_id), subst)

    def span_field(self, p_mask: torch.Tensor, L: int,
                   span_idx: torch.Tensor | None = None) -> torch.Tensor:
        """(B,L) bool, True where MASKED. p_mask (B,) is each row's target mask fraction.

        EXACT MARGINALS, CLUSTERED JOINT. White noise under CIRCULAR Gaussian smoothing stays a
        stationary field: every position is identically distributed and the joint is invariant under
        cyclic shift, so each position's rank is uniform on 0..L-1 and P(rank < k) = k/L for ALL of
        them -- no edge positions quietly corrupted at a different rate. Randomised rounding of the
        threshold (floor(p*L + u), u ~ U(0,1)) makes E[k] = p*L exactly, so the per-position marginal
        is exactly p rather than p rounded to 1/L. Smoothing correlates the ranks, which is what puts
        the masked positions in runs; it cannot change the marginal. Verified per position at every
        width: max deviation 0.010 against a target of 0.300 over 20k rows, whose binomial noise
        alone is 0.011, with the first and last 8 positions matching the middle.

        Every width is convolved from the SAME base noise and then selected, rather than looping over
        the rows of each width: 4 conv1d ops on (B,1,512) is free next to the model forward, and it
        keeps shapes static (no host sync, nothing data-dependent for the XPU to recompile on).
        """
        return span_mask_field(p_mask, L, self.span_width, span_idx)

    def sample_span_idx(self, B: int, device) -> torch.Tensor:
        """Per-row span width, exactly balanced within the batch when n_span divides B.

        A random PERMUTATION mod n_span rather than arange mod n_span: beta is assigned by
        `arange(B) % n_beta`, so a second arange-based rule would lock the two together and a whole
        (beta, width) sub-grid would never be sampled. Permuting decorrelates them while keeping the
        exact balance. argsort(rand) rather than randperm because randperm is not implemented for
        every XPU build.
        """
        if self.n_span == 1:
            return torch.zeros(B, dtype=torch.long, device=device)
        return torch.argsort(torch.rand(B, device=device)) % self.n_span

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
def span_mask_field(p_mask: torch.Tensor, L: int, widths: Sequence[int],
                    span_idx: torch.Tensor | None = None) -> torch.Tensor:
    """(B,L) bool mask indicator, True where masked. See CorruptionSchedule.span_field for the
    argument that the per-position marginal is exactly p_mask. Module-level so a caller that only
    wants the field (src/ce_curve.py) need not build a 500-matrix schedule to get it."""
    B = p_mask.shape[0]
    dev, dt = p_mask.device, p_mask.dtype
    z = torch.randn(B, 1, L, device=dev, dtype=dt)
    if len(widths) == 1:
        sel = _smooth(z, widths[0]).squeeze(1)
    else:
        stack = torch.stack([_smooth(z, w).squeeze(1) for w in widths], dim=0)
        if span_idx is None:
            span_idx = torch.zeros(B, dtype=torch.long, device=dev)
        sel = stack.gather(0, span_idx.view(1, B, 1).expand(1, B, L)).squeeze(0)
    k = torch.floor(p_mask * L + torch.rand(B, device=dev, dtype=dt)).long().clamp_(0, L)
    return sel.argsort(dim=1).argsort(dim=1) < k.unsqueeze(1)


# Mean zero-crossing interval of a Gaussian-smoothed Gaussian field is pi*sqrt(2)*sigma, so this
# converts a target run length into the smoothing sigma that produces it. Measured against the
# theory: width 8 -> 7.9, width 32 -> 28.2. Large widths undershoot (128 -> 83) because a 512 canvas
# at 50% corruption cannot hold many 128-residue runs; run_length() reports the achieved figure.
_RUN_PER_SIGMA = 4.44


def _smooth(z: torch.Tensor, width: int) -> torch.Tensor:
    """(B,1,L) white noise -> (B,1,L) smoothed so its level sets have runs of mean length ~`width`.
    width <= 1 is the identity, i.e. independent coin flips.

    GAUSSIAN, NOT A BOX FILTER. A box filter is the obvious choice and it is the wrong one: its
    frequency response is a sinc, so the smoothed field keeps enough high-frequency content to
    fragment its own level sets. Measured at box width 128 on a 512 canvas at 50% corruption, the
    median masked run was 3 positions and 29% of runs were a SINGLE position -- i.i.d. masking with
    a few blobs on top, which is precisely the thing this is meant to stop doing. The Gaussian has
    no sidelobes: at the same mean run length, 1-2% of runs are single positions and the median run
    is 47.

    CIRCULAR, NOT ZERO-PADDED. Zero padding shrinks the variance near the ends, so those positions
    would be masked at a systematically different rate -- measured at 0.257 against a target of 0.300
    over the first 8 positions, which would have quietly under-corrupted every N-terminus. Circular
    wrapping keeps the field stationary, which is the property the rank threshold needs to give an
    exact marginal.
    """
    if width <= 1:
        return z
    L = z.shape[-1]
    sigma = width / _RUN_PER_SIGMA
    r = min(max(1, int(round(3 * sigma))), L // 2)
    x = torch.arange(-r, r + 1, device=z.device, dtype=z.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = (k / k.sum()).view(1, 1, -1)
    w = 2 * r + 1
    lo, hi = w // 2, w - 1 - w // 2
    return torch.nn.functional.conv1d(
        torch.cat([z[..., L - lo:], z, z[..., :hi]], dim=-1), k)


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
