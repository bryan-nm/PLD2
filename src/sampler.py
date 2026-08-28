"""Unconditional confidence-ordered (MaskGIT-style) decoder.

Start from an all-MASK canvas of width Lmax; each step score every masked position, commit the most
confident few on a cosine schedule, leave the rest for later. Better than EvoDiff's random unmask
order, and it is what makes EOS-based length work: the boundary is committed when the model is
confident about it, not at a fixed position.

Carried over from ProLoopDiff (all of it earned, none of it implicated in the repetition problem):

  * EOS FIRST. `_place_eos_first` samples the boundary from P(EOS at i) over the all-MASK canvas --
    the model's marginal length prior -- and commits it before any residue. Decoding forbids
    emitting PAD, so without this a row that fails to place EOS is forced to invent residues across
    a region that was PAD throughout training, where it has no signal and falls back on copying its
    neighbours. That is where the repetitive tails came from. Deciding the boundary first makes that
    region legitimately PAD.

  * REPETITION PENALTY over periods 1..5 plus a hard max_run cap, in BOTH directions (decoding is
    any-order, so a tract can grow rightward or leftward). It needs n_steps ~= Lmax to bite: the
    penalty scores each position against the canvas BEFORE that step's commits, so co-committed
    positions cannot see one another. Measured on a 128 canvas at max_run=5, the longest homopolymer
    was 42 at 8 commits/step, 26 at 2/step and exactly the cap at 1/step.

  * CORRECTORS. `remask` works with any model; `substitution` resamples low-confidence residues
    directly and pairs with the D3PM half of PLD2's objective, which is what teaches token->token
    denoising. Because EOS is an allowed emission there, a substitution sweep can also MOVE the
    boundary.

Differences from ProLoopDiff: no text, no CFG, no guidance-weight plumbing -- PLD2 is unconditional.
And every per-row Python loop is gone. Each `for b in range(B)` cost a device->host sync, of which a
512-step decode ran thousands; the vectorised rank-threshold form below commits exactly the same
positions with no syncs at all.

`guidance_fn(canvas, logits) -> logits` is untouched and is the ProteinGuide hook: a property model
can reweight the per-position categorical without this file knowing anything about it.
"""

from __future__ import annotations
import math
from typing import Callable, Optional

import torch

from .d3pm import D3PMSchedule, sample_categorical, x0_probs
from .model import LoopedDiffusionLM


# --------------------------------------------------------------------------------------
# Per-step logits
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _step_logits(model, canvas, guidance_fn, ban_eos: bool = False, eos_min_pos: int = 0):
    lg = model(canvas)
    if guidance_fn is not None:
        lg = guidance_fn(canvas, lg)
    cfg = model.cfg
    lg = lg.clone()
    lg[..., cfg.mask_token_id] = float("-inf")      # never emit MASK...
    lg[..., cfg.pad_token_id] = float("-inf")       # ...or PAD (it arrives only via EOS enforcement)
    if ban_eos:
        # The boundary is already committed. A second EOS to its LEFT would silently shorten the
        # sequence, since _enforce_eos honours the leftmost one.
        lg[..., cfg.eos_token_id] = float("-inf")
    elif eos_min_pos > 0:
        # No EOS below the corpus floor: an EOS at position 0 yields the empty sequence, which is
        # where len=0 samples come from -- not from a short prediction.
        lg[:, :eos_min_pos, cfg.eos_token_id] = float("-inf")
    return lg


_WARNED_STEPS = False


def _warn_if_too_few_steps(n_steps, Lmax):
    global _WARNED_STEPS
    if not _WARNED_STEPS and n_steps < Lmax // 2:
        _WARNED_STEPS = True
        print(f"[sampler] WARNING: n_steps={n_steps} on a {Lmax}-wide canvas commits "
              f"~{Lmax / max(n_steps, 1):.0f} positions per step. The repetition penalty scores "
              f"each position against the canvas BEFORE that batch commits, so co-committed "
              f"positions are invisible to one another and the penalty is largely inert. "
              f"Use n_steps ~= {Lmax}.", flush=True)


# --------------------------------------------------------------------------------------
# Repetition penalty (guidance hook)
# --------------------------------------------------------------------------------------
def make_repetition_penalty(cfg, penalty: float = 1.5, periods=(1, 2, 3, 4, 5),
                            max_run: int = 5, ban: float = 1e4):
    """Suppress the periodic degeneration that confidence-ordered decoding invites.

    Confidence ordering commits the most PREDICTABLE positions first, and a repeat is maximally
    predictable -- each unit makes the next likelier, so a tract that starts by accident is then
    preferentially extended. This breaks that loop at the logit level.

    Two terms, both reading only COMMITTED residues (MASK/PAD/EOS are never repeat evidence):
      soft -- at position i, subtract `penalty` from the logit of whatever residue sits at i+-p, for
        each period p. Contributions accumulate over p, so a homopolymer (which matches at every
        period) is suppressed hardest and an isolated coincidence barely at all.
      hard -- if placing a residue at i would PRODUCE a run longer than max_run, counting neighbours
        on both sides plus itself, that residue is effectively banned. Checking one side only is not
        enough: a gap between a run of 3 and a run of 2 sees five-in-a-row on neither side, yet
        filling it yields six. A large finite subtraction, not -inf, so no NaN can reach the softmax.

    Deliberately a REPETITION penalty and NOT the SEG windowed-entropy statistic metrics.py reports,
    nor the k-mer statistic. Guiding on the metric you evaluate with would stop that metric being
    diagnostic; these must stay independent mechanisms.
    """
    n_aa = 20                      # ids 0..19 are residues; specials are not repeat evidence
    periods = tuple(p for p in periods if p > 0)

    def fn(canvas, logits):
        B, L, V = logits.shape
        pen = torch.zeros_like(logits)

        def add(ref, weight):
            valid = (ref >= 0) & (ref < n_aa)
            pen.scatter_add_(2, ref.clamp(min=0).unsqueeze(-1),
                             (valid.to(logits.dtype) * weight).unsqueeze(-1))

        for p in periods:
            if p >= L:
                continue
            back = canvas.new_full((B, L), -1)
            back[:, p:] = canvas[:, :-p]                                 # residue p positions back
            add(back, penalty)
            fwd = canvas.new_full((B, L), -1)
            fwd[:, :-p] = canvas[:, p:]                                  # residue p positions ahead
            add(fwd, penalty)

        if max_run and L > max_run:
            def _shift(x, n, forward):
                out = x.new_full(x.shape, -1)
                if n < L:
                    if forward:
                        out[:, :L - n] = x[:, n:]
                    else:
                        out[:, n:] = x[:, :L - n]
                return out

            def _runlen(forward):
                """Length of the committed identical run adjacent to each position, capped."""
                near = _shift(canvas, 1, forward)
                ok = (near >= 0) & (near < n_aa)
                eqs = torch.stack([(_shift(canvas, 1 + k, forward) == near) & ok
                                   for k in range(max_run)], dim=-1)
                return torch.cumprod(eqs.to(torch.int16), dim=-1).sum(-1), near, ok

            llen, lt, lok = _runlen(False)
            rlen, rt, rok = _runlen(True)
            joins = (lt == rt) & lok & rok                       # same residue on both sides
            for tok, own, other, valid in ((lt, llen, rlen, lok), (rt, rlen, llen, rok)):
                total = 1 + own + torch.where(joins, other, torch.zeros_like(other))
                hit = valid & (total > max_run)
                pen.scatter_add_(2, tok.clamp(min=0).unsqueeze(-1),
                                 (hit.to(logits.dtype) * ban).unsqueeze(-1))
        return logits - pen

    return fn


# --------------------------------------------------------------------------------------
# Vectorised canvas helpers (no per-row loops -> no host syncs)
# --------------------------------------------------------------------------------------
def _sample(probs, greedy):
    if greedy:
        conf, tok = probs.max(dim=-1)
        return tok, conf
    B, L, V = probs.shape
    tok = torch.multinomial(probs.reshape(-1, V), 1).reshape(B, L)
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return tok, conf


def _first_eos(canvas, is_masked, eos_id):
    """(B,) position of each row's leftmost COMMITTED EOS, or L if it has none."""
    B, L = canvas.shape
    pos = torch.arange(L, device=canvas.device).expand(B, L)
    hit = (canvas == eos_id) & ~is_masked
    return torch.where(hit, pos, torch.full_like(pos, L)).min(dim=1).values


def _enforce_eos(canvas, is_masked, cfg):
    """Everything right of the leftmost committed EOS becomes PAD and is done. In place."""
    B, L = canvas.shape
    first = _first_eos(canvas, is_masked, cfg.eos_token_id)
    after = torch.arange(L, device=canvas.device).expand(B, L) > first[:, None]
    canvas.masked_fill_(after, cfg.pad_token_id)
    is_masked.masked_fill_(after, False)


def _topk_mask(scores, k, largest=True):
    """(B,L) bool selecting each row's top-k (or bottom-k) entries, with a PER-ROW k tensor.

    torch.topk needs a scalar k, so ranks are taken instead: rank each row by sorting, then keep the
    entries whose rank is below that row's k. Same selection, one static shape, no loop.
    """
    order = scores.argsort(dim=1, descending=largest)
    rank = order.argsort(dim=1)
    return rank < k.clamp(min=0)[:, None]


def lengths_of(canvas, cfg):
    """Length = position of the first EOS, else the full width (a max-length generation)."""
    no_mask = torch.zeros_like(canvas, dtype=torch.bool)
    return _first_eos(canvas, no_mask, cfg.eos_token_id).tolist()


# --------------------------------------------------------------------------------------
# EOS-first boundary placement
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _place_eos_first(model, canvas, is_masked, guidance_fn, min_len, greedy, eos_temp):
    """Commit EOS before any residue, sampled from P(EOS at i) over the all-MASK canvas.

    The length prior itself is sound -- ProLoopDiff's unconditional samples that DID place EOS
    averaged 344 aa over 183-496 -- it is the committing that failed. argmax would collapse every
    sample onto one length, so this samples unless the caller asked for greedy.
    """
    cfg = model.cfg
    B, L = canvas.shape
    lg = _step_logits(model, canvas, guidance_fn)
    p_eos = torch.softmax(lg.float(), dim=-1)[..., cfg.eos_token_id].clone()      # (B, L)

    lo = min(max(int(min_len), 0), L - 1)
    p_eos[:, :lo] = 0.0                                             # corpus floor

    # A row whose EOS mass all sits below the floor (or is non-finite) has no usable opinion; fall
    # back to a uniform draw over the legal range rather than letting multinomial fail.
    row = p_eos.sum(dim=-1, keepdim=True)
    dead = ~torch.isfinite(row) | (row <= 0)
    if bool(dead.any()):
        unif = torch.zeros_like(p_eos)
        unif[:, lo:] = 1.0
        p_eos = torch.where(dead, unif, p_eos)
        row = p_eos.sum(dim=-1, keepdim=True)

    probs = p_eos / row
    if eos_temp != 1.0:                       # <1 sharpens toward the mode, >1 widens the spread
        probs = probs.clamp_min(1e-12) ** (1.0 / max(float(eos_temp), 1e-6))
        probs = probs / probs.sum(dim=-1, keepdim=True)

    pos = probs.argmax(dim=-1) if greedy else torch.multinomial(probs, 1).squeeze(-1)
    canvas.scatter_(1, pos[:, None], cfg.eos_token_id)
    is_masked.scatter_(1, pos[:, None], False)
    _enforce_eos(canvas, is_masked, cfg)
    return pos


# --------------------------------------------------------------------------------------
# D3PM revision channel (the OADM -> D3PM blend)
# --------------------------------------------------------------------------------------
def default_t_start(T: int) -> int:
    """Where the revision anneal begins. T // 10 -- measured, not guessed. See d3pm_timestep."""
    return max(1, min(T - 1, T // 10))


def d3pm_timestep(step: int, n_steps: int, t_start: int) -> int:
    """Anneal the D3PM timestep linearly from t_start down to 1 across the decode. -> int in [1, T-1]

    Indexed by STEP rather than by the committed fraction, deliberately. The two are near-identical
    (the cosine schedule ties commits to steps), but a step-indexed anneal is monotone and lands on
    exactly 1 at the final step no matter what the canvas does -- the convergence half of the
    guarantee, and not something worth making contingent on the sampling trajectory.

    WHICH DIRECTION, AND WHY t_start IS SMALL. A single reverse step's willingness to move a token is
    NOT monotone in the obvious direction. Measured, as P(x_{t-1} != x_t) for one step on a canvas
    the model actively disagrees with (uniform base, T=500; the BLOSUM base is within 0.1%):

        t     = 400    300    200    100     50     10      1
        move  = 1.1%   0.8%   0.8%   1.2%   2.2%  10.2%  100.0%      (model disagrees)
        move  = 0.13%  0.03%  0.01%  0.00%  0.00%  0.00%   0.00%     (model agrees)

    Authority is near zero at large t and total at t=1. The reason is Qbar_{t-1}: at large t it is
    nearly uniform, so `c = p~ Qbar_{t-1}` is nearly flat and cannot outvote the (1-b_t) mass that
    Q_t puts on staying put; at small t it is nearly the identity, so c[x_t] collapses to ~0 as soon
    as the model's x0 prediction points elsewhere and the token has to move.

    Two consequences:

      * Annealing t downward IS the "progressively more D3PM-like" direction. Early steps leave the
        canvas essentially alone (OADM-like); late steps rewrite anything the model disagrees with.
      * A linear anneal from T wastes itself. Between t=400 and t=100 the channel moves ~1% of
        disputed tokens per step, so a T -> 1 ramp spends ~90% of the decode doing nothing measurable.
        t_start = T // 10 = 50 puts the entire ramp inside the range where authority actually climbs.
        Raise it toward T to weaken the channel; lower it to make revision bite sooner.

    Convergence does not come from the step becoming an identity -- at t=1 it is at its most
    powerful. It comes from the second row of that table: once the model AGREES with the canvas, the
    move probability is 0.00% at every t. The late steps drive the canvas to a fixed point of the
    model's own x0 prediction and then stop, which is a stronger settling property than freezing.
    """
    return max(1, int(round(t_start * (1.0 - (step + 1) / n_steps))))


@torch.no_grad()
def _d3pm_revise(sched: D3PMSchedule, canvas, revisable, tempered_logits, t: int,
                 blend: float, greedy: bool, n_aa: int = 20):
    """One D3PM reverse transition x_t -> x_{t-1}, applied to already-committed RESIDUES.

    Returns the (B,L) bool mask of positions this call actually changed, for accounting.

    Three restrictions, each load-bearing:

      * ONLY committed positions. A masked position is the absorbing channel's business; D3PM has no
        MASK state to reason about it with (K = vocab_size - 1).
      * ONLY residues. EOS and PAD are excluded by the caller, so `[AA* EOS PAD*]` well-formedness is
        preserved structurally and the length stays owned by one mechanism (eos_first). D3PM was
        trained to repair boundaries too -- that capability lives in the post-decode substitution
        corrector, where it cannot fight the decode schedule.
      * p_tilde COMES FROM THE GUIDED, TEMPERED LOGITS -- the same ones the absorbing channel commits
        from. So the repetition penalty applies to revisions as well; without that, revision would be
        free to reintroduce exactly the tracts the penalty just suppressed.

    THE POSTERIOR IS CONSTRAINED TO RESIDUES, NOT JUST p_tilde. This is subtle and cost a real bug.
    The guided logits already carry -inf at MASK/PAD (and at EOS while ban_eos holds), so p_tilde
    puts zero mass on EOS -- but that does NOT make the posterior zero there:

        p(x_{t-1}=EOS) proportional to Q_t[EOS, x_t] * sum_i p~[i] Qbar_{t-1}[i, EOS]

    and Qbar_{t-1}[i, EOS] is strictly positive for every i, because the transition matrix is
    doubly stochastic over an alphabet that INCLUDES EOS -- which is exactly the property that lets
    the D3PM training branch learn to repair boundaries. So a residue could be revised INTO an EOS,
    and the next step's _enforce_eos would then dutifully truncate everything after it. Observed:
    a sequence went to length 0 because position 0 was revised to EOS. Masking the posterior's
    non-residue columns is the fix -- it samples p(x_{t-1} | x_t, x_{t-1} is a residue), which is
    what "revise residue identity" means and what the restriction above claims.
    """
    # AUTOCAST OFF, for the same reason as objective.d3pm_loss: `p_tilde @ Qbar_{t-1}` is a matmul,
    # matmul is on autocast's bf16 list, and Qbar_{t-1} is near-identity for small t with
    # off-diagonal entries ~1e-3 that an 8-bit mantissa cannot hold next to a diagonal near 1.
    # The canvas still holds MASK (id K) wherever the absorbing channel has not committed yet, and
    # MASK is by construction NOT a D3PM state -- indexing Q_t with it would run off the end of a
    # (K,K) matrix. Clamp for the gather: those rows of `a` are meaningless, and `revisable` (which
    # requires an already-committed residue) discards every one of them before anything is written.
    xt = canvas.clamp(max=sched.K - 1)
    with torch.autocast(device_type=canvas.device.type, enabled=False):
        p_tilde = x0_probs(tempered_logits, sched.K)                     # (B,L,K) fp32
        p_post = sched.p_reverse(xt, t, p_tilde)                         # (B,L,K)
        p_post = p_post.clone()
        p_post[..., n_aa:] = 0.0                          # residues only -- see the note above
        p_post = p_post / p_post.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    new = p_post.argmax(dim=-1) if greedy else sample_categorical(p_post)

    take = revisable
    if blend < 1.0:                       # Bernoulli gate: each accepted position still takes a
        take = take & (torch.rand_like(p_post[..., 0]) < blend)   # PROPER D3PM draw, never a
    changed = take & (new != canvas)                              # blurred mixture of two posteriors
    canvas[take] = new[take]
    return changed


# --------------------------------------------------------------------------------------
# Correctors
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _corrector_sweep(model, canvas, frac, guidance_fn, greedy, temperature, min_len=0):
    """Remask the lowest-confidence committed positions and redecode them.

    min_len is threaded through for the same reason it exists in the main loop: a redecoded position
    may emit EOS, and an EOS at position 0 collapses the row to the empty sequence. Measured on an
    untrained checkpoint, correctors WITHOUT this floor drove mean length from 88 to 2.2.
    """
    cfg = model.cfg
    lg = _step_logits(model, canvas, guidance_fn, eos_min_pos=min_len)
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, canvas.unsqueeze(-1)).squeeze(-1)
    eligible = canvas != cfg.pad_token_id                       # remask AAs / EOS, never PAD
    conf_e = conf.masked_fill(~eligible, float("inf"))          # inf -> never picked as lowest
    k = (frac * eligible.sum(dim=1).float()).long().clamp(min=1)
    pick = _topk_mask(conf_e, k, largest=False) & eligible

    canvas.masked_fill_(pick, cfg.mask_token_id)
    lg2 = _step_logits(model, canvas, guidance_fn, eos_min_pos=min_len)
    tok, _ = _sample(torch.softmax(lg2 / max(temperature, 1e-6), dim=-1), greedy)
    canvas[pick] = tok[pick]
    _enforce_eos(canvas, torch.zeros_like(canvas, dtype=torch.bool), cfg)


@torch.no_grad()
def _substitution_corrector_sweep(model, canvas, frac, guidance_fn, greedy, temperature,
                                  min_len=0):
    """Resample the lowest-confidence RESIDUES straight to new tokens -- no MASK detour.

    This is the corrector the D3PM half of the objective exists for: absorbing-state training only
    ever teaches MASK->token, while D3PM training teaches token->token, which is what a direct
    substitution needs. EOS is an allowed emission, so a sweep can also move the boundary -- but only
    to a position at or beyond min_len, or a single unlucky resample near the N-terminus truncates
    the whole sequence (measured: mean length 88 -> 2.2 without the floor).
    """
    cfg = model.cfg
    lg = _step_logits(model, canvas, guidance_fn, eos_min_pos=min_len)   # MASK/PAD out; EOS allowed
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, canvas.unsqueeze(-1)).squeeze(-1)
    eligible = ((canvas != cfg.pad_token_id) & (canvas != cfg.eos_token_id)
                & (canvas != cfg.mask_token_id))
    conf_e = conf.masked_fill(~eligible, float("inf"))
    k = (frac * eligible.sum(dim=1).float()).long().clamp(min=1)
    pick = _topk_mask(conf_e, k, largest=False) & eligible
    tok, _ = _sample(probs, greedy)
    canvas[pick] = tok[pick]
    _enforce_eos(canvas, torch.zeros_like(canvas, dtype=torch.bool), cfg)


# --------------------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model: LoopedDiffusionLM, Lmax: int, batch_size: int,
             n_steps: Optional[int] = None, temperature: float = 0.5,
             gumbel_temp: float = 0.1, greedy: bool = False,
             n_corrector: int = 0, corrector_frac: float = 0.1, corrector_type: str = "remask",
             guidance_fn: Optional[Callable] = None, device: str = "cpu",
             eos_first: bool = True, min_len: int = 30, eos_temp: float = 1.0,
             rep_penalty: float = 1.5, rep_periods=(1, 2, 3, 4, 5), max_run: int = 5,
             n_recurrence: Optional[int] = None,
             d3pm_sched: Optional[D3PMSchedule] = None, d3pm_blend: float = 0.0,
             d3pm_t_start: Optional[int] = None, stats: Optional[dict] = None):
    """Returns (canvas (B, Lmax) long, lengths list[int]). Always well-formed [AA* EOS PAD*].

    corrector_type: "remask" (works with any model) or "substitution" (wants D3PM training).
    A caller-supplied guidance_fn is applied AFTER the repetition penalty, so a ProteinGuide-style
    hook composes with it rather than replacing it.

    ------------------------------------------------------------------------------------------
    THE OADM -> D3PM BLEND (d3pm_blend > 0, needs d3pm_sched).

    PLD2 trains both objectives, so decoding with only the absorbing one leaves half the model
    unused at generation. This runs BOTH channels in every step, and the balance between them
    shifts on its own from OADM-like to D3PM-like as the canvas fills:

      absorbing channel  decides WHERE content exists. Unchanged: the cosine schedule commits the
                         most-confident masked positions and they stop being masked.
      D3PM channel       decides WHAT that content is. Every already-committed residue takes one
                         reverse transition p_theta(x_{t-1} | x_t) at an annealed t.

    Two ramps compose to give the blend its shape, and neither is a hand-tuned curve:

      COVERAGE ramps UP by itself. The D3PM channel acts on committed positions, and the committed
      set grows from ~0 to the whole canvas. Step 1 revises almost nothing; the last steps revise
      everything. No schedule needed -- this is just what the absorbing channel does.

      AUTHORITY ramps UP as t anneals t_start -> 1 (see d3pm_timestep for the measured curve, which
      runs the opposite way to intuition). Early steps barely touch the canvas; late steps rewrite
      any committed residue the model disagrees with, and leave alone every one it agrees with.

    So the early canvas is built OADM-style and the late canvas is pulled to a fixed point of the
    model's own x0 prediction. That is the attack on the repetition failure mode: in pure MaskGIT an
    accidental tract is committed early, frozen, and then preferentially EXTENDED because a repeat
    is maximally predictable. Here it stays revisable for the whole decode, and gets revised exactly
    when there is enough surrounding context to know it was wrong -- under the repetition penalty,
    which the revision path also sees.

    WHY THE TWO GUARANTEES HOLD.

      All masks decoded, in exactly n_steps: the absorbing schedule is untouched, and its cosine
      target is `round(n_active * cos(pi/2 * (step+1)/n_steps))`, which is exactly 0 on the final
      step -- so whatever is still masked is committed then. The D3PM channel CANNOT interfere,
      because its state space has no MASK to emit (K = vocab_size - 1). The two channels are
      separable by construction, not by careful scheduling.

      No extra compute: both channels read the SAME forward pass. The absorbing commit needs
      logits(canvas); the reverse step needs p_theta(x_0 | canvas) from those same logits. A blended
      decode costs exactly what an unblended one costs, to within one (K,K) matmul per step.

    WHAT THIS CHANNEL IS, AND ITS LIMIT. It is a faithful amplifier of the model's own x0 belief,
    nothing more. Measured on a deliberately overfit toy (4 memorised sequences, 6 residues corrupted
    in each, revision-only, t annealed 50 -> 1): two sequences went 6 errors -> 0, and two went
    6 -> 26 and 6 -> 15. That looks like the channel breaking things, but the model's own argmax
    p_tilde on those same corrupted inputs ALREADY differed from the truth at 24/28 and 15/16
    positions BEFORE any revision ran. The channel converged each sequence to the model's belief,
    exactly as designed; on two of them that belief was wrong. Flooring the anneal above t=1 does
    not help (swept t_end = 1, 3, 5, 10, 20, 50: 41, 41, 40, 40, 37, 35 total errors, all dominated
    by the belief rather than the schedule), which is why there is no t_end knob -- it would look
    like a safety control without being one.

    So this cleans up a good model and confidently corrupts a bad one, and its worth is not
    knowable from the mechanism alone. THAT is why config defaults sample_d3pm_blend to 0.0: it
    should be A/B'd against blend=0 on a real checkpoint (`src/sample.py --d3pm-blend`) before
    anything relies on it. Note the specific risk for PLD2's failure mode -- repeats are
    high-likelihood, so an unrestrained pull toward the model's belief could as easily drive samples
    INTO repetition as out of it. The repetition penalty applies to the revision path for exactly
    this reason, and the k-mer metrics are what would show it either way.

    d3pm_blend  0.0 disables the channel entirely (bit-identical to the absorbing-only sampler --
                no RNG is drawn on that path), 1.0 revises every eligible position every step.
                Values in between are a per-position Bernoulli gate, so each revision that does
                happen is still a proper D3PM draw rather than a blurred mixture of two posteriors.
    d3pm_t_start where the anneal begins; None -> default_t_start(T) = T//10, chosen from the
                measured authority curve rather than by feel. Raise it toward T to weaken the
                channel, lower it to make revision bite sooner.
    stats       optional dict, filled with revision accounting if given.
    """
    cfg = model.cfg
    B = batch_size
    n_steps = n_steps or Lmax
    was_training = model.training
    model.eval()

    guides = []
    if rep_penalty > 0 or max_run:
        guides.append(make_repetition_penalty(cfg, rep_penalty, rep_periods, max_run))
        _warn_if_too_few_steps(n_steps, Lmax)
    if guidance_fn is not None:
        guides.append(guidance_fn)

    def guide(cv, lg):
        for g in guides:
            lg = g(cv, lg)
        return lg
    guide = guide if guides else None

    use_d3pm = d3pm_blend > 0 and d3pm_sched is not None
    if d3pm_blend > 0 and d3pm_sched is None:
        raise ValueError("d3pm_blend > 0 needs a d3pm_sched (build it with train.build_d3pm_schedule)")
    # Clamped to T-1, never T: beta_T = 1.0 exactly (the calibration saturates there), so a reverse
    # step at t=T resamples ~95% of tokens whether or not the model agrees -- destruction, not
    # revision. That state is only meaningful when x_T really is uniform noise, which a canvas of
    # model-committed tokens is not.
    t_start = default_t_start(d3pm_sched.T) if d3pm_sched is not None else 1
    if d3pm_t_start is not None and d3pm_sched is not None:
        t_start = max(1, min(int(d3pm_t_start), d3pm_sched.T - 1))
    n_revised = 0

    canvas = torch.full((B, Lmax), cfg.mask_token_id, dtype=torch.long, device=device)
    is_masked = torch.ones((B, Lmax), dtype=torch.bool, device=device)

    if eos_first:
        _place_eos_first(model, canvas, is_masked, guide, min_len, greedy, eos_temp)

    # Per-ROW schedule budget. A cosine schedule over Lmax would assume the whole canvas is still in
    # play; after eos_first most of it is already committed PAD, so an Lmax-based target sits above
    # the true masked count for most of the run and commits nothing until a rush at the very end.
    n_active = is_masked.sum(dim=1).to(torch.float32)

    for step in range(n_steps):
        if not bool(is_masked.any()):
            break
        lg = _step_logits(model, canvas, guide, ban_eos=eos_first,
                          eos_min_pos=(0 if eos_first else min_len))
        probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
        tok, conf = _sample(probs, greedy)

        conf_masked = conf.masked_fill(~is_masked, float("-inf"))
        if gumbel_temp > 0:                                    # stochastic commit ORDER (MaskGIT)
            u = torch.rand_like(conf).clamp_min(1e-9)
            g = -torch.log(-torch.log(u).clamp_min(1e-9))
            conf_masked = conf_masked + gumbel_temp * g.masked_fill(~is_masked, 0.0)

        # cosine schedule: how many positions should remain masked after this step (per row)
        frac = (step + 1) / n_steps
        n_mask_target = (n_active * math.cos(math.pi / 2 * frac)).round().long()
        n_commit = (is_masked.sum(dim=1) - n_mask_target).clamp(min=0)

        # The D3PM channel acts on what was ALREADY committed at the top of this step -- residues
        # only, so EOS/PAD and therefore well-formedness are untouched. Captured before the commits
        # below, so a position is never both committed and revised in the same step: a token drawn
        # from this step's logits has nothing to gain from being immediately revised against them.
        revisable = (~is_masked) & (canvas < 20) if use_d3pm else None

        commit = _topk_mask(conf_masked, n_commit, largest=True) & is_masked
        canvas[commit] = tok[commit]
        is_masked &= ~commit
        _enforce_eos(canvas, is_masked, cfg)

        if use_d3pm:
            t = d3pm_timestep(step, n_steps, t_start)
            changed = _d3pm_revise(d3pm_sched, canvas, revisable,
                                   lg / max(temperature, 1e-6), t, d3pm_blend, greedy)
            n_revised += int(changed.sum())
            # A revision can only rewrite a residue with another residue, so no EOS can appear or
            # move and the [AA* EOS PAD*] invariant cannot be broken here. Re-enforcing would be a
            # no-op; asserting it in the test suite is cheaper than doing it 512 times.

    # Correctors run under the SAME composed guidance: redecoding a tract without the repetition
    # penalty would just reproduce it -- the flanking context still supports it and confidence
    # ordering would re-select it.
    for _ in range(n_corrector):
        if corrector_type == "substitution":
            _substitution_corrector_sweep(model, canvas, corrector_frac, guide, greedy, temperature,
                                          min_len=min_len)
        else:
            _corrector_sweep(model, canvas, corrector_frac, guide, greedy, temperature,
                             min_len=min_len)

    if was_training:
        model.train()
    if stats is not None:
        stats.update(n_steps=n_steps, n_revised=n_revised, d3pm_blend=d3pm_blend,
                     t_start=t_start if use_d3pm else 0,
                     revisions_per_seq=n_revised / max(B, 1))
    return canvas, lengths_of(canvas, cfg)


def decode_seqs(canvas, cfg, min_len: int = 0, max_len: Optional[int] = None):
    """Canvas rows -> amino-acid strings, truncated at the first EOS/PAD.

    Only ids 0..19 map to residues; a MASK that survived decoding is dropped. Rows outside
    [min_len, max_len] are OMITTED rather than clipped -- reporting a statistic for a molecule that
    was never generated is worse than reporting one fewer sample. Returns (sequences, n_skipped).
    """
    from .blosum import AA
    seqs, skipped = [], 0
    for row in canvas.cpu().tolist():
        out = []
        for t in row:
            if t == cfg.eos_token_id or t == cfg.pad_token_id:
                break
            if 0 <= t < len(AA):
                out.append(AA[t])
        if len(out) >= min_len and (max_len is None or len(out) <= max_len):
            seqs.append("".join(out))
        else:
            skipped += 1
    return seqs, skipped


def write_fasta(seqs, path, prefix="sample"):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i} len={len(s)}\n{s}\n")
    return len(seqs)
