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

from .corruption import sample_categorical
from .model import LoopedDiffusionLM


# --------------------------------------------------------------------------------------
# Per-step logits
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _step_logits(model, canvas, guidance_fn, ban_eos: bool = False, eos_min_pos: int = 0):
    """canvas (B, K, L) -> logits (B, K, L, V), K = model.cfg.n_tracks.

    The repetition penalty and any caller guidance see only the AMINO-ACID track: they are written
    against residue statistics (homopolymer runs, periodic repeats) and a 3Di run is a helix, not a
    defect. Applying them to the structure track would penalise the most ordinary secondary
    structure there is."""
    cfg = model.cfg
    K = cfg.n_tracks
    lg = model(canvas[:, 0], struct=canvas[:, 1]) if K == 2 else model(canvas[:, 0]).unsqueeze(1)
    if guidance_fn is not None:
        lg = torch.cat([guidance_fn(canvas[:, 0], lg[:, 0]).unsqueeze(1), lg[:, 1:]], dim=1)
    lg = lg.clone()
    lg[..., cfg.mask_token_id] = float("-inf")      # never emit MASK...
    lg[..., cfg.pad_token_id] = float("-inf")       # ...or PAD (it arrives only via EOS enforcement)
    if K > 1:
        # ONLY TRACK 0 DECIDES THE BOUNDARY. The tracks describe one molecule, so it has one length.
        # Left to itself the structure track will commit its own EOS at its own position, and with
        # eos_first=False it does: _enforce_eos then pads from the SEQUENCE track's boundary and the
        # structure track is left holding a stray EOS to the left of it, so the two tracks disagree
        # about where the protein ends and the decoded pair stops being position-aligned. The
        # structure track's EOS is a MIRROR of track 0's, written by _enforce_eos, never a decision.
        lg[:, 1:, :, cfg.eos_token_id] = float("-inf")
    if ban_eos:
        # The boundary is already committed. A second EOS to its LEFT would silently shorten the
        # sequence, since _enforce_eos honours the leftmost one.
        lg[..., cfg.eos_token_id] = float("-inf")
    elif eos_min_pos > 0:
        # No EOS below the corpus floor: an EOS at position 0 yields the empty sequence, which is
        # where len=0 samples come from -- not from a short prediction.
        # Indexed on the POSITION axis explicitly. lg is (B, K, L, V) now, so the old
        # `lg[:, :eos_min_pos, eos]` sliced the TRACK axis and then read eos_token_id as a POSITION,
        # blanking every logit at position 20 -- a softmax of all -inf, i.e. NaN.
        lg[:, :, :eos_min_pos, cfg.eos_token_id] = float("-inf")
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
    V = probs.shape[-1]
    tok = torch.multinomial(probs.reshape(-1, V), 1).reshape(probs.shape[:-1])
    conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    return tok, conf


def _first_eos(canvas, is_masked, eos_id):
    """(B,) position of each row's leftmost COMMITTED EOS, or L if it has none."""
    B, L = canvas.shape
    pos = torch.arange(L, device=canvas.device).expand(B, L)
    hit = (canvas == eos_id) & ~is_masked
    return torch.where(hit, pos, torch.full_like(pos, L)).min(dim=1).values


def _enforce_eos(canvas, is_masked, cfg):
    """Everything right of the leftmost committed EOS becomes PAD and is done. In place.

    Accepts (B,L) or the multi-track (B,K,L). The boundary is read from TRACK 0 and applied to every
    track: the tracks describe one molecule and must agree on where it ends. unsqueeze returns a
    view, so the in-place fills reach the caller's tensor either way."""
    L = canvas.shape[-1]
    cv = canvas if canvas.dim() == 3 else canvas.unsqueeze(1)
    im = is_masked if is_masked.dim() == 3 else is_masked.unsqueeze(1)
    first = _first_eos(cv[:, 0], im[:, 0], cfg.eos_token_id)
    pos = torch.arange(L, device=canvas.device)
    cv.masked_fill_(pos > first[:, None, None], cfg.pad_token_id)
    im.masked_fill_(pos > first[:, None, None], False)
    if cv.shape[1] > 1:
        # The structure track pads from the boundary INCLUSIVE -- it has no EOS of its own, so the
        # position holding the sequence track's EOS holds PAD here. That matches the training layout
        # exactly (data.ProteinShards.get_pair) and, more to the point, leaves no second boundary
        # token that could end up disagreeing with the first when the boundary moves. It can: a
        # corrector may remask the EOS and redecode it away entirely, and a stale mirrored EOS then
        # sat to the LEFT of the real boundary and silently unaligned the decoded pair.
        at_or_after = pos >= first[:, None, None]
        cv[:, 1:].masked_fill_(at_or_after, cfg.pad_token_id)
        im[:, 1:].masked_fill_(at_or_after, False)


def _topk_mask(scores, k, largest=True):
    """(B,L) bool selecting each row's top-k (or bottom-k) entries, with a PER-ROW k tensor.

    torch.topk needs a scalar k, so ranks are taken instead: rank each row by sorting, then keep the
    entries whose rank is below that row's k. Same selection, one static shape, no loop.
    """
    order = scores.argsort(dim=1, descending=largest)
    rank = order.argsort(dim=1)
    return rank < k.clamp(min=0)[:, None]


def aa_track(canvas):
    """(B,K,L) -> its amino-acid track (B,L); (B,L) passes through. One place to spell the
    n_tracks branch so callers written before the structure track keep reading."""
    return canvas[:, 0] if canvas.dim() == 3 else canvas


def struct_track(canvas):
    """(B,K,L) -> its 3Di track (B,L), or None if the canvas has no structure track."""
    return canvas[:, 1] if canvas.dim() == 3 and canvas.shape[1] > 1 else None


def lengths_of(canvas, cfg):
    """Length = position of the first EOS, else the full width (a max-length generation)."""
    canvas = aa_track(canvas)
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
    B, L = canvas.shape[0], canvas.shape[-1]
    lg = _step_logits(model, canvas, guidance_fn)
    # The boundary is drawn from the SEQUENCE track and mirrored into the structure track. The two
    # tracks describe the same molecule, so they have one length; letting each track place its own
    # EOS would let them disagree about where the protein ends.
    p_eos = torch.softmax(lg[:, 0].float(), dim=-1)[..., cfg.eos_token_id].clone()   # (B, L)

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
    # EOS into the SEQUENCE track only; _enforce_eos then pads every other track from that index.
    canvas[:, 0].scatter_(-1, pos[:, None], cfg.eos_token_id)
    is_masked[:, 0].scatter_(-1, pos[:, None], False)
    _enforce_eos(canvas, is_masked, cfg)
    return pos


# --------------------------------------------------------------------------------------
# Unified edit candidates (unmask OR substitute), scored on one confidence scale
# --------------------------------------------------------------------------------------
def _edit_candidates(probs, canvas, is_masked, n_aa, greedy):
    """One candidate edit per position, with a confidence comparable across both move types.

    masked position    -> the edit is an UNMASK.     token ~ p,          confidence = p(token)
    committed residue  -> the edit is a SUBSTITUTION. token ~ p|token!=current, confidence = p(token)

    Returns (token (B,L), conf (B,L), eligible (B,L) bool).

    THE CONFIDENCE IS DELIBERATELY NOT RENORMALISED for substitutions. The token is DRAWN from p
    with the current residue removed and the rest renormalised -- a substitution has to change
    something -- but it is SCORED by its probability under the original p. That asymmetry is the
    whole mechanism: a position the model is already happy with has all its mass on the current
    token, so the best alternative scores near zero and never wins a slot; a position the model
    thinks is wrong has mass sitting on some other residue, which scores high and does. Renormalising
    would erase exactly that signal -- a position the model was 99% sure about would see its leftover
    1% rescaled to ~100% and look maximally worth editing.

    Both move types therefore live on one scale -- "how much does the model want this token here" --
    which is what lets a single ranking choose between them.
    """
    conf_unmask, tok_unmask = probs.max(dim=-1) if greedy else (None, None)
    if not greedy:
        tok_unmask = sample_categorical(probs)
        conf_unmask = probs.gather(-1, tok_unmask.unsqueeze(-1)).squeeze(-1)

    # Substitution: same distribution with the incumbent removed.
    p_sub = probs.scatter(-1, canvas.clamp(max=probs.shape[-1] - 1).unsqueeze(-1), 0.0)
    if greedy:
        tok_sub = p_sub.argmax(dim=-1)
    else:
        tok_sub = sample_categorical(p_sub / p_sub.sum(dim=-1, keepdim=True).clamp_min(1e-30))
    conf_sub = probs.gather(-1, tok_sub.unsqueeze(-1)).squeeze(-1)     # scored under the ORIGINAL p

    is_residue = (canvas < n_aa) & ~is_masked
    token = torch.where(is_masked, tok_unmask, tok_sub)
    conf = torch.where(is_masked, conf_unmask, conf_sub)
    eligible = is_masked | is_residue
    return token, conf.masked_fill(~eligible, float("-inf")), eligible


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
    cv = aa_track(canvas)                        # a VIEW; edits write through to `canvas`
    # WITH A STRUCTURE TRACK THE BOUNDARY IS FROZEN. This sweep remasks committed positions and
    # redecodes them, so it can delete the EOS and let the boundary move RIGHT -- reclaiming a
    # position the structure track was already padded at, whose 3Di token is gone and cannot be
    # recovered. The two tracks then disagree about the length by one. Everything else the sweep
    # does is safe (a boundary moving LEFT just pads more), so only EOS is taken off the table.
    fixed = canvas.dim() == 3 and canvas.shape[1] > 1
    lg = _step_logits(model, canvas, guidance_fn, ban_eos=fixed, eos_min_pos=min_len)[:, 0]
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, cv.unsqueeze(-1)).squeeze(-1)
    eligible = cv != cfg.pad_token_id                           # remask AAs / EOS, never PAD
    if fixed:
        eligible = eligible & (cv != cfg.eos_token_id)
    conf_e = conf.masked_fill(~eligible, float("inf"))          # inf -> never picked as lowest
    k = (frac * eligible.sum(dim=1).float()).long().clamp(min=1)
    pick = _topk_mask(conf_e, k, largest=False) & eligible

    cv.masked_fill_(pick, cfg.mask_token_id)
    lg2 = _step_logits(model, canvas, guidance_fn, ban_eos=fixed, eos_min_pos=min_len)[:, 0]
    tok, _ = _sample(torch.softmax(lg2 / max(temperature, 1e-6), dim=-1), greedy)
    cv[pick] = tok[pick]
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
    cv = aa_track(canvas)                        # a VIEW; edits write through to `canvas`
    lg = _step_logits(model, canvas, guidance_fn,
                      eos_min_pos=min_len)[:, 0]            # MASK/PAD out; EOS allowed
    probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)
    conf = probs.gather(-1, cv.unsqueeze(-1)).squeeze(-1)
    eligible = ((cv != cfg.pad_token_id) & (cv != cfg.eos_token_id)
                & (cv != cfg.mask_token_id))
    conf_e = conf.masked_fill(~eligible, float("inf"))
    k = (frac * eligible.sum(dim=1).float()).long().clamp(min=1)
    pick = _topk_mask(conf_e, k, largest=False) & eligible
    tok, _ = _sample(probs, greedy)
    cv[pick] = tok[pick]
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
             subst_per_residue: float = 0.0, struct_first: float = 0.0,
             stats: Optional[dict] = None):
    """Returns (canvas, lengths). canvas is (B, Lmax) at n_tracks=1 and (B, 2, Lmax) at n_tracks=2
    -- use sampler.aa_track() / sampler.struct_track() rather than indexing. Always well-formed
    [AA* EOS PAD*], in every track.

    ------------------------------------------------------------------------------------------
    TWO TRACKS SHARE ONE RANKING, and that is the point of the layout. A 3Di edit and an amino-acid
    edit at the same position are SEPARATE candidates competing on the same confidence scale, so
    the decoder is free to lay down structure first and fill residues into it -- exactly the
    "global plan, then sequence" behaviour a joint (aa, 3Di) token could not express, because
    committing one such token commits both channels at once. Whether that ordering actually emerges
    is an empirical question the decode answers on its own: 3Di is the lower-entropy track
    (H = 2.52 nats against 2.89 for residues), so it should tend to win early slots without being
    told to.

    struct_first  forces the issue if it does not emerge. It is a decaying BIAS on structure
                confidences over the first `struct_first * n_steps` steps, not a gate: every slot
                stays eligible throughout, so the cosine floor and the termination proof below are
                untouched. 0.0 leaves the ranking alone, which is the default and the honest test.

    A caller-supplied guidance_fn is applied AFTER the repetition penalty, so a ProteinGuide-style
    hook composes with it rather than replacing it.

    ------------------------------------------------------------------------------------------
    UNIFIED EDIT DECODING (subst_per_residue > 0)

    The model is trained on ONE process in which a position can be masked or substituted (see
    src/corruption.py), so decoding offers the matching pair of moves and picks between them on one
    confidence scale. Every step, each position proposes exactly one candidate edit:

        masked            -> UNMASK to a drawn token
        committed residue -> SUBSTITUTE to a different token

    both scored by the model's probability for the token being written (see _edit_candidates for why
    substitutions are drawn from a renormalised distribution but SCORED under the original one).
    The highest-confidence edits win. There is no separate corrector pass and no second schedule:
    substitution is a first-class move throughout the decode, which is what the objective trains.

    WHY THE UNMASK QUOTA IS STILL A FLOOR. A single global ranking over both move types can starve
    the mask channel -- if substitution edits keep out-scoring unmask edits, masks are never
    committed and the canvas never resolves. So the cosine schedule's unmask count is taken FIRST
    and guaranteed, and only the remaining budget is allocated by global ranking (where an unmask
    can still win, so the floor never caps progress). That preserves the termination proof exactly:
    the cosine target is 0 at the final step, so everything still masked is committed then.

    WHY THE WORK IS BOUNDED. Both move types are chosen from the SAME forward pass -- the candidate
    tokens and their confidences all come from one call to the model. Nothing iterates to
    convergence. Model forwards are exactly
        1 (eos_first) + n_steps + n_corrector * (2 if remask else 1)
    independent of subst_per_residue, which src/tests_sampler.py asserts.

    subst_per_residue  the substitution budget, as expected edits per decodable position over the
                whole decode. The per-step allowance is `round(k * n_active / n_steps)`, constant,
                so early steps (few committed residues, most of the allowance unusable) spend little
                and late steps spend it all -- the OADM-like -> substitution-rich ramp falls out of
                the canvas filling up rather than needing a schedule. 0.0 disables substitution
                entirely, recovering pure absorbing decoding bit-for-bit.
    stats       optional dict, filled with edit accounting if given.
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

    use_subst = subst_per_residue > 0
    n_subst_done = 0

    K = cfg.n_tracks
    canvas = torch.full((B, K, Lmax), cfg.mask_token_id, dtype=torch.long, device=device)
    is_masked = torch.ones((B, K, Lmax), dtype=torch.bool, device=device)

    if eos_first:
        _place_eos_first(model, canvas, is_masked, guide, min_len, greedy, eos_temp)

    # Per-ROW schedule budget. A cosine schedule over Lmax would assume the whole canvas is still in
    # play; after eos_first most of it is already committed PAD, so an Lmax-based target sits above
    # the true masked count for most of the run and commits nothing until a rush at the very end.
    # Summed over BOTH tracks: the schedule's budget is slots, and a structure slot costs a commit
    # exactly like a residue slot does.
    n_active = is_masked.flatten(1).sum(dim=1).to(torch.float32)
    sf_steps = max(0.0, float(struct_first)) * n_steps

    # Per-step substitution allowance: constant, derived from the whole-decode budget. Held in a
    # tensor so it is per-row like every other quota here.
    n_subst_step = ((subst_per_residue * n_active / max(n_steps, 1)).round().long()
                    if use_subst else torch.zeros_like(n_active, dtype=torch.long))

    for step in range(n_steps):
        if not bool(is_masked.any()) and not use_subst:
            break
        lg = _step_logits(model, canvas, guide, ban_eos=eos_first,
                          eos_min_pos=(0 if eos_first else min_len))
        probs = torch.softmax(lg / max(temperature, 1e-6), dim=-1)

        if use_subst:
            tok, conf, eligible = _edit_candidates(probs, canvas, is_masked, 20, greedy)
        else:                                    # absorbing-only: identical to the previous sampler
            tok, conf = _sample(probs, greedy)
            eligible = is_masked
            conf = conf.masked_fill(~is_masked, float("-inf"))

        if gumbel_temp > 0:                      # stochastic edit ORDER (MaskGIT)
            u = torch.rand_like(conf).clamp_min(1e-9)
            g = -torch.log(-torch.log(u).clamp_min(1e-9))
            conf = conf + gumbel_temp * g.masked_fill(~eligible, 0.0)

        if K == 2 and step < sf_steps:
            # Decaying preference for the structure track. A BIAS, not a gate: -inf entries stay
            # -inf, every slot stays eligible, and the unmask floor below can still draw on the
            # sequence track if it has to -- so nothing here can starve the schedule.
            bias = torch.zeros_like(conf)
            bias[:, 1] = 1e4 * (1.0 - step / max(sf_steps, 1e-9))
            conf = conf + bias.masked_fill(~eligible, 0.0)

        # cosine schedule: how many positions should remain masked after this step (per row)
        frac = (step + 1) / n_steps
        n_mask_target = (n_active * math.cos(math.pi / 2 * frac)).round().long()
        n_commit = (is_masked.flatten(1).sum(dim=1) - n_mask_target).clamp(min=0)

        # Ranking is over the FLATTENED (track, position) slots, so one top-k chooses between an
        # unmask and a substitution and between the two tracks in a single comparison.
        shp = is_masked.shape
        flat = lambda x: x.reshape(shp[0], -1)

        # FLOOR: the scheduled unmask quota, chosen among masked positions only. Guaranteed, which
        # is what keeps the termination proof intact when substitutions compete for edits.
        commit = _topk_mask(flat(conf.masked_fill(~is_masked, float("-inf"))),
                            n_commit).view(shp) & is_masked
        sel = commit
        if use_subst:
            # EXTRA: the remaining budget, ranked globally across both move types. An unmask can win
            # here too, so the floor is a minimum and never a cap.
            rest = conf.masked_fill(sel, float("-inf"))
            extra = _topk_mask(flat(rest), n_subst_step).view(shp) & eligible & ~sel
            n_subst_done += int((extra & ~is_masked).sum())
            sel = sel | extra

        canvas[sel] = tok[sel]
        is_masked &= ~sel
        _enforce_eos(canvas, is_masked, cfg)

    # Correctors run under the SAME composed guidance: redecoding a tract without the repetition
    # penalty would just reproduce it -- the flanking context still supports it and confidence
    # ordering would re-select it.
    for _ in range(n_corrector):
        if corrector_type == "substitution":
            _substitution_corrector_sweep(model, canvas, corrector_frac, guide, greedy,
                                          temperature, min_len=min_len)
        else:
            _corrector_sweep(model, canvas, corrector_frac, guide, greedy, temperature,
                             min_len=min_len)

    if was_training:
        model.train()
    if stats is not None:
        stats.update(n_steps=n_steps, n_subst=n_subst_done, subst_per_residue=subst_per_residue,
                     subst_per_seq=n_subst_done / max(B, 1))
    # (B,L) at one track keeps every pre-structure caller working unchanged.
    return (canvas if K > 1 else canvas[:, 0]), lengths_of(canvas, cfg)


def decode_struct(canvas, cfg, min_len: int = 0, max_len: Optional[int] = None):
    """The generated 3Di track as strings, aligned 1:1 with decode_seqs' output on the same canvas.

    That alignment is what makes the self-consistency check possible: fold the amino-acid sequence,
    3Di-encode the resulting structure, and compare it to the string this returns. The model told
    you what fold it was building; ESMFold tells you what fold the sequence actually specifies."""
    from .data import DI
    st = struct_track(canvas)
    if st is None:
        return [], 0
    return _decode_track(st, aa_track(canvas), cfg, DI, min_len, max_len)


def decode_seqs(canvas, cfg, min_len: int = 0, max_len: Optional[int] = None):
    """Canvas rows -> amino-acid strings, truncated at the first EOS/PAD.

    Only ids 0..19 map to residues; a MASK that survived decoding is dropped. Rows outside
    [min_len, max_len] are OMITTED rather than clipped -- reporting a statistic for a molecule that
    was never generated is worse than reporting one fewer sample. Returns (sequences, n_skipped).
    """
    from .blosum import AA
    return _decode_track(aa_track(canvas), aa_track(canvas), cfg, AA, min_len, max_len)


def _decode_track(track, aa, cfg, alphabet, min_len, max_len):
    """Decode `track` up to the boundary, but judge the length filter on `aa`.

    Both tracks are filtered on the SEQUENCE track's length so decode_seqs and decode_struct keep
    and drop exactly the same rows: their outputs are index-aligned, which is what the
    self-consistency check needs. Applied to the aa track this is the original decode_seqs,
    character for character."""
    out, skipped = [], 0
    for trow, arow in zip(track.cpu().tolist(), aa.cpu().tolist()):
        cut = len(arow)
        for i, t in enumerate(arow):
            if t == cfg.eos_token_id or t == cfg.pad_token_id:
                cut = i
                break
        n_res = sum(1 for t in arow[:cut] if 0 <= t < 20)     # MASK survivors do not count
        if n_res >= min_len and (max_len is None or n_res <= max_len):
            out.append("".join(alphabet[t] for t in trow[:cut] if 0 <= t < len(alphabet)))
        else:
            skipped += 1
    return out, skipped


def write_fasta(seqs, path, prefix="sample"):
    with open(path, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f">{prefix}_{i} len={len(s)}\n{s}\n")
    return len(seqs)
