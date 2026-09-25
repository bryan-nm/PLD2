"""Preference tuning: IPO by default, DPO wired for comparison. ESM3's IRPO with three departures.

    mpiexec -n 12 python -m src.align --device xpu                 # IPO
    mpiexec -n 12 python -m src.align --device xpu --loss dpo      # the comparison
    mpiexec -n 12 python -m src.align --device xpu --alpha 0       # the SFT baseline

THE LOSS. ESM3's IRPO (Appendix A.4.1) is a supervised term plus a contrastive one:

    L = L_NLL + alpha * L_contrastive,     h = [log pi(y_w)-log pi_ref(y_w)] - [log pi(y_l)-log pi_ref(y_l)]
    DPO:  L_contrastive = -log sigmoid(beta * h)
    IPO:  L_contrastive = (h - margin)^2

WHY IPO IS THE DEFAULT. DPO's log-sigmoid is unbounded, so it keeps rewarding a larger margin
forever; Azar et al.'s point is that its implicit KL constraint collapses when preferences are
near-deterministic, because the sigmoid saturates and the effective penalty goes to zero. Ours are
deterministic BY CONSTRUCTION -- pairs come from a hard metric gap, not noisy human labels -- so we
sit exactly in that regime. Three further things make drift worse for us than for ESM3: positives
that are good only relative to their own prompt rather than in absolute terms; a reward that is a
folder's opinion about a generated sequence, with a known degenerate direction; and a starting
policy whose own likelihood is ANTI-correlated with success, so the required update is an inversion
rather than a sharpening and the distance travelled is large. IPO's squared loss has a finite
optimum and stops pushing. --loss dpo runs the comparison on the same pairs.

THE MARGIN IS PER POSITION. objective.surrogate_logp returns a mean over scored positions, so
`margin` is a nats-per-token budget: length-invariant, and something you can set on purpose.

WHAT TO WATCH, AND IT IS NOT THE MARGIN. DPO's characteristic pathology is driving log pi(y_w) DOWN
while the margin goes UP -- it lowers the preferred sample's likelihood, just more slowly than the
rejected one's. The margin looks healthy throughout. So the log line reports the ABSOLUTE
log-likelihood of the winner, and every eval reports the surrogate NLL on held-out NATURAL
sequences. That second number is the decisive one: if it rises, the policy has left the data
manifold, whatever the preference metrics say.

FOUR LIKELIHOODS, TWO FORWARDS. (y_w, y_l) are concatenated into one batch, so the policy costs one
forward with gradient and the reference one without. The reference is held in memory rather than
precomputed because the mask is redrawn every epoch; objective.surrogate_mask is nonetheless a pure
function of (pair_id, epoch), so precomputing becomes possible later without changing semantics.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from config import CFG, CKPT_DIR, AFDB_SHARDS
from .data import AA, DI, ProteinShards
from .dist import (average_gradients, barrier, broadcast_parameters, cleanup, init_distributed,
                   preallocate_grad_buffer)
from .model import LoopedDiffusionLM, count_params
from .objective import surrogate_logp, surrogate_mask
from .prompts import _unhex, read_manifest
from .train import find_latest_ckpt, save_checkpoint

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None

_AA_ID = {c: i for i, c in enumerate(AA)}
_DI_ID = {c: i for i, c in enumerate(DI)}


def encode_canvas(seq: str, di, mcfg, canvas: int, n_tracks: int):
    """(K, canvas) long, laid out exactly as training does: [AA* EOS PAD*] in track 0, and the 3Di
    track padded from the boundary INCLUSIVE with no EOS of its own."""
    L = len(seq)
    K = max(1, n_tracks)
    tok = torch.full((K, canvas), mcfg.pad_token_id, dtype=torch.long)
    tok[0, :L] = torch.tensor([_AA_ID.get(c, 0) for c in seq], dtype=torch.long)
    tok[0, L] = mcfg.eos_token_id
    if K > 1 and di:
        n = min(L, len(di))
        tok[1, :n] = torch.tensor([_DI_ID.get(c, 0) for c in di[:n]], dtype=torch.long)
    return tok


class PairSet:
    """pairs.jsonl joined to the prompt manifest, materialised lazily.

    Pairs are held as (tokens_w, tokens_l, generated-mask) triples built on demand rather than up
    front: at a thousand nodes the manifest is the only thing every rank reads, and a materialised
    canvas is 512 longs a side.
    """

    def __init__(self, pairs_path, manifest_path, mcfg, canvas, n_tracks):
        self.rows = []
        with open(pairs_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise SystemExit(f"{pairs_path} is empty. Run `python -m src.preference` first.")
        self.prompts = {r["pid"]: r for r in read_manifest(manifest_path)}
        self.mcfg, self.canvas, self.K = mcfg, canvas, n_tracks
        self.skipped = 0
        keep = []
        for r in self.rows:
            p = self.prompts.get(r["pid"])
            # A completion whose length disagrees with its prompt cannot be scored against that
            # prompt's generated-position mask. The prompt owns the length, so this should be
            # impossible; count it rather than trusting that.
            if p is None or len(r["w"]["seq"]) != p["L"] or len(r["l"]["seq"]) != p["L"]:
                self.skipped += 1
                continue
            keep.append(r)
        self.rows = keep

    def __len__(self):
        return len(self.rows)

    def get(self, i):
        r = self.rows[i]
        p = self.prompts[r["pid"]]
        L = int(p["L"])
        yw = encode_canvas(r["w"]["seq"], r["w"].get("di"), self.mcfg, self.canvas, self.K)
        yl = encode_canvas(r["l"]["seq"], r["l"].get("di"), self.mcfg, self.canvas, self.K)
        gen = torch.zeros(self.canvas, dtype=torch.bool)
        gen[:L] = torch.from_numpy(_unhex(p["masked"], L))
        return r, yw, yl, gen


def _keep_best(model, lr_sched, step, nat, out_dir, env):
    """Copy the current weights to policy/best.pt, with a one-line pointer beside it.

    A separate name so the ckpt_*.pt rotation cannot reach it. Weights only -- the same reasoning
    as the rolling saves, and nothing resumes from this file; it is a candidate to EVALUATE.
    """
    if not env.is_main:
        return
    from .train import _atomic_save
    os.makedirs(out_dir, exist_ok=True)
    _atomic_save({"model": model.state_dict(), "sched": lr_sched.state_dict(), "step": step},
                 os.path.join(out_dir, "best.pt"))
    _atomic_json(os.path.join(out_dir, "best.json"),
                 {"step": step, "natural_nll": nat,
                  "why": "lowest held-out natural NLL; the drift monitor has no sampling noise, "
                         "so this is the least-drifted policy of the run. A candidate to "
                         "evaluate, not a verdict."})


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def recommended_ranks(n_pairs, steps, epochs=2.0, pairs_per_rank=1, ppn=12):
    """How many ranks the tuner should use, rounded to whole nodes.

    steps * ranks * pairs_per_rank = n_pairs * epochs. Round 1 ran phase 5 on the whole 192-rank
    allocation and consumed 33.8 epochs of 5,680 pairs; the drift monitor turned over at 8.4. The
    fix is not more data, it is a smaller tuner -- and at 256 nodes it stops being optional, because
    3,072 ranks x 500 steps is 1.5M pair draws.
    """
    r = n_pairs * float(epochs) / max(int(steps) * max(int(pairs_per_rank), 1), 1)
    return max(ppn, int(round(r / ppn)) * ppn)


def loss_scale(acfg, kind, nll_weight=None):
    """-> (label, h_value, note) describing where each loss stops pushing. Printed at startup.

    IPO has an EQUILIBRIUM: its squared term's gradient opposes the anchor's, and they balance at
        h* = m + w_nll / (2*alpha)
    DPO has NONE. Both its term and the anchor push lp_w the same way, so nothing balances -- which
    is exactly Azar's unboundedness argument and the reason IPO is the default here. What stops DPO
    is the sigmoid saturating, once h is large against 1/beta.
    """
    w = acfg.nll_weight if nll_weight is None else nll_weight
    if kind == "ipo":
        a = acfg.alpha
        if a <= 0:
            return ("no contrastive term", float("nan"), "alpha=0: this is the SFT baseline")
        return ("equilibrium h*", acfg.ipo_margin + w / (2.0 * a),
                f"= margin {acfg.ipo_margin} + nll_weight {w} / (2*alpha {a})")
    return ("saturates for h >>", 1.0 / max(acfg.beta, 1e-9),
            "DPO has no equilibrium; the term simply fades once beta*h is large")


def contrastive(h, acfg, kind):
    """h is the per-position log-ratio difference; -> (B,) loss."""
    if kind == "ipo":
        # Finite optimum at h = margin. Bounded by construction: once the pair is separated by the
        # target amount the gradient is zero, which is the entire reason this is the default.
        return (h - acfg.ipo_margin) ** 2
    if kind == "dpo":
        return -torch.nn.functional.logsigmoid(acfg.beta * h)
    raise ValueError(f"unknown loss {kind!r}")


@torch.no_grad()
def natural_nll(model, shards, mcfg, canvas, n, device, seed=0, batch=8):
    """Surrogate NLL on held-out NATURAL sequences. The drift monitor.

    Every preference metric can look healthy while the policy walks off the data manifold; this is
    the number that notices. Fixed mask draw, so successive evals are comparable to each other and
    not just to zero.
    """
    if shards is None or len(shards) == 0:
        return float("nan")            # no corpus reachable; the caller reports it as unavailable
    model.eval()
    ys, ms = [], []
    for j in range(min(n, len(shards))):
        aa, di = shards.get_pair(j)
        L = len(aa) - 1
        if L < 30 or L + 1 > canvas:
            continue
        seq = "".join(AA[t] for t in aa[:L])
        dis = "".join(DI[t] for t in di[:L]) if di is not None else None
        ys.append(encode_canvas(seq, dis, mcfg, canvas, mcfg.n_tracks))
        gen = torch.zeros(canvas, dtype=torch.bool)
        gen[:L] = True
        # A FIXED rate and a fixed per-sequence seed, so successive evals differ only by the
        # policy. A resampled mask would put schedule noise on top of the drift being measured,
        # and the drift is the smaller quantity.
        ms.append(surrogate_mask(f"natural{j}", gen, canvas, device, epoch=seed, rate=0.5))
    vals = []
    amp = device.type in ("xpu", "cuda")
    for i in range(0, len(ys), batch):
        y = torch.stack(ys[i:i + batch]).to(device)
        m = torch.stack(ms[i:i + batch])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            vals += (-surrogate_logp(model, y, m, mcfg)).tolist()
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg, dcfg = CFG.align, CFG.data
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--dir", default=None, help="round directory (default: config align.round_dir)")
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--ref-ckpt", default=None,
                    help="the reference policy. Defaults to the checkpoint being tuned, which is "
                         "what IRPO means by pi_ref: the frozen base model at the same scale.")
    ap.add_argument("--ckpt", default=None, help="policy to tune (default: latest in CKPT_DIR)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default: <round>/policy)")
    ap.add_argument("--loss", default=acfg.loss, choices=("ipo", "dpo"))
    ap.add_argument("--alpha", type=float, default=None,
                    help="weight on the contrastive term. 0 = the SFT baseline (ESM3's A.4.6). "
                         "Default depends on the loss: the two need different scales because IPO's "
                         "term is quadratic in h and DPO's is a saturating sigmoid.")
    ap.add_argument("--beta", type=float, default=acfg.beta)
    ap.add_argument("--margin", type=float, default=acfg.ipo_margin, help="IPO target, nats/position")
    ap.add_argument("--epochs", type=float, default=acfg.target_epochs,
                    help="only used to report how far the tuner's rank count is from ideal")
    ap.add_argument("--steps", type=int, default=acfg.steps)
    ap.add_argument("--lr", type=float, default=acfg.lr)
    ap.add_argument("--pairs-per-rank", type=int, default=acfg.pairs_per_rank)
    ap.add_argument("--score-struct", action="store_true", default=acfg.score_struct)
    ap.add_argument("--shards", default=AFDB_SHARDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny randomly-initialised model, no checkpoint: exercises every path")
    ap.add_argument("--no-ipex", action="store_true")
    a = ap.parse_args()
    if a.alpha is None:
        a.alpha = acfg.ipo_alpha if a.loss == "ipo" else acfg.dpo_alpha
    acfg.loss, acfg.alpha, acfg.beta, acfg.ipo_margin = a.loss, a.alpha, a.beta, a.margin

    env = init_distributed(a.device)
    dev = env.device
    torch.manual_seed(a.seed + env.rank)
    mcfg = CFG.model_config()
    if a.smoke:
        from .align_sample import _smoke
        mcfg, dcfg = _smoke(mcfg, dcfg, a.canvas)
        acfg.eval_every, acfg.drift_natural_n, acfg.warmup_steps = 2, 4, 1
    elif a.canvas:
        dcfg.canvas = a.canvas
    rdir = a.dir or acfg.round_dir
    out_dir = a.out or os.path.join(rdir, "policy")

    data = PairSet(a.pairs or os.path.join(rdir, "pairs.jsonl"),
                   a.manifest or os.path.join(rdir, "prompts.jsonl"),
                   mcfg, dcfg.canvas, mcfg.n_tracks)

    # --- policy + frozen reference --------------------------------------------------------
    ckpt = ref_ckpt = "<smoke: random init>"
    model = LoopedDiffusionLM(mcfg).to(dev)
    if not a.smoke:
        ckpt = a.ckpt or find_latest_ckpt(CKPT_DIR)
        if not ckpt or not os.path.exists(ckpt):
            raise SystemExit(f"no checkpoint (looked at {a.ckpt or CKPT_DIR})")
        ref_ckpt = a.ref_ckpt or ckpt
        st = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(st["model"])
        model.to(dev)                   # see train.py: ipex leaves a CPU state dict on the host
    broadcast_parameters(model)

    ref = None
    if a.alpha != 0:
        # THE REFERENCE IS A SEPARATE COPY OF THE SAME WEIGHTS, frozen. Under IPO/DPO it is what
        # "how far have we moved" is measured against, so it must be the policy as it stood at the
        # start of this round -- not the base model of some earlier round.
        ref = LoopedDiffusionLM(mcfg).to(dev)
        ref.load_state_dict(model.state_dict())
        if not a.smoke and ref_ckpt != ckpt:
            rst = torch.load(ref_ckpt, map_location="cpu", mmap=True, weights_only=True)
            ref.load_state_dict(rst["model"])
            del rst
        ref.to(dev).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    if not a.smoke:
        del st

    opt = (torch.optim.RMSprop(model.parameters(), lr=a.lr) if acfg.optimizer == "rmsprop"
           else torch.optim.AdamW(model.parameters(), lr=a.lr))
    applied_ipex = ipex is not None and dev.type == "xpu" and not a.no_ipex and not a.smoke
    if applied_ipex:
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)
        if ref is not None:
            ref = ipex.optimize(ref, dtype=torch.bfloat16)
    warm = max(1, acfg.warmup_steps)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm)
        * (0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, a.steps - warm))))))

    stride_needed = env.world_size * a.pairs_per_rank
    if len(data) < stride_needed:
        raise SystemExit(
            f"{len(data)} pairs but one optimizer step consumes {stride_needed} "
            f"({env.world_size} ranks x {a.pairs_per_rank}). Generate more prompts, lower "
            f"--pairs-per-rank, or run on fewer nodes -- silently wrapping would have some ranks "
            f"train on the same pair in the same step, which double-counts its gradient.")

    n_grad = preallocate_grad_buffer(model, dev)
    # The drift monitor reads NATURAL sequences from the pretraining corpus, not from the alignment
    # reference set: it is asking whether the policy is still a model of proteins, and the cleanest
    # answer comes from data that has nothing to do with this round.
    shards = ProteinShards(a.shards, mcfg.eos_token_id, split="holdout",
                           holdout_stride=max(dcfg.holdout_stride, 2)) \
        if os.path.isdir(a.shards) else None

    if env.is_main:
        os.makedirs(out_dir, exist_ok=True)
        n_succ = sum(r["winner_success"] for r in data.rows)
        print(f"[align] {len(data):,} pairs from {len({r['pid'] for r in data.rows}):,} prompts"
              + (f" ({data.skipped} skipped for a length mismatch)" if data.skipped else ""),
              flush=True)
        print(f"[align] policy    {ckpt}\n[align] reference {ref_ckpt}"
              + ("   (alpha=0: SFT baseline, no reference model loaded)" if ref is None else ""),
              flush=True)
        print(f"[align] loss={a.loss} alpha={a.alpha} beta={a.beta} "
              + (f"margin={a.margin} nats/position " if a.loss == "ipo" else "")
              + f"| nll_weight={acfg.nll_weight} | {acfg.optimizer} lr={a.lr} "
              f"warmup={acfg.warmup_steps} clip={acfg.grad_clip}", flush=True)
        # WHERE THIS LOSS STOPS PUSHING, stated before the run rather than discovered from the log.
        # Round 1 used ESM3's alpha=0.8 against per-position log-likelihoods, which puts the IPO
        # equilibrium at h*=0.665 -- sixteen times the margin -- so the margin never bound and the
        # run was effectively SFT with a nudge. That is now impossible to ship unnoticed.
        lbl, val, note = loss_scale(acfg, a.loss)
        print(f"[align] {lbl} {val:.4g}   ({note})", flush=True)
        if a.loss == "ipo" and a.alpha > 0 and val > 3 * max(a.margin, 1e-9):
            print(f"[align] WARNING: h* is {val / max(a.margin, 1e-9):.0f}x the margin, so the "
                  f"margin will NOT bind and this run is closer to SFT than to IPO. For a "
                  f"tolerance of `tol` above the margin, use alpha = nll_weight/(2*tol) = "
                  f"{acfg.nll_weight / (2 * acfg.ipo_tol):.0f}.", flush=True)

        draws = a.steps * a.pairs_per_rank * env.world_size
        eps = draws / max(len(data), 1)
        want = recommended_ranks(len(data), a.steps, a.epochs, a.pairs_per_rank)
        print(f"[align] {a.steps} steps x {a.pairs_per_rank} pair(s)/rank x {env.world_size} ranks "
              f"= {draws:,} pair draws = {eps:.1f} epochs over {len(data):,} pairs | "
              f"params={count_params(model) / 1e6:.0f}M ipex={'ON' if applied_ipex else 'OFF'} "
              f"grad buffer {4 * (n_grad + 1) / 1e6:.0f}MB", flush=True)
        if abs(eps - a.epochs) > 0.5 * a.epochs:
            print(f"[align] {'WARNING: ' if eps > 5 else ''}for {a.epochs} epochs at {a.steps} "
                  f"steps you want ~{want} ranks, not {env.world_size}. Pairs consumed per step is "
                  f"world_size x pairs_per_rank, so the tuner's size sets the epoch count -- run "
                  f"phase 5 on a SUBSET of the allocation (scripts/align.pbs: ALIGN_RANKS).",
                  flush=True)
        print(f"[align] winner clears the absolute bar in {n_succ:,}/{len(data):,} pairs; "
              f"success_weight={acfg.success_weight}"
              f"{'  (OFF)' if acfg.success_weight == 1.0 else ''}", flush=True)
        print("[align] WATCH logpw (absolute), not just h: DPO's classic failure lowers the "
              "winner's own likelihood while the margin rises. natural NLL is the decisive one.",
              flush=True)

    use_amp = dev.type in ("xpu", "cuda")
    # THE BASELINE, measured before a single update. Every later reading of the drift monitor is
    # only interpretable against this one, and taking it at step eval_every instead means the first
    # point already contains 50 steps of movement.
    nat = []
    if env.is_main:
        nat.append((0, natural_nll(model, shards, mcfg, dcfg.canvas, acfg.drift_natural_n, dev)))
        print(f"[align] step 0 baseline: natural NLL {nat[0][1]:.4f} nats/position"
              if nat[0][1] == nat[0][1] else
              f"[align] natural NLL UNAVAILABLE -- no corpus at {a.shards}, drift monitor OFF. "
              f"Do not run a real round like this.", flush=True)

    # Ranks partition ONE shuffled order, so every pair is seen by exactly one rank per epoch and
    # an optimizer step covers world_size * pairs_per_rank distinct pairs. `base` walks the order in
    # strides of that width; rank r takes the r-th slot within each stride.
    order = np.random.default_rng(a.seed).permutation(len(data))
    stride = max(1, env.world_size * a.pairs_per_rank)
    base, epoch = 0, 0
    t0, acc, n_acc, last = time.perf_counter(), {}, 0, {}
    model.train()

    for step in range(a.steps):
        if base + stride > len(order):
            epoch += 1
            order = np.random.default_rng(a.seed + epoch).permutation(len(data))
            base = 0
        idx = [int(order[base + env.rank * a.pairs_per_rank + j])
               for j in range(a.pairs_per_rank)]
        base += stride

        ys, masks, weights = [], [], []
        for i in idx:
            r, yw, yl, gen = data.get(i)
            m = surrogate_mask(r["pair_id"], gen, dcfg.canvas, dev, epoch=epoch)
            ys.append((yw, yl))
            masks.append(m)
            weights.append(float(r["weight"]))
        B = len(ys)
        # [winners; losers] in one batch, so the policy costs one forward and the reference one.
        y = torch.stack([w for w, _ in ys] + [l for _, l in ys]).to(dev)
        mk = torch.stack(masks + masks)
        wt = torch.tensor(weights, device=dev, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            lp = surrogate_logp(model, y, mk, mcfg, score_struct=a.score_struct)
        lp_w, lp_l = lp[:B], lp[B:]
        nll = -(lp_w * wt).sum() / wt.sum()

        if ref is None:
            loss, h = acfg.nll_weight * nll, torch.zeros((), device=dev)
        else:
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                                 enabled=use_amp):
                rlp = surrogate_logp(ref, y, mk, mcfg, score_struct=a.score_struct)
            h = (lp_w - rlp[:B]) - (lp_l - rlp[B:])
            con = (contrastive(h, acfg, a.loss) * wt).sum() / wt.sum()
            loss = acfg.nll_weight * nll + a.alpha * con
            h = h.mean()
        loss.backward()

        nonfinite = average_gradients(model)
        if not nonfinite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        lr_sched.step()

        for k, v in (("loss", loss), ("nll", nll), ("h", h),
                     ("logpw", lp_w.mean()), ("logpl", lp_l.mean()),
                     ("acc", (lp_w > lp_l).float().mean())):
            acc[k] = acc.get(k, 0.0) + float(v.detach())
        n_acc += 1

        if env.is_main and (step % 10 == 0 or step == a.steps - 1):
            d = {k: v / max(n_acc, 1) for k, v in acc.items()}
            last = dict(d)
            el = time.perf_counter() - t0
            print(f"step {step:>5} | loss {d['loss']:.4f} nll {d['nll']:.4f} "
                  f"| h {d['h']:+.4f}"
                  + (f"/{a.margin}" if a.loss == "ipo" else "")
                  + f" | logpw {d['logpw']:+.4f} logpl {d['logpl']:+.4f} "
                  f"| win {d['acc']:.0%} | lr {lr_sched.get_last_lr()[0]:.2e} "
                  f"| {el / max(step + 1, 1):.2f}s/step"
                  + ("  SKIPPED (non-finite)" if nonfinite else ""), flush=True)
            acc, n_acc = {}, 0

        if acfg.eval_every and step > 0 and step % acfg.eval_every == 0:
            # Rank 0 only. There is no collective in here and every rank would compute the same
            # number from the same holdout with the same fixed masks; the others simply wait at the
            # next all-reduce.
            nn_ = natural_nll(model, shards, mcfg, dcfg.canvas, acfg.drift_natural_n,
                              dev) if env.is_main else float("nan")
            if env.is_main:
                nat.append((step, nn_))
                best = min((v for _, v in nat if v == v), default=float("nan"))
                # KEEP THE DRIFT-MINIMAL CHECKPOINT, because the rolling rotation deletes it.
                # A 500-step run saves ten times and keep_last discards all but the newest few, so
                # the end-of-run message named ckpt_00000100.pt as the one to evaluate after
                # already removing it. best.pt is written the moment the minimum moves.
                if nn_ == nn_ and nn_ <= best:
                    _keep_best(model, lr_sched, step, nn_, out_dir, env)
                print(f"[align] step {step}: natural NLL {nn_:.4f} nats/position "
                      f"({nn_ - nat[0][1]:+.4f} vs baseline, best {best:.4f}) -- rising means the "
                      f"policy is leaving the manifold, whatever h is doing" if nn_ == nn_ else
                      f"[align] step {step}: natural NLL UNAVAILABLE", flush=True)
            # keep_last=1 and no optimizer state: these runs are minutes long and restart from
            # the base checkpoint, so the only files worth their 5.4GB are the newest and the best.
            save_checkpoint(model, opt, lr_sched, step, out_dir, env,
                            keep_last=1, save_optimizer=False)

    if env.is_main:
        nn_ = natural_nll(model, shards, mcfg, dcfg.canvas, acfg.drift_natural_n, dev)
        nat.append((a.steps, nn_))
        vals = [(st, v) for st, v in nat if v == v]
        best_step, best = min(vals, key=lambda t: t[1]) if vals else (0, float("nan"))
        print(f"[align] {'baseline' if a.steps == 0 else 'final'} natural NLL {nn_:.4f}"
              if nn_ == nn_ else "[align] natural NLL unavailable (no corpus)", flush=True)
        if vals and a.steps:
            # The drift monitor is the only number here with NO sampling noise: fixed sequences,
            # fixed masks, so the only thing changing between readings is the policy. When its
            # minimum is not the last step, that step is a checkpoint worth evaluating.
            print(f"[align] drift monitor: baseline {nat[0][1]:.4f} -> min {best:.4f} at step "
                  f"{best_step} -> final {nn_:.4f} ({nn_ - nat[0][1]:+.4f})", flush=True)
            if best_step not in (a.steps, a.steps - 1) and best_step > 0:
                print(f"[align] the minimum is not the last step: it is at {best_step}, saved as "
                      f"{os.path.join(out_dir, 'best.pt')}. The drift monitor has no sampling "
                      f"noise, so that is the least-drifted policy of the run -- a candidate to "
                      f"evaluate, not a verdict. The fold numbers settle it.", flush=True)
        # One machine-readable record per policy, so src/align_compare.py can put the tuning
        # metrics next to the generation metrics without anyone re-reading a log.
        _atomic_json(os.path.join(out_dir, "metrics.json"), {
            "loss": a.loss, "alpha": a.alpha, "beta": a.beta, "margin": a.margin,
            "nll_weight": acfg.nll_weight, "optimizer": acfg.optimizer, "lr": a.lr,
            "steps": a.steps, "warmup": acfg.warmup_steps, "ranks": env.world_size,
            "pairs_per_rank": a.pairs_per_rank, "n_pairs": len(data), "epochs": eps,
            "equilibrium": loss_scale(acfg, a.loss)[1],
            "ckpt": ckpt, "ref_ckpt": ref_ckpt,
            "nat_nll": [[st, v] for st, v in nat],
            "nat_nll_baseline": nat[0][1], "nat_nll_final": nn_,
            "nat_nll_min": best, "nat_nll_min_step": best_step,
            "h_final": last.get("h"), "win_final": last.get("acc"),
            "logpw_final": last.get("logpw"), "logpl_final": last.get("logpl"),
            "nll_final": last.get("nll"),
        })
    if a.steps == 0:
        # --steps 0 is the BASELINE measurement: the drift monitor on the untouched policy, which
        # is the only thing a later reading of it can be compared against. It must not write a
        # checkpoint -- an untuned policy saved into the round's policy/ directory is exactly the
        # thing a later `find_latest_ckpt` would pick up and mistake for a tuned one.
        if env.is_main:
            print("[align] --steps 0: baseline only, no checkpoint written.", flush=True)
        barrier()
        cleanup()
        return
    save_checkpoint(model, opt, lr_sched, a.steps - 1, out_dir, env,
                    keep_last=1, save_optimizer=False)
    if env.is_main:
        import glob as _g
        gb = sum(os.path.getsize(f) for f in _g.glob(os.path.join(out_dir, "*.pt"))) / 1e9
        print(f"[align] {out_dir} holds {gb:.1f}GB of checkpoints (weights only, newest + best).",
              flush=True)
        print(f"[align] done. Tuned policy in {out_dir}. Evaluate it the way the base model was "
              f"evaluated -- and generate the NEXT round's pairs from it, on a fresh prompt set: "
              f"pairs drawn from a policy that no longer exists are not merely stale, they are "
              f"mislabelled, because the reference term measures movement away from a model that "
              f"is no longer the one being moved.", flush=True)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
