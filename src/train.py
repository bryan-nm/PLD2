"""Training entry point for PLD2 (Aurora XPU / oneCCL).

Wires config -> dist -> data -> model -> objective, plus the training-time generative eval.

Manual data-parallel (broadcast + one coalesced grad all-reduce) rather than DDP: every parameter is
used by every row of every batch -- PLD2 has no conditional pathway at all -- so the param-grad set
is identical across ranks every step, which is what makes the single collective safe.

TRAINING-TIME EVAL (instruction 8). Every `eval_every` steps, every rank decodes `eval_n` sequences
at temperature 0.5 and computes the sequence-level metrics; the counts are summed in ONE all-reduce
and rank 0 prints them. The first `eval_fasta_ranks` ranks also write their samples to
SAMPLES_DIR/step_XXXXXXXX.rankNNN.fasta. Folding is NOT done here: src/fold_fasta.py picks those
FASTAs up in a separate single-rank process. That separation is not a preference -- ESMFold on
Aurora aborts the process outright with a GPU page fault often enough that it must not be able to
take the trainer with it, and it installs process-global monkey-patches on torch.linalg and F.linear
that have no business inside an ipex-optimised training process.

Launch (Aurora): scripts/train.pbs. Local smoke:
    PLD2_UNIREF_SHARDS=/path/to/shards python -m src.train --smoke
"""
from __future__ import annotations
import argparse
import io
import math
import os
import time

import torch

from config import CFG, UNIREF_SHARDS, BLOSUM_MAT, CKPT_DIR, SAMPLES_DIR
from .blosum import substitution_kernel, uniform_substitution_kernel
from .corruption import CorruptionSchedule
from .data import ProteinShards, ShardDataset, StepBatchSampler, make_collate
from .dist import (init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
                   broadcast_checkpoint_bytes, preallocate_grad_buffer, preallocate_stats_buffer,
                   allreduce_stats)
from .metrics import flatten_kmer, kmer_counts, kmer_line, lcr_counts, unflatten_kmer
from .model import LoopedDiffusionLM, count_params
from .objective import training_step
from .sampler import decode_seqs, generate, write_fasta

try:
    import intel_extension_for_pytorch as ipex
    # IPEX logs "split master weight ... only support sgd" and two BatchNorm-folding lines at INFO on
    # every ipex.optimize call, on every rank. Its logger is named "IPEX", not the module name.
    import logging as _logging
    _logging.getLogger("IPEX").setLevel(_logging.WARNING)
except Exception:
    ipex = None


def lr_lambda(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def build_schedule(ocfg, mcfg, device):
    """The unified absorbing+substitution corruption process (src/corruption.py)."""
    n = mcfg.vocab_size - 1                       # non-MASK states; MASK is the absorbing one
    if ocfg.sub_kernel == "blosum":
        if not os.path.exists(BLOSUM_MAT):
            raise FileNotFoundError(
                f"sub_kernel='blosum' needs {BLOSUM_MAT}. Set PLD2_BLOSUM, or use "
                f"sub_kernel='uniform'.")
        B = substitution_kernel(BLOSUM_MAT, n, temp=ocfg.sub_kernel_temp)
    elif ocfg.sub_kernel == "uniform":
        B = uniform_substitution_kernel(n)
    else:
        raise ValueError(f"unknown sub_kernel {ocfg.sub_kernel!r}")
    return CorruptionSchedule(B, mcfg.vocab_size, mcfg.mask_token_id,
                              betas=ocfg.betas, T=ocfg.d3pm_T, device=device)


# --- checkpointing (rank 0 writes; atomic via tmp+rename; latest.txt is the resume pointer) ---
def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, opt, lr_sched, step, ckpt_dir, env, keep_last=3):
    if not env.is_main:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{step:08d}.pt")
    _atomic_save({"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": lr_sched.state_dict(), "step": step}, path)
    tmp = os.path.join(ckpt_dir, "latest.txt.tmp")   # update the pointer only after a full write
    with open(tmp, "w") as f:
        f.write(os.path.basename(path))
    os.replace(tmp, os.path.join(ckpt_dir, "latest.txt"))
    print(f"[ckpt] saved {path}", flush=True)
    if keep_last:                                    # rotate: model+opt is ~650MB a copy
        import glob
        for old in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt")))[:-keep_last]:
            try:
                os.remove(old)
            except OSError:
                pass


def find_latest_ckpt(ckpt_dir):
    p = os.path.join(ckpt_dir, "latest.txt")
    if not os.path.exists(p):
        return None
    full = os.path.join(ckpt_dir, open(p).read().strip())
    return full if os.path.exists(full) else None


def _device_sync(dev):
    # XPU/CUDA ops are async; sync before timing so tok/s reflects real compute, not enqueue time.
    if dev.type == "xpu":
        torch.xpu.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def _peak_mem_gb(dev):
    if dev.type == "xpu":
        return torch.xpu.max_memory_allocated() / 1e9
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0


# --------------------------------------------------------------------------------------
# Training-time generative eval
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generative_eval(model, dev, mcfg, ocfg, step, env, samples_dir, use_amp):
    """Decode eval_n sequences per rank at temperature 0.5, write FASTA, aggregate the metrics.

    EVERY rank runs this, in lockstep, because the metric all-reduce is a collective. Only the first
    `eval_fasta_ranks` ranks write files. Returns rank 0's aggregated dict (other ranks get the same
    structure with their own totals, which they do not print).
    """
    t0 = time.perf_counter()
    # Seeded per (run, step, rank): every rank draws different sequences, and the same checkpoint
    # re-evaluated at the same step draws the same ones. The previous RNG state is restored
    # afterwards so that whether an eval ran does not perturb the training corruption stream --
    # otherwise two runs with different eval_every would diverge for a reason no one would suspect.
    rng_state = torch.get_rng_state()
    torch.manual_seed(ocfg.seed + 100003 * (step + 1) + env.rank)

    gen_stats = {}
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
        canvas, lengths = generate(
            model, Lmax=ocfg.eval_canvas, batch_size=ocfg.eval_n, n_steps=ocfg.eval_steps,
            device=str(dev), temperature=ocfg.sample_temperature,
            gumbel_temp=ocfg.sample_gumbel_temp, greedy=False,
            n_corrector=ocfg.sample_corrector, corrector_type=ocfg.sample_corrector_type,
            eos_first=ocfg.sample_eos_first, min_len=ocfg.sample_min_len,
            eos_temp=ocfg.sample_eos_temp, rep_penalty=ocfg.sample_rep_penalty,
            rep_periods=ocfg.sample_rep_periods, max_run=ocfg.sample_max_run,
            subst_per_residue=ocfg.sample_subst_per_residue, stats=gen_stats)

    seqs, _ = decode_seqs(canvas, mcfg)
    if env.rank < ocfg.eval_fasta_ranks:
        os.makedirs(samples_dir, exist_ok=True)
        path = os.path.join(samples_dir, f"step_{step:08d}.rank{env.rank:03d}.fasta")
        tmp = path + ".tmp"                          # tmp+rename so the folding watcher, which polls
        write_fasta(seqs, tmp, prefix=f"s{step}r{env.rank}")   # this directory, never reads a
        os.replace(tmp, path)                                  # half-written file

    lcr, lcr_tot = lcr_counts(seqs)
    n_no_eos = sum(1 for v in lengths if v >= ocfg.eval_canvas)
    head = [float(len(lengths)), float(sum(lengths)), float(sum(v * v for v in lengths)),
            float(n_no_eos), float(lcr), float(lcr_tot), float(gen_stats.get("n_subst", 0))]
    flat = allreduce_stats(head + flatten_kmer(kmer_counts(seqs, ocfg.kmer_ks), ocfg.kmer_ks), dev)

    n, s1, s2, no_eos, lcr, lcr_tot, n_subst = flat[:7]
    torch.set_rng_state(rng_state)
    n = max(int(n), 1)
    mean = s1 / n
    var = s2 / n - mean * mean
    return {"n": n, "mean": mean, "sd": var ** 0.5 if var > 0 else 0.0,
            "no_eos": int(no_eos), "lcr": lcr / max(lcr_tot, 1), "subst": n_subst / n,
            "kmer": unflatten_kmer(flat[7:], ocfg.kmer_ks),
            "seconds": time.perf_counter() - t0}


def _eval_line(step, m, ocfg):
    rev = f" | sub {m['subst']:.0f}/seq" if ocfg.sample_subst_per_residue > 0 else ""
    return (f"[eval] step {step:>7} | len {m['mean']:.1f}+-{m['sd']:.1f} | "
            f"no-EOS {m['no_eos']}/{m['n']} | LCR {m['lcr']:.1%} | "
            f"{kmer_line(m['kmer'], ocfg.kmer_ks)}{rev} | {m['seconds']:.1f}s")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--smoke", action="store_true", help="tiny run: small batch, a few steps, CPU-ok")
    ap.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start at step 0")
    ap.add_argument("--max-steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--no-ipex", action="store_true",
                    help="skip ipex.optimize (eager XPU; robust to shape churn)")
    ap.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    env = init_distributed(args.device)
    dev = env.device
    torch.manual_seed(CFG.opt.seed + env.rank)
    mcfg, dcfg, ocfg = CFG.model_config(), CFG.data, CFG.opt
    if args.grad_checkpoint is not None:
        mcfg.grad_checkpoint = args.grad_checkpoint
    if args.smoke:
        # A smoke run exercises every code path -- both objective branches, the eval, the FASTA
        # write, checkpoint save/resume -- at the REAL model dimensions, on a canvas small enough to
        # finish on a laptop CPU. Only the shapes shrink; nothing is stubbed out.
        dcfg.canvas, dcfg.num_workers = 128, 0
        ocfg.eval_n, ocfg.eval_canvas, ocfg.eval_steps = 2, 128, 32
        ocfg.eval_fasta_ranks, ocfg.sample_min_len, ocfg.warmup_steps = 1, 10, 5

    # Before anything else touches the allocator, so oneCCL's cached registration for the one buffer
    # it reduces every eval is taken from a clean heap and held for the life of the run.
    preallocate_stats_buffer(dev)

    # --- model ---
    model = LoopedDiffusionLM(mcfg).to(dev)
    if env.is_main:
        print(f"[train] params={count_params(model)/1e6:.1f}M device={dev} "
              f"grad_checkpoint={mcfg.grad_checkpoint}", flush=True)
    broadcast_parameters(model)

    # --- objective ---
    sched = build_schedule(ocfg, mcfg, dev)
    batch_size = 2 if args.smoke else CFG.batch_size
    if env.is_main:
        print(f"[train] corruption: kernel={ocfg.sub_kernel} T={sched.T} betas={sched.betas} "
              f"| eos_w={ocfg.eos_loss_weight} pad_w={ocfg.pad_loss_weight}", flush=True)
        print(f"[train] objective: L = {ocfg.vb_weight} * T*E_t[KL] + {ocfg.ce_weight} * L_ce"
              f" | L_ce on corrupted positions"
              f"{'' if ocfg.ce_uncorrupted_weight == 0 else f' (+{ocfg.ce_uncorrupted_weight} on uncorrupted)'}"
              f" | vb is T-scaled (T={sched.T}), so vb and ce should be COMPARABLE -- a vb three "
              f"orders below ce means the scaling was lost", flush=True)
        for bi, b in enumerate(sched.betas):
            print(f"[train]   beta={b:<5} P(x_T=MASK)={sched.terminal_mask_fraction(bi):.4f} "
                  f"(L_T ~ 0) | mask fraction {sched.mask_fraction(bi)} "
                  f"| substituted among survivors at T/2: {sched.substitution_fraction(bi):.1%}",
                  flush=True)

    # --- data ---
    shards = ProteinShards(UNIREF_SHARDS, mcfg.eos_token_id,
                           split="train" if dcfg.holdout_stride else "all",
                           holdout_stride=max(dcfg.holdout_stride, 2))
    if len(shards) == 0:
        raise RuntimeError(f"no shards in {UNIREF_SHARDS}. Run src.preprocess_fasta first "
                           f"(or set PLD2_UNIREF_SHARDS).")
    ds = ShardDataset(shards)
    total = args.max_steps or (12 if args.smoke else ocfg.total_steps)

    # --- resume (rank 0 reads latest.txt and broadcasts the BYTES; see dist.py) ---
    start_step = 0
    ckpt_path = find_latest_ckpt(CKPT_DIR) if (env.is_main and not args.fresh) else None
    raw = None if args.fresh else broadcast_checkpoint_bytes(ckpt_path, dev)

    opt = torch.optim.AdamW(model.parameters(), lr=ocfg.lr, weight_decay=ocfg.weight_decay,
                            betas=(0.9, 0.98))
    use_ipex = ocfg.use_ipex and not args.no_ipex
    applied_ipex = ipex is not None and dev.type == "xpu" and use_ipex
    if applied_ipex:
        model, opt = ipex.optimize(model, optimizer=opt, dtype=torch.bfloat16)
    if env.is_main:
        print(f"[train] ipex.optimize {'ON' if applied_ipex else 'OFF'} (device={dev.type}, "
              f"requested={use_ipex}, ipex={'available' if ipex is not None else 'missing'})",
              flush=True)
    # Built AFTER ipex.optimize so it binds to the optimizer that actually steps.
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, ocfg.warmup_steps, total))

    if raw is not None:
        ck = torch.load(io.BytesIO(raw), map_location=dev, weights_only=True)
        del raw
        model.load_state_dict(ck["model"])
        try:
            opt.load_state_dict(ck["opt"])
        except Exception as ex:
            if env.is_main:
                print(f"[ckpt] optimizer state not restored ({ex}); re-warming moments.", flush=True)
        lr_sched.load_state_dict(ck["sched"])
        start_step = int(ck["step"]) + 1
        if env.is_main:
            print(f"[ckpt] resumed from {ckpt_path}: continuing at step {start_step}", flush=True)

    # The sampler is a pure function of (seed, step, rank), so a resumed run walks EXACTLY the batch
    # sequence the original would have. ProLoopDiff's epoch-seeded permutation did not, which is why
    # its resumed jobs could fault a few steps past a checkpoint the original had sailed through.
    sampler = StepBatchSampler(len(ds), batch_size, rank=env.rank, world=env.world_size,
                               seed=ocfg.seed, start_step=start_step, total_steps=total)
    nw = dcfg.num_workers
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=nw,
        collate_fn=make_collate(mcfg, dcfg.canvas),
        **({"prefetch_factor": dcfg.prefetch_factor, "persistent_workers": True} if nw > 0 else {}))
    if env.is_main:
        epochs = batch_size * env.world_size * total / max(len(shards), 1)
        print(f"[train] corpus={len(shards):,} of {shards.n_total:,} sequences in "
              f"{shards.n_shards_total} shards "
              f"(holdout=every {dcfg.holdout_stride}th)" if dcfg.holdout_stride else "(no holdout)")
        print(f"[train] "
              f"canvas={dcfg.canvas} B={batch_size}/rank x {env.world_size} ranks | "
              f"{total} steps ~= {epochs:.1f} epochs", flush=True)

    n_grad = preallocate_grad_buffer(model, dev)
    if env.is_main and n_grad:
        print(f"[train] all-reduce buffer preallocated: {4 * (n_grad + 1) / 1e6:.0f}MB fp32",
              flush=True)

    log_every = 5 if args.smoke else ocfg.log_every
    eval_every = (10 if args.smoke else ocfg.eval_every)
    use_amp = dev.type in ("xpu", "cuda")
    step = start_step
    tok_win, steps_win, t_win, t_eval = 0, 0, time.perf_counter(), 0.0
    skipped, skip_streak = 0, 0
    model.train()

    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        n_tok = batch["tokens"].numel()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, m = training_step(model, batch, sched,
                                    eos_loss_weight=ocfg.eos_loss_weight,
                                    pad_loss_weight=ocfg.pad_loss_weight,
                                    ce_weight=ocfg.ce_weight, vb_weight=ocfg.vb_weight,
                                    ce_uncorrupted_weight=ocfg.ce_uncorrupted_weight)
        loss.backward()
        nonfinite = average_gradients(model)
        if nonfinite:
            skipped += 1
            skip_streak += 1
            if env.is_main:
                print(f"[warn] step {step}: non-finite gradient -> update skipped "
                      f"({skip_streak} consecutive, {skipped} total)", flush=True)
            if skip_streak >= ocfg.max_skip_streak:
                raise RuntimeError(
                    f"{skip_streak} consecutive non-finite gradients ending at step {step}; "
                    f"aborting rather than spinning. Last good checkpoint is in {CKPT_DIR}.")
        else:
            skip_streak = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), ocfg.grad_clip)
            opt.step()
        lr_sched.step()
        tok_win += n_tok
        steps_win += 1

        if env.is_main and step % log_every == 0:
            _device_sync(dev)                       # finish queued work before reading the clock
            dt = max(time.perf_counter() - t_win, 1e-9)
            peak = _peak_mem_gb(dev)
            print(f"step {step:>7} | loss {float(m['loss']):.3f} "
                  f"(vb {float(m['vb']):.3f}[{float(m['vb_step']):.5f}/step] ce {float(m['ce']):.3f}) "
                  f"| corrupt {float(m['masked']):.0%}m {float(m['subst']):.0%}s "
                  f"| lr {lr_sched.get_last_lr()[0]:.2e} | "
                  f"{tok_win / dt / 1e3:.0f}k tok/s | {dt / steps_win:.2f}s/step"
                  f"{f' | peak {peak:.1f}GB' if peak else ''}"
                  f"{f' | eval {100 * t_eval / (t_eval + dt):.1f}% of wall' if t_eval else ''}"
                  f"{f' | skipped {skipped}' if skipped else ''}", flush=True)
            tok_win, steps_win, t_win, t_eval = 0, 0, time.perf_counter(), 0.0

        if eval_every and step > 0 and step % eval_every == 0:
            te = time.perf_counter()
            stats = generative_eval(model, dev, mcfg, ocfg, step, env, SAMPLES_DIR, use_amp)
            model.train()
            t_eval += time.perf_counter() - te
            if env.is_main:
                print(_eval_line(step, stats, ocfg), flush=True)

        if step > 0 and step % ocfg.ckpt_every == 0:
            save_checkpoint(model, opt, lr_sched, step, CKPT_DIR, env)
        step += 1
        if step >= total:
            break

    save_checkpoint(model, opt, lr_sched, step - 1, CKPT_DIR, env)     # final checkpoint
    if env.is_main:
        print(f"[train] done at step {step}", flush=True)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
