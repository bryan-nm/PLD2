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

GRADIENT ACCUMULATION. `step` throughout this file means an OPTIMIZER step: the LR schedule, the
eval cadence, checkpointing and total_steps all count those. Each one runs `grad_accum` forward/
backward micro-batches, and the single coalesced all-reduce happens once at the end of them. That
decouples two things that were previously the same knob:

  * the global batch, which wants to be large enough for the model (a 1.35B model on a 0.7M-token
    batch is under-batched), and
  * how often the full fp32 gradient crosses the fabric, which is what actually caps model size
    here -- traffic goes as params / step-time, so a 25x parameter increase at a smaller
    micro-batch would have pushed ~16x the bytes per second of the 55M run.

The log reports the measured all-reduce share of each step, because that number decides whether
grad_accum is tuned right and it was previously invisible.

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

from config import CFG, UNIREF_SHARDS, AFDB_SHARDS, BLOSUM_MAT, CKPT_DIR, SAMPLES_DIR
from .blosum import substitution_kernel, uniform_substitution_kernel
from .corruption import CorruptionSchedule
from .data import ProteinShards, ShardDataset, StepBatchSampler, make_collate
from .dist import (init_distributed, barrier, cleanup, broadcast_parameters, average_gradients,
                   broadcast_checkpoint_bytes, preallocate_grad_buffer, preallocate_stats_buffer,
                   allreduce_stats)
from .metrics import flatten_kmer, kmer_counts, kmer_line, lcr_counts, unflatten_kmer
from .model import LoopedDiffusionLM, count_params
from .objective import training_step
from .sampler import decode_seqs, decode_struct, generate, write_fasta

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
    return CorruptionSchedule(B, mcfg.vocab_size, mcfg.mask_token_id, betas=ocfg.betas,
                              T=ocfg.d3pm_T, span_width=ocfg.span_width, device=device)


def build_struct_schedule(ocfg, mcfg, device):
    """The 3Di track's corruption process. Same absorbing MASK, same betas, same schedule, same
    span widths -- only the substitution kernel differs, because BLOSUM is a matrix over amino acids
    and carries no meaning over structural states."""
    n = mcfg.vocab_size - 1
    if ocfg.struct_sub_kernel == "uniform":
        B = uniform_substitution_kernel(n)
    elif ocfg.struct_sub_kernel == "blosum":
        raise ValueError("struct_sub_kernel='blosum' is a category error: BLOSUM scores amino-acid "
                         "exchangeability and the 3Di alphabet only borrows its letters. Use "
                         "'uniform', or wire in Foldseek's mat3di.")
    else:
        raise ValueError(f"unknown struct_sub_kernel {ocfg.struct_sub_kernel!r}")
    return CorruptionSchedule(B, mcfg.vocab_size, mcfg.mask_token_id, betas=ocfg.betas,
                              T=ocfg.d3pm_T, span_width=ocfg.span_width, device=device)


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
            subst_per_residue=ocfg.sample_subst_per_residue,
            struct_first=ocfg.sample_struct_first, stats=gen_stats)

    seqs, _ = decode_seqs(canvas, mcfg)
    if env.rank < ocfg.eval_fasta_ranks:
        os.makedirs(samples_dir, exist_ok=True)
        path = os.path.join(samples_dir, f"step_{step:08d}.rank{env.rank:03d}.fasta")
        tmp = path + ".tmp"                          # tmp+rename so the folding watcher, which polls
        write_fasta(seqs, tmp, prefix=f"s{step}r{env.rank}")   # this directory, never reads a
        os.replace(tmp, path)                                  # half-written file
        if mcfg.n_tracks == 2:
            # The 3Di track the model generated ALONGSIDE that sequence, same record ids, so the
            # pair can be rejoined downstream: fold the sequence, 3Di-encode the result, and compare
            # it to this. That is the self-consistency check the structure track exists to make
            # possible -- the model states which fold it was building and ESMFold says which fold
            # the sequence actually specifies.
            #
            # ".3di.fa", NOT ".3di.fasta": src/fold_fasta.py globs SAMPLES_DIR/*.fasta, and a file
            # of 3Di strings picked up by that glob would be folded as though it were protein --
            # silently, since 3Di uses the amino-acid letters and every string would parse.
            dpath = os.path.join(samples_dir, f"step_{step:08d}.rank{env.rank:03d}.3di.fa")
            dis, _ = decode_struct(canvas, mcfg)
            write_fasta(dis, dpath + ".tmp", prefix=f"s{step}r{env.rank}")
            os.replace(dpath + ".tmp", dpath)

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
        # A smoke run exercises every code path -- the objective, gradient accumulation, the eval,
        # the FASTA write, checkpoint save/resume -- on a laptop CPU. Unlike earlier versions this
        # SHRINKS THE MODEL too: the real config is 1.35B parameters, which is 5.4GB of weights
        # before optimizer state and not something to instantiate on a workstation. `python
        # config.py` reports the real shape and parameter count; the job validates it for real.
        mcfg.d_model, mcfg.n_heads, mcfg.d_ff = 256, 4, 768
        mcfg.n_upstream, mcfg.n_middle, mcfg.n_downstream = 2, 6, 2
        mcfg.checkpoint_chunk = 3
        dcfg.canvas, dcfg.num_workers = 128, 0
        ocfg.eval_n, ocfg.eval_canvas, ocfg.eval_steps = 2, 128, 32
        ocfg.eval_fasta_ranks, ocfg.sample_min_len, ocfg.warmup_steps = 1, 10, 5
        ocfg.grad_accum = 2

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
    two = mcfg.n_tracks == 2
    sched_struct = build_struct_schedule(ocfg, mcfg, dev) if two else None
    batch_size = 4 if args.smoke else CFG.batch_size
    if env.is_main:
        print(f"[train] tracks: {mcfg.n_tracks} "
              + ("(amino acid + Foldseek 3Di at the same L positions, separate embeddings and "
                 "heads, INDEPENDENT noise level per track -- shared t would reveal both tracks "
                 f"together and neither could ever inform the other) | 3Di kernel="
                 f"{ocfg.struct_sub_kernel} struct_weight={ocfg.struct_weight}"
                 if two else "(amino acids only)"), flush=True)
        print(f"[train] corruption: kernel={ocfg.sub_kernel} T={sched.T} betas={sched.betas} "
              f"span_width={sched.span_width} "
              f"| eos_w={ocfg.eos_loss_weight} pad_w={ocfg.pad_loss_weight}", flush=True)
        if sched.n_span > 1 or sched.span_width[0] > 1:
            rl = sched.run_length(frac=0.5)
            print("[train]   span corruption ON. MEASURED (masked run, visible run) at 50% mask, "
                  "512 canvas: " + "  ".join(f"w={w}: {m}/{v}" for w, (m, v) in rl.items()),
                  flush=True)
            print("[train]   L_vb is now a MEAN-FIELD SURROGATE, not an ELBO -- per-position "
                  "marginals are still exact, the joint posterior's correlation is not modelled. "
                  "Compare vb only to other span runs. L_ce and the CE curve are unaffected.",
                  flush=True)
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
    # At n_tracks=2 the paired AFDB shards are the corpus; the aa-only UniRef shards have no
    # structure to train the second track on.
    shard_dir = AFDB_SHARDS if two else UNIREF_SHARDS
    shards = ProteinShards(shard_dir, mcfg.eos_token_id,
                           split="train" if dcfg.holdout_stride else "all",
                           holdout_stride=max(dcfg.holdout_stride, 2))
    if len(shards) == 0:
        raise RuntimeError(f"no shards in {shard_dir}. Run "
                           f"{'src.preprocess_3di' if two else 'src.preprocess_fasta'} first "
                           f"(or set {'PLD2_AFDB_SHARDS' if two else 'PLD2_UNIREF_SHARDS'}).")
    if two and not shards.has_struct:
        raise RuntimeError(
            f"n_tracks=2 but no shard in {shard_dir} has a .3di sibling, so the structure track "
            f"would train against an all-MASK input on every row and contribute nothing. Run "
            f"src.preprocess_3di, or set data.n_tracks=1.")
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

    # A checkpoint at or past total_steps yields ZERO batches, so the job would load the old
    # weights, train nothing, re-save, and exit 0 -- burning a queue slot and looking like success.
    # Refuse instead. This is the shape of the mistake that follows every completed run: the next
    # experiment is launched into a CKPT_DIR that still holds the finished one.
    if start_step >= total:
        raise RuntimeError(
            f"checkpoint is already at step {start_step - 1} and total_steps is {total}, so this "
            f"run would train 0 steps and exit successfully. Pick one:\n"
            f"  --fresh                      start over, ignoring {CKPT_DIR}\n"
            f"  --max-steps N (N > {start_step})   continue the SAME run for longer\n"
            f"  archive {CKPT_DIR} first     start a new run and keep the old one for comparison\n"
            f"NOTE: those weights were trained under whatever objective was current then; "
            f"continuing them is not a test of any change made since.")

    # The sampler indexes MICRO-batches, so a resumed run walks exactly the micro-batch sequence the
    # original would have -- the reproducibility property survives accumulation only if the sampler
    # counts the same thing the data loader does.
    accum = max(1, int(ocfg.grad_accum))
    # objective.training_step assigns betas round-robin over the MICRO-batch, so a micro-batch that
    # is not a multiple of len(betas) gives some betas more rows than others -- a silently skewed
    # corruption mix rather than the balanced one the round-robin exists to guarantee.
    if batch_size % len(ocfg.betas) and env.is_main:
        print(f"[train] WARNING: micro-batch {batch_size} is not a multiple of "
              f"len(betas)={len(ocfg.betas)}, so the beta mix is skewed "
              f"({[sum(1 for i in range(batch_size) if i % len(ocfg.betas) == b) for b in range(len(ocfg.betas))]} "
              f"rows per beta). Pick a micro-batch divisible by {len(ocfg.betas)}.", flush=True)
    sampler = StepBatchSampler(len(ds), batch_size, rank=env.rank, world=env.world_size,
                               seed=ocfg.seed, start_step=start_step * accum,
                               total_steps=total * accum)
    nw = dcfg.num_workers
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, num_workers=nw,
        collate_fn=make_collate(mcfg, dcfg.canvas, n_tracks=mcfg.n_tracks),
        **({"prefetch_factor": dcfg.prefetch_factor, "persistent_workers": True} if nw > 0 else {}))
    if env.is_main:
        epochs = batch_size * accum * env.world_size * total / max(len(shards), 1)
        print(f"[train] batch: {batch_size}/rank x {accum} accum x {env.world_size} ranks = "
              f"{batch_size * accum * env.world_size:,} sequences "
              f"({batch_size * accum * env.world_size * dcfg.canvas / 1e6:.1f}M tokens) per "
              f"optimizer step", flush=True)
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
    tok_win, steps_win, t_win, t_eval, t_comm = 0, 0, time.perf_counter(), 0.0, 0.0
    skipped, skip_streak = 0, 0
    span_off = sched.n_span == 1 and sched.span_width[0] <= 1
    n_micro = 0
    acc = {}
    model.train()
    opt.zero_grad(set_to_none=True)

    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        tok_win += batch["tokens"].numel()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, m = training_step(model, batch, sched,
                                    eos_loss_weight=ocfg.eos_loss_weight,
                                    pad_loss_weight=ocfg.pad_loss_weight,
                                    ce_weight=ocfg.ce_weight, vb_weight=ocfg.vb_weight,
                                    ce_uncorrupted_weight=ocfg.ce_uncorrupted_weight,
                                    sched_struct=sched_struct,
                                    struct_weight=ocfg.struct_weight)
        # Scale so the accumulated gradient equals one pass over the full effective batch, rather
        # than `accum` times it -- otherwise the effective learning rate scales with grad_accum.
        (loss / accum).backward()
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + float(v) / accum
        n_micro += 1
        if n_micro < accum:
            continue
        n_micro = 0

        # ONE all-reduce per optimizer step. Timed behind a device sync so the number is the
        # collective itself and not queued compute draining into it.
        _device_sync(dev)
        t_c = time.perf_counter()
        nonfinite = average_gradients(model)
        t_comm += time.perf_counter() - t_c

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
        opt.zero_grad(set_to_none=True)
        steps_win += 1

        if env.is_main and step % log_every == 0:
            _device_sync(dev)
            dt = max(time.perf_counter() - t_win, 1e-9)
            peak = _peak_mem_gb(dev)
            run_str = "" if span_off else f"run{acc['mrun']:.0f} "
            # The structure track's own CE is directly comparable to the sequence track's (both are
            # 23-state categoricals), so one glance says whether the two are balanced.
            s_str = (f"| 3Di ce {acc['s_ce']:.3f} vb {acc['s_vb']:.3f} {acc['s_masked']:.0%}m "
                     f"lbl {acc['s_frac']:.0%} " if two else "")
            print(f"step {step:>7} | loss {acc['loss']:.3f} "
                  f"(vb {acc['vb']:.3f}[{acc['vb_step']:.5f}/step] ce {acc['ce']:.3f}) "
                  f"| corrupt {acc['masked']:.0%}m {acc['subst']:.0%}s {run_str}"
                  f"{s_str}"
                  f"| lr {lr_sched.get_last_lr()[0]:.2e} | {tok_win / dt / 1e3:.0f}k tok/s "
                  f"| {dt / steps_win:.2f}s/step | comm {100 * t_comm / dt:.0f}%"
                  f"{f' | peak {peak:.1f}GB' if peak else ''}"
                  f"{f' | eval {100 * t_eval / dt:.0f}%' if t_eval else ''}"
                  f"{f' | skipped {skipped}' if skipped else ''}", flush=True)
            tok_win, steps_win, t_win, t_eval, t_comm = 0, 0, time.perf_counter(), 0.0, 0.0
        acc = {}

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
