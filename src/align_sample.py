"""Generate N completions per prompt: the y's that preference pairs are cut from.

    mpiexec -n 12 python -m src.align_sample --device xpu           # one pass, resumable
    mpiexec -n 12 python -m src.align_sample --device xpu --gamma 0 # no caption guidance

NO PROCESS GROUP, BY DESIGN. Generation is embarrassingly parallel over prompts and nothing here
needs aggregating, so init_distributed(no_dist=True) pins each rank to its own tile without ever
initialising oneCCL. That is the same decision src/fold_fasta.py made and for a related reason: a
job that never creates node-local IPC peer mappings has one fewer way to die, and this job runs
right next to the folding pass that established the problem.

RESUMABLE AT PROMPT GRANULARITY. Every prompt's completions are appended to this rank's own JSONL
and fsynced before the next prompt starts, and a rerun skips pids already recorded. A crash 300
prompts into 500 costs the one in flight. Ranks partition prompts by a stable crc32 of the pid
(fold_fasta.owns), not by a stride over a list, so ownership is a property of the prompt and two
ranks can never disagree about it -- which also means adding ranks on a rerun is safe.

WHAT IS WRITTEN, AND WHY IT IS SPELT THIS WAY
    gen.rank000.fasta     >p0000042_7   -- the residues. fold_fasta prefixes ids with the file stem,
                             so these become "gen|p0000042_7", and self_consistency.record_key
                             strips that back off. The header carries prompt and draw index, which
                             is the whole join key downstream.
    gen.rank000.3di.fa    the structure track the model emitted alongside, same headers.
    gen.rank000.jsonl     one record per completion: pid, k, sequence, 3Di, length.

SCALE. At 16 nodes this is ~5 prompts a rank; at 200 nodes and 100k prompts it is ~40. Nothing here
is O(world): no collectives, no shared-file writes, no manifest broadcast. The one shared read is
the prompt manifest, which is deliberately ~180 bytes a row (src/prompts.py) so a thousand ranks
opening it at once is a few tens of megabytes rather than a few hundred.
"""
from __future__ import annotations
import argparse
import glob as glob_
import json
import os
import sys
import time

import torch

from config import CFG, CKPT_DIR, FILIP_CACHE, FILIP_CKPT
from .dist import init_distributed
from .fold_fasta import partition
from .model import LoopedDiffusionLM
from .objective import surrogate_logp, surrogate_mask
from .prompts import _unhex, materialize, read_manifest
from .sampler import decode_seqs, decode_struct, generate
from .train import find_latest_ckpt

try:
    import intel_extension_for_pytorch as ipex
except Exception:
    ipex = None


def done_pids(rdir):
    """pids already generated, across EVERY rank's JSONL. Tolerates a truncated final line -- a GPU
    fault aborts the process mid-write and leaves one.

    Every rank's file, not just this one, so that changing the rank count on a rerun costs nothing.
    Ownership is a positional stride now (fold_fasta.partition), which redistributes when `world`
    changes; reading the done set globally means a redistributed prompt is still recognised as
    finished. The partition remains the sole authority on ASSIGNMENT, so this can only cause a rank
    to skip work that is genuinely complete -- never to miss work.
    """
    out = set()
    for path in sorted(glob_.glob(os.path.join(rdir, "gen.rank*.jsonl"))):
        with open(path) as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)["pid"])
                except Exception:
                    continue
    return out



def _smoke(mcfg, dcfg, canvas=None):
    """Shrink the model to something a laptop can hold, keeping every code path.

    The real config is 1.35B parameters, which is 5.4GB of weights before optimizer state. A smoke
    run has to exercise prompt materialisation, the freeze, decoding, the surrogate and the loss --
    none of which care about width -- so it shrinks the model and leaves the logic alone.
    """
    mcfg.d_model, mcfg.n_heads, mcfg.d_ff = 128, 4, 384
    mcfg.n_upstream, mcfg.n_middle, mcfg.n_downstream = 1, 2, 1
    mcfg.n_recurrence, mcfg.checkpoint_chunk, mcfg.grad_checkpoint = 1, 2, False
    dcfg.canvas = canvas or 64
    return mcfg, dcfg


def main():
    sys.stdout.reconfigure(line_buffering=True)
    acfg, ocfg, dcfg = CFG.align, CFG.opt, CFG.data
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=CFG.device)
    ap.add_argument("--ckpt", default=None, help="policy to sample from (default: latest)")
    ap.add_argument("--dir", default=None, help="round directory (default: config align.round_dir)")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n-gen", type=int, default=acfg.n_gen)
    ap.add_argument("--steps", type=int, default=acfg.gen_steps)
    ap.add_argument("--temperature", type=float, default=acfg.gen_temperature)
    ap.add_argument("--gamma", type=float, default=acfg.gen_gamma,
                    help="FILIP caption guidance weight; ignored when a prompt has no caption")
    ap.add_argument("--filip-ckpt", default=FILIP_CKPT)
    ap.add_argument("--filip-cache", default=FILIP_CACHE)
    ap.add_argument("--limit", type=int, default=0, help="stop after N prompts per rank (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canvas", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny randomly-initialised model, no checkpoint: exercises every path")
    ap.add_argument("--no-ipex", action="store_true")
    a = ap.parse_args()

    env = init_distributed(a.device, no_dist=True)
    dev = env.device
    rank, world = env.rank, env.world_size
    rdir = a.dir or acfg.round_dir
    os.makedirs(rdir, exist_ok=True)
    manifest = a.manifest or os.path.join(rdir, "prompts.jsonl")

    rows = read_manifest(manifest)
    # PARTITION THE MANIFEST, WHICH IS IMMUTABLE, then filter by what is already done -- never the
    # other way round. Partitioning the REMAINING work would make ownership depend on a set that
    # changes while ranks are starting, and two ranks a few seconds apart would split different
    # lists: some prompts done twice, some by nobody.
    by_pid = {r["pid"]: r for r in rows}
    out_jsonl = os.path.join(rdir, f"gen.rank{rank:03d}.jsonl")
    already = done_pids(rdir)
    mine = partition(list(by_pid), rank, world)
    todo = [by_pid[p] for p in mine if p not in already]
    if a.limit:
        todo = todo[:a.limit]
    if rank == 0:
        bal = (f" (balanced: {len(rows) // world} or {len(rows) // world + 1} each)"
               if world > 1 else "")
        print(f"[gen] {len(rows):,} prompts, {len(mine):,} on rank 0{bal}, "
              f"{len(already):,} already done, {len(todo):,} to do | n_gen={a.n_gen} "
              f"steps={a.steps} T={a.temperature}", flush=True)
    if not todo:
        print(f"[gen] rank {rank}: nothing to do", flush=True)
        return

    mcfg = CFG.model_config()
    if a.smoke:
        mcfg, dcfg = _smoke(mcfg, dcfg, a.canvas)
    elif a.canvas:
        dcfg.canvas = a.canvas
    model = LoopedDiffusionLM(mcfg).to(dev).eval()
    if a.smoke:
        print(f"[gen] SMOKE: random init, d_model={mcfg.d_model}, canvas={dcfg.canvas}. The "
              f"sequences are noise; what is being tested is that the pipeline runs.", flush=True)
    else:
        ckpt = a.ckpt or find_latest_ckpt(CKPT_DIR)
        if not ckpt or not os.path.exists(ckpt):
            raise SystemExit(f"no checkpoint (looked at {a.ckpt or CKPT_DIR})")
        st = torch.load(ckpt, map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        if ipex is not None and dev.type == "xpu" and not a.no_ipex:
            model = ipex.optimize(model, dtype=torch.bfloat16)
        if rank == 0:
            print(f"[gen] policy {ckpt} (step {st.get('step', '?')}) on {dev}", flush=True)

    # --- caption guidance -----------------------------------------------------------------
    # Every prompt carries the FILIP cache row of its protein's caption (src/reference_set.py), so
    # guidance composes with the scaffold rather than replacing it. Gamma is low on purpose: the
    # best-of-N multiplier collapses as it rises (5.7x at 10, 2.7x at 30, 1.9x at 100) because
    # guidance and best-of-N spend the same resource, within-prompt spread, which is what every
    # preference pair is cut from.
    guide = None
    has_caption = any("caption" in r for r in rows)
    if a.gamma > 0 and has_caption:
        from .filip_guidance import FilipGuidance
        guide = FilipGuidance(mcfg, dev, ckpt=a.filip_ckpt, cache_dir=a.filip_cache,
                              gamma=a.gamma, verbose=(rank == 0))
    elif rank == 0:
        why = ("NO PROMPT CARRIES A CAPTION -- the manifest was not built from a reference set"
               if not has_caption else f"--gamma {a.gamma}")
        print(f"[gen] caption guidance OFF ({why}); conditioning is the scaffold only", flush=True)

    fa = open(os.path.join(rdir, f"gen.rank{rank:03d}.fasta"), "a")
    f3 = open(os.path.join(rdir, f"gen.rank{rank:03d}.3di.fa"), "a")
    fj = open(out_jsonl, "a")
    use_amp = dev.type in ("xpu", "cuda")
    t0 = time.perf_counter()
    two = mcfg.n_tracks == 2

    for i, row in enumerate(todo):
        pid, B = row["pid"], int(a.n_gen)
        tok, given = materialize(row, mcfg, dcfg.canvas, n_tracks=mcfg.n_tracks)
        prompt = tok.unsqueeze(0).expand(B, -1, -1).contiguous().to(dev)
        pmask = given.unsqueeze(0).expand(B, -1, -1).contiguous().to(dev)
        # Seeded per PROMPT, not per rank: the N draws of a prompt are reproducible on their own, so
        # a rerun after a crash regenerates the same pool and the pairs do not change underneath a
        # partially built dataset.
        torch.manual_seed(a.seed + int(pid[1:]) * 7919)
        if guide is not None and "caption" in row:
            guide.set_target(row["caption"])
        gfn = guide if (guide is not None and "caption" in row) else None
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                             enabled=use_amp):
            cv, _ = generate(model, Lmax=dcfg.canvas, batch_size=B, n_steps=a.steps,
                             device=str(dev), temperature=a.temperature,
                             gumbel_temp=ocfg.sample_gumbel_temp,
                             subst_per_residue=ocfg.sample_subst_per_residue,
                             rep_penalty=ocfg.sample_rep_penalty,
                             rep_periods=ocfg.sample_rep_periods,
                             max_run=ocfg.sample_max_run,
                             min_len=ocfg.sample_min_len,
                             guidance_fn=gfn,
                             prompt=prompt, prompt_mask=pmask)
        seqs, _ = decode_seqs(cv, mcfg)
        dis = decode_struct(cv, mcfg)[0] if two else [None] * B

        # THE MODEL'S OWN OPINION OF WHAT IT JUST MADE, recorded here because the model is already
        # loaded and it costs one extra forward per prompt. It is the diagnosis, not a nicety:
        # best-of-N by this number is WORSE THAN A RANDOM DRAW (0.007 at N=1 falling to 0.000 at
        # N=8) while pLDDT selection tracks the oracle exactly. The generator makes good proteins
        # and ranks them below the bad ones. Preference tuning exists to close that, so the column
        # has to be measurable every round or there is no way to see it closing.
        gen_pos = torch.zeros(dcfg.canvas, dtype=torch.bool)
        gen_pos[:row["L"]] = torch.from_numpy(_unhex(row["masked"], row["L"]))
        smask = surrogate_mask(pid, gen_pos, dcfg.canvas, dev, epoch=0,
                               rate=0.5).unsqueeze(0).expand(B, -1)
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16,
                                             enabled=use_amp):
            logl = surrogate_logp(model, cv if cv.dim() == 3 else cv.unsqueeze(1),
                                  smask, mcfg).tolist()

        for k, seq in enumerate(seqs):
            gid = f"{pid}_{k}"
            fa.write(f">{gid}\n{seq}\n")
            if dis[k] is not None:
                f3.write(f">{gid}\n{dis[k]}\n")
            fj.write(json.dumps({"gid": gid, "pid": pid, "k": k, "L": len(seq),
                                 "seq": seq, "di": dis[k], "rate": row["rate"],
                                 "bin": row["bin"], "ref_len": row["L"], "rid": row["rid"],
                                 "caption": row.get("caption"),
                                 "loglik": round(float(logl[k]), 6)}) + "\n")
        # Fsync AFTER the whole prompt, so the resume unit is a prompt: a half-written pool would
        # otherwise be skipped as done and the prompt would carry fewer than n_gen draws forever.
        for fh in (fa, f3, fj):
            fh.flush()
            os.fsync(fh.fileno())
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.perf_counter() - t0
            print(f"[gen] rank {rank}: {i + 1}/{len(todo)} prompts "
                  f"({(i + 1) * B:,} sequences, {el:.0f}s, {el / (i + 1):.1f}s/prompt)", flush=True)

    for fh in (fa, f3, fj):
        fh.close()
    print(f"[gen] rank {rank}: done, {len(todo)} prompts in "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
