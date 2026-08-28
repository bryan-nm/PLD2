"""Central configuration for PLD2 (Aurora conventions).

THIS FILE OWNS EVERY PATH. Job scripts must not pass model/dataset locations on the command line;
`python config.py` prints what a run resolves to, and every job script banners that output so the
.o log is the record.

PLD2_* env vars are the escape hatch for a workstation whose data lives elsewhere. They are
deliberately NOT set by any job script. PLD2_*_DIR swaps a base dir; per-item vars override one path.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

from src.model import Config as ModelConfig
from src.metrics import KMER_KS

REPO_ROOT = Path(__file__).resolve().parent

# --- base dirs (Aurora flare defaults; override for workstation smoke tests) ---
MODELS_DIR = os.environ.get("PLD2_MODELS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/models")
DATASETS_DIR = os.environ.get("PLD2_DATASETS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/datasets")
RUNS_DIR = os.environ.get("PLD2_RUNS_DIR", "/flare/NLDesignProtein/bryan/Diffusion-dev-space/runs/pld2")

# --- data ---
# The ONLY corpus: UniRef90 filtered to the same 30-500 aa window ProLoopDiff used, pre-tokenised to
# packed uint8 .bin + int64 .idx shards (src/preprocess_fasta.py). PLD2 is unconditional, so there
# is no labelled corpus and no text cache. These shards ALREADY EXIST from the ProLoopDiff
# preprocess job -- it also dropped exact SwissProt matches, which is harmless here (a ~0.3%
# thinning of the corpus, no leakage risk in either direction) -- so point at them and skip the
# 6-hour rebuild. Rebuild only if you want the SwissProt sequences back.
UNIREF_FASTA = os.environ.get("PLD2_UNIREF_FASTA", f"{DATASETS_DIR}/uniref90.fasta.gz")
UNIREF_SHARDS = os.environ.get("PLD2_UNIREF_SHARDS", f"{DATASETS_DIR}/uniref90_shards")

BLOSUM_MAT = os.environ.get("PLD2_BLOSUM", f"{MODELS_DIR}/blosum62-special-MSA.mat")

# --- run outputs ---
CKPT_DIR = os.environ.get("PLD2_CKPT_DIR", f"{RUNS_DIR}/checkpoints")
SAMPLES_DIR = os.environ.get("PLD2_SAMPLES_DIR", f"{RUNS_DIR}/samples")   # training-time eval FASTA
FOLDS_JSONL = os.environ.get("PLD2_FOLDS_JSONL", f"{RUNS_DIR}/folds.jsonl")  # ESMFold results

# ESMFold2-Fast weights for the structural eval (pLDDT/pTM). NOT under MODELS_DIR: this is where the
# EsmFold repo's speed_test.pbs already has them staged. The ESM-C 6B backbone named in its config
# resolves from the HuggingFace cache -- pre-cache it from a login node and set HF_HUB_OFFLINE=1 on
# compute nodes. See the EsmFold repo's README.
ESMFOLD_WEIGHTS = os.environ.get("PLD2_ESMFOLD_WEIGHTS",
                                 "/flare/NLDesignProtein/bryan/models/ESMFold2-Fast")


@dataclass
class DataCfg:
    # FIXED CANVAS (instruction 7). Every batch is exactly (B, 512): one static shape for the whole
    # run, so XPU compiles it once. ProLoopDiff's length buckets are gone -- with a single width
    # there is nothing to bucket. The corpus is 30-500 aa, so 512 always holds [AA* EOS PAD*].
    canvas: int = 512
    num_workers: int = 4
    prefetch_factor: int = 4         # batches prefetched per worker (only used when num_workers > 0)
    # Every Nth sequence GLOBALLY is held out for the fold/repetition baselines. Strided, not
    # by-shard: shard order is FASTA order, and reserving the last shard silently returned the
    # shortest ~1% of a length-sorted corpus (observed: a "natural" baseline of 33.9 +- 2.3 aa from
    # data filtered to 30-500). A stride is order-agnostic. 0 disables the holdout entirely.
    holdout_stride: int = 100


@dataclass
class OptCfg:
    # Per-rank token budget per step -> batch size B = global_batch_tokens // canvas = 64 at 512.
    global_batch_tokens: int = 32768
    lr: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 50_000        # instruction 4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # A non-finite gradient norm makes clip_grad_norm_'s coefficient NaN, which turns every parameter
    # NaN on the next opt.step(). The trainer skips those updates; this many CONSECUTIVE skips means
    # something is structurally wrong, so abort rather than spin.
    max_skip_streak: int = 20

    # --- objective: ONE process, MASK absorbing, beta as the slider (src/corruption.py) ---
    # Per step a non-MASK position stays, goes to MASK (prob beta*c_t), or takes a BLOSUM-weighted
    # substitution (prob (1-beta)*c_t). beta = "of the corruption events that happen, what fraction
    # are maskings". beta=1 is exactly OADM's corruption; beta<1 enriches the SAME trajectories with
    # substitution moves. MASK is absorbing at every beta, so the stationary distribution is always
    # all-MASK -- generation cold-starts from a fully-masked canvas regardless, and the ELBO's
    # dropped L_T term is genuinely ~0 (measured P(x_T=MASK) = 0.999-1.000, not an approximation).
    #
    # A DISTRIBUTION over beta, not a single value. Under the old two-branch design mixing beta
    # would have blurred two different bounds together -- ProLoopDiff's mistake. Here beta only
    # picks which transition matrix a row is corrupted by and one ELBO scores them all, so a row per
    # beta is simply training over a family of corruption processes. Rows are assigned round-robin,
    # so the mix is exactly balanced every step. 1.0 keeps pure-OADM trajectories in the mix; the
    # rest carry progressively more substitution (measured footprint among still-unmasked positions
    # at t=T/2: 0%, 7.4%, 20.3%, 47.8%).
    betas: tuple = (1.0, 0.9, 0.75, 0.5)
    d3pm_T: int = 500                # diffusion steps; the Q/Qbar stacks are ~17MB at 4 betas
    sub_kernel: str = "blosum"       # "blosum" (biologically informed) or "uniform"
    sub_kernel_temp: float = 1.0     # BLOSUM softmax sharpness (higher -> flatter -> less informed)
    ce_weight: float = 1.0           # EvoDiff's lambda on the x0 cross-entropy. They used 0; at
                                     # beta=1 this term IS the OADM cross-entropy, so keeping it
                                     # means the objective that demonstrably works is still a
                                     # component rather than replaced. The KL alone is numerically
                                     # tiny at small t.
    # EOS UPWEIGHTING. One EOS per 512-wide canvas is 0.2% of an unweighted loss, and ProLoopDiff
    # duly learned everything except where to stop (53% of its samples at 181k steps never placed
    # EOS at all). At 20.0 against ~350 residues at 1.0 and ~160 PAD at 0.1, EOS becomes ~5% of the
    # per-sequence loss. Raise it if EOS placement is still weak at 10k steps; the failure mode of
    # raising it too far is over-eager EOS, i.e. short samples, which the length column of the eval
    # line makes obvious.
    eos_loss_weight: float = 20.0
    pad_loss_weight: float = 0.1     # the fixed canvas's PAD tail must not swamp the AA/EOS signal

    use_ipex: bool = True            # ipex.optimize: fused/faster, recompiles on new shapes (XPU)

    # --- training-time generative eval (instruction 8) ---
    # The loss says nothing about whether the model places EOS sensibly or repeats itself, so sample
    # and measure. Cost: eval runs eval_steps forwards on (eval_n, eval_canvas) while a training step
    # is ~3 forward-units on (B, canvas). At 512/8/512 vs 64/512 that is ~21 training steps, so
    # eval_every=1000 predicts ~2% of wall time (the trainer measures and logs the real figure).
    eval_every: int = 1000           # 0 disables; aligned with ckpt_every so every ckpt gets numbers
    eval_n: int = 8                  # sequences per RANK per eval (x world = the real sample count)
    eval_canvas: int = 512           # the training canvas = the model's length prior
    # Decoding steps. MUST stay ~= eval_canvas while the repetition penalty is on: the penalty scores
    # each position against the canvas BEFORE that step's commits, so positions committed in the same
    # step cannot see one another. Measured at max_run=5 on a 128 canvas: longest homopolymer 42 at
    # 8 commits/step, 26 at 2/step, and exactly the cap at 1/step.
    eval_steps: int = 512
    # How many ranks write their samples to FASTA. All ranks contribute to the aggregated statistics;
    # only these write files, because the folder downstream is a SINGLE tile at ~1.1 s/sequence and
    # 8 ranks x 8 sequences = 64 per round (~70s) fits comfortably inside the checkpoint interval.
    eval_fasta_ranks: int = 8
    # k-mer lengths for the long-range repetition metric. 13 is one above the SEG window, i.e. the
    # shortest k that neither the LCR scan nor the sampler's period-1..5 penalty can act on. See
    # src/metrics.py -- this is the metric instruction 8 asks for.
    kmer_ks: tuple = KMER_KS

    # --- sampling controls (src/sampler.generate) ---
    # EVERY generate() argument that is not a per-call concern lives HERE, so src.train's eval and
    # src.sample cannot drift apart. In ProLoopDiff they agreed only because hardcoded literals in
    # eval happened to equal sample.py's CLI defaults, so tuning either would have silently left the
    # eval sampling differently from the thing being shipped.
    sample_temperature: float = 0.5  # instruction 8. At ProLoopDiff's 181k checkpoint T=1.0 folded
                                     # to pLDDT 33.9 (BELOW its 37.5 shuffled baseline) while T=0.5
                                     # gave 55.2 with 21% of samples over 70. Temperature was the
                                     # difference between "folds like a shuffle" and "folds".
    sample_eos_first: bool = True    # commit the boundary before any residue (see sampler.py)
    sample_min_len: int = 30         # corpus floor; also stops EOS at position 0 (the len=0 bug)
    sample_eos_temp: float = 1.0     # <1 sharpens the length draw, >1 widens it
    sample_rep_penalty: float = 1.5  # per-period logit penalty for continuing a repeat
    sample_max_run: int = 5          # hard cap on identical consecutive residues (0 disables)
    sample_rep_periods: tuple = (1, 2, 3, 4, 5)   # repeat periods scored (1 = homopolymer)
    sample_gumbel_temp: float = 0.1  # noise on the COMMIT ORDER; raising it breaks the
                                     # commit-the-most-predictable-first loop that feeds repeats
    sample_corrector: int = 0        # post-decode corrector sweeps (see sampler._corrector_sweep)
    sample_corrector_type: str = "remask"   # or "substitution"
    # SUBSTITUTION BUDGET AT DECODE TIME, as expected edits per decodable position over the whole
    # decode. The decoder offers one candidate edit per position -- unmask if masked, substitute if
    # already committed -- and takes the highest-confidence ones, which is the inference-time mirror
    # of the training process. The scheduled unmask count is a guaranteed FLOOR so substitutions can
    # never starve the mask channel (see sampler.generate).
    #
    # 0.0 disables substitution and recovers pure absorbing decoding bit-for-bit. Left at 0.0 for
    # now: the right value depends on how much the model actually wants to revise itself, which is
    # not knowable before a real checkpoint. A/B it once one exists --
    #     python -m src.sample --n 128 --subst-per-residue 0   --out off.fasta
    #     python -m src.sample --n 128 --subst-per-residue 1.0 --out on.fasta
    # -- and compare the k-mer repetition columns and folded pLDDT. Setting it here turns it on for
    # the training-time eval too, which then reports a `sub N/seq` column.
    sample_subst_per_residue: float = 0.0

    # --- structural eval (ESMFold2-Fast), run by src/fold_fasta.py in its OWN process ---
    # Folding NEVER shares a process with generation. ProLoopDiff established that the hard way:
    # ESMFold on Aurora aborts with "Segmentation fault from GPU ... NotPresent" for reasons that
    # survived four separate refutations, and it installs process-global monkey-patches on
    # torch.linalg and F.linear that have no business near an ipex-optimised trainer. So the trainer
    # writes FASTA and a separate single-rank watcher folds it, appending every result to disk
    # immediately -- a crash 60 sequences in costs 40, not 100.
    fold_min_len: int = 10           # too short to fold meaningfully
    fold_max_len: int = 512          # ESMFold's pair tensors are ~L^2; also the canvas width
    # DO NOT lower fold_steps below 20: pLDDT does not degrade gracefully, it collapses to the ~0.25
    # no-information floor between 10 and 20 steps, which would silently look like a metric.
    fold_steps: int = 20
    fold_loops: int = 1              # trunk recycling; measured not to move pLDDT at all
    plddt_confident: float = 0.70    # "confident fold" (0-1 scale; = 70 on AlphaFold's 0-100)
    # pTM is global (is the OVERALL topology right) where pLDDT is local (is each residue placed
    # confidently). They come apart: a chain of well-formed helices floating in the wrong arrangement
    # scores high pLDDT and low pTM, so watching only pLDDT can miss that the model makes good
    # secondary structure and no real fold. 0.5 is the usual "topology likely correct" line.
    ptm_confident: float = 0.50
    # Release cached HBM every N folded sequences. Keep at 1: the EsmFold README warns that NOT
    # clearing lets a long, length-varied batch fragment HBM, "which on Aurora XPU manifests as a GPU
    # page fault rather than a clean OOM". Setting it to 0 buys a failure mode instead of avoiding one.
    fold_empty_cache_every: int = 1
    n_baseline: int = 200            # held-out sequences per baseline FASTA (natural + shuffled)

    # bookkeeping
    log_every: int = 50
    ckpt_every: int = 1000           # crashes are common on many tiles; save often
    seed: int = 0


@dataclass
class RunCfg:
    device: str = "auto"                 # auto -> xpu on Aurora, cpu on a laptop
    data: DataCfg = field(default_factory=DataCfg)
    opt: OptCfg = field(default_factory=OptCfg)

    def model_config(self) -> ModelConfig:
        # Same dims as ProLoopDiff (~55M params) so the two runs are comparable. What changed is
        # what is NOT here: no pb_layers, no pb_dim, no n_pb_heads, no text_dim. Every layer is
        # d_model wide (instruction 1) and there is no conditioning pathway (instruction 5).
        return ModelConfig(
            vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
            d_model=512, n_heads=8, d_ff=1536,
            n_upstream=4, n_middle=8, n_downstream=4, n_recurrence=3,
        )

    @property
    def batch_size(self) -> int:
        return max(1, self.opt.global_batch_tokens // self.data.canvas)


CFG = RunCfg()

if __name__ == "__main__":
    m = CFG.model_config()
    print("REPO_ROOT      :", REPO_ROOT)
    print("UNIREF_SHARDS  :", UNIREF_SHARDS, " exists:", os.path.isdir(UNIREF_SHARDS))
    print("UNIREF_FASTA   :", UNIREF_FASTA, " exists:", os.path.exists(UNIREF_FASTA))
    print("BLOSUM_MAT     :", BLOSUM_MAT, " exists:", os.path.exists(BLOSUM_MAT))
    print("ESMFOLD_WEIGHTS:", ESMFOLD_WEIGHTS, " exists:", os.path.exists(ESMFOLD_WEIGHTS))
    print("CKPT_DIR       :", CKPT_DIR)
    print("SAMPLES_DIR    :", SAMPLES_DIR)
    print("FOLDS_JSONL    :", FOLDS_JSONL)
    print(f"model          : d_model={m.d_model} d_ff={m.d_ff} heads={m.n_heads} "
          f"layers={m.n_upstream}+{m.n_middle}(x{m.n_recurrence})+{m.n_downstream} "
          f"vocab={m.vocab_size} (eos={m.eos_token_id} pad={m.pad_token_id} mask={m.mask_token_id})")
    print(f"batch          : canvas={CFG.data.canvas} B={CFG.batch_size}/rank "
          f"(rows split round-robin across {len(CFG.opt.betas)} betas)")
    print(f"objective      : betas={CFG.opt.betas} T={CFG.opt.d3pm_T} "
          f"kernel={CFG.opt.sub_kernel} lambda={CFG.opt.ce_weight} "
          f"eos_w={CFG.opt.eos_loss_weight} pad_w={CFG.opt.pad_loss_weight}")
    print(f"sampling       : T={CFG.opt.sample_temperature} steps={CFG.opt.eval_steps} "
          f"eos_first={CFG.opt.sample_eos_first} "
          f"subst/residue={CFG.opt.sample_subst_per_residue}"
          f"{' (absorbing-only decode)' if CFG.opt.sample_subst_per_residue <= 0 else ' (unified edit decode)'}")
    print(f"schedule       : {CFG.opt.total_steps} steps, warmup {CFG.opt.warmup_steps}, "
          f"lr {CFG.opt.lr}, eval every {CFG.opt.eval_every}, ckpt every {CFG.opt.ckpt_every}")
