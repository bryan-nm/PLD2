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

# PAIRED (amino acid, Foldseek 3Di) shards from the AlphaFold DB corpus -- src/preprocess_3di.py.
# Same .bin/.idx layout as the UniRef shards plus a .3di sibling, so a directory of either kind
# reads correctly and the two can be mixed. Measured on the source FASTAs: 17,070,828 records,
# perfectly aligned (zero header or length mismatches), 15,316,921 inside the 30-500 window for
# 2.79B residues. That is ~5.6B tokens across both tracks -- under Chinchilla for a 1.35B model, so
# expect ~2-3 epochs on an 18k-step run rather than the single pass UniRef gave.
AFDB_SHARDS = os.environ.get("PLD2_AFDB_SHARDS", f"{DATASETS_DIR}/afdb_3di_shards")

BLOSUM_MAT = os.environ.get("PLD2_BLOSUM", f"{MODELS_DIR}/blosum62-special-MSA.mat")

# Foldseek's 3Di substitution matrix, shipped in the foldseek distribution as data/mat3di.out. It is
# to the structure track what BLOSUM is to the sequence track, and it ships with the same binary
# that produced the training labels -- so the corruption process and the tokeniser agree on what
# "a similar structural state" means.
MAT3DI = os.environ.get("PLD2_MAT3DI", f"{MODELS_DIR}/mat3di.out")

# --- FILIP prompt guidance (src/filip_guidance.py, optional) ---------------------------------
# A trained text<->protein co-embedding used as a ProteinGuide-style Bayesian classifier:
# generation is steered by reweighting PLD2's unconditional logits by p(prompt | sequence)^gamma.
# MINI_EMBED_REPO is imported rather than vendored -- loading AMPLIFY on Aurora needs an xformers
# stub, a RoPE-cache rematerialisation and two SDPA patches that were established the hard way in
# that repo, and a second copy of them would rot.
MINI_EMBED_REPO = os.environ.get(
    "PLD2_MINI_EMBED_REPO",
    "/flare/NLDesignProtein/bryan/FILIP-dev-space/small_scale_training/mini-embed-filip")
FILIP_CKPT = os.environ.get(
    "PLD2_FILIP_CKPT", f"{MINI_EMBED_REPO}/checkpoints/hard_masking/epoch49.pt")
# Packed per-token encoder cache: text_h.bin / text_offsets.pt / text_mask.bin plus pair_ids.json.
# The TEXT side is precomputed, so guidance never loads BioLinkBERT -- only the 768->16 projection
# head out of the checkpoint. The PROTEIN side cannot be cached: the canvas changes every step.
FILIP_CACHE = os.environ.get("PLD2_FILIP_CACHE", f"{MINI_EMBED_REPO}/cache")

# The SwissProt text<->protein pair corpus the FILIP cache was built from. It lives under PLD2's own
# DATASETS_DIR, NOT inside the mini-embed repo -- an earlier version of src/reference_set.py derived
# it from MINI_EMBED_REPO's parent and sent a job looking in FILIP-dev-space, which cost a queue
# slot. Columns are fixed by the SwissProt-full file and mirrored from mini-embed's config.DataCfg;
# captions contain commas, so anything reading this must use the csv module.
SWISSPROT_CSV = os.environ.get("PLD2_SWISSPROT_CSV",
                               f"{DATASETS_DIR}/fully_annotated_swiss_prot_080326.csv")
SWISSPROT_COLS = ("primary_Accession", "protein_sequence", "[final]text_caption")

# --- preference tuning (src/align.py and friends) ---------------------------------------------
# One directory per alignment ROUND. IRPO is iterative: round 2 must generate from the policy round
# 1 produced, on a fresh prompt set, and its pairs must not be mixed with round 1's -- data from a
# policy that no longer exists is exactly what a preference loss must not train on. The round number
# is in the path so that is structurally hard to get wrong.
ALIGN_DIR = os.environ.get("PLD2_ALIGN_DIR", f"{RUNS_DIR}/align")

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
    # 1 = amino acids only. 2 = amino acids + Foldseek 3Di at the same L positions (see
    # model.Config.n_tracks for why two tracks rather than a joint alphabet or a 2L sequence).
    # At 2, shards without a .3di sibling still train the sequence track; their structure track is
    # all-MASK and excluded from the structure loss.
    n_tracks: int = 2
    # Fraction of training rows drawn from the PAIRED (structure-carrying) corpus, the rest from
    # aa-only UniRef. Only meaningful at n_tracks=2.
    #
    #   1.0  AFDB only. Structure on every row, but 2.79B residues (~2-3 epochs at 18k steps), a
    #        narrower and shorter sequence distribution, and -- measured -- a fold ceiling that
    #        moves: natural 73.0 pLDDT / 0.568 pTM at length 165, against UniRef's 81.8 / 0.677 at
    #        250. That last one is the expensive part: the run stops being comparable to every
    #        earlier one.
    #   0.0  UniRef only, i.e. no structure signal at all. Pointless at n_tracks=2.
    #   ~0.14 what simple concatenation would give, since UniRef is several times larger. Too thin
    #        to train a second track on.
    #
    # 0.5 keeps structure on half the rows while the sequence track still sees UniRef's diversity,
    # and it keeps the fold table readable against the same baselines as the previous three runs.
    # Rounded to sixteenths (data.MixedShards._K); the training log's `lbl` column reports the
    # fraction actually achieved, which is the check that this knob does what it says.
    struct_frac: float = 0.5
    num_workers: int = 4
    prefetch_factor: int = 4         # batches prefetched per worker (only used when num_workers > 0)
    # Every Nth sequence GLOBALLY is held out for the fold/repetition baselines. Strided, not
    # by-shard: shard order is FASTA order, and reserving the last shard silently returned the
    # shortest ~1% of a length-sorted corpus (observed: a "natural" baseline of 33.9 +- 2.3 aa from
    # data filtered to 30-500). A stride is order-agnostic. 0 disables the holdout entirely.
    holdout_stride: int = 100


@dataclass
class OptCfg:
    # Per-rank MICRO-batch, as a token budget -> B = global_batch_tokens // canvas = 8 at 512.
    global_batch_tokens: int = 4096
    # Micro-batches per optimizer step. Everything else in this file counts OPTIMIZER steps.
    #
    # This is what makes a 1.35B model reachable on this cluster. The trainer all-reduces the full
    # fp32 gradient once per optimizer step, so fabric traffic goes as params / step-time: at 8
    # sequences per rank the 1.35B model would push ~3.8 GB/s per rank against the 55M run's 0.24,
    # and if the all-reduce were even 10% of a step there it would spend longer communicating than
    # computing. Accumulating 4 micro-batches divides that by 4 AND lifts the global batch from
    # 0.7M to 2.9M tokens, which is the right size for a model this large. Two problems, one knob.
    #
    # The log now reports the measured comm share of each step. If it comes back small, 16 x accum 2
    # is the same effective batch with half the micro-step overhead; if large, raise accum.
    #
    # Estimated peak ~31GB of the 64GB tile: 27GB of weights + Adam + gradient + all-reduce buffer
    # (20 B/param at 1.35B), and only ~4GB of activations because checkpoint_chunk=6 caps the
    # recompute peak at 6 middle layers instead of all 36. Memory is not the binding constraint
    # here; communication is, which is what this knob is for.
    grad_accum: int = 4
    # 2e-4 for 1.35B at a 2.9M-token batch, near GPT-3-1.3B's 2e-4 at 1M. The 3e-4 that a 55M model
    # tolerated is not safe at 25x the parameters, especially now that the T-scaled vb term is live.
    lr: float = 2e-4
    warmup_steps: int = 2_000        # ~10% of the run; generous, and cheap insurance for the
                                     # T-scaled vb term. Raise it if `skipped` climbs early.
    # OPTIMIZER steps. MEASURED at 3.85s/step on Aurora (2.83s compute + 1.02s all-reduce), not the
    # 5.63s the linear FLOP extrapolation predicted -- d=1536 GEMMs run far closer to peak than the
    # d=512 ones the estimate was calibrated on, so the model came out 1.5x faster than planned.
    #
    # 18k x 2.9M tokens = 26B real residues, which is ~0.95x the 20-per-parameter point for 1.35B,
    # in ~20h against a 24h walltime. The previous 55M run was 129x PAST that point, which is what
    # made capacity rather than data the diagnosis.
    #
    # A COMPLETED cosine beats an interrupted one: if the queue cuts this short, resubmit and it
    # resumes on the same schedule. Do not raise this to fill the walltime exactly -- the margin is
    # what stops the LR being left mid-anneal.
    total_steps: int = 18_000
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
    # --- the 3Di structure track (n_tracks=2) ---
    # Its own substitution kernel, because BLOSUM is a matrix over AMINO ACIDS and has no meaning
    # over 3Di states -- the two alphabets share their 20 letters and nothing else. Uniform is the
    # honest default: Foldseek ships a real 3Di substitution matrix (mat3di.out) and that is the
    # right input here if you can get it, but inventing a similarity structure is worse than
    # declaring none. beta=1 rows are unaffected either way (no substitution channel at all).
    # "mat3di" is Foldseek's own 3Di substitution matrix (config.MAT3DI, shipped as
    # data/mat3di.out with the binary that produced our labels). "uniform" was the honest default
    # while nothing was known; the first two-track run then measured the structure track's own local
    # grammar at H(x_i|x_i-1) = 2.17 against natural 3Di's 2.01, FLAT for all 18k steps -- the model
    # generating implausible 3Di and matching sequence to it. A uniform kernel means a quarter of
    # the structure track's corruption (at beta<1) is uniform-random 3Di, i.e. the model is asked to
    # denoise toward no structure at all, which is a plausible cause of exactly that.
    # "blosum" is refused: it scores amino-acid exchangeability and the 3Di alphabet only borrows
    # its letters.
    struct_sub_kernel: str = "mat3di"
    # ENTROPY-MATCHED TO BLOSUM, not left at 1.0. mat3di's scores span -17..+9 against BLOSUM62's
    # -4..+11, so at temp=1.0 its rows come out at 1.24 nats against BLOSUM's 2.10 -- one row puts
    # 0.98 on a single substitution. The structure track's substitution channel would then be a
    # near-deterministic process where the sequence track's is a broad one, and beta would not mean
    # the same thing on the two tracks. 2.28 is the temperature at which the two match (measured).
    struct_sub_kernel_temp: float = 2.28
    # Multiplier on the structure track's loss. 1.0 weights the two tracks equally per labelled row.
    # Lower it if the structure track dominates early; the tracks' CE values are directly comparable
    # (both are 23-state categoricals) so the log shows immediately whether they are balanced.
    struct_weight: float = 1.0
    # SPAN CORRUPTION: the SHAPE of the mask, not its amount. Two runs (55M/50k, 1.35B/18k) put pTM
    # at the shuffled baseline -- 0.184 and 0.176 against a shuffle's 0.169, with naturals at 0.677 --
    # and the held-out CE curve was flat below 50% corruption: going from 40% of the sequence visible
    # to 95% bought 0.18 nats, and 25x the parameters bought 0.08 nats of mean CE. Under i.i.d.
    # masking that is the predicted outcome, because a masked position's nearest VISIBLE neighbour is
    # under two residues away at 40% corruption, so local context saturates early and long-range
    # dependency is never on the gradient's path. Drawing the mask as a spatially correlated field
    # puts the masked positions in RUNS, so filling the middle of a hole requires reaching past it.
    # The corruption AMOUNT and every per-position marginal are unchanged (src/corruption.py).
    #
    # Each entry is a TARGET MEAN MASKED-RUN LENGTH in residues at 50% corruption. What it buys,
    # measured (src/tests_corruption.py section 7 reproduces this) -- the distance a masked position
    # must reach to find ANY revealed residue, at 50% corruption:
    #
    #   span_width      1      8     32    128
    #   achieved run  2.0    7.9   28.2   82.6     (128 undershoots: a 512 canvas at 50% corruption
    #   nearest vis.                                cannot hold many 128-residue runs)
    #     mean        1.3    3.7   12.7   39.6
    #     median      1.0    3.0    9.0   30.0
    #
    # The first column is the diagnosis as a number: under i.i.d. masking the median masked position
    # has a revealed residue DIRECTLY ADJACENT to it, which is why local context saturates so early.
    #
    # A TUPLE, assigned per row like `betas`, and 1 stays in the mix for a specific reason: the
    # sampler's mid-decode canvas is a SCATTERED set of confidence-committed positions among masks,
    # which IS the i.i.d. regime. A model trained only on wide spans would be off-distribution
    # exactly where it is used. The rest of the ladder is geometric -- 8 is secondary-structure
    # scale, 32 sub-domain, 128 domain scale on a 512 canvas. The trainer logs the measured runs per
    # width at startup and the mean masked-run length (`run`) on every log line.
    #
    # WHAT THIS COSTS: L_vb stops being an exact ELBO and becomes a mean-field surrogate (see
    # src/objective.py). L_ce, the per-position marginals, the cold start and the CE curve are all
    # unaffected. Set to (1,) to restore the original process bit-for-bit.
    span_width: tuple = (1, 8, 32, 128)
    ce_weight: float = 1.0           # EvoDiff's lambda on the x0 cross-entropy. They used 0; at
                                     # beta=1 this term IS the OADM cross-entropy, so keeping it
                                     # means the objective that demonstrably works is still a
                                     # component rather than replaced.
    vb_weight: float = 1.0           # multiplier on the (T-scaled) variational term. L_vb is a SUM
                                     # over t = 1..T and one t is sampled, so the unbiased estimator
                                     # of that sum multiplies by T -- src/objective.py does this now.
                                     # The first run omitted it: vb sat at 0.005 against ce ~1.3, so
                                     # the term that makes this a diffusion model rather than a
                                     # masked LM carried 0.4% of the gradient. Lower this only if
                                     # training destabilises with both terms live.
    # Weight on UNCORRUPTED positions in L_ce. 0.0 scores only positions the model cannot copy the
    # answer for, which is what OADM does. The first run used the D3PM convention of scoring every
    # position; with mask fraction ~ U(0,1) that put about half the objective's weight on a copy
    # task solved in the first few hundred steps, and made the logged number uninterpretable. The
    # "keep this token" signal is not lost -- it lives in L_vb, whose posterior at an uncorrupted
    # position is peaked on the current token. 1.0 restores the old behaviour.
    ce_uncorrupted_weight: float = 0.0
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
    eval_every: int = 500            # 0 disables
    eval_n: int = 4                  # sequences per RANK per eval (x world = the real sample count)
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
    # 1.0 = the model's own distribution, unmodified. ProLoopDiff needed T=0.5 (at its 181k
    # checkpoint T=1.0 folded to pLDDT 33.9, BELOW its own 37.5 shuffled baseline, while T=0.5 gave
    # 55.2 with 21% over 70) -- but that was a fix for ProLoopDiff's repetition pathology, and
    # carrying it over as a default would quietly assume PLD2 inherits the same pathology. It may
    # not: the PB bottlenecks are gone, EOS is upweighted, and substitution is a native decode move.
    # Sample at 1.0, measure, and turn it down only if the metrics say to.
    sample_temperature: float = 1.0
    sample_eos_first: bool = True    # commit the boundary before any residue (see sampler.py)
    sample_min_len: int = 30         # corpus floor; also stops EOS at position 0 (the len=0 bug)
    sample_eos_temp: float = 1.0     # <1 sharpens the length draw, >1 widens it
    # ANTI-REPETITION: OFF. It was inherited from ProLoopDiff, whose samples contained homopolymer
    # runs of 42, and PLD2 shows no sign of that pathology. Measured on the 50k checkpoint, turning
    # it off moved four statistics at once and all toward natural:
    #     LCR    0.0% -> 7.4%   (natural 7.9%)   -- matching, not merely "better"
    #     k13    0.0% -> 0.3%   (natural 0.5%)
    #     pLDDT  35.0 -> 43.4   crossing the shuffled baseline of 39.0 for the first time
    #     >70      0% -> 6%
    # An LCR of exactly 0.0% across 17k residues, against 4.7% for a random shuffle, was the tell:
    # the penalty was suppressing local composition BELOW chance -- manufacturing the pathology it
    # existed to prevent. The sweep then separated the two halves cleanly:
    #     penalty_off (penalty off, run cap KEPT)  45.3  ~= no_reppen 45.4  -> cap is harmless
    #     maxrun_off  (penalty on,  run cap off)   35.8  ~= default   35.7  -> penalty was ALL of it
    # So the periodic logit penalty is off and the hard run cap stays as free insurance.
    sample_rep_penalty: float = 0.0  # per-period logit penalty for continuing a repeat
    sample_max_run: int = 5          # hard cap on identical consecutive residues (0 disables)
    sample_rep_periods: tuple = (1, 2, 3, 4, 5)   # repeat periods scored (1 = homopolymer)
    # Noise on the COMMIT ORDER. THIS is the anti-repetition mechanism that works, and it is
    # load-bearing: the sweep's gumbel_temp=0 configuration scored the HIGHEST pLDDT of all (69.0,
    # 66% over 70) with 98.7% of residues inside a repeated 13-mer. Committing strictly
    # most-confident-first is exactly the loop the sampler docstring warns about -- a repeat is
    # maximally predictable, so it wins every slot and extends itself. Order noise breaks that; the
    # logit penalty tried to and only flattened the composition.
    # Structure-first decoding (n_tracks=2). A DECAYING BIAS toward 3Di edits over the first
    # `sample_struct_first * eval_steps` steps -- not a gate, so every slot stays eligible and the
    # cosine floor's termination guarantee is untouched. 0.0 leaves the unified ranking alone, which
    # is the honest test: 3Di is the lower-entropy track (H = 2.52 nats against 2.89 for residues)
    # so a trained model may well lead with it unprompted. Raise it only if the decode is measured
    # to commit residues before the fold they are supposed to sit in.
    sample_struct_first: float = 0.0
    sample_gumbel_temp: float = 0.1
    sample_corrector: int = 0        # post-decode corrector sweeps (see sampler._corrector_sweep)
    sample_corrector_type: str = "remask"   # or "substitution"
    # SUBSTITUTION BUDGET AT DECODE TIME, as expected edits per decodable position over the whole
    # decode. The decoder offers one candidate edit per position -- unmask if masked, substitute if
    # already committed -- and takes the highest-confidence ones, which is the inference-time mirror
    # of the training process. The scheduled unmask count is a guaranteed FLOOR so substitutions can
    # never starve the mask channel (see sampler.generate).
    #
    # NON-ZERO BY DEFAULT, because the training-time eval has to sample the way we intend to
    # generate. Substitution is a first-class move in this model's process; an absorbing-only eval
    # would measure a decoder we do not plan to ship, and every generation number in the log would
    # describe something nobody runs. 1.0 = each decodable position gets, on average, one
    # substitution opportunity across the decode.
    #
    # Costs nothing: both move types come from the same forward pass, so the eval's forward count is
    # unchanged (src/tests_sampler.py asserts it). To attribute a bad repetition number to the
    # decoder rather than the model, A/B it -- `python -m src.sample --subst-per-residue 0`.
    sample_subst_per_residue: float = 1.0

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
    # A 1.35B checkpoint is ~16GB (fp32 weights + Adam moments), so this is no longer free: at
    # keep_last=2 that is 32GB resident on flare and ~a minute of Lustre write per save.
    ckpt_every: int = 1000
    seed: int = 0


@dataclass
class AlignCfg:
    """Preference tuning, following ESM3's IRPO (Appendix A.4) with three deliberate departures.

    IPO, NOT DPO, BY DEFAULT. DPO's log-sigmoid is unbounded, so its implicit KL constraint
    collapses when preferences are near-deterministic (Azar et al.) -- and ours are deterministic by
    construction, because pairs are built from a hard metric gap rather than noisy human labels.
    Three things make drift a bigger risk here than it was for ESM3: that determinism, positives
    that are good only relative to their own prompt, and a reward that is a folder's opinion about a
    generated sequence. IPO's squared loss has a finite optimum, so the margin stops growing.
    `loss="dpo"` switches back for comparison; the two differ by four lines.

    RELATIVE, WITHIN-PROMPT THRESHOLDS, NOT ABSOLUTE ONES. ESM3 could demand pTM > 0.8 and
    cRMSD < 1.5A because its base model produced such samples in quantity. Ours clears an absolute
    bar about 1% of the time, so at n_gen=16 only ~11% of prompts would contain a single qualifying
    sample and 89% of the fold budget would be discarded. Ranking within a prompt yields a usable
    pair at every difficulty instead. success_weight is the knob that reintroduces absolute quality
    -- it upweights pairs whose winner clears the bar outright -- and it is 1.0, i.e. OFF.

    THE MARGIN IS PER POSITION. Our surrogate log-likelihood comes out of a weighted row MEAN
    (objective._weighted_row_mean), so the IPO target is a per-token nat budget rather than a
    sequence-level one: length-invariant, and a number that can be reasoned about.
    """
    dir: str = ALIGN_DIR
    round: int = 1

    @property
    def round_dir(self) -> str:
        return os.path.join(self.dir, f"round{int(self.round)}")

    # --- the reference set (src/reference_set.py) ---
    # Captioned SwissProt proteins, ESMFolded once. The scaffold's 3Di, the TM target and the
    # caption all come from that single fold, so no predictor change sits inside the measurement --
    # which is what carving prompts out of the AlphaFold-derived AFDB shards would have done.
    n_refs_factor: float = 1.2       # draw this many x n_prompts: folding and foldseek lose some
    ref_split: str = "test"          # rows the FILIP co-embedding never trained on
    ref_min_plddt: float = 0.70      # a reference ESMFold is unsure about is a bad prompt AND a bad
                                     # TM target: its 3Di is a guess in both roles

    # --- prompts (src/prompts.py) ---
    n_prompts: int = 1_000
    prompt_seed: int = 0
    # References reserved for evaluation and never used to build a pair. Without this there is no
    # set on which two checkpoints can be compared without one of them having trained on it, and
    # the eval manifest is a FILE -- carry round 1's forward to compare rounds on identical prompts.
    n_eval_prompts: int = 200
    # Mask rate = the fraction of residues the model must generate. 1.0 is the cold start and takes
    # the largest single share on purpose: it is the condition we deploy in, and the supervised half
    # of the loss would otherwise fill up with easy low-mask completions.
    mask_bins: tuple = (0.5, 0.7, 0.85, 1.0)
    bin_weights: tuple = (1.0, 1.0, 1.0, 1.5)      # -> 22/22/22/33%
    span_widths: tuple = (8, 32, 128)

    # --- generation (src/align_sample.py) ---
    # Also records the model's OWN surrogate log-likelihood for each completion. That column is the
    # whole diagnosis: best-of-N by log-likelihood is worse than a random draw (0.007 at N=1 falling
    # monotonically to 0.000 at N=8) while pLDDT selection tracks the oracle exactly. The model
    # makes good proteins and ranks them below the bad ones, and the gap between those two columns
    # is what preference tuning is for -- so watch it shrink across rounds.
    n_gen: int = 16                  # completions per prompt. This sets the CEILING: the winner is
                                     # a best-of-n draw, and the model is being taught to imitate
                                     # it, so n is the height of the target. Prompts buy fidelity to
                                     # that target; only n moves it.
    gen_temperature: float = 1.0
    gen_steps: int = 512
    # FILIP caption guidance during generation, when a prompt carries a caption. Measured: best-of-N
    # gain collapses monotonically as guidance rises (5.7x at g10, 2.7x at g30, 1.9x at g100) and by
    # N=10 the ordering reverses. Guidance and best-of-N draw on the same resource -- within-prompt
    # spread -- and spread is the raw material every preference pair is cut from. So: a little.
    gen_gamma: float = 10.0

    # --- reward (src/preference.py) ---
    # The success criterion, measured: pLDDT > 70 AND TM > 0.5 to the ESMFolded query. TM is what
    # makes this safe to optimise -- a poly-alanine helix scores well on pLDDT and nowhere on TM, so
    # the degenerate solution is not on the reward's frontier.
    plddt_success: float = 0.70
    tm_success: float = 0.50
    tm_field: str = "ttmscore"       # normalised by the TARGET, which is the natural query
    reward_plddt: float = 1.0        # reward = w_plddt * pLDDT + w_tm * TM, for RANKING only
    reward_tm: float = 1.0
    top_k: int = 2                   # winners per prompt
    bot_k: int = 2                   # losers per prompt
    # Reward gap a pair must clear. NOT zero, and this matters more than it looks: both DPO and
    # IPO treat preference as BINARY, so a mislabelled pair contributes a gradient of exactly the
    # same size as a correct one. Measured, the successes cluster into a minority of prompts --
    # oracle pass@10 is 0.040 against 0.068 for independent draws at the same per-draw rate -- so
    # most prompts produce 16 uniformly mediocre completions whose ordering is metric noise. This
    # discards them, which is what ESM3 did with a much larger gap (dpTM >= 0.2) for the same reason.
    min_gap: float = 0.05
    success_weight: float = 1.0      # >1 upweights pairs whose winner clears the absolute bar. OFF.
    matched_frac: float = 0.5        # share of pairs built matched-on-pLDDT / split-on-TM, which
                                     # holds quality fixed so the gradient can only carry
                                     # prompt-consistency. 0 disables.
    matched_plddt_tol: float = 0.05  # "same pLDDT" for that construction

    # --- the loss (src/align.py) ---
    # ESM3's alpha=0.8 / beta=0.05 DO NOT TRANSFER, and round 1 is the evidence. They
    # length-normalise only L_NLL and leave the contrastive term as a sequence-level SUM; we make
    # both a per-position mean, so our contrastive term is ~L times smaller relative to the anchor.
    # At L~250 that is a factor of 250 on the gradient and 62,500 on the loss -- measured, round 1
    # ran with the contrastive term at 0.5% of the loss and h settled 3x past its target. The
    # margin was never an attractor, because:
    #
    #     L = w_nll*(-lp_w) + alpha*(h - m)^2      dL/dlp_w = -w_nll + 2*alpha*(h - m) = 0
    #     =>  h* = m + w_nll / (2 * alpha)
    #
    # At w_nll=1, alpha=0.8 that is h* = 0.665, sixteen times the margin. Choosing alpha from the
    # tolerance you will accept instead of copying a number gives alpha = w_nll / (2 * tol).
    # src/align.py prints h* at startup so this can never be a surprise again.
    loss: str = "ipo"                # "ipo" | "dpo"
    ipo_tol: float = 0.02            # how far above the margin h may settle
    ipo_alpha: float = 25.0          # = nll_weight / (2 * ipo_tol)
    # DPO HAS NO EQUILIBRIUM -- that is Azar's point and the reason IPO is the default. Both terms
    # push lp_w the same way; what stops DPO is the sigmoid SATURATING, which happens for h >> 1/beta.
    # So beta sets the scale at which the term still has gradient, and it must match the scale of h.
    # Ours is per-position and lands around 0.1, so beta ~ 10; ESM3's 0.05 against a sequence-level
    # sum at L~250 is the same choice expressed in different units.
    dpo_alpha: float = 0.8           # ESM3's value, which is right ONCE beta is on the right scale
    beta: float = 10.0
    ipo_margin: float = 0.04         # IPO target margin, NATS PER POSITION
    nll_weight: float = 1.0          # the supervised anchor. ESM3 keeps it at 1 and leans on it.
    steps: int = 500
    lr: float = 1e-5
    warmup_steps: int = 50           # 150 was 15% of ESM3's single epoch; ours was five epochs
    grad_clip: float = 1.0
    optimizer: str = "rmsprop"       # ESM3 used RMSProp for every IRPO run
    pairs_per_rank: int = 1          # pairs per rank per optimizer step
    # EPOCHS ARE A TUNER-SCALE PROBLEM, NOT A DATA PROBLEM. Pairs consumed per step is
    # world_size * pairs_per_rank, so running phase 5 on the whole allocation is what burned 33.8
    # epochs in round 1 -- not a shortage of pairs. align.recommended_ranks() inverts the relation
    # and scripts/align.pbs sizes the tuner from it; the tuner's rank count is now independent of
    # the job's, which it has to be at 256 nodes.
    target_epochs: float = 2.0
    score_struct: bool = False       # score the 3Di track in the surrogate too, not just residues
    eval_every: int = 50
    drift_natural_n: int = 64        # held-out natural sequences for the drift monitor


@dataclass
class RunCfg:
    device: str = "auto"                 # auto -> xpu on Aurora, cpu on a laptop
    data: DataCfg = field(default_factory=DataCfg)
    opt: OptCfg = field(default_factory=OptCfg)
    align: AlignCfg = field(default_factory=AlignCfg)

    def model_config(self) -> ModelConfig:
        # Same dims as ProLoopDiff (~55M params) so the two runs are comparable. What changed is
        # what is NOT here: no pb_layers, no pb_dim, no n_pb_heads, no text_dim. Every layer is
        # d_model wide (instruction 1) and there is no conditioning pathway (instruction 5).
        # 1.35B: d=1536, 44 distinct layers, NO loop. The 55M run was undersized, not
        # undertrained -- it saw 142B residues, 129x past 20-per-parameter, and the compute already
        # spent implies a Chinchilla-optimal ~0.9-1.3B.
        #
        # n_recurrence=1 is the substantive change beyond width. The loop spends n_recurrence x the
        # middle stack's COMPUTE to avoid n_recurrence x its PARAMETERS -- the right trade when
        # memory-bound, and the 55M run used 22% of a 64GB tile. At this width, 4+12x3+4 would cost
        # 1.35B params' worth of FLOPs per token and keep only 613M of capacity; 4+36x1+4 has the
        # same applied depth and the same cost, and keeps all 1.35B.
        #
        # 24 heads keeps head_dim at 64 (1536/24), the value ESM-2 uses at every scale. Leaving it
        # at 8 would give 192-wide heads. d_ff stays 3x d_model.
        return ModelConfig(
            vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
            d_model=1536, n_heads=24, d_ff=4608,
            n_upstream=4, n_middle=36, n_downstream=4, n_recurrence=1,
            checkpoint_chunk=6,
            n_tracks=self.data.n_tracks,
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
    print("ALIGN_DIR      :", ALIGN_DIR)
    # The alignment inputs, checked here because every job script banners this output: a path that
    # is only resolved inside phase 0 fails minutes in, after the queue slot is already spent.
    print("SWISSPROT_CSV  :", SWISSPROT_CSV, " exists:", os.path.exists(SWISSPROT_CSV))
    print("FILIP_CACHE    :", FILIP_CACHE, " exists:", os.path.isdir(FILIP_CACHE))
    print("FILIP_CKPT     :", FILIP_CKPT, " exists:", os.path.exists(FILIP_CKPT))
    print(f"model          : d_model={m.d_model} d_ff={m.d_ff} heads={m.n_heads} "
          f"layers={m.n_upstream}+{m.n_middle}(x{m.n_recurrence})+{m.n_downstream} "
          f"vocab={m.vocab_size} (eos={m.eos_token_id} pad={m.pad_token_id} mask={m.mask_token_id})")
    n_par = 13 * m.d_model ** 2 * (m.n_upstream + m.n_middle + m.n_downstream) / 1e6
    print(f"params         : ~{n_par:.0f}M in {m.n_upstream + m.n_middle + m.n_downstream} distinct "
          f"layers, {m.n_upstream + m.n_middle * m.n_recurrence + m.n_downstream} applied/token "
          f"(head_dim={m.d_model // m.n_heads}, checkpoint_chunk={m.checkpoint_chunk})")
    print(f"batch          : canvas={CFG.data.canvas} micro={CFG.batch_size}/rank x "
          f"accum {CFG.opt.grad_accum} (rows split round-robin across {len(CFG.opt.betas)} betas)")
    print(f"corpora        : "
          + (f"{CFG.data.struct_frac:.0%} paired AFDB + {1 - CFG.data.struct_frac:.0%} aa-only "
             f"UniRef (rows without a structure label get an all-MASK structure track and are "
             f"excluded from the structure loss)" if m.n_tracks == 2 else "UniRef only"))
    print(f"tracks         : {m.n_tracks} "
          + ("(amino acid + Foldseek 3Di at the same L positions; separate embeddings and heads, "
             f"INDEPENDENT noise level per track) | 3Di kernel={CFG.opt.struct_sub_kernel} "
             f"struct_weight={CFG.opt.struct_weight} struct_first={CFG.opt.sample_struct_first}"
             if m.n_tracks == 2 else "(amino acids only)"))
    print("shards         :", AFDB_SHARDS if m.n_tracks == 2 else UNIREF_SHARDS,
          " exists:", os.path.isdir(AFDB_SHARDS if m.n_tracks == 2 else UNIREF_SHARDS))
    print(f"corruption     : span_width={CFG.opt.span_width}"
          + ("  (i.i.d. per position -- the original process)"
             if tuple(CFG.opt.span_width) == (1,) else
             "  (mask drawn as a correlated field; amount and per-position marginals unchanged, "
             "L_vb is a mean-field surrogate not an ELBO)"))
    print(f"objective      : betas={CFG.opt.betas} T={CFG.opt.d3pm_T} "
          f"kernel={CFG.opt.sub_kernel} | L = {CFG.opt.vb_weight}*T*E[KL] + "
          f"{CFG.opt.ce_weight}*L_ce(corrupted, uncorr_w={CFG.opt.ce_uncorrupted_weight}) | "
          f"eos_w={CFG.opt.eos_loss_weight} pad_w={CFG.opt.pad_loss_weight}")
    print(f"sampling       : T={CFG.opt.sample_temperature} steps={CFG.opt.eval_steps} "
          f"eos_first={CFG.opt.sample_eos_first} "
          f"subst/residue={CFG.opt.sample_subst_per_residue}"
          f"{' (absorbing-only decode)' if CFG.opt.sample_subst_per_residue <= 0 else ' (unified edit decode)'}")
    print(f"schedule       : {CFG.opt.total_steps} steps, warmup {CFG.opt.warmup_steps}, "
          f"lr {CFG.opt.lr}, eval every {CFG.opt.eval_every}, ckpt every {CFG.opt.ckpt_every}")
