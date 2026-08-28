# PLD2

An unconditional, sequence-only protein generator: a **looped, uniform-width** bidirectional
transformer trained jointly on the **OADM** (absorbing / order-agnostic) and **D3PM** (discrete
substitution) diffusion objectives, EvoDiff-style, on size-filtered UniRef.

Successor to [ProLoopDiff](../ProLoopDiff). Same trunk topology and much of the same hard-won
machinery, with the conditioning apparatus removed and the objective made honest.

## Why PLD2 exists

ProLoopDiff at 180k steps did not generate plausible sequences — it collapsed into repetition. Its
most unusual feature was a set of small **privileged-basis** layers inside the recurrent stack:
512 → 16 → 512 bottlenecks, placed there as an information-bottlenecked entry point for text
cross-attention. Because they lived in the *looped* middle stack, every forward pass pushed the
representation through them `n_recurrence` times. A low-rank writeback re-applied that often is a
strong attractor, and in a bidirectional denoiser a strong attractor reads as periodic output.

PLD2 removes them. Every layer is `d_model` wide, there is no cross-attention, and generation is
unconditional. If conditioning returns it will be ProteinGuide-style decoding-time steering, which
touches only output logits and needs nothing from the architecture.

## What changed, point by point

| # | Instruction | Where |
|---|---|---|
| 1 | All layers the same width | `src/model.py` — `PBBlock`/`PBConditioning`/`PBCrossAttention` deleted; one `Block` type at `d_model` |
| 2 | Keep the OADM ↔ D3PM slider | `config.OptCfg.p_d3pm` (0 = pure OADM, 1 = pure D3PM) |
| 3 | Train on **both** objectives | `src/objective.training_step` — a fixed row split runs both in one forward pass |
| 4 | ~50k steps | `config.OptCfg.total_steps = 50_000`, cosine after 2k warmup |
| 5 | No cross-attention, no labelled data | no text pathway anywhere; `src/data.py` reads UniRef shards only |
| 6 | Upweight EOS | `config.OptCfg.eos_loss_weight = 20.0`, applied in both loss branches |
| 7 | Fixed 512 canvas | `config.DataCfg.canvas = 512`; length bucketing removed entirely |
| 8 | Eval at T=0.5 → FASTA → ESMFold; repetition above the LCR scale | `src/train.generative_eval` + `src/fold_fasta.py` + `src/metrics.kmer_counts` |

### 2 & 3 — the slider is now real

ProLoopDiff's `beta` mixed the **corruption** per position (MASK vs substitution) while always
scoring with the absorbing-state surrogate cross-entropy. The D3PM half of the model was therefore
trained against the wrong likelihood; its own README listed "tight D3PM KL-ELBO" as the missing
piece. PLD2 implements it (`src/d3pm.py`), and `p_d3pm` switches the **loss along with the
corruption**:

- **OADM rows** — absorbing corruption, `n ~ U(1, 512)` positions → MASK, scored by the reweighted
  mean cross-entropy (the ARDM/OADM ELBO term, exact).
- **D3PM rows** — `t ~ U(1, T)`, `x_t ~ Cat(x_0 Q̄_t)` over the 22-token non-MASK alphabet, scored by
  `KL(q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t))` plus EvoDiff's optional λ·CE term.

Both run in **one forward pass**: rows `[0, n_d3pm)` are D3PM, the rest OADM, with `n_d3pm` a Python
int fixed for the run — one static graph, no boolean gathers, no host syncs.

Two details worth knowing:

- **EOS and PAD are in the D3PM state space.** A D3PM row has its boundary corrupted like anything
  else, so this branch trains the model to *repair a misplaced EOS* — directly on ProLoopDiff's
  failure (53% of its samples never placed EOS at all).
- **The β-schedule is calibrated, not copied.** Sohl-Dickstein's `β_t = 1/(T-t+1)` is exactly linear
  for a *uniform* base but not for BLOSUM, which is strongly self-preferring: borrowing it leaves
  44% of tokens unchanged at `t = T`, nowhere near the stationary distribution the ELBO assumes when
  it drops `L_T`. `d3pm.calibrate_betas` solves numerically for the schedule that puts BLOSUM on the
  uniform case's corruption curve — and reproduces `1/(T-t+1)` to 1e-6 for a uniform base, so
  there is one code path. `D3PMSchedule.stationary_tv()` reports the residual and warns if it is large.

### 6 — the EOS arithmetic

A 512 canvas holds ~350 residues, ~160 PAD and exactly **one** EOS. Unweighted, the token that
decides sequence length carries 0.2% of the loss. At `eos_loss_weight=20` against `pad_loss_weight=0.1`
the split is ~350 : 16 : 20, i.e. EOS is ~5% of the per-sequence loss. If EOS placement is still weak
at 10k steps, raise it; the failure mode of raising it too far is over-eager EOS, which the length
column of the eval line makes obvious immediately.

This is a loss-share argument, not a measured one — the toy in `python -m src.objective` is too
small to separate `eos_loss_weight=1` from `=20`, and says so. The measurement that matters is the
`no-EOS` column during the real run.

### 8 — repetition above what the LCR detector can see

ProLoopDiff measured repetition one way (SEG-like window-12 entropy) and penalised it another
(sampler periods 1–5, hard run cap 5). **Both are short-range.** A period-`p` repeat only puts two
copies inside a 12-residue window when `p ≤ 6`, so a 20-residue motif repeated twice 150 residues
apart is invisible to the guidance meant to prevent it *and* to the metric meant to detect it.

`src/metrics.kmer_counts` measures repetition at **k ≥ 13** — one above the SEG window, the shortest
k provably outside both mechanisms. Three numbers per k: within-sequence repeat coverage, the
distinct-k-mer ratio, and the fraction of k-mers shared across *different* samples (mode collapse,
which no per-sequence statistic can see). `python -m src.metrics` demonstrates the gap: a hidden
20-mer repeat leaves LCR at the random-sequence level and shows up only in `k13`/`k20`.

Always read these against `src/make_baselines.py`, which writes **natural** (held-out UniRef) and
**shuffled** (composition-matched) FASTAs through the same code path. Real proteins contain real
repeats; the question is never "is `rep_frac > 0`" but "how far above natural, and above the shuffle".

## Architecture

- 4 upstream + **8 distinct middle layers looped `n_recurrence=3` times** + 4 downstream — 16
  distinct layers, 32 applications. `d_model=512`, `d_ff=1536` (SwiGLU), 8 heads, ~54.6M params.
- Bidirectional (a diffusion denoiser is not causal), RoPE on self-attention Q/K only, pre-norm
  RMSNorm, zero-init gated re-injection of the upstream output at the head of each loop pass.
- Vocab 23: 20 AA + EOS(20) + PAD(21) + MASK(22). MASK is last **by contract** — the D3PM alphabet
  is exactly `logits[..., :22]`.
- **EOS is the length.** PAD after it is modelled and attended, which is what lets an all-MASK canvas
  resolve to a well-formed `[AA* EOS PAD*]`.

## Sampling

Confidence-ordered (MaskGIT) decoding with **EOS committed first**, a repetition penalty over
periods 1–5 with a hard run cap, and optional remask / substitution corrector sweeps. Defaults live
in `CFG.opt` so `src/train.py`'s eval and `src/sample.py` cannot drift apart.

Temperature defaults to **0.5**: at ProLoopDiff's 181k checkpoint, T=1.0 folded to pLDDT 33.9 —
*below* its 37.5 shuffled baseline — while T=0.5 gave 55.2 with 21% of samples over 70.

### The OADM → D3PM blended decode

Training both objectives but decoding with only the absorbing one leaves half the model unused at
generation. `d3pm_blend > 0` runs **both channels in every step**:

| channel | decides | mechanism |
|---|---|---|
| absorbing | *where* content exists | unchanged — cosine-scheduled confidence-ordered commits |
| D3PM | *what* that content is | one reverse step `p_θ(x_{t-1}\|x_t)` on every already-committed residue |

The balance shifts from OADM-like to D3PM-like on its own, from two ramps, neither hand-tuned:

- **Coverage ramps up by itself** — the D3PM channel acts on committed positions, and that set grows
  from ~0 to the whole canvas. Step 1 revises nothing; the last steps revise everything.
- **Authority ramps up as `t` anneals `t_start → 1`**, which runs *opposite to intuition*. Measured
  single-step `P(x_{t-1} ≠ x_t)`:

  | t | 400 | 300 | 200 | 100 | 50 | 10 | 1 |
  |---|---|---|---|---|---|---|---|
  | model **disagrees** | 1.1% | 0.8% | 0.8% | 1.2% | 2.2% | 10.2% | **100%** |
  | model **agrees** | 0.13% | 0.03% | 0.01% | 0.00% | 0.00% | 0.00% | **0.00%** |

  At large `t`, `Q̄_{t-1}` is near-uniform so `c = p̃ Q̄_{t-1}` is flat and cannot outvote the
  stay-mass in `Q_t`. At small `t` it is near-identity, so `c[x_t]` collapses the moment the model
  points elsewhere. This is also why `t_start` defaults to `T//10 = 50` rather than `T`: a linear
  `T → 1` ramp spends ~90% of the decode in the ~1% dead band.

**Both guarantees hold by construction, not by tuning.**

- *All masks decoded in exactly `n_steps`* — the absorbing schedule is untouched and its cosine
  target is exactly 0 on the final step. The D3PM channel **cannot** interfere, because its state
  space has no MASK to emit (`K = vocab_size − 1`). The channels are separable structurally.
- *No extra compute* — both read the **same forward pass**. A blended decode costs an unblended one
  plus one (22×22) matmul per step.

Well-formedness is preserved by restricting revision to committed **residues** and constraining the
posterior to residue columns. That second part is not redundant and cost a real bug: `p̃(EOS)=0`
does *not* make the posterior zero at EOS, because `Q̄_{t-1}[i, EOS] > 0` for every `i` — the very
property that lets the D3PM training branch learn boundary repair. A residue was revised into an
EOS and the next step's `_enforce_eos` truncated the sequence to length 0.

**The honest limit, and why it defaults to off.** This channel is a faithful amplifier of the
model's own `x₀` belief. On a deliberately overfit toy (4 memorised sequences, 6 residues corrupted
in each), two sequences went 6 errors → **0** and two went 6 → **26** and 6 → **15**. That looks
like the sampler breaking things — but the model's own `argmax p̃` on those same inputs already
differed from truth at 24/28 and 15/16 positions *before any revision ran*. The channel converged
each sequence to the model's belief, exactly as designed; on two of them the belief was wrong.
Flooring the anneal above `t=1` does not help (swept `t_end` = 1, 3, 5, 10, 20, 50 → 41, 41, 40, 40,
37, 35 total errors, all dominated by the belief), which is why there is **no `t_end` knob** — it
would look like a safety control without being one.

So it cleans up a good model and confidently corrupts a bad one, and that cannot be settled from the
mechanism alone. `sample_d3pm_blend` defaults to **0.0**; A/B it on the first real checkpoint:

```bash
python -m src.sample --n 128 --d3pm-blend 0   --out off.fasta
python -m src.sample --n 128 --d3pm-blend 1.0 --out on.fasta
```

and compare the k-mer repetition columns and folded pLDDT. Note the specific risk for PLD2's failure
mode: **repeats are high-likelihood**, so an unrestrained pull toward the model's belief could as
easily drive samples *into* repetition as out of it. The repetition penalty is applied to the
revision path for exactly this reason, and the k-mer metrics are what would show it either way.
Setting `sample_d3pm_blend` in `config.py` turns it on for the training-time eval too (which reports
a `rev N/seq` column), which is why it lives there rather than only behind a flag.

`n_steps` must stay ≈ the canvas width whenever the repetition penalty is on. The penalty scores each
position against the canvas *before* that step's commits, so co-committed positions are invisible to
one another; measured at `max_run=5` on a 128 canvas, the longest homopolymer was 42 at 8
commits/step, 26 at 2/step, and exactly the cap at 1/step. The sampler warns if you get this wrong.

## Layout

```
config.py             # owns ALL paths + model/data/opt config (python config.py prints them)
src/
  model.py            # looped uniform-width transformer (no PB layers, no cross-attention)
  blosum.py           # BLOSUM62 -> substitution matrix + doubly-stochastic D3PM base (Sinkhorn)
  d3pm.py             # calibrated beta schedule, Q/Qbar stacks, posteriors, reverse step, KL
  objective.py        # OADM + D3PM losses, EOS/PAD weighting, one-forward training_step
  sampler.py          # confidence-ordered decoding, EOS-first, rep penalty, D3PM blend, correctors
  metrics.py          # SEG-LCR + long-range k-mer repetition (k > SEG window)
  data.py             # tokenizer, shard reader w/ holdout, stateless step-keyed batch sampler
  train.py            # training loop + training-time generative eval -> FASTA
  sample.py           # ad-hoc unconditional sampling -> FASTA
  make_baselines.py   # held-out natural + shuffled reference FASTAs
  fold_fasta.py       # ESMFold on FASTA -> JSONL, resumable, --watch mode
  preprocess_fasta.py # UniRef FASTA -> packed uint8 shards
  dist.py             # Aurora XPU + oneCCL bootstrap, grad/stat all-reduce buffers
  xpu_linalg_guard.py # CPU round-trip for torch.linalg ops that lack XPU kernels
scripts/              # pbs_common.sh, train.pbs, fold.pbs, sample.pbs, preprocess.pbs
```

## Run

```bash
python config.py                       # resolve and print every path a run will use

# local smoke: real model dims, 128 canvas, both objective branches, eval + FASTA + checkpoint
PLD2_UNIREF_SHARDS=/path/to/shards python -m src.train --smoke --fresh --device cpu

# self-tests, each with a stated expectation
python -m src.model            # uniform widths; init cross-entropy ~ ln(23)
python -m src.blosum <mat>     # row/doubly-stochastic checks; A->S I->V K->R W->Y
python -m src.objective        # both objective branches wired up and optimising
python -m src.metrics          # a hidden 20-mer repeat is invisible to LCR, visible at k13

# Aurora
qsub scripts/train.pbs                 # training + co-scheduled ESMFold watcher
python -m src.fold_fasta --summarize   # the results table, any time, no GPU
```

## Data

`config.UNIREF_SHARDS` points at the packed shards **ProLoopDiff already built** with the same
30–500 aa filter, so there is nothing to preprocess. `src/preprocess_fasta.py` + `scripts/preprocess.pbs`
rebuild them if you want a different size window.

`data.ProteinShards` reserves the **last shard** as a held-out split, which `src/make_baselines.py`
draws its reference sequences from. ProLoopDiff had no held-out split at all.

## Folding is a separate process, always

ESMFold on Aurora aborts with `Segmentation fault from GPU ... NotPresent` for reasons that survived
four hypotheses and four refutations (oneCCL — a 1-rank run crashed identically; sequence content —
scrambled naturals fold fine; unpatched aten fallbacks — `PYTORCH_DEBUG_XPU_FALLBACK` printed nothing
beyond det/svd; memory pressure — 14.2 GiB peak of 64, flat, dying on the *first* row). It also
installs process-global monkey-patches on `torch.linalg` and `F.linear`.

So PLD2 does not try to prevent the crash; it makes it cheap. The trainer only ever writes FASTA.
`src/fold_fasta.py` runs in its own single-rank process, fsyncs every scored sequence, and skips ids
already present on restart — a crash 60 into 100 costs 40, not 100.

## Deviations from ProLoopDiff beyond the eight instructions

Three, all documented at their definitions:

1. **Initialisation.** PyTorch's default `nn.Embedding` init is `N(0, 1)`, unscaled; with tied
   embeddings that put step-0 cross-entropy at ~210 nats instead of `ln(23) = 3.14`. PLD2 uses the
   GPT-2 scheme (`N(0, 0.02)`, residual-writing projections scaled by `1/√(2·32)`). Measured init CE
   is now 2.43 (`python -m src.model`). This is a fix, not a design change.
2. **Stateless data sampling.** Batches derive from `(seed, step, rank)` alone, so a resumed run
   walks *exactly* the batch sequence the original would have. ProLoopDiff's epoch-seeded permutation
   did not — its `repro.py` documents resumed jobs faulting a few steps past a checkpoint the
   original had sailed through. It also removes a ~1.2 GB per-rank permutation array.
3. **`min_len` in the correctors.** ProLoopDiff's corrector sweeps bypassed the EOS floor, so one
   unlucky resample near the N-terminus truncated the whole sequence. Measured on an untrained
   checkpoint: mean length 88 → 2.2. The floor is now threaded through.

Not changed, and worth an ablation later: `tie_embeddings=True`. At |V|=23 the tie saves 11,776
parameters out of 54.6M — no real benefit, and it constrains the output geometry to the input's.
Left as-is so PLD2's comparison against ProLoopDiff stays clean; `Config.tie_embeddings=False` flips it.

## What to watch during the run

The `[eval]` line every 1000 steps carries the whole story:

```
[eval] step   12000 | len 287.4+-91.2 | no-EOS 3/1440 | LCR 4.1% | k13 rep 2.2% dist 0.994 shar 0.4% | ...
```

- **`no-EOS`** should collapse toward zero early. If it does not, `eos_loss_weight` is too low.
- **`len`** should settle near the corpus mean with real spread. A collapsing `sd` means the length
  prior is degenerating.
- **`LCR`** catches short-range degeneration, **`k13/k20/k30 rep`** catches the long-range kind the
  sampler's penalty cannot touch, and **`shar`** catches collapse across samples.
- **pLDDT / pTM** arrive separately in the `[fold]` table. The number that matters is not pLDDT
  itself but pLDDT *relative to the shuffled row*: ProLoopDiff's 181k checkpoint at T=1.0 scored
  below its own shuffled baseline, which is what "learned composition, not structure" looks like.
