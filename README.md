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
| 2 | Keep the OADM ↔ substitution slider | `config.OptCfg.betas` — β is a parameter of one corruption process |
| 3 | Train on **both** move types | `src/objective.training_step` — one ELBO covers unmasking *and* substitution |
| 4 | ~50k steps | `config.OptCfg.total_steps` — 50k at 55M; 18k at 1.35B, cosine after 2k warmup |
| 5 | No cross-attention, no labelled data | no text pathway anywhere; `src/data.py` reads UniRef shards only |
| 6 | Upweight EOS | `config.OptCfg.eos_loss_weight = 20.0`, applied in both loss branches |
| 7 | Fixed 512 canvas | `config.DataCfg.canvas = 512`; length bucketing removed entirely |
| 8 | Eval → FASTA → ESMFold; repetition above the LCR scale | `src/train.generative_eval` + `src/fold_fasta.py` + `src/metrics.kmer_counts` |

### 2 & 3 — one process with MASK absorbing, β as its parameter

PLD2 is **a single diffusion whose absorbing state is MASK**. Per step, a non-MASK position

    stays          with probability 1 − c_t
    → MASK         with probability β · c_t          (absorbing)
    → substitute   with probability (1 − β) · c_t    (BLOSUM-weighted, zero diagonal)

so β is literally *"of the corruption events that happen, what fraction are maskings"*. β=1 is
exactly OADM's corruption; β<1 enriches **the same trajectories** with substitution moves.

**Why MASK-absorbing rather than EvoDiff's D3PM.** EvoDiff's D3PM corrupts toward a *uniform*
stationary distribution with no mask state, so it cold-starts generation from uniform-random amino
acids and never gets the clean "this position is definitely unknown" signal or the implicit
known-count signal. That is a large part of why it lost to OADM and failed to scale. Here MASK is
absorbing and reachable from every state at every β > 0, so:

- the stationary distribution is **all-MASK for the whole family** — generation always cold-starts
  from a fully-masked canvas, and β does not change what the sampler starts from;
- the known-count signal that makes OADM scale is preserved;
- `L_T` is genuinely ~0 rather than dropped on credit — measured `P(x_T = MASK)` is 0.999–1.000 at
  every β, i.e. `q(x_T|x_0)` *is* the canvas the sampler starts from;
- and the BLOSUM-graded substitution channel is enriched into those same trajectories, buying the
  learned token→token repair that pure absorbing genuinely lacks.

**One ELBO, not two objectives.** `L = L_vb + λ·L_ce` over the unified `Q`. `L_vb` covers *both*
move types in one term — unmasking an absorbed position and substituting a placed one — which is the
whole point of doing it this way. At `t=1` it reduces to `L_0 = −log p_θ(x_0|x_1)` with no special
case. `λ` defaults to 1 (EvoDiff used 0) because at β=1 that term *is* the OADM cross-entropy, so
the objective that demonstrably works remains a component rather than being replaced.

**Span corruption: β sets *what* is corrupted, `span_width` sets *where*.** Two runs — 55M/50k and
1.35B/18k — put generated pTM at the shuffled baseline (0.184 and 0.176 against a shuffle's 0.169,
with naturals at 0.677), and the held-out CE curve was nearly flat below 50% corruption: going from
40% of the sequence visible to 95% bought 0.18 nats, and 25× the parameters bought 0.08. Under
i.i.d. per-position masking that is the *predicted* outcome — the median masked position has a
revealed residue **directly adjacent** to it, so local context saturates almost immediately and
long-range dependency never lands on the gradient's path. The result was a competent local infiller
(ppl 7.6 at 5% corruption, in the range published protein MLMs report) that cannot place a fold.

`span_width` draws the mask as a spatially correlated field instead of independent coin flips, so
masked positions arrive in **runs** and filling the middle of a hole means reaching past it. Each
entry is a target mean masked-run length in residues; measured at 50% corruption:

| `span_width` | 1 | 8 | 32 | 128 |
|---|---|---|---|---|
| achieved mean run | 2.0 | 7.9 | 28.2 | 82.6 |
| distance to nearest visible residue (mean) | 1.3 | 3.7 | 12.7 | 39.6 |
| — (median) | 1.0 | 3.0 | 9.0 | 30.0 |

It is a **tuple assigned per row**, like `betas`, and `1` stays in the mix deliberately: mid-decode
the sampler's canvas is a *scattered* set of confidence-committed positions among masks, which is the
i.i.d. regime, so a model trained only on wide spans would be off-distribution exactly where it gets
used. The field is Gaussian-smoothed white noise under a circular rank threshold — Gaussian because a
box filter's sinc sidelobes fragment its own level sets (at box width 128, 29% of "spans" were a
single position), circular because zero padding under-corrupts the ends (0.257 against a 0.300
target over the first 8 positions).

**What span corruption costs, stated plainly.** The corruption *amount* and every *per-position
marginal* are exactly unchanged — `P(x_t = MASK | x_0)` is still exactly `Qbar_t[x_0, MASK]` at every
position — so `L_ce`, the all-MASK stationary distribution, the cold start and the CE curve stay
directly comparable to both earlier runs, and **the sampler is untouched** (it runs the model's
reverse process, which is factorised either way). What is lost is the exact ELBO: D3PM's
`q(x_{t−1}|x_t,x_0)` factorises over positions only because the forward corruption does, so `L_vb`
becomes a **mean-field surrogate** — the same computation, still a sensible denoising objective, but
no longer a bound on `−log p(x_0)`. A `vb` figure from a span run is comparable only to other span
runs. Set `span_width = (1,)` to restore the original process bit-for-bit.

---

## The 3Di structure track (`n_tracks = 2`)

Two runs put generated pTM at the shuffled baseline, and span corruption moved it only part way
(0.190 → 0.222 against 0.677 for naturals). Sequence-only any-order diffusion learns local
composition and secondary-structure propensity; global fold is a constraint that lives in structure
space. A **Foldseek 3Di token encodes which residue a position sits against in 3D** — long-range
information expressed as a per-position symbol — which is exactly the channel the sequence-only
model has no way to represent.

**Corpus** (`src/preprocess_3di.py` → `AFDB_SHARDS`): 17,070,828 paired records, verified aligned
(zero header or length mismatches), 15,316,921 inside the 30–500 window for 2.79B residues. The
pairing is checked per record and a mismatch aborts the run — a shifted corpus would attach the
wrong structure to every sequence, train happily, and be invisible downstream.

### Why two tracks at L, and not SaProt's joint alphabet

| layout | verdict |
|---|---|
| **2 × L, separate embeddings + heads** | **chosen** |
| 1 × L over a 22×22 alphabet (SaProt) | buys ~nothing, costs the corruption machinery |
| 1 × 2L, one sequence | doubles length, quadruples attention |

A joint alphabet exists to capture **within-position** correlation between the two channels. On this
corpus that correlation is **I(aa; 3Di) = 0.044 nats — 1.8% of H(3Di)**, and H(aa | 3Di) = 2.845
against H(aa) = 2.890. There is essentially nothing there to capture. That is not a defect: a single
residue barely constrains its own local 3D environment, which is determined by its *neighbours* —
which is precisely why 3Di is worth carrying as a separate channel rather than folded into the
residue token. The tracks are complementary, not redundant, and the information the structure track
supplies is at the whole-chain level, not the position level.

Against that ~0.044 nats, the joint alphabet costs: D3PM's `Q`/`Qbar` stacks are
`(n_beta, T+1, V, V)`, which is **4.2 MB at V=23 and 1.9 GB *each* at V=484**, against a measured
36.7 GB peak on a 40 GB tile. And a product alphabet with a product transition kernel is
*mathematically identical* to two independent tracks — it just pays V² for it. The 2L layout instead
doubles the sequence (4× attention, or a 256-residue canvas that truncates most of the corpus).

Two further things two tracks get for free: the aa embedding is **shared** across all structural
contexts (a joint alphabet gives "A with structure V" and "A with structure D" no common
parameters), and an **unlabelled** row is expressible as an all-MASK structure track with its loss
masked off, rather than needing a wildcard sub-alphabet.

The cost, stated plainly: two heads assume the tracks are conditionally independent *within one
decoding step*. That is worth 0.044 nats by the measurement above, and the sequential decoder
recovers even that — commit the 3Di token, re-run, and the residue head conditions on it.

### The two choices that make the track do any work

**Independent noise level per track.** Sharing `t` would mask the same fraction of both at once, so
at every position sequence and structure would be revealed or hidden together and neither could ever
inform the other — two parallel unconditional generators sharing a trunk. Drawing `t`, `beta` and
`span_width` independently per track fills every batch with rows that are structure-revealed and
sequence-masked (inverse folding) and rows the other way round (structure prediction). Those are the
rows that force cross-track conditioning. `src/tests_tracks.py` asserts the two mask fractions are
uncorrelated (measured r = −0.02; a shared `t` would pin it at +1.0).

**One confidence ranking over both tracks.** A 3Di edit and a residue edit at the same position are
*separate candidates* on one scale, so the decoder can lay down a fold and fill sequence into it —
which a joint token cannot express at all, since committing one commits both channels at once.
`sample_struct_first` biases the ordering if it does not emerge on its own; it is a decaying bias,
never a gate, so the cosine floor's termination guarantee is untouched.

**One molecule, one boundary.** Only the sequence track carries EOS; the structure track marks the
same boundary by where its PAD begins. Giving it its own EOS let the tracks disagree about the
length whenever the boundary moved, silently unaligning the decoded pair.

**The eval writes both tracks** — `step_XXXXXXXX.rankNNN.fasta` and `.3di.fa` with matching record
ids — so self-consistency is measurable: fold the sequence, 3Di-encode the result, compare it to the
structure the model said it was building. Note the `.3di.fa` extension: `src/fold_fasta.py` globs
`*.fasta`, and 3Di reuses the amino-acid letters, so a structure file reaching ESMFold would parse
cleanly and produce confident nonsense rather than an error.

---

**The β-schedule is closed-form, not calibrated.** Pinning the cumulative mask fraction to the
linear target `t/T` gives `β·c_t = 1/(T−t+1)`, hence `c_t = min(1, 1/(β(T−t+1)))`. The clamp binds
only over the last ⌈1/β⌉ steps, where survival is already ~1e-3. Measured mask fraction at every β:
`{0: 0.0, 100: 0.2, 250: 0.5, 400: 0.8, 500: ≈1.0}` — exactly linear. The numerical Sinkhorn +
bisection calibration an earlier version needed is gone along with the uniform-stationary
requirement that motivated it.

**A distribution over β is the default**, `(1.0, 0.9, 0.75, 0.5)`, assigned round-robin across rows.
This is only coherent *because* there is one process: β selects which transition matrix a row is
corrupted by, and one ELBO scores them all. Mixing β under a two-objective design would blur two
different bounds together — which is exactly ProLoopDiff's mistake. Measured substitution footprint
among still-unmasked positions at `t=T/2`: **0%, 7.4%, 20.3%, 47.8%**.

### Decoding mirrors the process

Every step, each position proposes exactly one candidate edit — **unmask** if masked, **substitute**
if already committed — and the highest-confidence edits win. No staged decode-then-correct pass, no
second schedule: substitution is a first-class move throughout, which is what the objective trains.

Both are scored by the model's probability for the token being written. Substitutions are *drawn*
from a distribution with the incumbent removed (an edit has to change something) but *scored* under
the original — that asymmetry is the mechanism. A position the model is already happy with has all
its mass on the current token, so its best alternative scores near zero and never wins a slot.
Renormalising would destroy exactly that signal.

**The unmask quota is a floor, and has to be.** A single global ranking can starve the mask channel:
if substitutions keep out-scoring unmask edits, masks never commit and the canvas never resolves.
So the cosine schedule's unmask count is taken first and guaranteed, and only the remaining budget is
ranked globally (where an unmask can still win, so the floor is never a cap). Termination is
therefore exactly as before — the cosine target is 0 at the final step.

`subst_per_residue` is the budget, in expected edits per decodable position over the whole decode.
The per-step allowance is constant, so early steps (few committed residues) spend little and late
steps spend it all — the OADM-like → substitution-rich ramp falls out of the canvas filling up
rather than needing a schedule. It defaults to **1.0** — one substitution opportunity per decodable position across the decode —
so the training-time eval measures the decoder we actually intend to use.

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
periods 1–5 with a hard run cap, and — when `subst_per_residue > 0` — substitution as a first-class
move alongside unmasking (see *Decoding mirrors the process*). Optional post-decode corrector sweeps
remain. Defaults live in `CFG.opt` so `src/train.py`'s eval and `src/sample.py` cannot drift apart.

### What the sampler sweep measured (50k checkpoint)

| configuration | pLDDT | >70 | LCR | k13 | len | |
|---|---|---|---|---|---|---|
| natural | 81.8 | 80% | 7.9% | 0.5% | 250 | |
| `nr_no_gumbel` | **69.0** | **66%** | 100% | **98.7%** | 272 | **degenerate** |
| `nr_t0.8` | 54.1 | 17% | 35.4% | 7.4% | 272 | **degenerate** |
| `nr_no_eos_1st` | 50.2 | 2% | 6.3% | 0.0% | **67** | length artifact |
| `no_reppen` / `penalty_off` / `nr_recur6` | ~45.4 | 2% | 7.5% | 0.0% | 272 | **best clean** |
| shuffled | 39.0 | 2% | 4.7% | 0.0% | 250 | |
| `maxrun_off` / `default` | ~35.7 | 0% | 0.0% | 0.0% | 272 | |

**The two highest-pLDDT rows are both traps.** `nr_no_gumbel` scores 69.0 with 66% of samples over
70 — and 98.7% of its residues sit inside a repeated 13-mer. ESMFold is confident about simple
repetitive structure, so on pLDDT alone that reads as a 34-point win. The k-mer metric
(instruction 8) is the only thing that makes it visible; the summary now flags such rows as
`DEGENERATE` against the natural row's own rate. `nr_no_eos_1st`'s 50.2 is a length artifact — its
chains are 67 residues, and short chains score higher.

This also casts doubt on ProLoopDiff's "T=0.5 beats T=1.0 on pLDDT" finding, which was never checked
against a repetition metric: `nr_t0.8` reproduces exactly that trade here (+9 pLDDT, k13 0.0% → 7.4%).

Three clean conclusions: the **periodic logit penalty was the entire damage** (`maxrun_off` 35.8 ≈
`default` 35.7 vs `penalty_off` 45.3 ≈ `no_reppen` 45.4), so the hard run cap stays as free
insurance; **gumbel commit-order noise is the anti-repetition mechanism that actually works**; and
**inference-time recurrence buys nothing** (`nr_recur6` 45.5 vs 45.4), so decode compute is not the
binding constraint.

The **anti-repetition penalty defaults to off** (`sample_rep_penalty=0`, `sample_max_run=0`). It
came from ProLoopDiff, whose samples had homopolymer runs of 42; PLD2 shows no such pathology, and
measured on the 50k checkpoint the penalty cost ~8 pLDDT and every sample above 70 while driving SEG
low-complexity to exactly 0.0% — *below* the 4.7% a random shuffle produces. Turning it off put LCR
at 7.4% against natural's 7.9%. `scripts/sweep.pbs` re-tests the two halves separately.

**Bounded wall time.** Both move types are chosen from the *same* forward pass and nothing iterates
to convergence, so model forwards are exactly `1 (eos_first) + n_steps + n_corrector × (2 if remask
else 1)` regardless of the substitution budget. `python -m src.tests_sampler` asserts that count
alongside well-formedness across 36 option combinations.

**The eval samples the way we intend to generate**, which is the whole reason every `generate()`
argument lives in `CFG.opt` rather than in the callers. Two defaults follow from that:

*Temperature is **1.0*** — the model's own distribution, unmodified. ProLoopDiff needed T=0.5 (at
its 181k checkpoint T=1.0 folded to pLDDT 33.9, *below* its own 37.5 shuffled baseline, while T=0.5
gave 55.2 with 21% over 70), but that was a fix for ProLoopDiff's repetition pathology. Carrying it
over as a default would quietly assume PLD2 inherits the same pathology — and the PB bottlenecks are
gone, EOS is upweighted, and substitution is now a native decode move. Sample at 1.0, measure, turn
it down only if the metrics say to.

*`subst_per_residue` is **1.0***, not 0 — substitution is a first-class move in this model's
process, so an absorbing-only eval would measure a decoder we do not plan to ship. It costs nothing
(both move types come from the same forward pass). To attribute a bad repetition number to the
decoder rather than the model, A/B with `python -m src.sample --subst-per-residue 0`.

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
  corruption.py       # the unified process: absorbing MASK + BLOSUM substitution, Q/Qbar, ELBO
  objective.py        # the single ELBO (L_vb + lambda L_ce), EOS/PAD weighting, training_step
  sampler.py          # unified edit decoding (unmask|substitute), EOS-first, rep penalty
  metrics.py          # SEG-LCR + long-range k-mer repetition (k > SEG window)
  data.py             # tokenizer, shard reader w/ holdout, stateless step-keyed batch sampler
  train.py            # training loop + training-time generative eval -> FASTA
  sample.py           # ad-hoc unconditional sampling -> FASTA
  make_baselines.py   # held-out natural + shuffled reference FASTAs
  tests_sampler.py    # cross-product regression: well-formedness + exact forward count
  tests_corruption.py # the corruption math, checked against independent computations
  inspect_shards.py   # shard length distribution, corpus ordering, .idx/.bin integrity
  ce_curve.py         # held-out CE by corruption level vs uniform/unigram -- what the loss hides
  sweep_sampler.py    # one FASTA per sampler config, scored side by side by the fold pipeline
  fold_fasta.py       # ESMFold on FASTA -> JSONL, resumable, --watch mode
  preprocess_fasta.py # UniRef FASTA -> packed uint8 shards
  dist.py             # Aurora XPU + oneCCL bootstrap, grad/stat all-reduce buffers
  xpu_linalg_guard.py # CPU round-trip for torch.linalg ops that lack XPU kernels
scripts/              # pbs_common.sh, train.pbs, fold.pbs, sweep.pbs, sample.pbs, preprocess.pbs
```

## Run

```bash
python config.py                       # resolve and print every path a run will use

# local smoke: real model dims, 128 canvas, both objective branches, eval + FASTA + checkpoint
PLD2_UNIREF_SHARDS=/path/to/shards python -m src.train --smoke --fresh --device cpu

# self-tests, each with a stated expectation
python -m src.model            # uniform widths; init cross-entropy ~ ln(23)
python -m src.blosum <mat>     # row/doubly-stochastic checks; A->S I->V K->R W->Y
python -m src.objective        # one loss trains both move types; corruption line shows both
python -m src.metrics          # a hidden 20-mer repeat is invisible to LCR, visible at k13
python -m src.tests_sampler    # 36 decode configs: well-formed, and exactly N forwards
python -m src.tests_corruption # MASK absorbing, L_T ~ 0, Qbar vs explicit product, KL identities

# diagnosis, when generations fold at the floor
python -m src.ce_curve         # does the model know anything at COLD START, or only composition?
python -m src.sweep_sampler    # is the sampler destroying what it does know?
qsub scripts/sweep.pbs         # ...the same thing on Aurora: generate, fold, one table

# Aurora
qsub scripts/train.pbs                 # training + co-scheduled ESMFold watcher
python -m src.fold_fasta --summarize   # the results table, any time, no GPU
```

## Data

`config.UNIREF_SHARDS` points at the packed shards **ProLoopDiff already built** with the same
30–500 aa filter, so there is nothing to preprocess. `src/preprocess_fasta.py` + `scripts/preprocess.pbs`
rebuild them if you want a different size window.

`data.ProteinShards` holds out **every 100th sequence globally** (`holdout_stride`), which
`src/make_baselines.py` draws its reference sequences from. ProLoopDiff had no held-out split at all.

The stride is not incidental. An earlier version reserved the *last shard*, which is only unbiased
if the corpus order is unbiased — and shard order is FASTA order. On the real UniRef shards that
returned a "natural" baseline of **33.9 ± 2.3 aa** from a corpus filtered to 30–500: the shortest
~1% of the data, pinned against the floor, with a spread far too tight for any random draw. The
reader was fine; the split was wrong. A stride is order-agnostic.

`python -m src.inspect_shards` reports the per-shard length distribution, a rank correlation between
shard index and mean length (with an effect size, so a handful of near-identical shards isn't
flagged as "sorted"), and each `.idx`'s final offset against its `.bin` size. That last check matters
because numpy slices a memmap past its end **without error** — a mismatched pair would feed silently
truncated sequences into training, and `ProteinShards` now refuses to open one.

## Folding is a separate process, always

ESMFold on Aurora aborts with `Segmentation fault from GPU ... NotPresent` for reasons that survived
four hypotheses and four refutations (oneCCL — a 1-rank run crashed identically; sequence content —
scrambled naturals fold fine; unpatched aten fallbacks — `PYTORCH_DEBUG_XPU_FALLBACK` printed nothing
beyond det/svd; memory pressure — 14.2 GiB peak of 64, flat, dying on the *first* row). It also
installs process-global monkey-patches on `torch.linalg` and `F.linear`.

So PLD2 does not try to prevent the crash; it makes it cheap. The trainer only ever writes FASTA.
`src/fold_fasta.py` runs in its own process, fsyncs every scored sequence, and skips work already
done on restart — a crash 60 into 100 costs 40, not 100.

It runs on **all 12 tiles of the fold node with no process group**. The distinction matters: what
faulted was oneCCL's node-local Level-Zero IPC peer mappings, not rank count — 12 ranks *with* a
process group died 3 for 3 in the IPC address range, while 1 rank (which returns before
`init_process_group`) folded cleanly. `dist.init_distributed(no_dist=True)` reads the MPI topology
and pins each tile without ever initialising oneCCL. Ranks need no collective: they partition by a
stable CRC32 of each sequence id — *not* a stride over the todo list, which shifts under them in
`--watch` mode as any rank writes — and each appends to its own `<out>.rankNNN.jsonl`, since
concurrent appends to one file on Lustre can interleave mid-line. This is reasoned but not yet proven
at 12 ranks; `FOLD_RANKS_PER_NODE=1` is the fallback with direct evidence behind it.

**Resume is keyed on (id, sequence), not id.** An id is `<file stem>|<fasta header>` — pure position.
Regenerating `natural.fasta` with entirely different sequences reuses `natural|natural_0…199`, so an
id-only key silently reported "400 already scored | 0 to do" for two files whose contents had
completely changed. The summary also keeps the newest record per id, so stale rows heal themselves
rather than needing the JSONL deleted.

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
  itself but pLDDT *relative to the shuffled row*: a checkpoint scoring below its own shuffled
  baseline is what "learned composition, not structure" looks like.
- **LCR below the *shuffled* row is a red flag, not a good sign.** Chance alone produces ~4.7%;
  anything under that means the sampler is suppressing local compositional structure that real
  proteins have. The first PLD2 run posted 0.0% across 17k residues — see `src/sweep_sampler.py`.

If pLDDT sits at the floor and does not move with training, two very different things can cause it
and they need different fixes. `python -m src.ce_curve` asks whether the model knows anything at the
100%-corrupted end where generation actually starts; `python -m src.sweep_sampler` asks whether the
sampler is destroying what it does know. Run both before changing anything.
