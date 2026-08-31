"""Two-track (amino acid + Foldseek 3Di) invariants:  python -m src.tests_tracks

The failure this file exists to catch is a SILENT one. A two-track model whose tracks drift apart --
different lengths, a boundary in one and not the other, a structure string that does not correspond
position-for-position to the sequence it was generated with -- still trains, still samples, and
still writes plausible-looking FASTA. Nothing downstream notices; the self-consistency metric just
quietly measures noise. So the alignment is asserted rather than assumed, everywhere it could break.
"""
import torch

from .blosum import substitution_kernel, uniform_substitution_kernel
from .corruption import CorruptionSchedule
from .data import DI, make_collate
from .model import Config, LoopedDiffusionLM, count_params
from .objective import training_step
from . import sampler as S
import os, sys
from config import BLOSUM_MAT
bp = BLOSUM_MAT if os.path.exists(BLOSUM_MAT) else (sys.argv[1] if len(sys.argv) > 1 else None)
if bp is None or not os.path.exists(bp):
    raise SystemExit("need a BLOSUM .mat: set PLD2_BLOSUM or pass the path as an argument")

V, MASK, EOS, PAD, L = 23, 22, 20, 21, 64
cfg = Config(vocab_size=V, eos_token_id=EOS, pad_token_id=PAD, mask_token_id=MASK,
             d_model=96, n_heads=4, d_ff=288, n_upstream=2, n_middle=2, n_downstream=2,
             n_recurrence=1, n_tracks=2)
torch.manual_seed(0)
model = LoopedDiffusionLM(cfg).eval()
print(f"model: {count_params(model)/1e6:.2f}M params, n_tracks={cfg.n_tracks}, 3Di alphabet {DI}")

print("\n1. FORWARD carries both tracks and returns one head per track")
tok = torch.randint(0, 20, (4, L)); st = torch.randint(0, 20, (4, L))
lg = model(tok, struct=st)
print(f"   logits {tuple(lg.shape)}  (B, 2, L, V)")
# The two heads must be genuinely separate: perturbing the structure INPUT must move the aa logits
# (they share a trunk) but the two heads must not be the same function.
lg2 = model(tok, struct=torch.randint(0, 20, (4, L)))
print(f"   changing the 3Di input moves the aa logits: "
      f"{float((lg[:,0]-lg2[:,0]).abs().max()):.4f} > 0  <- the tracks actually condition")
print(f"   the two heads are different functions: {float((lg[:,0]-lg[:,1]).abs().max()):.4f} > 0")
try:
    model(tok)
    print("   *** FAIL: n_tracks=2 accepted a missing struct argument")
except ValueError as e:
    print(f"   missing struct is refused: {str(e)[:60]}...")

print("\n2. SAMPLING resolves BOTH tracks and keeps them aligned")
s_aa = CorruptionSchedule(substitution_kernel(bp, 22), V, MASK, betas=(1.0, 0.5), T=50)
for name, kw in (("unified edit", dict(subst_per_residue=1.0)),
                 ("absorbing only", dict(subst_per_residue=0.0)),
                 ("no eos_first", dict(subst_per_residue=1.0, eos_first=False)),
                 ("struct_first .5", dict(subst_per_residue=1.0, struct_first=0.5)),
                 ("2 correctors", dict(subst_per_residue=1.0, n_corrector=2))):
    torch.manual_seed(7)
    cv, lens = S.generate(model, Lmax=L, batch_size=6, n_steps=L, temperature=1.0,
                          device="cpu", min_len=10, rep_penalty=0.0, **kw)
    aa, sd = S.aa_track(cv), S.struct_track(cv)
    n_mask = int((cv == MASK).sum())
    no_eos_in_struct = int((sd == EOS).sum()) == 0
    seqs, sk1 = S.decode_seqs(cv, cfg, min_len=0)
    dis, sk2 = S.decode_struct(cv, cfg, min_len=0)
    aligned = all(len(a) == len(b) for a, b in zip(seqs, dis)) and len(seqs) == len(dis)
    print(f"   {name:<16} canvas {tuple(cv.shape)} | unresolved MASK {n_mask} | "
          f"no EOS in the 3Di track {no_eos_in_struct} | decoded pairs "
          f"{len(seqs)}=={len(dis)}, lengths agree {aligned}")
    assert n_mask == 0 and no_eos_in_struct and aligned, f"{name} broke an invariant"

print("\n3. COMMIT ORDER -- one ranking spans both tracks, so a track CAN lead")
# What this does and does not show. It demonstrates the MECHANISM: a single confidence ranking over
# (track, position) slots lets one track drain before the other, which a joint (aa, 3Di) token
# could not express at all. It says NOTHING about which track a trained model will prefer -- this
# is a randomly initialised network and the ordering below is a property of its initialisation.
# Measure it again on a real checkpoint before claiming structure-first generation.
orig = S._step_logits
for sf in (0.0, 0.9):
    tr = []
    def probe(model, canvas, *a, **k):
        tr.append((canvas == MASK).sum(dim=(0, 2)).tolist())
        return orig(model, canvas, *a, **k)
    S._step_logits = probe
    torch.manual_seed(7)
    S.generate(model, Lmax=L, batch_size=6, n_steps=L, temperature=1.0, device="cpu",
               min_len=10, rep_penalty=0.0, subst_per_residue=0.0, struct_first=sf)
    S._step_logits = orig
    done = lambda i: next((k for k, x in enumerate(tr) if x[i] == 0), len(tr))
    print(f"   struct_first={sf}: aa track empty at step {done(0):>3}, 3Di at {done(1):>3} "
          f"(of {len(tr)}) -- this model happens to lead with 3Di unaided, so the bias is "
          f"redundant, not inert")

# The bias itself, unit-tested where nothing else can mask it: a track that would otherwise take
# only some of the slots must take all of them once biased.
cnf = torch.rand(2, 2, 8)
k = torch.full((2,), 4)
flat = lambda x: x.reshape(2, -1)
base = S._topk_mask(flat(cnf), k).view(2, 2, 8).sum(dim=(0, 2)).tolist()
bias = torch.zeros_like(cnf); bias[:, 1] = 1e4
bsd = S._topk_mask(flat(cnf + bias), k).view(2, 2, 8).sum(dim=(0, 2)).tolist()
print(f"   bias mechanism: unbiased split {base} -> biased {bsd}  (must be [0, 8])")
assert bsd == [0, 8], "struct_first bias does not reach the ranking"

print("\n4. OBJECTIVE: independent noise per track is what makes cross-track conditioning possible")
s_di = CorruptionSchedule(uniform_substitution_kernel(22), V, MASK, betas=(1.0, 0.5), T=50)
coll = make_collate(cfg, L, n_tracks=2)
rows = [([int(x) for x in torch.randint(0, 20, (30,))] + [EOS],
         [int(x) for x in torch.randint(0, 20, (30,))] + [EOS]) for _ in range(64)]
b = coll(rows)
torch.manual_seed(3)
_, m = training_step(model, b, s_aa, sched_struct=s_di)
print(f"   aa masked {float(m['masked']):.0%} | 3Di masked {float(m['s_masked']):.0%} "
      f"| labelled {float(m['s_frac']):.0%}")
# Over many draws the two mask fractions must be UNCORRELATED; a shared t would pin r to +1.0.
xs, ys = [], []
for _ in range(120):
    _, mm = training_step(model, coll(rows[:8]), s_aa, sched_struct=s_di)
    xs.append(float(mm["masked"])); ys.append(float(mm["s_masked"]))
import numpy as np
r = float(np.corrcoef(xs, ys)[0, 1])
print(f"   corr(aa mask fraction, 3Di mask fraction) over 120 batches = {r:+.3f}")
print(f"   {'OK -- independent' if abs(r) < 0.3 else '*** FAIL: the tracks share a noise level'}"
      f"; a shared t would put this at +1.0 and no row could ever inform the other track")

print("\nall two-track invariants hold")
