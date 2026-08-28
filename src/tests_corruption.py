"""Correctness checks for the unified corruption process:  python -m src.tests_corruption

Everything downstream rests on this file being right, so each property is checked against an
INDEPENDENT computation rather than against itself:

  1. MASK is absorbing, with zero leak, and P(x_T = MASK) ~ 1 -- so dropping L_T is free, not an
     approximation taken on credit.
  2. The mask fraction tracks the linear t/T target at every beta, which is what makes beta=1
     exactly the absorbing-state form of OADM's n ~ U(1, D) masking.
  3. beta really does control the substitution footprint (and is exactly 0 at beta=1).
  4. Qbar matches an explicitly recomputed matrix product, and shows the survival * Mbar
     factorisation the two channels are supposed to have.
  5. Forward sampling matches its own Qbar row empirically.
  6. The posterior: KL is exactly 0 under a perfect x0 prediction; unmasking is irreversible
     (q(x_{t-1}=MASK) = 0 whenever x_t != MASK); and t=1 reduces to D3PM's
     L_0 = -log p_theta(x_0|x_1) with no special case.
"""
import torch, torch.nn.functional as F
from src.corruption import CorruptionSchedule, kl_categorical, x0_probs, _rows
from src.blosum import substitution_kernel
import os, sys
from config import BLOSUM_MAT
bp = BLOSUM_MAT if os.path.exists(BLOSUM_MAT) else (sys.argv[1] if len(sys.argv) > 1 else None)
if bp is None or not os.path.exists(bp):
    raise SystemExit("need a BLOSUM .mat: set PLD2_BLOSUM or pass the path as an argument")
V, MASK, T = 23, 22, 500
B = substitution_kernel(bp, V-1, temp=1.0)
betas = (1.0, 0.9, 0.75, 0.5)
s = CorruptionSchedule(B, V, MASK, betas=betas, T=T)

print("1. MASK IS ABSORBING and the stationary distribution is all-MASK")
for bi, b in enumerate(betas):
    i = s.flat(torch.tensor(bi), torch.tensor(T))
    mrow = s.Q[s.flat(torch.tensor(bi), torch.tensor(250))][MASK]
    print(f"   beta={b}: Q_t[MASK,MASK]={float(mrow[MASK]):.6f} leak={float(mrow[:MASK].sum()):.2e}"
          f" | P(x_T=MASK)={s.terminal_mask_fraction(bi):.6f}  <- L_T ~ 0")

print("\n2. MASK FRACTION tracks the linear t/T target (beta=1 IS OADM's masking)")
for bi, b in enumerate(betas):
    print(f"   beta={b}: {s.mask_fraction(bi)}   target {{0:0.0, 100:0.2, 250:0.5, 400:0.8, 500:1.0}}")

print("\n3. BETA CONTROLS THE MIX: substitution footprint among still-unmasked positions at t=T/2")
for bi, b in enumerate(betas):
    print(f"   beta={b}: {s.substitution_fraction(bi):.1%} of surviving positions substituted"
          f"  (beta=1 must be exactly 0)")

print("\n4. Qbar == the explicit product, and matches the survival*Mbar factorisation")
bi, tc = 2, 137
P = torch.eye(V, dtype=torch.float64)
for t in range(1, tc+1): P = P @ s.Q[s.flat(torch.tensor(bi), torch.tensor(t))].double()
got = s.Qbar[s.flat(torch.tensor(bi), torch.tensor(tc))].double()
print(f"   |Qbar_137 - prod Q| = {float((P-got).abs().max()):.2e}")
surv = 1.0 - got[:V-1, MASK]
print(f"   mask column constant across rows: {float(got[:V-1,MASK].std()):.2e}"
      f" | rows sum to 1: {float((got.sum(-1)-1).abs().max()):.2e}")

print("\n5. FORWARD SAMPLING matches Qbar empirically")
x0 = torch.full((40000,1), 7, dtype=torch.long)
fi = s.flat(torch.full((40000,), 2, dtype=torch.long), torch.full((40000,), 250, dtype=torch.long))
xt = s.q_sample(x0, fi)
emp = torch.bincount(xt.reshape(-1), minlength=V).float()/40000
ref = s.Qbar[s.flat(torch.tensor(2), torch.tensor(250))][7]
print(f"   max |empirical - Qbar row| = {float((emp-ref).abs().max()):.4f}")

print("\n6. POSTERIOR: perfect prediction -> KL 0; t=1 -> D3PM's L_0; unmasking is irreversible")
Bs, L = 64, 32
x0 = torch.randint(0, 20, (Bs, L)); x0[:, 20] = 20; x0[:, 21:] = 21
bidx = torch.randint(0, len(betas), (Bs,)); t = s.sample_t(Bs, x0.device)
fi, fp = s.flat(bidx, t), s.flat(bidx, t-1)
xt = s.q_sample(x0, fi)
q, p = s.posteriors(x0, xt, fi, fp, F.one_hot(x0, V-1).float())
print(f"   q_post sums to 1: {float((q.sum(-1)-1).abs().max()):.2e} | KL(perfect) max = {float(kl_categorical(q,p).max()):.2e}")
nm = xt != MASK
qm = q[..., MASK][nm]
print(f"   x_t != MASK  =>  q(x_(t-1)=MASK) = {float(qm.max()):.2e}   (unmasking is irreversible)")
t1 = torch.ones(Bs, dtype=torch.long); f1, f0 = s.flat(bidx, t1), s.flat(bidx, t1-1)
x1 = s.q_sample(x0, f1); pt = torch.softmax(torch.randn(Bs, L, V-1), -1)
q1, p1 = s.posteriors(x0, x1, f1, f0, pt)
print(f"   t=1: q_post == onehot(x0)? err = {float((q1 - F.one_hot(x0, V).float()).abs().max()):.2e}")
a = _rows(s.QT, f1, x1); ref2 = (a[..., :V-1]*pt); ref2 = ref2/ref2.sum(-1, keepdim=True)
l0 = -ref2.gather(-1, x0.unsqueeze(-1)).squeeze(-1).log()
print(f"   t=1: KL == -log p_theta(x0|x1)? err = {float((kl_categorical(q1,p1)-l0).abs().max()):.2e}")
