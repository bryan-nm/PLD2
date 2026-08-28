"""Cross-product regression for the sampler:  python -m src.tests_sampler

Every decoding option against every blend setting, checking BOTH kinds of guarantee:

  well-formedness  no MASK survives, at most one EOS, [AA* EOS PAD*], length >= min_len
  bounded work     the number of model forwards is EXACTLY predictable from the arguments --
                   1 (eos_first) + n_steps + correctors. This is the wall-time guarantee: the D3PM
                   channel adds no forward pass at all, because it reads the same logits the
                   absorbing commit does, and nothing anywhere iterates to convergence.

The eos_first=False x blend>0 cell is why this file exists. Testing the options one axis at a time
missed it: `revisable` was snapshotted before _enforce_eos, so a step that committed a new EOS could
turn a committed residue into PAD and revision would write a residue back into the PAD tail. It
broke every row of the batch, and no single-axis test touched it.
"""
import itertools, torch
import src.sampler as S
from src.model import LoopedDiffusionLM, Config
from src.d3pm import D3PMSchedule
from src.blosum import uniform_transition

cfg = Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22,
             d_model=128, n_heads=4, d_ff=384, n_upstream=2, n_middle=2, n_downstream=2,
             n_recurrence=2, grad_checkpoint=False)
torch.manual_seed(0); m = LoopedDiffusionLM(cfg).eval()
sch = D3PMSchedule(uniform_transition(22), T=500)
L, B, MIN = 96, 6, 20

calls = {"n": 0}; orig = S._step_logits
def counted(*a, **k):
    calls["n"] += 1; return orig(*a, **k)
S._step_logits = counted

fails = 0; total = 0
for blend, eos_first, greedy, corr in itertools.product(
        (0.0, 0.5, 1.0), (True, False), (False, True),
        ((0, "remask"), (2, "remask"), (2, "substitution"))):
    total += 1
    calls["n"] = 0; torch.manual_seed(13)
    cv, lens = S.generate(m, Lmax=L, batch_size=B, n_steps=L, device="cpu", min_len=MIN,
                          eos_first=eos_first, greedy=greedy,
                          n_corrector=corr[0], corrector_type=corr[1],
                          d3pm_sched=sch, d3pm_blend=blend)
    # forwards: 1 (eos_first) + L decode + correctors (remask costs 2, substitution 1)
    want = (1 if eos_first else 0) + L + corr[0] * (2 if corr[1] == "remask" else 1)
    ok = {
        "no MASK":     bool((cv != 22).all()),
        "<=1 EOS":     all(int((cv[b] == 20).sum()) <= 1 for b in range(B)),
        "AA prefix":   all(bool((cv[b, :lens[b]] < 20).all()) for b in range(B)),
        "PAD tail":    all(bool((cv[b, lens[b]+1:] == 21).all()) if lens[b]+1 < L else True
                           for b in range(B)),
        "len>=min":    all(l >= MIN for l in lens),
        "forwards":    calls["n"] == want,
    }
    if not all(ok.values()):
        fails += 1
        print(f"  FAIL blend={blend} eos_first={eos_first} greedy={greedy} corr={corr}: "
              f"{[k for k, v in ok.items() if not v]} (forwards {calls['n']} want {want})")
S._step_logits = orig
print(f"{total - fails}/{total} configurations pass "
      f"(well-formed, min_len honoured, and EXACTLY the predicted number of model forwards)")
