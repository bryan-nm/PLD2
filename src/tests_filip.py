"""FILIP guidance invariants that do NOT need AMPLIFY:  python -m src.tests_filip

The classifier itself needs the 350M protein encoder and the packed caption cache, both of which
live on the cluster. What can and must be checked anywhere is the BRIDGE, because every way it can
be wrong is silent. PLD2 and AMPLIFY both write residues as letters over ids that do not line up;
a bridge that mapped by id rather than by letter would hand the classifier a different protein and
still return a perfectly plausible score. And TAG maps a gradient at an AMPLIFY position back onto
a PLD2 position, so an off-by-one in the cls/eos framing would apply each position's guidance to
its neighbour -- again with no error, just worse samples.
"""
import torch

from .blosum import AA
from .filip_guidance import CanvasBridge
from .model import Config


class _MockAmplifyTokenizer:
    """AMPLIFY-shaped, with an id layout deliberately unlike PLD2's so a by-id bug cannot pass."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<cls>": 1, "<eos>": 2, "<unk>": 3, "<mask>": 4}
        for i, a in enumerate("LAGVSERTIDPKQNFYMHWCXBUZO"):
            self.vocab[a] = 5 + i
        self.pad_token_id, self.cls_token_id, self.eos_token_id = 0, 1, 2
        self.unk_token_id, self.mask_token_id, self.bos_token_id = 3, 4, 1

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)


cfg = Config(vocab_size=23, eos_token_id=20, pad_token_id=21, mask_token_id=22, d_model=32,
             n_heads=2, d_ff=64, n_upstream=1, n_middle=1, n_downstream=1, n_tracks=2)
tok = _MockAmplifyTokenizer()
br = CanvasBridge(cfg, tok, torch.device("cpu"))

print("1. RESIDUES MAP BY LETTER, not by id")
wrong = [a for i, a in enumerate(AA) if int(br.register[i]) != tok.vocab[a]]
print(f"   all 20 residues reach their own AMPLIFY id: {not wrong}")
print(f"   e.g. PLD2 'A'={AA.index('A')} -> AMPLIFY {int(br.register[AA.index('A')])}; "
      f"'W'={AA.index('W')} -> {int(br.register[AA.index('W')])}")
print(f"   MASK -> {int(br.register[22])} (<mask>)  EOS -> {int(br.register[20])} (<eos>)  "
      f"PAD -> {int(br.register[21])} (<pad>)")
assert not wrong and int(br.register[22]) == tok.mask_token_id

print("\n2. ALIGNMENT: AMPLIFY position i+1 IS canvas position i")
cv = torch.full((2, 10), cfg.pad_token_id)
cv[0, :6] = torch.tensor([AA.index(c) for c in "MKVLAG"])
cv[0, 3] = cfg.mask_token_id
cv[0, 6] = cfg.eos_token_id
cv[1, :4] = torch.tensor([AA.index(c) for c in "WYCD"])
cv[1, 4] = cfg.eos_token_id
ids, attn, live = br.to_amplify(cv)
aligned = all(int(ids[b, i + 1]) == int(br.register[cv[b, i]])
              for b in range(cv.shape[0]) for i in range(cv.shape[1]))
print(f"   ids {tuple(ids.shape)} for a canvas of {tuple(cv.shape)} (cls + L + eos)")
print(f"   ids[:, i+1] == register[canvas[:, i]] everywhere: {aligned}")
print(f"   framing: ids[0,0]={int(ids[0,0])} (<cls>), ids[0,-1]={int(ids[0,-1])} (<eos>)")
assert aligned and int(ids[0, 0]) == tok.cls_token_id and int(ids[0, -1]) == tok.eos_token_id

print("\n3. `live` SELECTS COMMITTED RESIDUES ONLY -- the FILIP score reads nothing else")
print(f"   canvas : {[int(x) for x in cv[0]]}")
print(f"   live   : {[int(x) for x in live[0]]}  (3 is MASK, 6 is EOS, 7+ are PAD)")
assert live[0].tolist() == [1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
assert attn[0, 1:11].tolist() == [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
print(f"   attn over the canvas span: {[int(x) for x in attn[0, 1:11]]}  (PAD attends 0)")

print("\n4. aa_ids ARE IN PLD2 ORDER, so a [B, L, 20] guidance tensor adds to logits[..., :20]")
print(f"   br.aa_ids[:6] {br.aa_ids[:6].tolist()}  ==  AMPLIFY ids of {AA[:6]} "
      f"{[tok.vocab[a] for a in AA[:6]]}")
assert br.aa_ids.tolist() == [tok.vocab[a] for a in AA]

print("\n5. A FULLY MASKED CANVAS HAS NOTHING TO SCORE, and guidance must know that")
empty = torch.full((1, 8), cfg.mask_token_id)
_, _, live0 = br.to_amplify(empty)
print(f"   all-MASK canvas -> live.any() = {bool(live0.any())}  "
      f"(FilipGuidance returns the logits untouched here; the classifier excludes masked "
      f"positions, so its score would be 0 for every candidate alike)")
assert not bool(live0.any())

print("\nall bridge invariants hold")
