"""BLOSUM62 -> the two matrices the corruption processes need.

Two different objects come out of the same file, for two different jobs:

  * blosum_sub_probs()  -- a 20x20 ROW-STOCHASTIC matrix with a zero diagonal: "if residue i is
    corrupted, what does it become". Used by the sampler's substitution corrector.

  * d3pm_transition()   -- a KxK DOUBLY-STOCHASTIC matrix over the full non-MASK alphabet (20 AA +
    EOS + PAD). Doubly stochastic is not a nicety: it is what makes the uniform distribution the
    stationary distribution of the D3PM Markov chain, which is what the D3PM ELBO assumes when it
    drops L_T. Built EvoDiff's way -- exponentiate the BLOSUM scores, then Sinkhorn-Knopp.

Alphabet order: our canonical AA string (== token id), NOT BLOSUM column order
(A R N D C Q E G H I L K M F P S T W Y V B Z X J O U -), so the file is reordered on load.

EvoDiff includes uniform rows for the non-standard codes and <GAP>; the analogue here is EOS and
PAD, which get a BLOSUM score of 0 against everything (0 = "as often as chance", BLOSUM's neutral
point) before the exponentiation and Sinkhorn. That keeps every entry strictly positive, which is
what Sinkhorn-Knopp needs to converge.
"""
from __future__ import annotations
import torch

# Canonical amino-acid alphabet: token id == index here (matches EvoDiff vocab.txt's first 20).
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_blosum62(path: str):
    """Parse a BLOSUM .mat file into {(row_aa, col_aa): score} plus the column-label list."""
    labels = None
    score = {}
    with open(path) as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = line.split()
            if labels is None:
                labels = parts                          # header row of column labels
                continue
            rlabel, vals = parts[0], [int(x) for x in parts[1:]]
            for clabel, v in zip(labels, vals):
                score[(rlabel, clabel)] = v
    return score, labels


def blosum_sub_probs(path: str, temp: float = 1.0, aa: str = AA) -> torch.Tensor:
    """(20, 20) row-stochastic substitution matrix, ZERO diagonal, BLOSUM-weighted.

    A BLOSUM score is a log-odds substitution score, so softmax_{j != i}(S[i,j]/temp) is a proper
    "what does residue i turn into" distribution favouring biochemically similar residues.
    temp -> inf recovers uniform-over-19; small temp peaks on the single most similar residue.
    """
    score, _ = load_blosum62(path)
    n = len(aa)
    S = torch.full((n, n), -1e9)
    for i, ai in enumerate(aa):
        for j, aj in enumerate(aa):
            if i != j:
                S[i, j] = score[(ai, aj)] / temp
    return torch.softmax(S, dim=1)                      # diagonal -> ~0; rows sum to 1


def uniform_sub_probs(num_aa: int = 20) -> torch.Tensor:
    """Uniform over the 19 non-identity residues -- the no-biology fallback for the corrector."""
    m = torch.ones(num_aa, num_aa)
    m.fill_diagonal_(0.0)
    return m / m.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# D3PM transition matrices (doubly stochastic over the full non-MASK alphabet)
# ---------------------------------------------------------------------------
def sinkhorn_knopp(M: torch.Tensor, iters: int = 500, tol: float = 1e-9) -> torch.Tensor:
    """Alternately normalise rows and columns until M is doubly stochastic.

    Converges for any strictly positive square matrix (Sinkhorn's theorem). M must be positive --
    a zero row or column has no scaling that makes it sum to one, and the loop would divide by 0.
    """
    assert M.dim() == 2 and M.shape[0] == M.shape[1], "need a square matrix"
    assert bool((M > 0).all()), "Sinkhorn-Knopp needs a strictly positive matrix"
    M = M.double()
    for _ in range(iters):
        M = M / M.sum(dim=1, keepdim=True)
        M = M / M.sum(dim=0, keepdim=True)
        err = max((M.sum(1) - 1).abs().max().item(), (M.sum(0) - 1).abs().max().item())
        if err < tol:
            break
    return (M / M.sum(dim=1, keepdim=True)).float()


def uniform_transition(k: int) -> torch.Tensor:
    """The D3PM-Uniform base matrix: every token goes to every token equally often."""
    return torch.full((k, k), 1.0 / k)


def blosum_transition(path: str, k: int, temp: float = 1.0, aa: str = AA) -> torch.Tensor:
    """The D3PM-BLOSUM base matrix: (k, k) doubly stochastic over [AA..., EOS, PAD].

    k must be >= len(aa); the extra rows/columns are the special tokens (EOS, PAD) and get a
    neutral BLOSUM score of 0 against everything. Scores are exponentiated at `temp` and then made
    doubly stochastic by Sinkhorn-Knopp, exactly as EvoDiff does.

    The diagonal survives (BLOSUM62's self-scores are +4..+11), which is the point: a self-preferring
    base matrix is what makes the corruption GRADUAL rather than one-shot uniform noise.
    """
    n = len(aa)
    assert k >= n, f"k={k} is smaller than the amino-acid alphabet ({n})"
    score, _ = load_blosum62(path)
    S = torch.zeros(k, k)                               # 0 = BLOSUM-neutral, used for EOS/PAD
    for i, ai in enumerate(aa):
        for j, aj in enumerate(aa):
            S[i, j] = float(score[(ai, aj)])
    return sinkhorn_knopp(torch.exp(S / max(temp, 1e-6)))


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/bryan/Documents/models/EvoDiff-d3pm-blosum-38M/blosum62-special-MSA.mat"
    K = 22                                              # 20 AA + EOS + PAD (MASK is not a D3PM state)

    for temp in (1.0, 2.0, 100.0):
        P = blosum_sub_probs(path, temp=temp)
        top = {AA[i]: AA[int(P[i].argmax())] for i in (AA.index("A"), AA.index("I"),
                                                       AA.index("K"), AA.index("F"))}
        print(f"sub_probs  temp={temp:5.0f} | rowsum in [{P.sum(1).min():.4f},{P.sum(1).max():.4f}] "
              f"diag_max={P.diagonal().abs().max():.1e} | A->{top['A']} I->{top['I']} "
              f"K->{top['K']} F->{top['F']}")

    for temp in (1.0, 2.0):
        B = blosum_transition(path, K, temp=temp)
        print(f"d3pm base  temp={temp:5.1f} | rowsum err={float((B.sum(1)-1).abs().max()):.2e} "
              f"colsum err={float((B.sum(0)-1).abs().max()):.2e} | "
              f"mean diag={float(B.diagonal().mean()):.3f} min={float(B.min()):.2e}")
    U = uniform_transition(K)
    print(f"d3pm unif        | rowsum err={float((U.sum(1)-1).abs().max()):.2e} "
          f"colsum err={float((U.sum(0)-1).abs().max()):.2e}")
