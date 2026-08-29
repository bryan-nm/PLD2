"""PLD2 backbone: a looped, UNIFORM-WIDTH bidirectional transformer.

Same trunk topology as ProLoopDiff -- n_upstream distinct blocks, then n_middle distinct blocks
applied n_recurrence times (the "loop"), then n_downstream distinct blocks -- with the two features
that PLD2 removes on purpose:

  * NO privileged-basis (PB) sublayers. In ProLoopDiff a subset of the middle blocks projected the
    residual stream down to pb_dim=16, did work there, and wrote the result back through a
    zero-init gate. That 512 -> 16 -> 512 bottleneck sat INSIDE the looped stack, so every
    recurrence pass pushed the representation through it three times per forward. It is the prime
    suspect for ProLoopDiff's collapse into repetitive sequences: a 16-d writeback re-applied every
    pass is a strong low-rank attractor, and low-rank attractors in a bidirectional denoiser show up
    as periodic output. Here every layer is exactly d_model wide.

  * NO cross-attention, no text pathway, no learned null token, no CFG. PLD2 generates
    unconditionally. Conditioning, if it comes, arrives later as ProteinGuide-style decoding-time
    steering, which only ever touches the output logits -- so it needs nothing from this file.

Kept from ProLoopDiff because none of it was implicated:
  * Bidirectional self-attention (an any-order denoiser is not causal).
  * RoPE on self-attention Q/K only; the residual stream is never rotated.
  * Pre-norm blocks, RMSNorm, SwiGLU.
  * The zero-init gated re-injection of the upstream output at the head of each recurrence pass.
  * PAD is a MODELLED, attended token. EOS marks the length; PAD is what follows it. Predicting the
    PAD tail is what lets an all-MASK canvas resolve to a well-formed [AA* EOS PAD*].

Aurora / Intel XPU: device-agnostic, no CUDA-only paths. Attention goes through
F.scaled_dot_product_attention, which dispatches to oneDNN on XPU via IPEX. ipex.optimize /
autocast / oneCCL live in the trainer, not here.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # Vocab: 20 AA (ids 0..19) + EOS + PAD + MASK. MASK is deliberately the LAST id, so the D3PM
    # non-MASK state space -- the states x_0 can actually take -- is exactly
    # logits[..., :vocab_size-1]. corruption.py relies on that; assert_vocab() below is the contract.
    vocab_size: int = 23
    eos_token_id: int = 20    # length marker: the model predicts EOS; its position = sequence length
    pad_token_id: int = 21    # post-EOS filler; MODELLED and attended, not attention filler
    mask_token_id: int = 22   # absorbing state (OADM only; never part of the D3PM state space)

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1536              # SwiGLU inner dim (~8/3 * d_model -> param parity with 4x GELU)

    n_upstream: int = 4
    n_middle: int = 8             # distinct middle layers, shared across recurrence passes
    n_downstream: int = 4
    n_recurrence: int = 3         # passes through the middle stack (adaptive-compute knob)

    grad_checkpoint: bool = True
    # How many middle layers form ONE checkpoint segment. 0 = the whole stack as a single unit,
    # which is what the looped design implied (n_middle=8 recomputed per recurrence pass) and what
    # n_recurrence=1 with a deep middle stack makes wrong: 36 layers recomputed in one go holds all
    # 36 sets of activations at peak. Segmenting stores one boundary tensor per segment (a few tens
    # of MB) and caps the recompute peak at `checkpoint_chunk` layers, for identical total FLOPs.
    # sqrt(n_middle) is the classic optimum; 6 for a 36-layer stack.
    checkpoint_chunk: int = 6

    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def assert_vocab(self):
        """MASK must be the last id and EOS/PAD must precede it -- see the vocab note above."""
        assert self.mask_token_id == self.vocab_size - 1, \
            f"mask_token_id must be vocab_size-1 ({self.vocab_size - 1}), got {self.mask_token_id}"
        assert self.eos_token_id < self.mask_token_id and self.pad_token_id < self.mask_token_id
        return self

    @property
    def n_states(self) -> int:
        """Number of states x_0 can take: everything except the absorbing MASK state."""
        return self.vocab_size - 1


# --------------------------------------------------------------------------------------
# Norm + FFN
# --------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.affine = affine
        self.w = nn.Parameter(torch.ones(d)) if affine else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x.to(dt)
        return self.w * x if self.affine else x


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False)
        self.w3 = nn.Linear(d, ff, bias=False)
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --------------------------------------------------------------------------------------
# RoPE  (self-attention Q/K only; the residual stream is never rotated)
# --------------------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    """Cos/sin tables for RoPE.

    inv_freq is deliberately NOT a registered buffer: ipex.optimize(dtype=bfloat16) casts float
    buffers along with parameters, and a bf16 inv_freq costs up to ~1.9 rad of phase error by
    position 511 -- i.e. RoPE's high-frequency bands become noise. It is rebuilt in fp32 here.

    The (L, device, dtype) tables are cached: this runs once per self-attention (32x per forward at
    n_recurrence=3) and the canvas is a fixed 512, so exactly one entry is ever live.
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim, self.base = head_dim, base
        self._cache: dict = {}                           # (L, device, dtype) -> (cos, sin)

    def forward(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device,
                                                     dtype=torch.float32) / self.head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                 # (L, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (L, hd)  split-half convention
        cos = emb.cos()[None, None, :, :].to(dtype)      # (1,1,L,hd)
        sin = emb.sin()[None, None, :, :].to(dtype)
        self._cache[key] = (cos, sin)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, H, L, hd) ; cos, sin: (1, 1, L, hd)
    q = (q * cos) + (_rotate_half(q) * sin)
    k = (k * cos) + (_rotate_half(k) * sin)
    return q, k


# --------------------------------------------------------------------------------------
# Bidirectional self-attention (an any-order / discrete-diffusion denoiser is NOT causal)
# --------------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_heads
        self.hd = cfg.d_model // cfg.n_heads
        assert self.h * self.hd == cfg.d_model
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.rope = RotaryEmbedding(self.hd, cfg.rope_base)
        self.drop_p = cfg.dropout

    def forward(self, x: torch.Tensor, keep_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                 # (B, H, L, hd)

        cos, sin = self.rope(L, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)                # rotates the head subspace only

        attn_mask = None
        if keep_mask is not None:                        # keep_mask: (B, L) True = attendable
            attn_mask = keep_mask[:, None, None, :]      # (B,1,1,L) bool
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.drop_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o(out)


# --------------------------------------------------------------------------------------
# Block  (the ONLY block type -- every layer is d_model wide)
# --------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, keep_mask):
        x = x + self.attn(self.n1(x), keep_mask)
        x = x + self.ff(self.n2(x))
        return x


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _run_segment(layers, x, keep_mask):
    """Run a contiguous run of blocks. Module-level so torch_checkpoint can call it directly."""
    for l in layers:
        x = l(x, keep_mask)
    return x


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


class LoopedDiffusionLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg.assert_vocab()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.up_layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_upstream))
        self.mid_layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_middle))
        self.down_layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_downstream))

        # Gated re-injection of the upstream representation at the head of every recurrence pass.
        # Zero-init (tanh(0)=0), so at init the loop is a plain deep stack and this opens only if
        # unrolling starts to lose the input context. Costs one scalar.
        self.reinject_gate = nn.Parameter(torch.zeros(1))

        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # INITIALISATION -- a fix, not a design change (see README "deviations"). PyTorch's default
        # nn.Embedding init is N(0, 1), unscaled, and with tied embeddings the output logits are
        # <h, e_i> with ||h|| ~ sqrt(d_model): measured, that put the cross-entropy at step 0 near
        # 210 nats instead of ln(23) = 3.14. The model spends its warmup undoing the initialisation
        # rather than learning, under gradients large enough to be their own hazard. GPT-2's scheme
        # instead: everything N(0, 0.02), with the two projections that WRITE INTO the residual
        # stream (attn.o, ff.w2) scaled by 1/sqrt(2 * n_residual_writes) so the stream's variance
        # does not grow with depth. n_residual_writes counts layer APPLICATIONS, not distinct
        # layers -- a looped middle stack writes into the residual n_recurrence times per layer, and
        # it is the number of writes that sets the accumulated variance.
        n_apply = cfg.n_upstream + cfg.n_middle * cfg.n_recurrence + cfg.n_downstream
        self.apply(_init_weights)
        for blk in list(self.up_layers) + list(self.mid_layers) + list(self.down_layers):
            for lin in (blk.attn.o, blk.ff.w2):
                nn.init.normal_(lin.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_apply))

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, tokens, canvas_mask=None, n_recurrence=None):
        """
        tokens:      (B, L) long. MASK where absorbed (OADM), or a substituted token (D3PM).
                     EOS marks the length; PAD after EOS is modelled and attended.
        canvas_mask: (B, L) bool, True = position is part of the modelled canvas. None -> full
                     bidirectional attention, which is the case for PLD2's fixed 512 canvas (every
                     position is AA, EOS or PAD -- there is no beyond-canvas filler).
        n_recurrence: override the number of loop passes (adaptive compute at inference).
        returns:     logits (B, L, vocab_size)
        """
        cfg = self.cfg
        N = n_recurrence or cfg.n_recurrence
        keep_mask = canvas_mask

        x = self.embed(tokens)
        for l in self.up_layers:
            x = l(x, keep_mask)
        h_up = x                                                # upstream output for re-injection

        ckpt = cfg.grad_checkpoint and self.training
        chunk = cfg.checkpoint_chunk or len(self.mid_layers)
        for _ in range(N):
            x = x + torch.tanh(self.reinject_gate) * h_up
            for i in range(0, len(self.mid_layers), chunk):
                seg = self.mid_layers[i:i + chunk]
                if ckpt:
                    x = torch_checkpoint(_run_segment, seg, x, keep_mask, use_reentrant=False)
                else:
                    x = _run_segment(seg, x, keep_mask)

        for l in self.down_layers:
            x = l(x, keep_mask)

        return self.lm_head(self.final_norm(x))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    cfg = Config()
    model = LoopedDiffusionLM(cfg).to(device)
    widths = {p.shape[-1] for n, p in model.named_parameters()
              if p.dim() == 2 and "lm_head" not in n and "embed" not in n}
    print(f"device={device}  params={count_params(model)/1e6:.1f}M")
    print(f"distinct layers={cfg.n_upstream + cfg.n_middle + cfg.n_downstream} "
          f"applications={cfg.n_upstream + cfg.n_middle * cfg.n_recurrence + cfg.n_downstream}")
    print(f"linear input widths (must all be d_model or d_ff): {sorted(widths)} "
          f"-> uniform: {widths <= {cfg.d_model, cfg.d_ff}}")

    B, L = 2, 128
    tokens = torch.randint(0, 20, (B, L), device=device)
    tokens[:, L // 2:] = cfg.mask_token_id
    tokens[:, -9] = cfg.eos_token_id
    tokens[:, -8:] = cfg.pad_token_id
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device != "cpu")):
        logits = model(tokens)
    print("logits:", tuple(logits.shape))
    # At init the model should be near-uniform over the vocabulary: CE ~ ln(vocab_size).
    ce = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), tokens.reshape(-1)).item()
    print(f"init cross-entropy: {ce:.3f} (uniform = ln({cfg.vocab_size}) = "
          f"{math.log(cfg.vocab_size):.3f})")
