"""Route EVERY torch.linalg op that may lack an XPU kernel through an explicit CPU round-trip,
and record which ones actually fired.

WHY. EsmFold's scorer patches exactly two ops, torch.linalg.svd and torch.linalg.det, because
those are the ones its speed test hit. Its README is explicit about the stakes: PyTorch's automatic
aten XPU->CPU fallback "corrupts GPU memory on Aurora's compute-runtime -- inference aborts with
'Segmentation fault from GPU' immediately after the fallback warning", and a NEW fallback after an
esm or torch upgrade reintroduces exactly that. Any other linalg op esm reaches is unpatched.

That matches the evidence here. The speed test folded 1100 sequences of 63-297 aa without incident;
our runs die after roughly 300 sequences of up to 512 aa. A different length regime can take a
different branch through a structure module, and a branch that calls, say, linalg.inv or linalg.qr
would fall back silently and fault -- with no traceback, because the abort happens in the driver.

WHAT THIS DOES. Wraps the ops below so an XPU input is copied to CPU in float32, computed there,
and copied back. That is what the scorer already does for svd/det; this just stops the coverage
being a guess. The tensors these ops see in a structure module are small (per-residue 3x3 frames),
so the round-trip is cheap -- and an op that is NEVER called costs exactly nothing.

IT ALSO ANSWERS THE QUESTION. fired() reports which wrapped ops actually ran. If a run that used to
crash now survives and fired() names an op beyond svd/det, that op WAS the fault, and the fix
belongs upstream in EsmFold's own patch list. If nothing beyond svd/det fires, the fallback theory
is wrong and the cause is elsewhere -- which is worth knowing just as much.

This does not replace PYTORCH_DEBUG_XPU_FALLBACK=1, which reports fallbacks from every aten op
rather than just linalg. Run that too when you can; this is the belt to its braces.
"""
from __future__ import annotations
import torch

# Ops a protein structure module plausibly reaches. svd/det are included so this module's
# accounting is complete even though EsmFold patches them first (its guard makes ours a no-op).
_OPS = ("svd", "det", "slogdet", "inv", "pinv", "solve", "lstsq",
        "eigh", "eigvalsh", "eig", "eigvals", "qr", "cholesky", "cholesky_ex",
        "matrix_exp", "matrix_power", "matrix_rank", "norm", "cross")

_fired: set[str] = set()
_patched = False


def fired() -> set:
    """Names of wrapped ops that actually executed on an XPU tensor this process."""
    return set(_fired)


def _to_cpu(x):
    return x.detach().to("cpu", torch.float32) if torch.is_tensor(x) and x.is_xpu else x


def _wrap(fn, name):
    def inner(*args, **kwargs):
        src = next((a for a in args if torch.is_tensor(a) and a.is_xpu), None)
        if src is None:
            src = next((v for v in kwargs.values() if torch.is_tensor(v) and v.is_xpu), None)
        if src is None:
            return fn(*args, **kwargs)

        dev, dt = src.device, src.dtype
        out = fn(*[_to_cpu(a) for a in args],
                 **{k: _to_cpu(v) for k, v in kwargs.items()})
        _fired.add(name)

        def back(x):
            if not torch.is_tensor(x):
                return x
            # Only floating results go back to the input dtype. Integer results -- pivots, ranks,
            # the `info` codes from the _ex variants -- must keep their own dtype; casting those to
            # bfloat16 would silently corrupt them.
            return x.to(dev, dt) if x.is_floating_point() else x.to(dev)

        if torch.is_tensor(out):
            return back(out)
        vals = tuple(back(t) for t in out)
        try:                                   # preserve the named-tuple interface where there is one
            return type(out)(vals)
        except Exception:
            return vals

    inner._xpu_roundtrip = True
    return inner


def patch(verbose: bool = True) -> list:
    """Wrap the ops. Idempotent, and skips anything already wrapped (by EsmFold or by us)."""
    global _patched
    if _patched:
        return []
    done = []
    for name in _OPS:
        fn = getattr(torch.linalg, name, None)
        if fn is None:
            continue
        # EsmFold marks its own svd wrapper with _xpu_patched; ours uses _xpu_roundtrip. Either
        # means the op is already routed and must not be double-wrapped.
        if getattr(fn, "_xpu_patched", False) or getattr(fn, "_xpu_roundtrip", False):
            continue
        setattr(torch.linalg, name, _wrap(fn, name))
        done.append(name)
    _patched = True
    if verbose and done:
        print(f"[xpu-linalg] CPU round-trip installed for {len(done)} linalg op(s): "
              f"{', '.join(done)}", flush=True)
    return done


def report(prefix: str = "[xpu-linalg]") -> str:
    f = sorted(_fired)
    if not f:
        return (f"{prefix} no wrapped linalg op ran on an XPU tensor. If a crash persisted, an "
                f"unpatched aten fallback is NOT the cause -- look elsewhere.")
    extra = [n for n in f if n not in ("svd", "det")]
    msg = f"{prefix} linalg ops that ran on XPU tensors: {', '.join(f)}"
    if extra:
        msg += (f"\n{prefix} NOTE: {', '.join(extra)} ran and is NOT covered by EsmFold's own "
                f"svd/det patch. If this run survived where it used to crash, that is the culprit "
                f"and the fix belongs in EsmFold's _patch_linalg_cpu_roundtrip.")
    return msg
