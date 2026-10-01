"""Satisfy ``import intel_extension_for_pytorch`` for packages that still insist on it.

Aurora's 26.181.0 image (frameworks 2026.1.0, torch 2.13) ships no IPEX, because XPU support
moved into torch itself. Every ipex call site *in this repo* degrades to ``ipex = None``, so the
job banner reported the absence as harmless -- but EsmFold does not. ``esmfold_scorer.device``
treats a failed ipex import as proof that there is no XPU, so on round 3 all 192 fold ranks died
five seconds into model construction with

    RuntimeError: XPU requested but intel_extension_for_pytorch is not installed or
                  no XPU device is visible.

and the round produced zero structures. install() registers a stub under the real module name so
that import succeeds. It forwards the only two APIs callers actually use -- ``optimize``, whose
fusions torch now applies on its own, and ``xpu``, which *is* ``torch.xpu`` -- and leaves every
other attribute to raise AttributeError rather than quietly answering wrongly.

The stub goes in ONLY when torch can really see an XPU. That error message conflates two very
different failures, and papering over "this node has no GPU" would turn a loud crash into a run
that folds on CPU at a hundredth of the speed, which is far worse than not folding at all.
"""

import contextlib
import importlib.machinery
import sys
import types

import torch

MODULE = "intel_extension_for_pytorch"


def _optimize(model, optimizer=None, **_kw):
    """ipex.optimize's contract: model in, model out -- or ``(model, optimizer)`` when an
    optimizer is passed. torch 2.13 already applies the fusions, so there is nothing left to do."""
    return (model, optimizer) if optimizer is not None else model


def _stub() -> types.ModuleType:
    m = types.ModuleType(MODULE)
    # A NON-NONE __spec__ IS LOAD-BEARING. types.ModuleType leaves it None, and
    # importlib.util.find_spec() raises ValueError on an imported module with no spec rather than
    # reporting absence. transformers probes every optional backend that way at import time
    # (utils/import_utils.py: _is_package_available -> find_spec), so a spec-less stub takes down
    # the first `import transformers` underneath esm -- which is how job 8884342 died.
    #
    # Deliberately NO fake dist-info to go with it: transformers' next step is
    # importlib.metadata.version(), whose PackageNotFoundError makes it conclude ipex is absent.
    # That IS the truth, and it is the answer that keeps it off the ipex code paths.
    m.__spec__ = importlib.machinery.ModuleSpec(MODULE, loader=None)
    m.__version__ = f"0.0.0+pld2.shim.torch.{torch.__version__}"
    m.__file__ = __file__
    m.optimize = _optimize
    m.xpu = torch.xpu
    m.pld2_stub = True                 # so anything downstream can tell this from the real thing
    return m


def have_xpu() -> bool:
    xpu = getattr(torch, "xpu", None)
    try:
        return bool(xpu is not None and xpu.is_available())
    except Exception:
        return False


def have_ipex() -> bool:
    """True if the real package (or a stub already installed) imports."""
    if MODULE in sys.modules:
        return True
    try:
        __import__(MODULE)
    except Exception:
        return False
    return True


def needed() -> bool:
    """True when a caller would ask for ipex, not find it, and wrongly conclude there is no GPU."""
    return have_xpu() and not have_ipex()


def install(verbose: bool = False) -> bool:
    """Register the stub if it is both missing and wanted. -> whether one was installed."""
    if not needed():
        return False
    sys.modules[MODULE] = _stub()
    if verbose:
        print(f"[ipex-shim] intel_extension_for_pytorch is absent and torch {torch.__version__} "
              f"owns the XPU backend; registered a stub forwarding to torch.xpu so EsmFold's "
              f"device check does not mistake the missing package for a missing GPU.", flush=True)
    return True


def uninstall() -> bool:
    """Take OUR stub back out of sys.modules. Never touches a real ipex. -> whether it removed one."""
    m = sys.modules.get(MODULE)
    if m is not None and getattr(m, "pld2_stub", False):
        del sys.modules[MODULE]
        return True
    return False


@contextlib.contextmanager
def only_for_import(verbose: bool = False):
    """Hold the stub just long enough for one import, then remove it.

    THE STUB'S BLAST RADIUS IS THE PROBLEM, not its contents. EsmFold reads its ipex flag once, at
    module-import time, so the stub only has to exist while esmfold_scorer is being imported.
    Leaving it in sys.modules afterwards is what killed job 8884342: esm goes on to import
    transformers, which probes every optional backend with importlib.util.find_spec, and found
    ours. Scoping it to the import means nothing downstream can trip over a module that is not
    really there -- and we cannot enumerate everything downstream, which is the whole point.

    -> whether a stub was installed for the duration.
    """
    installed = install(verbose=verbose)
    try:
        yield installed
    finally:
        if installed:
            uninstall()


def status() -> str:
    """One line for a preflight table."""
    if MODULE in sys.modules and getattr(sys.modules[MODULE], "pld2_stub", False):
        return "stub installed (forwards to torch.xpu)"
    if have_ipex():
        return "real package present; no shim needed"
    if have_xpu():
        return "absent, XPU visible -- shim WILL be installed before folding"
    return "absent and no XPU visible -- nothing to shim (expected on a login node)"


if __name__ == "__main__":
    print(f"torch       : {torch.__version__}")
    print(f"xpu visible : {have_xpu()}")
    print(f"ipex        : {status()}")
