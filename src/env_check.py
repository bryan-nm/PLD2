"""Does this environment still run the pipeline?  python -m src.env_check [--deep]

RUN THIS AFTER EVERY AURORA IMAGE CHANGE, on a login node, before spending a queue slot. An image
update swaps the frameworks module underneath the venv, and the failures it causes surface one
phase at a time, hours apart, each costing an allocation. This imports everything the pipeline
touches in dependency order and says which half of the run each break would take down.

It is deliberately import-only and CPU-only: no model weights, no GPU, seconds to run. --deep adds
the two heavyweight loads (the ESMFold scorer and the FILIP/AMPLIFY stack) that nothing else
exercises until a job is already running.

REQUIRED means the pipeline cannot run. OPTIONAL means a phase degrades or is skipped -- notably
intel_extension_for_pytorch, which newer Aurora frameworks drop because XPU support moved into
torch itself. Every ipex.optimize call in this repo is guarded, so losing it costs whatever weight
prepacking was worth -- but EsmFold reads a failed ipex import as "there is no XPU" and refuses to
build a scorer at all, which killed all 192 fold ranks of round 3. src/ipex_shim.py covers that;
the 'ipex shim' row below says whether it applies here.
"""
from __future__ import annotations
import argparse
import importlib
import os
import sys
import traceback

OK, WARN, FAIL = "ok  ", "WARN", "FAIL"
_rows, _fatal = [], 0


def row(status, name, detail=""):
    global _fatal
    _rows.append((status, name, detail))
    if status == FAIL:
        _fatal += 1
    print(f"  [{status}] {name:<34} {detail}", flush=True)


def check(name, fn, required=True, detail=""):
    try:
        d = fn()
        row(OK, name, d or detail)
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".replace("\n", " ")[:160]
        row(FAIL if required else WARN, name, msg)
        return False


def _esmfold_model_class():
    """Can EsmFold find the ESMFold2 class at all?  The check job 8883902 needed: the venv rebuild
    left esmfold_scorer importable and ESMFOLD_WEIGHTS on disk, but nothing providing the model
    class, so _resolve_model_class() raised an ImportError after a queue slot was already spent.
    Resolving the class touches no weights and no GPU -- a second on a login node."""
    import inspect
    sc = importlib.import_module("esmfold_scorer.scorer")
    cls = sc.StructureScorer
    probe = getattr(cls, "_resolve_model_class", None)
    if probe is None:
        return "no _resolve_model_class() in this EsmFold build; nothing to probe"
    # It is a @staticmethod today, so it takes nothing; an instance method would want a self.
    # Supply one with object.__new__, which skips __init__ -- __init__ pulls the weights onto a
    # device to answer a question that is purely about which packages are installed.
    static = isinstance(inspect.getattr_static(cls, "_resolve_model_class"),
                        (staticmethod, classmethod))
    args = () if static else (object.__new__(cls),)
    try:
        mc = probe(*args)
    except ImportError as e:
        # EsmFold's own message here says to install esm, which alone fixes nothing: its
        # esm.models.esmfold2 branch cannot succeed (3.3.0 does not export EsmFold2Model), so the
        # Biohub transformers FORK is what actually resolves the class. Say so.
        raise ImportError(f"{e}  <-- really: pip install -r requirements-aurora.txt "
                          f"(the Biohub transformers fork; upstream PyPI has no "
                          f"models/esmfold2)") from None
    return f"{getattr(mc, '__module__', '?')}.{getattr(mc, '__name__', mc)}"


def _esmfold_device_check():
    """Would EsmFold accept device='xpu' here?  Separates the two failures its own error message
    runs together: a missing ipex (which src/ipex_shim.py fixes) from a missing GPU (which it must
    not hide). On a login node the honest answer is the second one."""
    from src import ipex_shim
    dev = importlib.import_module("esmfold_scorer.device")
    resolve = getattr(dev, "resolve_device", None)
    if resolve is None:
        return "no resolve_device() in this EsmFold build; nothing to shim"
    installed = ipex_shim.install()
    try:
        return f"accepts xpu -> {resolve('xpu')}" + ("  (via the shim)" if installed else "")
    except Exception as e:
        if not ipex_shim.have_xpu():
            return f"no XPU on this node, so untestable here: {type(e).__name__}: {e}"[:150]
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deep", action="store_true",
                    help="also load the ESMFold scorer and the FILIP/AMPLIFY stack (slow)")
    a = ap.parse_args()

    print(f"python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"prefix   {sys.prefix}")
    print(f"base     {sys.base_prefix}")
    cfg = os.path.join(sys.prefix, "pyvenv.cfg")
    if os.path.exists(cfg):
        home = next((l.split("=", 1)[1].strip() for l in open(cfg)
                     if l.startswith("home")), "")
        # THE FAILURE MODE THAT STARTED THIS: a venv keeps pointing at the interpreter it was built
        # from. When the frameworks module moves, --system-site-packages silently serves a
        # different torch and the first symptom is a missing MKL shared object, four phases in.
        same = home and os.path.realpath(home).startswith(os.path.realpath(sys.base_prefix))
        print(f"venv built against: {home or '<unknown>'}"
              + ("" if same else "   <-- DOES NOT MATCH base; rebuild the venv"))
    print()

    print("core")
    import torch
    check("torch", lambda: torch.__version__)
    check("torch.xpu available", lambda: (f"{torch.xpu.device_count()} device(s)"
                                          if torch.xpu.is_available()
                                          else "no XPU visible (expected on a login node)"),
          required=False)
    check("xccl backend", lambda: ("available" if getattr(
        torch.distributed, "is_xccl_available", lambda: False)() else "NOT available"),
        required=False)
    check("intel_extension_for_pytorch",
          lambda: importlib.import_module("intel_extension_for_pytorch").__version__,
          required=False,
          detail="absent is fine for this repo's own call sites; see the 'ipex shim' row")
    check("numpy", lambda: importlib.import_module("numpy").__version__)

    print("\nthis repo (generation, pairing, tuning)")
    for m in ("src.model", "src.corruption", "src.objective", "src.sampler", "src.data",
              "src.metrics", "src.dist", "src.prompts", "src.preference", "src.align",
              "src.align_sample", "src.align_compare", "src.tm_align", "src.reference_set"):
        check(m, lambda m=m: importlib.import_module(m) and "")

    print("\npaths (config.py owns every one)")
    try:
        import config
        for label, path, isdir in (
                ("UNIREF_SHARDS", config.UNIREF_SHARDS, True),
                ("AFDB_SHARDS", config.AFDB_SHARDS, True),
                ("SWISSPROT_CSV", config.SWISSPROT_CSV, False),
                ("FILIP_CACHE", config.FILIP_CACHE, True),
                ("FILIP_CKPT", config.FILIP_CKPT, False),
                ("ESMFOLD_WEIGHTS", config.ESMFOLD_WEIGHTS, False),
                ("BLOSUM_MAT", config.BLOSUM_MAT, False),
                ("MAT3DI", config.MAT3DI, False)):
            ex = os.path.isdir(path) if isdir else os.path.exists(path)
            row(OK if ex else WARN, label, path if ex else f"MISSING: {path}")
    except Exception as e:
        row(FAIL, "config", f"{type(e).__name__}: {e}")

    print("\nexternal tools")
    # RUN IT, do not just find it. `which foldseek` on a login node reports MISSING purely because
    # scripts/pbs_common.sh only puts it on PATH inside a job -- a warning that is always there and
    # therefore tells you nothing. config.FOLDSEEK resolves the same way a job will, and executing
    # it is the only check that covers what an image update could actually break: a static binary
    # that no longer runs against the new system libraries.
    import subprocess
    from shutil import which

    def _foldseek():
        import config as _c
        exe = _c.FOLDSEEK
        if not (os.access(exe, os.X_OK) or which(exe)):
            raise FileNotFoundError(f"not executable and not on PATH: {exe} "
                                    f"(set PLD2_FOLDSEEK or PLD2_FOLDSEEK_DIR)")
        r = subprocess.run([exe, "version"], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"{exe} version exited {r.returncode}: "
                               f"{(r.stderr or r.stdout).strip()[:120]}")
        return f"{(r.stdout or r.stderr).strip().splitlines()[0][:40]}  at {exe}"

    check("foldseek runs", _foldseek, required=False)

    print("\nfolding (phase 0b, 2) -- no folds means no rewards, so no alignment at all")
    # NOT a cosmetic row. Round 3 lost a 16-node job in two minutes because EsmFold's
    # resolve_device() demands ipex before it will admit an XPU exists, and the image had
    # dropped ipex. On a login node there is no XPU, so this can only report what WILL happen.
    check("ipex shim", lambda: importlib.import_module("src.ipex_shim").status(),
          required=False)
    # Both of these must be Biohub forks, not PyPI -- see requirements-aurora.txt. Upstream
    # transformers has no models/esmfold2 at any version, so a plain `pip install transformers`
    # satisfies this row and still cannot fold.
    check("esm", lambda: importlib.import_module("esm").__version__, required=False)
    check("transformers (needs the Biohub fork)",
          lambda: importlib.import_module("transformers").__version__, required=False)
    if a.deep:
        check("esmfold_scorer.StructureScorer",
              lambda: importlib.import_module("esmfold_scorer").StructureScorer and "importable")
        check("esmfold_scorer device check", _esmfold_device_check, required=False)
        check("esmfold_scorer model class", _esmfold_model_class)
        check("src.filip_guidance (AMPLIFY stack)",
              lambda: importlib.import_module("src.filip_guidance") and "importable",
              required=False)
    else:
        # Be specific about what is being skipped. Job 8883902 died on a model class that only
        # --deep looks for: esmfold_scorer imported, the weights were on disk, and the one thing
        # missing from the rebuilt venv surfaced 23 seconds into a queue slot instead of here.
        row(WARN, "esmfold_scorer / filip_guidance",
            "not checked -- pass --deep after ANY venv rebuild or image change")

    print()
    n_ok = sum(1 for s, _, _ in _rows if s == OK)
    n_warn = sum(1 for s, _, _ in _rows if s == WARN)
    print(f"{n_ok} ok, {n_warn} warn, {_fatal} fail")
    if _fatal:
        print("\nREQUIRED checks failed; the pipeline cannot run in this environment.")
        return 1
    print("\nNo required check failed. Warnings above are phases that would degrade or be "
          "skipped,\nnot the whole run. Next: the test suites, then a --smoke pass, then one "
          "short job.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
