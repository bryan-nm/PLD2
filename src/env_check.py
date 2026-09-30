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
torch itself. Every ipex.optimize call in this repo is already guarded; losing it costs whatever
weight prepacking was worth and nothing else.
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
          detail="absent is fine: every ipex.optimize in this repo is guarded")
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
    from shutil import which
    row(OK if which("foldseek") else WARN, "foldseek on PATH",
        which("foldseek") or "MISSING: phases 0c and 3 need it")

    print("\nfolding (phase 0b, 2) -- the half a broken esm takes down")
    check("esm", lambda: importlib.import_module("esm").__version__, required=False)
    check("transformers", lambda: importlib.import_module("transformers").__version__,
          required=False)
    if a.deep:
        check("esmfold_scorer.StructureScorer",
              lambda: importlib.import_module("esmfold_scorer").StructureScorer and "importable")
        check("src.filip_guidance (AMPLIFY stack)",
              lambda: importlib.import_module("src.filip_guidance") and "importable",
              required=False)
    else:
        row(WARN, "esmfold_scorer / filip_guidance", "not checked -- pass --deep")

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
