#!/usr/bin/env bash
# Build (or repair) the venv the alignment pipeline needs, on an Aurora LOGIN node.
#
#   module load frameworks
#   source /flare/NLDesignProtein/bryan/envs/<env>/bin/activate
#   scripts/aurora_env.sh
#
# WHY THIS IS A SCRIPT AND NOT ONE requirements FILE. esm must go in with --no-deps -- it pins
# torch>=2.11,<2.12 and two CUDA cuequivariance wheels whose platform markers match Aurora, so a
# plain install replaces the frameworks torch 2.13 and folding loses the GPU -- and a requirements
# file cannot express per-package --no-deps. Its own dependencies then go in normally.
#
# DO NOT ADD --ignore-installed. It was here to force transformers past the 5.x copy the frameworks
# module bundles, but it is a GLOBAL flag: pip stops counting anything as installed and reinstalls
# the whole dependency closure, which shadowed the frameworks numpy with PyPI numpy 2.5.3 and broke
# numba's `numpy<2.5` pin inside the shared module. The `<5` upper bound below does the same job
# precisely, because a 5.x install genuinely does not satisfy it.
#
# WHICH esm COMMIT, AND WHY IT MATTERS. EsmFold's _resolve_model_class() looks for an ESMFold2
# model class in esm first and transformers second. No transformers release has ever shipped one,
# so esm is the only source -- and esm only started exporting EsmFold2Model after 3.3.0. Pin a
# commit known to export it, not @main, so a rebuild repeats a working environment. 3.3.0 is the
# trap: it HAS esm.models.esmfold2 (input prep and tokenisation) and no model in it, which is how
# job 8883902 got an ImportError telling it to install a package it already had.
set -euo pipefail

ESM_COMMIT=${ESM_COMMIT:-43b4548b86762edfa747b07d5f440aad3c33acee}   # 3.4.1.post1, 2026-09-16

[ -n "${VIRTUAL_ENV:-}" ] || { echo "Activate the target venv first." >&2; exit 1; }
cd "$(dirname "$0")/.." || exit 1
echo "venv   : ${VIRTUAL_ENV}"
echo "python : $(python -V 2>&1)  ($(command -v python))"
python -c "import torch, sys; print(f'torch  : {torch.__version__}')" || {
    echo "torch does not import; run 'module load frameworks' first." >&2; exit 1; }

python -m pip install --no-deps "esm @ git+https://github.com/Biohub/esm.git@${ESM_COMMIT}"
# requirements-aurora.txt pins transformers<5 and lists the rest of esm's dependencies. Everything
# the frameworks module already provides reports "already satisfied" and is left alone -- that is
# the behaviour --ignore-installed would destroy.
python -m pip install -r requirements-aurora.txt

# The three resolver complaints this leaves are EXPECTED and must not be chased: esm asks for
# torch>=2.11,<2.12 (we deliberately keep the frameworks 2.13) and for two CUDA cuequivariance
# wheels (there is no CUDA here). That is precisely what --no-deps was for.

echo
python -m src.env_check --deep
