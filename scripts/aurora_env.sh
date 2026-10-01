#!/usr/bin/env bash
# Build (or repair) the venv the alignment pipeline needs, on an Aurora LOGIN node.
#
#   module load frameworks
#   source /flare/NLDesignProtein/bryan/envs/<env>/bin/activate
#   scripts/aurora_env.sh
#
# WHY THIS IS A SCRIPT AND NOT ONE requirements FILE. The three steps below need three different
# pip behaviours, and a requirements file cannot express per-package --no-deps or --ignore-installed:
#
#   1. transformers must be forced into the venv. The frameworks module bundles an older copy that
#      silently satisfies the requirement through --system-site-packages.
#   2. esm must go in with --no-deps. It pins torch>=2.11,<2.12 and two CUDA cuequivariance wheels
#      whose platform markers match Aurora, so a plain install replaces the frameworks torch and
#      folding loses the GPU.
#   3. its remaining dependencies then go in normally, from requirements-aurora.txt.
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

# --ignore-installed applies to transformers ALONE, on purpose. Passing it to the whole
# requirements file would reinstall numpy and every other system package from PyPI.
python -m pip install --ignore-installed 'transformers>=4.57.6,<5'
python -m pip install --no-deps "esm @ git+https://github.com/Biohub/esm.git@${ESM_COMMIT}"
python -m pip install -r requirements-aurora.txt

echo
python -m src.env_check --deep
