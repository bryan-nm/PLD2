#!/bin/bash
# Shared setup for every PBS job in this repo (Aurora). Every job script starts with:
#   cd "${PBS_O_WORKDIR:?submit with qsub from the repo root}" || exit 1
#   source ./scripts/pbs_common.sh
# THE cd MUST COME FIRST: PBS runs a spooled copy with $HOME as cwd. config.py owns all
# model/dataset paths; job scripts must NOT export PLD2_* (that would override the config silently).
set -o pipefail

if [ ! -f config.py ] || [ ! -d src ]; then
    echo "FATAL: $(pwd) is not the PLD2 repo root (need config.py and src/)." >&2
    exit 1
fi

module load frameworks          # torch + oneCCL (NOT ipex: image 26.181.0 dropped it, see ipex_shim)
# Create once: python -m venv --system-site-packages, then
#   pip install -r requirements-aurora.txt    <- the two Biohub forks folding needs
# and confirm with `python -m src.env_check --deep` BEFORE spending a queue slot.
PLD2_VENV=${PLD2_VENV:-/flare/NLDesignProtein/bryan/envs/ProLoopDiff-env}
# shellcheck disable=SC1091
source "${PLD2_VENV}/bin/activate"

# --- the environment must work BEFORE anything else runs -------------------------------------
# An Aurora image update swaps the frameworks module, and a venv built with --system-site-packages
# against the previous one inherits a Python whose torch no longer resolves its MKL runtime. The
# failure is not subtle -- but without this check it surfaced as a config.py traceback that the
# banner PRINTED AND IGNORED, after which every derived path was the empty string and the job went
# on to mkdir '/pdb' at the filesystem root. Fail here, with the facts needed to fix it.
env_preflight() {
    if python -c "import torch" >/dev/null 2>&1; then
        # Record what this job is actually running against. An image update changes all of it
        # silently, and a log that does not say so cannot be compared with last week's.
        python - <<'PYEOF'
import os, sys
import torch
ipex = None
try:
    import intel_extension_for_pytorch as ipex
except Exception:
    pass
cfg = os.path.join(sys.prefix, "pyvenv.cfg")
home = ""
if os.path.exists(cfg):
    home = next((l.split("=", 1)[1].strip() for l in open(cfg) if l.startswith("home")), "")
ipexv = getattr(ipex, "__version__", "ABSENT -- src/ipex_shim.py stands in for it so that "
                                     "EsmFold does not read a missing package as a missing GPU")
print(f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | ipex {ipexv}")
print(f"[env] base {sys.base_prefix}")
if home and not os.path.realpath(home).startswith(os.path.realpath(sys.base_prefix)):
    print(f"[env] WARNING: venv was built against {home}, which is NOT the interpreter now in "
          f"use. That is how an Aurora image update breaks a run. Rebuild the venv if anything "
          f"below misbehaves; `python -m src.env_check` says what still imports.")
PYEOF
        return 0
    fi
    {
        echo "FATAL: this environment cannot import torch, so nothing downstream can run."
        echo "  python      : $(command -v python)"
        echo "  version     : $(python -V 2>&1)"
        echo "  VIRTUAL_ENV : ${VIRTUAL_ENV:-<none>}"
        echo "  venv built against:"
        sed 's/^/      /' "${PLD2_VENV}/pyvenv.cfg" 2>/dev/null || echo "      <no pyvenv.cfg>"
        echo "  the import fails with:"
        python -c "import torch" 2>&1 | tail -4 | sed 's/^/      /'
        echo
        echo "  MOST LIKELY: the frameworks module changed under the venv. Compare the version"
        echo "  above with the module's own interpreter, and if the minor version moved, rebuild:"
        echo "      module load frameworks"
        echo "      python -m venv --system-site-packages <NEW_ENV>"
        echo "      source <NEW_ENV>/bin/activate"
        echo "      pip install 'transformers>=4.57' biopython biotite cloudpathlib"
        echo "      pip install --no-deps esm"
        echo "  then re-run with PLD2_VENV=<NEW_ENV>. Keep the old env until the new one works."
        echo '  python -m src.env_check --deep  reports exactly which phases survive.' 
    } >&2
    exit 1
}
env_preflight

# --- guards for values a job DERIVES, so an empty one cannot reach a command ------------------
# require NAME VALUE [abs|int]
require() {
    local name="$1" val="$2" kind="${3:-}"
    if [ -z "${val}" ]; then
        echo "FATAL: ${name} is empty. Something that computes it failed; refusing to run with a" >&2
        echo "       blank path or count -- that is how 'mkdir /pdb' happens." >&2
        exit 1
    fi
    case "${kind}" in
        abs) case "${val}" in /?*) ;; *) echo "FATAL: ${name}='${val}' is not an absolute path." >&2; exit 1 ;; esac ;;
        int) case "${val}" in ''|*[!0-9]*) echo "FATAL: ${name}='${val}' is not a number." >&2; exit 1 ;; esac
             [ "${val}" -gt 0 ] || { echo "FATAL: ${name}=${val} must be positive." >&2; exit 1; } ;;
    esac
}

# --- topology: 12 tiles/node, one rank per tile ---
RANKS_PER_NODE=${RANKS_PER_NODE:-12}
NNODES=$(wc -l < "$PBS_NODEFILE" | tr -d " ")
NRANKS=$(( NNODES * RANKS_PER_NODE ))

# --- device selector + rendezvous (set AFTER module load frameworks) ---
# frameworks defaults ONEAPI_DEVICE_SELECTOR to opencl+level_zero, which double-enumerates tiles and
# breaks one-rank-per-tile. Force Level-Zero only. Without ZE_FLAT_DEVICE_HIERARCHY=FLAT one
# "device" is a whole 2-tile GPU with implicit scaling. dist.py warns if either is wrong.
export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=${MASTER_PORT:-29500}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- oneCCL / fabric (KVS exchange over MPI/PMIx, not CCL's internal TCP KVS which melts at scale) ---
export CCL_PROCESS_LAUNCHER=pmix
export CCL_ATL_TRANSPORT=mpi
export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=600
export FI_PROVIDER=cxi
export CCL_ZE_IPC_EXCHANGE=pidfd

# --- ESMFold2-Fast structural eval (src/fold_fasta.py; harmless for training ranks) ---
# The scorer lives in a sibling repo. Its deps (transformers>=4.57, esm --no-deps, biopython,
# biotite, cloudpathlib) must already be in the venv above; see EsmFold/README.md. HF_HUB_OFFLINE
# keeps the ESM-C 6B backbone resolving from ~/.cache/huggingface on compute nodes, which have no
# network -- pre-cache it once from a login node.
ESMFOLD_REPO=${ESMFOLD_REPO:-/flare/NLDesignProtein/bryan/Diffusion-dev-space/EsmFold}
if [ -d "${ESMFOLD_REPO}/src" ]; then
    export PYTHONPATH="${ESMFOLD_REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Add Foldseek path
# config.py owns this path (FOLDSEEK_DIR) like every other; exported here only so the binary is
# on PATH for anything that shells out by bare name. Override with PLD2_FOLDSEEK_DIR.
export PATH="${PLD2_FOLDSEEK_DIR:-/flare/NLDesignProtein/bryan/tools/foldseek/bin}:$PATH"

MPI_LAUNCH=(mpiexec -n "${NRANKS}" -ppn "${RANKS_PER_NODE}" --pmi=pmix --cpu-bind depth -d 8)

job_banner() {
    local title="$1"; shift
    echo "================================================================"
    echo "  ${title}"
    echo "  job        : ${PBS_JOBID:-<interactive>}  started $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "  repo       : $(pwd)   commit $(git rev-parse --short HEAD 2>/dev/null || echo '<no git>')"
    echo "  topology   : ${NNODES} nodes x ${RANKS_PER_NODE} ranks/node = ${NRANKS} ranks"
    local e
    for e in ONEAPI_DEVICE_SELECTOR CCL_PROCESS_LAUNCHER CCL_KVS_MODE FI_PROVIDER OMP_NUM_THREADS; do
        printf '    %-24s = %s\n' "$e" "${!e}"
    done
    echo "  config.py resolves to:"
    local cfg_out cfg_rc
    cfg_out=$(python config.py 2>&1); cfg_rc=$?
    echo "${cfg_out}" | sed 's/^/    /'
    if [ "${cfg_rc}" -ne 0 ]; then
        echo "FATAL: config.py exited ${cfg_rc}. Every path this job uses comes from it, so there" >&2
        echo "       is nothing safe to do next. The traceback is above." >&2
        exit 1
    fi
    local v
    for v in "$@"; do printf '    %-24s = %s\n' "$v" "${!v}"; done
    echo "================================================================"
}

# ---------------------------------------------------------------------------
# Slack notifications: START now (topology is known), FINISH via an EXIT trap.
# The FINISH trap reads $? at exit, so each job script MUST end with `exit $rc` rather than a
# status-resetting command like a trailing `echo`, or the reported code is always 0.
# ---------------------------------------------------------------------------
if [ -f ~/bin/slack_notify.sh ]; then
    source ~/bin/slack_notify.sh
else
    slack() { :; }
fi

_slack_finish() {
    local rc=$1
    if [ "$rc" -eq 0 ]; then
        slack ":white_check_mark: DONE ${PBS_JOBID} ${PBS_JOBNAME}"
    else
        slack ":x: FAILED (exit ${rc}) ${PBS_JOBID} ${PBS_JOBNAME}"
    fi
    echo "[job] finished $(date '+%Y-%m-%dT%H:%M:%S%z') exit=${rc}"
}
trap '_slack_finish $?' EXIT

slack ":rocket: START ${PBS_JOBID} ${PBS_JOBNAME} on ${NNODES} nodes"
