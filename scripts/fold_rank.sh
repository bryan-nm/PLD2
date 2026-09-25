#!/bin/bash
# Per-rank ESMFold supervisor. Runs INSIDE one MPI rank and relaunches its command a FEW times
# before surrendering. Used as:  mpiexec ... ./scripts/fold_rank.sh python -m src.fold_fasta ...
#
# WHY THE RETRY UNIT MOVED FROM THE JOB TO THE RANK. ESMFold aborts on Aurora at a rate that
# survived four hypotheses and four refutations (see src/fold_fasta.py). The retry used to wrap the
# whole mpiexec, and that does not scale: when one rank of 192 dies, mpiexec SIGTERMs the other 191
# and the job-level loop reloads ESMFold on all of them. Measured on a 24,000-generation pass --
# twenty attempts, every one ending in exit 143, 66% coverage, out of tries while still making
# progress. mpiexec only watches its DIRECT child, so making that child a shell means the fault
# kills the python grandchild and one rank reloads instead of 192.
#
# WHY IT SURRENDERS EARLY, WHICH THE FIRST VERSION DID NOT. That version retried 40 times over two
# hours per rank, and a 6-hour job that should have taken 1.6 spent 5.25 hours in the fold phase
# with 165 of 192 ranks already finished -- the other 27 had aborted (SIGABRT, 134), relaunched,
# and neither completed nor gave up. The job-level teardown is not merely a crude retry: it is the
# only thing that returns every tile to a clean state, and a rank that cannot recover locally has
# to let that happen. So: a few quick in-rank retries for a transient fault, then exit and let
# mpiexec tear the launch down for the outer loop to restart. Both layers, each doing its own job.
#
# AND A HUNG CHILD IS NOW KILLED. The first version handled a child that DIES and not one that
# hangs, so a wedged process held its rank -- and the whole mpiexec -- indefinitely. Every attempt
# runs under `timeout`, and a timed-out attempt counts as a failure like any other.
#
# A HANG IS AN ABSENCE OF PROGRESS, NOT A LONG RUNTIME, and conflating the two does not survive a
# change of scale. The first watchdog killed any attempt past RANK_ATTEMPT=1200s, which is fine when
# a rank holds ~80 structures and catastrophic when it holds 833: at ten times the prompts every
# WORKING rank would be killed at 48% done and the phase would churn forever. So the supervisor
# watches the file the rank appends to -- fold_fasta fsyncs one record per sequence -- and kills
# only when it stops growing. That bound does not care how much work a rank was given.
#
#   RANK_TRIES     in-rank relaunches before surrendering   (default 3)
#   RANK_BUDGET    wall-clock seconds for this rank         (default 1800)
#   RANK_PROGRESS  file this rank appends to; %r -> its id  (default unset)
#   RANK_STALL     seconds of no growth before killing      (default 900)
#   RANK_ATTEMPT   absolute cap on ONE attempt, 0 = off     (default 0)
#   RANK_MIN_RUN   below this, a failure is not a fault     (default 30)
#   RANK_HARD      consecutive fast failures to abort       (default 3)
#
# RANK_BUDGET bounds how long one mpiexec can run, so the caller's own budget stays meaningful:
# worst case for a fold phase is FOLD_BUDGET + RANK_BUDGET, and that has to fit the job's walltime.
set -o pipefail

TRIES=${RANK_TRIES:-3}
BUDGET=${RANK_BUDGET:-1800}
ATTEMPT_MAX=${RANK_ATTEMPT:-0}
PROGRESS=${RANK_PROGRESS:-}
STALL=${RANK_STALL:-900}
MIN_RUN=${RANK_MIN_RUN:-30}
HARD=${RANK_HARD:-3}
ME=${PALS_RANKID:-${PMIX_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-?}}}}

[ $# -gt 0 ] || { echo "[rank-fold] usage: $0 <command> [args...]" >&2; exit 2; }

# A WATCHDOG IN BASH RATHER THAN coreutils `timeout`. Bounding a hang is the entire point of this
# rewrite, and making it conditional on an optional binary would leave the bug in place wherever
# that binary is missing -- which is not hypothetical: it is absent on macOS, where this is
# developed and tested. -> 124 when it had to kill, otherwise the child's own status.
CHILD=""
# A SIGNAL TO THE SUPERVISOR MEANS STOP, NOT RETRY. Forward it to the child and leave: the usual
# sender is PBS at the walltime limit, and relaunching ESMFold into a job that is being torn down
# wastes the only thing left. Without the exit, the loop treats the killed child as one more
# failure and starts another attempt.
_forward() {
    [ -n "${CHILD}" ] && kill -TERM "${CHILD}" 2>/dev/null
    echo "[rank-fold] rank ${ME}: signalled; stopping (child ${CHILD:-none} sent TERM)"
    exit 143
}
trap _forward TERM INT

# The file this rank appends to. %r becomes its zero-padded id, matching fold_fasta.rank_path.
# Unavailable (no RANK_PROGRESS, or a non-numeric rank id) falls back to the absolute cap alone.
_progress_path() {
    [ -n "${PROGRESS}" ] || return 1
    case "${ME}" in ''|*[!0-9]*) return 1 ;; esac
    printf '%s' "${PROGRESS//%r/$(printf '%03d' "${ME}")}"
}
_progress_size() {
    local f; f=$(_progress_path) || return 1
    wc -c < "${f}" 2>/dev/null || echo 0
}

_kill_child() {
    kill -TERM "${CHILD}" 2>/dev/null
    sleep 20
    kill -KILL "${CHILD}" 2>/dev/null
    wait "${CHILD}" 2>/dev/null
    CHILD=""
}

run_bounded() {
    "$@" &
    CHILD=$!
    local waited=0 rc now last_size quiet
    last_size=$(_progress_size) || last_size=""
    quiet=0
    while kill -0 "${CHILD}" 2>/dev/null; do
        if [ "${ATTEMPT_MAX}" -gt 0 ] && [ "${waited}" -ge "${ATTEMPT_MAX}" ]; then
            _kill_child; return 124
        fi
        if [ -n "${last_size}" ]; then
            now=$(_progress_size) || now="${last_size}"
            if [ "${now}" != "${last_size}" ]; then
                last_size="${now}"; quiet=0
            else
                quiet=$(( quiet + 5 ))
                if [ "${quiet}" -ge "${STALL}" ]; then
                    _kill_child; return 125
                fi
            fi
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    wait "${CHILD}"; rc=$?
    CHILD=""
    return ${rc}
}

if _progress_path >/dev/null 2>&1; then
    echo "[rank-fold] rank ${ME}: stall detection on $(_progress_path) (${STALL}s without growth)"
elif [ "${ATTEMPT_MAX}" -eq 0 ]; then
    echo "[rank-fold] rank ${ME}: WARNING: neither RANK_PROGRESS nor RANK_ATTEMPT is set, so a"\
         "hung attempt cannot be detected. Set one."
fi

t0=$SECONDS
fast=0
for attempt in $(seq 1 "${TRIES}"); do
    elapsed=$(( SECONDS - t0 ))
    if [ "${elapsed}" -ge "${BUDGET}" ]; then
        echo "[rank-fold] rank ${ME}: budget ${BUDGET}s spent after $(( attempt - 1 )) attempt(s);"\
             "surrendering so mpiexec can tear down and the outer loop can restart clean."
        exit 1
    fi
    started=$SECONDS
    run_bounded "$@"
    rc=$?
    ran=$(( SECONDS - started ))

    if [ "${rc}" -eq 0 ]; then
        [ "${attempt}" -gt 1 ] && echo "[rank-fold] rank ${ME}: clean on attempt ${attempt}"\
            "after $(( SECONDS - t0 ))s"
        exit 0
    fi

    # 124/125 are the watchdog's: the attempt hung rather than faulted. Never counted as a "fast"
    # failure -- it ran for the full window by definition.
    if [ "${rc}" -eq 125 ]; then
        fast=0
        echo "[rank-fold] rank ${ME}: attempt ${attempt} wrote nothing for ${STALL}s and was"\
             "killed after ${ran}s (${elapsed}s into this rank's budget)"
    elif [ "${rc}" -eq 124 ]; then
        fast=0
        echo "[rank-fold] rank ${ME}: attempt ${attempt} hit the absolute cap ${ATTEMPT_MAX}s and"\
             "was killed (${elapsed}s into this rank's budget)"
    elif [ "${ran}" -lt "${MIN_RUN}" ]; then
        fast=$(( fast + 1 ))
        echo "[rank-fold] rank ${ME}: attempt ${attempt} exited ${rc} after only ${ran}s --"\
             "too fast to be a GPU fault (${fast}/${HARD} consecutive)"
        if [ "${fast}" -ge "${HARD}" ]; then
            echo "[rank-fold] rank ${ME}: ${fast} immediate failures in a row; this is not"\
                 "transient. Aborting with rc=${rc} rather than spinning."
            exit "${rc}"
        fi
    else
        fast=0
        if [ "${attempt}" -lt "${TRIES}" ]; then
            echo "[rank-fold] rank ${ME}: attempt ${attempt} exited ${rc} after ${ran}s"\
                 "(${elapsed}s into budget); relaunching (the other ranks are untouched)"
        fi
    fi
done
echo "[rank-fold] rank ${ME}: ${TRIES} attempts exhausted in $(( SECONDS - t0 ))s; surrendering to"\
     "the outer loop, whose mpiexec teardown is what resets every tile."
exit 1
