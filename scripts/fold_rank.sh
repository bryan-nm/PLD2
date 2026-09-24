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
#   RANK_TRIES     in-rank relaunches before surrendering   (default 3)
#   RANK_BUDGET    wall-clock seconds for this rank         (default 1800)
#   RANK_ATTEMPT   hard timeout on ONE attempt              (default 1200)
#   RANK_MIN_RUN   below this, a failure is not a fault     (default 30)
#   RANK_HARD      consecutive fast failures to abort       (default 3)
#
# RANK_BUDGET bounds how long one mpiexec can run, so the caller's own budget stays meaningful:
# worst case for a fold phase is FOLD_BUDGET + RANK_BUDGET, and that has to fit the job's walltime.
set -o pipefail

TRIES=${RANK_TRIES:-3}
BUDGET=${RANK_BUDGET:-1800}
ATTEMPT_MAX=${RANK_ATTEMPT:-1200}
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

run_bounded() {
    "$@" &
    CHILD=$!
    local waited=0 rc
    while kill -0 "${CHILD}" 2>/dev/null; do
        if [ "${waited}" -ge "${ATTEMPT_MAX}" ]; then
            kill -TERM "${CHILD}" 2>/dev/null
            sleep 20
            kill -KILL "${CHILD}" 2>/dev/null
            wait "${CHILD}" 2>/dev/null
            CHILD=""
            return 124
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    wait "${CHILD}"; rc=$?
    CHILD=""
    return ${rc}
}

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

    # 124 is the watchdog's: the attempt hung rather than faulted. Same treatment, louder, and
    # never counted as a "fast" failure -- it ran for the full attempt window by definition.
    if [ "${rc}" -eq 124 ]; then
        fast=0
        echo "[rank-fold] rank ${ME}: attempt ${attempt} HUNG past ${ATTEMPT_MAX}s and was killed"\
             "(${elapsed}s into this rank's budget)"
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
