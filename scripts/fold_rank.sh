#!/bin/bash
# Per-rank ESMFold supervisor. Runs INSIDE one MPI rank and relaunches its command until it exits
# clean. Used as:  mpiexec ... ./scripts/fold_rank.sh python -m src.fold_fasta ...
#
# WHY THE RETRY UNIT HAD TO MOVE FROM THE JOB TO THE RANK. ESMFold aborts on Aurora at a rate that
# survived four hypotheses and four refutations (see src/fold_fasta.py), and the pipeline is built
# around that rather than against it. But the retry lived in the PBS script, around the whole
# mpiexec, and that does not scale: when one rank of 192 dies, mpiexec SIGTERMs the other 191, and
# the job-level loop then reloads ESMFold on ALL of them. Measured on a 24,000-generation fold
# pass -- twenty attempts, every one ending in exit 143, 66% coverage, and the loop simply ran out
# of tries while still making progress on every attempt:
#
#     fold attempt  1 exited 143:      0 -> 1126 records
#     fold attempt 20 exited 143:  15740 -> 15858 records
#
# mpiexec only watches its DIRECT child. Make that child a shell and the GPU fault kills the
# python grandchild instead: the shell survives, relaunches it, and the other 191 ranks never
# notice. One fault now costs one model reload rather than 192.
#
# THE LOOP IS BOUNDED, because a command that is simply wrong would otherwise spin forever. A
# transient GPU fault cannot happen in under ~30s -- loading the ESM-C backbone alone takes longer
# than that -- so a failure faster than RANK_MIN_RUN is treated as a hard error, and RANK_HARD of
# those in a row aborts the rank with the real exit code. Bad arguments fail in under a second and
# stop immediately; a fault after real work is retried.
#
#   RANK_TRIES    max relaunches for this rank          (default 40)
#   RANK_BUDGET   wall-clock seconds for this rank      (default 7200)
#   RANK_MIN_RUN  below this, a failure is not a fault  (default 30)
#   RANK_HARD     consecutive fast failures to abort    (default 3)
set -o pipefail

TRIES=${RANK_TRIES:-40}
BUDGET=${RANK_BUDGET:-7200}
MIN_RUN=${RANK_MIN_RUN:-30}
HARD=${RANK_HARD:-3}
# Best-effort rank id, for logs only. PALS sets PALS_RANKID on Aurora; the others are fallbacks.
ME=${PALS_RANKID:-${PMIX_RANK:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-?}}}}

[ $# -gt 0 ] || { echo "[rank-fold] usage: $0 <command> [args...]" >&2; exit 2; }

t0=$SECONDS
fast=0
for attempt in $(seq 1 "${TRIES}"); do
    elapsed=$(( SECONDS - t0 ))
    if [ "${elapsed}" -ge "${BUDGET}" ]; then
        echo "[rank-fold] rank ${ME}: budget ${BUDGET}s spent after $(( attempt - 1 )) attempt(s)"
        exit 1
    fi
    started=$SECONDS
    "$@"
    rc=$?
    ran=$(( SECONDS - started ))

    if [ "${rc}" -eq 0 ]; then
        [ "${attempt}" -gt 1 ] && echo "[rank-fold] rank ${ME}: clean on attempt ${attempt}"
        exit 0
    fi

    if [ "${ran}" -lt "${MIN_RUN}" ]; then
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
            echo "[rank-fold] rank ${ME}: attempt ${attempt} exited ${rc} after ${ran}s;"\
                 "relaunching (the other ranks are untouched)"
        fi
    fi
done
echo "[rank-fold] rank ${ME}: ${TRIES} attempts exhausted"
exit 1
