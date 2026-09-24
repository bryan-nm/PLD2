#!/bin/bash
# Behavioural tests for scripts/fold_rank.sh:  bash src/tests_fold_rank.sh
#
# Every case here is one a real run hit. The hang test is the important one: the first version of
# the supervisor handled a child that DIES and not one that HANGS, and a 6-hour job spent 5.25
# hours in its fold phase with 165 of 192 ranks already finished.
cd "$(dirname "$0")/.." || exit 1
S=scripts/fold_rank.sh
T=$(mktemp -d); trap 'rm -rf "$T"; pkill -f PLD2_TEST_HANG 2>/dev/null' EXIT
pass=0; fail=0
ok() { if [ "$2" = "$3" ]; then pass=$((pass+1)); else fail=$((fail+1)); echo "  FAIL $1: got '$2' want '$3'"; fi; }

cat > "$T/flaky" <<'X'
#!/bin/bash
n=$(cat "$1" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$1"
sleep "${3:-1}"; [ "$n" -ge "$2" ] && exit 0; exit 134
X
cat > "$T/hang" <<'X'
#!/bin/bash
exec -a PLD2_TEST_HANG sleep 400
X
chmod +x "$T/flaky" "$T/hang"

# a transient fault is recovered inside the rank -- the whole point of the supervisor
RANK_MIN_RUN=0 RANK_TRIES=3 bash $S "$T/flaky" "$T/c1" 2 1 >/dev/null 2>&1
ok "transient fault recovers" $? 0

# an unrecoverable rank surrenders rather than holding the job
t0=$SECONDS
RANK_MIN_RUN=0 RANK_TRIES=3 bash $S "$T/flaky" "$T/c2" 9999 1 >/dev/null 2>&1
rc=$?; el=$(( SECONDS - t0 ))
ok "unrecoverable surrenders" $rc 1
[ "$el" -lt 30 ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "  FAIL surrender took ${el}s"; }

# a hung child is killed and does not hold the rank -- the 5.25h bug
t0=$SECONDS
RANK_ATTEMPT=5 RANK_TRIES=2 RANK_BUDGET=60 bash $S "$T/hang" >/dev/null 2>&1
rc=$?; el=$(( SECONDS - t0 ))
ok "hung child is killed" $rc 1
[ "$el" -lt 60 ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "  FAIL hang test took ${el}s"; }

# a wrong command aborts at once instead of spinning through its budget
t0=$SECONDS
RANK_TRIES=40 bash $S "$T/flaky" "$T/c3" 9999 0 >/dev/null 2>&1
rc=$?; el=$(( SECONDS - t0 ))
ok "bad command aborts with the real code" $rc 134
[ "$el" -lt 20 ] && pass=$((pass+1)) || { fail=$((fail+1)); echo "  FAIL bad-command took ${el}s"; }

# a signal means stop, not retry: the child dies and nothing is orphaned
RANK_ATTEMPT=999 bash $S "$T/hang" >/dev/null 2>&1 &
sup=$!; sleep 3
kill -TERM $sup 2>/dev/null; sleep 3
left=$(pgrep -f PLD2_TEST_HANG 2>/dev/null | wc -l | tr -d ' ')
ok "signal leaves no orphan" "$left" 0
kill -0 $sup 2>/dev/null && { fail=$((fail+1)); echo "  FAIL supervisor survived its TERM"; } \
                         || pass=$((pass+1))

# usage
bash $S >/dev/null 2>&1; ok "no args is a usage error" $? 2

echo "$pass/$((pass+fail)) fold_rank checks pass"
[ "$fail" -eq 0 ]
