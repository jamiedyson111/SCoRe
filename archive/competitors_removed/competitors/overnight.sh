#!/bin/bash
# Start every competitor unattended. Run ONCE on a BlueBEAR LOGIN node, detached,
# so it survives logging out:
#
#   cd /rds/projects/c/chenhp-graphsyn && nohup setsid bash competitors/overnight.sh > competitors/logs/overnight.log 2>&1 &
#
# 1. submits the DiGress memory test (minutes of GPU);
# 2. builds the ICLR [2] env on this login node (it needs the internet), then
#    queues CHAIN training jobs of 12 h, each starting when the previous ends
#    (afterany, so a timeout still lets the next one resume);
# 3. the same for HiGen.
# A setup failure skips that method's jobs and is logged; the rest carry on.
# Progress: competitors/logs/overnight.log, *_setup.log, and squeue.

set -u
CHAIN=${1:-4}                       # 4 x 12 h = 2 days of training each
BASE=/rds/projects/c/chenhp-graphsyn
cd "$BASE"
mkdir -p competitors/logs
say () { echo "$(date '+%F %T')  $*"; }

chain () {   # chain NAME SCRIPT N
    local prev="" jid dep
    for i in $(seq 1 "$3"); do
        dep=${prev:+--dependency=afterany:$prev}
        jid=$(sbatch --parsable $dep "$2") || { say "$1: sbatch failed"; return 1; }
        say "$1 training job $i/$3: $jid"
        prev=$jid
    done
}

say "DiGress scale test: $(cd D_intra && sbatch --parsable competitors/run_digress_scale.sh)"

say "ICLR [2] setup starting (log: competitors/logs/iclr2_setup.log)"
if bash competitors/iclr2/setup_bluebear.sh > competitors/logs/iclr2_setup.log 2>&1; then
    say "ICLR [2] setup OK"
    chain ICLR2 competitors/iclr2/run_train.sh "$CHAIN"
else
    say "ICLR [2] setup FAILED -- see competitors/logs/iclr2_setup.log; no jobs queued"
fi

say "HiGen setup starting (log: competitors/logs/higen_setup.log)"
if bash competitors/higen/setup_bluebear.sh > competitors/logs/higen_setup.log 2>&1; then
    say "HiGen setup OK"
    chain HiGen competitors/higen/run_train.sh "$CHAIN"
else
    say "HiGen setup FAILED -- see competitors/logs/higen_setup.log; no jobs queued"
fi
say "done -- check with: squeue -u \$USER"
