#!/bin/bash
# Submit A, B and C as a CHAIN -- each starts only if the previous SUCCEEDED.
#
# This is a SUBMITTER, not a job. Run it on the login node with bash:
#
#     cd /rds/projects/c/chenhp-graphsyn/D_intra
#     bash scripts/run_all_transfer.sh
#
# Optionally pick a subset, in order:
#
#     bash scripts/run_all_transfer.sh B C
#
# WHY CHAINED rather than one long job: run_transfer.sh asks for 6 hours. Doing
# all three inside a single allocation would need 18, which the QOS will not
# grant. Chaining keeps three separate 6h jobs that happen to run in order.
#
# WHY NOT three independent sbatch calls: all three configs share
# data/cluster_lux/ as their PyG root. The per-config caches
# (pairs_A_..._train.pt and friends) do not collide, but pre_filter.pt and
# pre_transform.pt have the same name for every dataset under that root, so
# three jobs starting together can race on them the first time each config is
# processed. Chaining removes the race for free. If you have already processed
# all three once, submitting in parallel is fine and finishes sooner.
set -euo pipefail

cd "$(dirname "$0")/.."

CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=(A B C)

DEP=""
for CFG in "${CFGS[@]}"; do
    case "$CFG" in
        A|B|C) ;;
        *) echo "usage: bash scripts/run_all_transfer.sh [A] [B] [C]" >&2; exit 1 ;;
    esac

    if [ -z "$DEP" ]; then
        JID=$(sbatch --parsable --job-name="transfer_$CFG" scripts/run_transfer.sh "$CFG")
        echo "  $CFG -> job $JID"
    else
        JID=$(sbatch --parsable --job-name="transfer_$CFG" \
                     --dependency=afterok:"$DEP" scripts/run_transfer.sh "$CFG")
        echo "  $CFG -> job $JID   (waits for $DEP)"
    fi
    DEP=$JID
done

cat <<'EOF'

  watch     squeue -u $USER -o "%.10i %.14j %.9T %.10M %.12L %R"
            the last column is WHY a job is pending
  logs      ls -t logs/transfer_*.out | head -3
  check     grep -m1 "input features" $(ls -t logs/transfer_*.out | head -1)
            -> must read "Size of the input features 7 2 22"
  cancel    scancel <jobid> [<jobid> ...]
            scancel takes ONE --name, so cancel by job id, not by name

If a link FAILS, the ones behind it do not run -- they sit in state
DependencyNeverSatisfied and must be cancelled by hand. Read the failed job's
.err before resubmitting.

To release a pending link EARLY without losing the priority it has accrued,
strip its dependency in place rather than cancelling and resubmitting:

    scontrol update jobid=<jobid> Dependency=

Cancelling resets the job's submit time and sends it to the back of the queue,
which after a long wait is the opposite of what you want. And cancelling a link
strands everything chained behind it.
EOF
