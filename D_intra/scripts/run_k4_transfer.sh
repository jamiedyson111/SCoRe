#!/bin/bash
# Submit everything needed for the Minnesota k=4 transfer tables.
#
# This is a SUBMITTER, not a job. Run it on the login node with bash:
#
#     cd /rds/projects/c/chenhp-graphsyn/D_intra
#     bash scripts/run_k4_transfer.sh
#
# Luxembourg stays at k=7; only the private graph is recoarsened. That asymmetry
# is what makes the work split unevenly:
#
#   B, B_nohop   NO TRAINING. B trains and validates on Luxembourg alone, so the
#                private graph's k changes B's TEST split and nothing else --
#                same training records, same standardisation statistics, same
#                weights. The existing transfer_B / transfer_B_nohop checkpoints
#                ARE the k=4 B models. Sampling only: ~15 min each.
#   C, C_nohop   FULL TRAINING. C's training set now contains k=4 clusters, so
#                these are genuinely new models and need the 6h allocation.
#
# The two C jobs are chained rather than submitted together: on a first run they
# both process new pairs files under the shared data/cluster_lux/ root, and
# pre_filter.pt / pre_transform.pt carry the same name for every dataset under
# that root, so simultaneous first runs can race on them. The B sampling jobs
# read caches that already exist and are safe to run alongside.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "sampling B from the EXISTING Luxembourg-trained checkpoints (no retrain):"
for CFG in B_k4 B_k4_nohop; do
    JID=$(sbatch --parsable --job-name="sample_$CFG" \
                 scripts/run_sample_conditional.sh "$CFG")
    echo "  $CFG -> job $JID"
done

echo
echo "training C (its training set contains k=4 clusters, so it is a new model):"
DEP=""
for CFG in C_k4 C_k4_nohop; do
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
  check     grep -m1 "input features" $(ls -t logs/transfer_*.out | head -1)
            -> "7 2 22" for C_k4, "7 2 15" for C_k4_nohop
            The k=4 datasets keep the 7-wide hop histogram, so the widths are
            UNCHANGED from the k=7 configurations. A width of 15 on C_k4 means
            the nohop pairs file was picked up by mistake.

Collect the four samples, then build the table locally in the kmis env:

  for c in B_k4 B_k4_nohop C_k4 C_k4_nohop; do
      cp "$(ls -t outputs/*/*transfer_${c}_sample*/conditional_samples.txt | head -1)" \
         "results/samples/${c}.txt"
  done

  cd evaluation && python make_intra_table.py --tag lux2minn_k7k4 \
      --b results/samples/B_k4.txt       --b-nohop results/samples/B_k4_nohop.txt \
      --c results/samples/C_k4.txt       --c-nohop results/samples/C_k4_nohop.txt \
      --latex
EOF
