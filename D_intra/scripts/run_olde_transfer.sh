#!/bin/bash
# Submit everything needed for the Luxembourg -> Oldenburg transfer tables.
#
# This is a SUBMITTER, not a job. Run it on the login node with bash:
#
#     cd /rds/projects/c/chenhp-graphsyn/D_intra
#     bash scripts/run_olde_transfer.sh            # the base model, hops and no-hops
#     bash scripts/run_olde_transfer.sh _v3        # version 3 (or _centroid, _hopmlp)
#
# Upload data/cluster_lux/raw/pairs_*_lux2olde_k7*.pt first. Both graphs are
# coarsened at k=7, and the work splits unevenly:
#
#   A            NOTHING. A never sees the private graph: pairs_A_lux2olde_k7.pt
#                is byte-identical to pairs_A_lux2minn_k7.pt, so the A samples
#                already in ../results/samples/D_intra/ are the Oldenburg A samples.
#   B, B_nohop   NO TRAINING. B trains and validates on Luxembourg alone, so the
#                private graph changes B's TEST split and nothing else -- same
#                training records, same standardisation statistics, same
#                weights. The existing transfer_B / transfer_B_nohop checkpoints
#                ARE the Oldenburg B models. Sampling only: ~15 min each.
#   C, C_nohop   FULL TRAINING. C's training set contains Oldenburg clusters, so
#                these are new models (transfer_C_olde*) and need the 6h slot.
#
# The C jobs are chained rather than submitted together: on a first run they
# both process new pairs files under the shared data/cluster_lux/ root, and
# pre_filter.pt / pre_transform.pt carry the same name for every dataset under
# that root, so simultaneous first runs can race on them. The B sampling jobs
# are NOT immune: the lux2olde pairs files are new, so they process on first
# run too. The window is the first minute or two and a collision crashes loudly
# rather than corrupting results -- check each log reaches "input features".
# Running the script twice (plain, then _v3) puts C and C_v3 in the queue
# unchained; `scontrol hold` the second until the first is past processing.
set -euo pipefail

cd "$(dirname "$0")/.."

SFX="${1:-}"
case "$SFX" in
    ""|_centroid)  CFGS_B="B B_nohop"; CFGS_C="C C_nohop" ;;
    # MLP_h embeds h, and the _nohop condition has no h to embed.
    _v3|_hopmlp)   CFGS_B="B";         CFGS_C="C" ;;
    *) echo "usage: bash scripts/run_olde_transfer.sh [_v3|_centroid|_hopmlp]" >&2; exit 1 ;;
esac

echo "sampling B from the EXISTING Luxembourg-trained checkpoints (no retrain):"
for CFG in $CFGS_B; do
    CFG="$CFG$SFX"
    JID=$(sbatch --parsable --job-name="sample_$CFG" \
                 scripts/run_sample_conditional.sh "$CFG")
    echo "  $CFG -> job $JID"
done

echo
echo "training C (its training set contains Oldenburg clusters, so it is a new model):"
DEP=""
for CFG in $CFGS_C; do
    CFG="$CFG$SFX"
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

cat <<EOF2

  watch     squeue -u \$USER -o "%.10i %.14j %.9T %.10M %.12L %R"
  check     grep -m1 "input features" \$(ls -t logs/transfer_*.out | head -1)
            -> "7 2 22" for C, "7 2 15" for C_nohop, "8 2 16" for C_v3

Collect the samples into ../results/samples/D_intra/oldenburg/, then build the tables
locally with ./make_tables.sh. A comes across unchanged; B and C are new:

  mkdir -p ../results/samples/D_intra/oldenburg
  cp ../results/samples/D_intra/A{,_nohop,_v3}.txt ../results/samples/D_intra/oldenburg/
  for c in B B_nohop B_v3; do
      cp "\$(ls -t outputs/*/*transfer_\${c}_olde_sample*/conditional_samples.txt | head -1)" \\
         "../results/samples/D_intra/oldenburg/\${c}.txt"
  done
  for c in C C_nohop C_v3; do
      cp "\$(ls -t outputs/*/*transfer_\${c/#C/C_olde}_sample*/conditional_samples.txt | head -1)" \\
         "../results/samples/D_intra/oldenburg/\${c}.txt"
  done
EOF2
