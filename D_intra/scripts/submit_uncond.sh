#!/bin/bash
# Plain DiGress, one supernode at a time (no condition), as one chain of SLURM jobs.
#
# A SUBMITTER, not a job. Run it once on the login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/D_intra/scripts/submit_uncond.sh
#
# It submits, each job waiting (afterok) for the model it samples:
#
#   train      A_uncond  (Luxembourg; also B's model)   C_uncond  (Luxembourg + Oldenburg, jointly)
#              200 epochs, the same architecture and settings as D_intra, condition removed
#   evaluate   5 seeds of B_uncond and C_uncond (63 Oldenburg test supernodes) and
#              A_uncond (1763 Luxembourg), each graph generated at its target's size n_i
#
# 5 jobs. Needs the _uncond datasets (extraction/make_transfer_datasets.py --no-cond)
# and configs/dataset/cluster_{A,B,C}_lux2olde_k7_s80_uncond.yaml uploaded first.
# Fetch the samples with fetch_s80.sh locally, then score with
#     python repeat_summary.py --configs B_uncond B_nohop_centroid B_v3 C_uncond C_nohop_centroid C_v3 BS1 BS2
set -euo pipefail

INTRA=/rds/projects/c/chenhp-graphsyn/D_intra
for c in A B C; do
    f=$INTRA/data/cluster_lux/raw/pairs_${c}_lux2olde_k7_s80_uncond.pt
    [ -f "$f" ] || { echo "ERROR: $f is missing -- upload D_intra/data first" >&2; exit 1; }
done

sub() { (cd "$INTRA" && sbatch --parsable "$@"); }

echo "== training (plain DiGress, no condition)"
TA=$(sub --time=12:00:00 scripts/run_transfer.sh A_uncond); echo "  A_uncond -> $TA"
TC=$(sub --time=12:00:00 scripts/run_transfer.sh C_uncond); echo "  C_uncond -> $TC"

echo "== evaluation (5 seeds of every test supernode)"
for CFG in B_uncond C_uncond A_uncond; do
    case $CFG in B*|A*) D=$TA ;; C*) D=$TC ;; esac
    TIME=01:00:00; [[ $CFG == A* ]] && TIME=04:00:00          # A: 1763 supernodes
    J=$(sub --dependency=afterok:$D --job-name="rep_$CFG" --time=$TIME \
            scripts/run_sample_conditional.sh --repeats 5 "$CFG")
    echo "  $CFG -> $J"
done

cat <<'EOF'

All submitted. Watch with:
    squeue -u $USER -o "%.10i %.20j %.9T %.10M %.12L %R"
In each training log, check:  "Size of the input features 7 2 12"  (12 = no condition)
EOF
