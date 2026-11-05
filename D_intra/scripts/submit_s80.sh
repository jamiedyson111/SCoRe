#!/bin/bash
# EVERYTHING on BlueBEAR for the 80/20 split, as one chain of SLURM jobs.
#
# A SUBMITTER, not a job. Run it once on the login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/D_intra/scripts/submit_s80.sh
#
# It submits, each job waiting (afterok) for the ones it needs:
#
#   train      D_intra  A_v3, A_nohop_centroid, C_v3, C_nohop_centroid   (200 epochs)
#              D_inter  pair_lux_s80 (A and B's model), pair_C_s80        (180 epochs)
#   evaluate   D_intra  5 seeds of B_*, C_* (63 Oldenburg test supernodes)
#                       and A_* (1763 Luxembourg)
#              D_inter  5 seeds of B, C (187 Oldenburg test pairs) and A (3998 Luxembourg)
#   assemble   D_intra on all 317 supernodes, then D_inter on all 552 pairs
#              around them, 5 seeds, for B (B_v3 + B) and C (C_v3 + C)
#
# 20 jobs. With the 2-GPU limit, expect about a day of wall clock, most of it
# training. If a job fails, the jobs that depend on it are never started:
# squeue shows them as (DependencyNeverSatisfied) -- scancel them, fix, and
# submit those steps by hand (see D_intra/README.md and D_inter/README.md).
#
# When the queue is empty, fetch everything with fetch_s80.sh on the LOCAL machine.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
INTRA=$P/D_intra
INTER=$P/D_inter
[ -f "$INTRA/private_inference/out/release_oldenburg_k7.pt" ] || {
    echo "ERROR: $INTRA/private_inference/out/release_oldenburg_k7.pt is missing" >&2
    exit 1; }

# sbatch from the project's own directory, so SLURM_SUBMIT_DIR is its root.
sub() {                      # sub DIR [sbatch options...] SCRIPT [args...]
    local dir=$1; shift
    (cd "$dir" && sbatch --parsable "$@")
}
after() { local IFS=:; echo "--dependency=afterok:$*"; }

echo "== training"
declare -A T
for CFG in A_v3 A_nohop_centroid C_v3 C_nohop_centroid; do
    T[$CFG]=$(sub "$INTRA" --time=12:00:00 scripts/run_transfer.sh "$CFG")
    echo "  D_intra $CFG -> ${T[$CFG]}"
done
PL=$(sub "$INTER" scripts/run_pair.sh pair_lux_s80 pair_lux_s80);         echo "  D_inter pair_lux_s80 -> $PL"
PC=$(sub "$INTER" scripts/run_pair.sh pair_C_lux2olde_k7_s80 pair_C_s80); echo "  D_inter pair_C_s80   -> $PC"

echo "== D_intra evaluation (5 seeds of every test supernode)"
declare -A R Q
for CFG in B_v3 B_nohop_centroid C_v3 C_nohop_centroid A_v3 A_nohop_centroid; do
    MODEL=${CFG/#B/A}                                    # B samples A's model
    TIME=01:00:00; [[ $CFG == A* ]] && TIME=04:00:00     # A: 1763 supernodes
    R[$CFG]=$(sub "$INTRA" "$(after "${T[$MODEL]}")" --job-name="rep_$CFG" --time=$TIME \
                  scripts/run_sample_conditional.sh --repeats 5 "$CFG")
    echo "  $CFG -> ${R[$CFG]}"
done

echo "== D_inter evaluation (5 seeds of every test pair, real insides)"
for CFG in B C A; do
    case $CFG in B) D=$PL; TIME=02:00:00 ;; C) D=$PC; TIME=02:00:00 ;; A) D=$PL; TIME=12:00:00 ;; esac
    Q[$CFG]=$(sub "$INTER" "$(after "$D")" --job-name="pairs_$CFG" --time=$TIME \
                  scripts/run_sample_pairs_transfer.sh --repeats 5 "$CFG")
    echo "  $CFG -> ${Q[$CFG]}"
done

echo "== full graph (every supernode and pair, 5 seeds)"
# Each also waits for the evaluation job on the same dataset: the s80 datasets
# are new, and two jobs building one PyG cache at once can corrupt it.
for CFG in B C; do
    MODEL=${CFG/#B/A}
    case $CFG in B) D=$PL ;; C) D=$PC ;; esac
    J1=$(sub "$INTRA" "$(after "${T[${MODEL}_v3]}" "${R[${CFG}_v3]}")" \
             private_inference/run_private_clusters_repeats.sh --repeats 5 "${CFG}_v3")
    J2=$(sub "$INTER" "$(after "$J1" "$D" "${Q[$CFG]}")" \
             private_inference/run_private_blocks_repeats.sh --repeats 5 "$CFG" "${CFG}_v3")
    echo "  $CFG: D_intra ${CFG}_v3 -> $J1, then D_inter $CFG -> $J2"
done

cat <<'EOF'

All submitted. Watch with:
    squeue -u $USER -o "%.10i %.20j %.9T %.10M %.12L %R"
Logs: D_intra/logs/, D_inter/logs/. When the queue is empty, run fetch_s80.sh locally.
EOF
