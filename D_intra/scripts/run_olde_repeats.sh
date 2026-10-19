#!/bin/bash
# Re-sample every Oldenburg configuration 5 times, for mean / std tables.
#
# This is a SUBMITTER, not a job. Run it on the login node with bash:
#
#     cd /rds/projects/c/chenhp-graphsyn/D_intra
#     bash scripts/run_olde_repeats.sh                     # B, C: no hops+centroid and v3
#     bash scripts/run_olde_repeats.sh 10                  # or any other count
#     bash scripts/run_olde_repeats.sh 5 B_nohop_centroid C_nohop_centroid   # named only
#
# The comparison is no hops against v3, BOTH with the centroid node type, so the
# two differ in h_i alone and the hop error E_h can be measured from node 0 in
# each. Other configurations (plain *_nohop, raw-hop B, C) can still be named.
#
# NO TRAINING. Every job loads an existing checkpoint once and samples the
# 160-supernode Oldenburg test split N times with seeds 0..N-1, so the spread it
# measures is SAMPLING noise -- not what retraining with another seed would do.
#   B_nohop_centroid, B_v3   the Luxembourg-only checkpoints (transfer_B*)
#   C_nohop_centroid, C_v3   the Oldenburg C checkpoints (transfer_C_olde*)
# The *_nohop_centroid checkpoints must be trained first (run_transfer.sh).
# Each single sample took 3-4 min, so a 5-repeat job is ~20 min. The datasets
# were all processed by the first Oldenburg runs, so the jobs can run together;
# the 2-GPU-per-user limit queues the rest.
#
# A is not in the default set: it is tested on Luxembourg (882 supernodes), not
# Oldenburg, so its numbers do not depend on the private graph. Name it to run
# it -- a single A sample took ~7 min, so 5 repeats are ~40 min and fit the same
# 1h limit. repeat_summary.py writes A to its own sheet.
set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-5}"
shift || true
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=(B_nohop_centroid B_v3 C_nohop_centroid C_v3)
# run_sample_conditional.sh asks for 2h, sized for the 882-supernode Luxembourg
# split. Five Oldenburg repeats take ~20 min, and a shorter request backfills
# into gaps the 2h one does not fit. --time on the command line overrides the
# script's #SBATCH line. TIME=hh:mm:ss to change it; raise it for N well above 5.
TIME="${TIME:-01:00:00}"
echo "sampling each configuration $N times (seeds 0..$((N - 1))), time limit $TIME:"
for CFG in "${CFGS[@]}"; do
    # --repeats as an argument: BlueBEAR does not pass env vars like REPEATS=N
    # through to the job (see run_sample_conditional.sh).
    JID=$(sbatch --parsable --job-name="rep_$CFG" --time="$TIME" \
                 scripts/run_sample_conditional.sh --repeats "$N" "$CFG")
    echo "  $CFG -> job $JID"
done

cat <<EOF

  watch     squeue -u \$USER -o "%.10i %.14j %.9T %.10M %.12L %R"

Each job writes conditional_samples_s0.txt .. _s$((N - 1)).txt into
outputs/<date>/<time>-transfer_<CFG>_olde_sample_x$N/ (e.g. transfer_C_v3_olde_sample_x$N).
Copy them to ../results/samples/D_intra/oldenburg/repeats/<CFG>_s<seed>.txt locally, then:

  cd evaluation && python repeat_summary.py                       # B and C
  cd evaluation && python repeat_summary.py --configs A_nohop_centroid A_v3  # A
EOF
