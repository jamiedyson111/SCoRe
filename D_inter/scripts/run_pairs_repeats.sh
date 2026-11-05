#!/bin/bash
# Sample D_inter's transfer configurations N times each, for mean / std tables.
#
# A SUBMITTER, not a job. Run it on the login node with bash:
#
#     cd /rds/projects/c/chenhp-graphsyn/D_inter
#     bash scripts/run_pairs_repeats.sh               # B and C, 5 seeds each
#     bash scripts/run_pairs_repeats.sh 5 B           # B only (C not trained yet)
#     bash scripts/run_pairs_repeats.sh 5 A           # A, on Luxembourg (own sheet)
#
# No training: each job loads one checkpoint and samples every test pair N times
# (seeds 0..N-1), with the internal-degree D_inter (pair_lux_s80_deg /
# pair_C_s80_deg; B and C read X^ from D_intra's generated blocks). The true-X
# ablation is run_sample_pairs_transfer.sh --real-x; scripts/submit_inter_deg.sh
# submits everything. B and C ask for 2h
# (~1h expected at 5 seeds); A keeps the job script's 4h.
set -euo pipefail

cd "$(dirname "$0")/.."

N="${1:-5}"
shift || true
CFGS=("$@")
[ ${#CFGS[@]} -eq 0 ] && CFGS=(B C)

echo "sampling each configuration $N times (seeds 0..$((N - 1))):"
for CFG in "${CFGS[@]}"; do
    TIME=()
    if [ "$CFG" = A ]; then TIME=(--time=12:00:00); else TIME=(--time=02:00:00); fi   # A: 3998 pairs, ~2 h a seed
    JID=$(sbatch --parsable --job-name="pairs_$CFG" ${TIME[@]+"${TIME[@]}"} \
                 scripts/run_sample_pairs_transfer.sh --repeats "$N" "$CFG")
    echo "  $CFG -> job $JID"
done

cat <<EOF

  watch     squeue -u \$USER -o "%.10i %.14j %.9T %.10M %.12L %R"
  check     grep "repeat\|exactly" \$(ls -t logs/pair_transfer_*.out | head -1)

Each job writes pair_samples_s0.pt .. _s$((N - 1)).pt into
outputs/<date>/<time>-pairs_<CFG>_s80_deg_sample_x$N/. Fetch them locally with
fetch_s80.sh deg, then:

  cd evaluation && python transfer_summary.py --configs B_deg C_deg BS1 BS2
  cd evaluation && python transfer_summary.py --configs A_deg A_BS1 A_BS2
EOF
