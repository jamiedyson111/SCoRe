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
# (seeds 0..N-1). C needs its pair_C_olde run to exist first. B and C ask for 2h
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
    [ "$CFG" != A ] && TIME=(--time=02:00:00)
    JID=$(sbatch --parsable --job-name="pairs_$CFG" ${TIME[@]+"${TIME[@]}"} \
                 scripts/run_sample_pairs_transfer.sh --repeats "$N" "$CFG")
    echo "  $CFG -> job $JID"
done

cat <<EOF

  watch     squeue -u \$USER -o "%.10i %.14j %.9T %.10M %.12L %R"
  check     grep "repeat\|exactly" \$(ls -t logs/pair_transfer_*.out | head -1)

Each job writes pair_samples_s0.pt .. _s$((N - 1)).pt into
outputs/<date>/<time>-pairs_<CFG>_sample_x$N/. Copy them locally to
../results/samples/D_inter/transfer/<CFG>_s<seed>.pt, then:

  cd evaluation && python transfer_summary.py                  # B and C
  cd evaluation && python transfer_summary.py --configs A      # A
EOF
