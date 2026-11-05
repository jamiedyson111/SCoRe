#!/bin/bash
# The internal-degree D_inter (the paper's Section B), retrained and evaluated, as
# one chain of SLURM jobs.
#
# A SUBMITTER, not a job. Run it once on the login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/D_inter/scripts/submit_inter_deg.sh
#
#   X_i      normalised internal degree d_in(v) / (n_i - 1) instead of L_i
#   c_ij     [n_i, n_j, w_ij, r_{i->j}, r_{j->i}]
#   A / B    one model, trained on Luxembourg from scratch (180 epochs)
#   C        A's model FINE-TUNED on the Oldenburg training pairs (true X; 100
#            epochs at lr 5e-5, scripts/run_pair.sh)
#   inference  B and C read X^ computed from D_intra's GENERATED internal blocks
#            (the existing s80 whole-graph runs, clusters_oldenburg_k7_{B,C}_v3_s80_s*.pt);
#            A, and the B / C "realX" ablation, read the true X
#
# Jobs, each waiting (afterok) for what it needs:
#   train      pair_lux_s80_deg (A, B)  ->  pair_C_s80_deg (C, fine-tuned from it)
#   test pairs 5 seeds of B, C (187 Oldenburg pairs, generated X^), B_realX, C_realX,
#              and A (3998 Luxembourg pairs)
#   full graph 5 seeds of D_inter on all 552 Oldenburg pairs around D_intra's B_v3 / C_v3
#
# 9 jobs. D_intra is NOT retrained: its s80 models and samples are reused.
# Needs uploaded first: D_inter (code, configs, data/pair_lux/raw/*_s80_deg.pt) and
# the new release (with r_ij / r_ji) at D_intra/private_inference/out/.
# When the queue is empty, fetch with fetch_s80.sh deg on the LOCAL machine.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
INTER=$P/D_inter
OUT=$P/D_intra/private_inference/out
for f in $INTER/data/pair_lux/raw/pairs_inter_{luxembourg_osm,B_lux2olde,C_lux2olde}_k7_s80_deg.pt \
         $OUT/clusters_oldenburg_k7_{B,C}_v3_s80_s{0,1,2,3,4}.pt; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
grep -q finetune_from $INTER/src/main.py || { echo "ERROR: $INTER/src is the old code -- upload D_inter" >&2; exit 1; }

sub() { (cd "$INTER" && sbatch --parsable "$@"); }
after() { local IFS=:; echo "--dependency=afterok:$*"; }

echo "== training"
TA=$(sub --job-name=pair_A_deg scripts/run_pair.sh pair_lux_s80_deg pair_lux_s80_deg)
echo "  A / B  pair_lux_s80_deg -> $TA"
TC=$(sub "$(after "$TA")" --job-name=pair_C_deg --time=03:00:00 \
         scripts/run_pair.sh pair_C_lux2olde_k7_s80_deg pair_C_s80_deg pair_lux_s80_deg)
echo "  C      pair_C_s80_deg (fine-tunes A) -> $TC"

echo "== test pairs (5 seeds each)"
# The realX job waits for its generated-X twin: both use one dataset, and two jobs
# building one PyG cache at once can corrupt it.
QB=$(sub "$(after "$TA")" --job-name=pairs_B_deg --time=02:00:00 \
         scripts/run_sample_pairs_transfer.sh --repeats 5 B)
QBr=$(sub "$(after "$QB")" --job-name=pairs_B_degR --time=02:00:00 \
          scripts/run_sample_pairs_transfer.sh --repeats 5 --real-x B)
QC=$(sub "$(after "$TC")" --job-name=pairs_C_deg --time=02:00:00 \
         scripts/run_sample_pairs_transfer.sh --repeats 5 C)
QCr=$(sub "$(after "$QC")" --job-name=pairs_C_degR --time=02:00:00 \
          scripts/run_sample_pairs_transfer.sh --repeats 5 --real-x C)
QA=$(sub "$(after "$TA")" --job-name=pairs_A_deg --time=12:00:00 \
         scripts/run_sample_pairs_transfer.sh --repeats 5 A)
echo "  B -> $QB   B_realX -> $QBr   C -> $QC   C_realX -> $QCr   A -> $QA"

echo "== full graph (D_inter on every Oldenburg pair, around D_intra's generated blocks)"
FB=$(sub "$(after "$QBr")" private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3)
FC=$(sub "$(after "$QCr")" private_inference/run_private_blocks_repeats.sh --repeats 5 C C_v3)
echo "  B -> $FB   C -> $FC"

cat <<'EOF'

All submitted. Watch with:
    squeue -u $USER -o "%.10i %.20j %.9T %.10M %.12L %R"
In the first training log check:  "Size of the input features 9 2 ..."  (9 = 2 sides
+ 6 extra features + 1 internal degree) and, for C, "fine-tuning from ...".
EOF
