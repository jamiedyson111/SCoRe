#!/bin/bash
# Vermont in place of Oldenburg: the final configurations, as one chain of SLURM
# jobs. A SUBMITTER, not a job: run it once on the BlueBEAR login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/submit_vermont.sh
#
# Luxembourg -> Vermont (us_roads VT, 97,975 nodes; k = 7: 6,075 supernodes, 5,522
# with n >= 3, 4,418 train / 1,104 test; 7,951 pairs, 5,069 train / 2,881 test).
# The Luxembourg models (D_intra A_v3, A_nohop_centroid; D_inter pair_lux_s80_deg,
# pair_lux_s80_deg_nor) are reused as A / B. New:
#
#   D_intra  train  Cft_v3, Cft_nohop_centroid (A fine-tuned on Vermont train, 100
#                   epochs, lr 5e-5), O_uncond (plain DiGress on Vermont, ~22k steps)
#            test   5 seeds on the 1,104 test supernodes: B_nohop_centroid, B_v3,
#                   Cft_nohop_centroid, Cft_v3, O_uncond
#            full   every Vermont supernode, 5 seeds: B_v3, Cft_v3
#   D_inter  train  pair_C_verm_s80_deg, pair_C_verm_s80_deg_nor (fine-tuned from A)
#            test   5 seeds on the 2,881 test pairs, X^ from D_intra's generated
#                   supernodes: B, B_nor (on B_v3), Cft, Cft_nor (on Cft_v3)
#            full   every Vermont pair around D_intra's B_v3 / Cft_v3, 5 seeds
#
# 18 jobs, each waiting (afterok) for what it needs; jobs sharing a dataset run one
# after the other, so no PyG cache is built twice at once. Sampling uses batches of
# 128 (+sample_batch, set by --graph vermont). Needs uploaded first: D_intra and
# D_inter code, the lux2verm datasets and D_intra/private_inference/out/
# release_vermont_k7.pt. When the queue is empty, fetch with
# `bash fetch_s80.sh vermont` on the LOCAL machine.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
INTRA=$P/D_intra
INTER=$P/D_inter
for f in $INTRA/data/cluster_lux/raw/pairs_{B,Cft}_lux2verm_k7_s80{,_nohop}.pt \
         $INTRA/data/cluster_lux/raw/pairs_O_lux2verm_k7_s80_uncond.pt \
         $INTER/data/pair_lux/raw/pairs_inter_{B,C}_lux2verm_k7_s80_deg{,_nor}.pt \
         $INTRA/private_inference/out/release_vermont_k7.pt; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
grep -q GTAG $INTRA/scripts/run_transfer.sh || { echo "ERROR: upload D_intra's scripts" >&2; exit 1; }
grep -q GTAG $INTER/scripts/run_sample_pairs_transfer.sh || { echo "ERROR: upload D_inter's scripts" >&2; exit 1; }

intra() { (cd "$INTRA" && sbatch --parsable "$@"); }
inter() { (cd "$INTER" && sbatch --parsable "$@"); }
after() { local IFS=:; echo "--dependency=afterok:$*"; }
V=(--graph vermont)

echo "== D_intra: train (Vermont)"
TCv3=$(intra --job-name=vCft_v3 --time=04:00:00 scripts/run_transfer.sh "${V[@]}" Cft_v3)
TCnh=$(intra --job-name=vCft_noh --time=04:00:00 scripts/run_transfer.sh "${V[@]}" Cft_nohop_centroid)
TO=$(intra --job-name=vO_uncond --time=08:00:00 scripts/run_transfer.sh "${V[@]}" O_uncond)
echo "  Cft_v3 -> $TCv3   Cft_nohop_centroid -> $TCnh   O_uncond -> $TO"

echo "== D_intra: 5 seeds on the 1,104 Vermont test supernodes"
SBnh=$(intra --job-name=vrep_B_noh --time=03:00:00 \
             scripts/run_sample_conditional.sh "${V[@]}" --repeats 5 B_nohop_centroid)
SBv3=$(intra --job-name=vrep_B_v3 --time=03:00:00 \
             scripts/run_sample_conditional.sh "${V[@]}" --repeats 5 B_v3)
SCv3=$(intra "$(after "$TCv3")" --job-name=vrep_Cft_v3 --time=03:00:00 \
             scripts/run_sample_conditional.sh "${V[@]}" --repeats 5 Cft_v3)
SCnh=$(intra "$(after "$TCnh")" --job-name=vrep_Cft_noh --time=03:00:00 \
             scripts/run_sample_conditional.sh "${V[@]}" --repeats 5 Cft_nohop_centroid)
SO=$(intra "$(after "$TO")" --job-name=vrep_O_uncond --time=03:00:00 \
           scripts/run_sample_conditional.sh "${V[@]}" --repeats 5 O_uncond)
echo "  B_nohop_centroid -> $SBnh   B_v3 -> $SBv3   Cft_v3 -> $SCv3   Cft_noh -> $SCnh   O -> $SO"

echo "== D_intra: every Vermont supernode (full graph, stage 1)"
FIB=$(intra "$(after "$SBv3")" --job-name=vfull_B_v3 --time=08:00:00 \
            private_inference/run_private_clusters_repeats.sh "${V[@]}" --repeats 5 B_v3)
FIC=$(intra "$(after "$SCv3")" --job-name=vfull_Cft_v3 --time=08:00:00 \
            private_inference/run_private_clusters_repeats.sh "${V[@]}" --repeats 5 Cft_v3)
echo "  B_v3 -> $FIB   Cft_v3 -> $FIC"

echo "== D_inter: fine-tune C on Vermont (with r, no r)"
TCd=$(inter --job-name=vpair_C --time=06:00:00 \
            scripts/run_pair.sh pair_C_lux2verm_k7_s80_deg pair_C_verm_s80_deg pair_lux_s80_deg)
TCn=$(inter --job-name=vpair_C_nor --time=06:00:00 \
            scripts/run_pair.sh pair_C_lux2verm_k7_s80_deg_nor pair_C_verm_s80_deg_nor pair_lux_s80_deg_nor)
echo "  C -> $TCd   C no-r -> $TCn"

echo "== D_inter: test pairs (5 seeds each)"
QB=$(inter "$(after "$FIB")" --job-name=vpairs_B --time=08:00:00 \
           scripts/run_sample_pairs_transfer.sh "${V[@]}" --repeats 5 B)
QBn=$(inter "$(after "$FIB")" --job-name=vpairs_B_nor --time=08:00:00 \
            scripts/run_sample_pairs_transfer.sh "${V[@]}" --repeats 5 B_nor)
QC=$(inter "$(after "$FIC" "$TCd")" --job-name=vpairs_Cft --time=08:00:00 \
           scripts/run_sample_pairs_transfer.sh "${V[@]}" --repeats 5 Cft)
QCn=$(inter "$(after "$FIC" "$TCn")" --job-name=vpairs_Cft_nor --time=08:00:00 \
            scripts/run_sample_pairs_transfer.sh "${V[@]}" --repeats 5 Cft_nor)
echo "  B -> $QB   B_nor -> $QBn   Cft -> $QC   Cft_nor -> $QCn"

echo "== D_inter: every Vermont pair (full graph, stage 2)"
FB=$(inter "$(after "$QB")" --job-name=vfull_B --time=16:00:00 \
           private_inference/run_private_blocks_repeats.sh "${V[@]}" --repeats 5 B B_v3)
FC=$(inter "$(after "$QC")" --job-name=vfull_C --time=16:00:00 \
           private_inference/run_private_blocks_repeats.sh "${V[@]}" --repeats 5 C Cft_v3)
echo "  B on B_v3 -> $FB   C on Cft_v3 -> $FC"

cat <<'EOF'

All 18 submitted. Watch with:
    squeue -u $USER -o "%.10i %.14j %.9T %.10M %.12L %R"
Check early in the logs:
  D_intra vCft_*   "fine-tuning from .../transfer_A_v3_s80/..." and edge marginal [0.877, 0.123]
  D_intra vO       "Size of the input features 7 2 12", marginal [0.9113, 0.0887]
  D_inter vpair_C  "fine-tuning from ...pair_lux_s80_deg..."
EOF
