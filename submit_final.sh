#!/bin/bash
# The final road-network runs, as one chain of SLURM jobs. A SUBMITTER, not a job:
# run it once on the BlueBEAR login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/submit_final.sh
#
# Only what the final tables need that does not exist yet (everything else -- A_v3,
# B_v3, B_nohop_centroid, BS1 / BS2, D_inter A / B / C and their samples -- is reused):
#
#   D_intra
#     Cft_v3              C as a FINE-TUNE of A_v3 on the 254 Oldenburg train supernodes
#     Cft_nohop_centroid  the same for the no-h model (A_nohop_centroid)
#     O_uncond            plain DiGress trained on Oldenburg alone (254 train / 63 test)
#     each: train -> 5 seeds on the 63 Oldenburg test supernodes;
#     Cft_v3 also generates every Oldenburg supernode (5 seeds) for the full graph
#   D_inter
#     pair_lux_s80_deg_nor -> pair_C_s80_deg_nor   the no-r ablation, c_ij = [n_i, n_j, w_ij]
#     test pairs (5 seeds, X^ from D_intra's generated supernodes):
#       Cft      D_inter's C on D_intra's fine-tuned C (Cft_v3)
#       B_nor    no-r B, on D_intra's B_v3
#       Cft_nor  no-r C, on D_intra's Cft_v3
#     full graph: D_inter C on every Oldenburg pair around D_intra Cft_v3 (5 seeds)
#
# 13 jobs, each waiting (afterok) for what it needs. Needs uploaded first: D_intra and
# D_inter code, D_intra/data/cluster_lux/raw/pairs_{Cft,O}_*.pt and
# D_inter/data/pair_lux/raw/*_deg_nor.pt. When the queue is empty, fetch with
# `bash fetch_s80.sh final` on the LOCAL machine.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
INTRA=$P/D_intra
INTER=$P/D_inter
for f in $INTRA/data/cluster_lux/raw/pairs_Cft_lux2olde_k7_s80{,_nohop}.pt \
         $INTRA/data/cluster_lux/raw/pairs_O_lux2olde_k7_s80_uncond.pt \
         $INTER/data/pair_lux/raw/pairs_inter_{luxembourg_osm,B_lux2olde,C_lux2olde}_k7_s80_deg_nor.pt \
         $INTRA/private_inference/out/clusters_oldenburg_k7_B_v3_s80_s{0,1,2,3,4}.pt; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
grep -q finetune_from $INTRA/src/main.py || { echo "ERROR: $INTRA/src is the old code -- upload D_intra" >&2; exit 1; }
grep -q _nor $INTER/scripts/run_sample_pairs_transfer.sh || { echo "ERROR: upload D_inter's scripts" >&2; exit 1; }

intra() { (cd "$INTRA" && sbatch --parsable "$@"); }
inter() { (cd "$INTER" && sbatch --parsable "$@"); }
after() { local IFS=:; echo "--dependency=afterok:$*"; }

echo "== D_intra: train"
TCv3=$(intra --job-name=Cft_v3 --time=02:00:00 scripts/run_transfer.sh Cft_v3)
TCnh=$(intra --job-name=Cft_noh --time=02:00:00 scripts/run_transfer.sh Cft_nohop_centroid)
TO=$(intra --job-name=O_uncond --time=08:00:00 scripts/run_transfer.sh O_uncond)
echo "  Cft_v3 -> $TCv3   Cft_nohop_centroid -> $TCnh   O_uncond -> $TO"

echo "== D_intra: 5 seeds on the 63 Oldenburg test supernodes"
SCv3=$(intra "$(after "$TCv3")" --job-name=rep_Cft_v3 --time=01:00:00 \
             scripts/run_sample_conditional.sh --repeats 5 Cft_v3)
SCnh=$(intra "$(after "$TCnh")" --job-name=rep_Cft_noh --time=01:00:00 \
             scripts/run_sample_conditional.sh --repeats 5 Cft_nohop_centroid)
SO=$(intra "$(after "$TO")" --job-name=rep_O_uncond --time=01:00:00 \
           scripts/run_sample_conditional.sh --repeats 5 O_uncond)
echo "  Cft_v3 -> $SCv3   Cft_nohop_centroid -> $SCnh   O_uncond -> $SO"

echo "== D_intra: every Oldenburg supernode, Cft_v3 (full graph, stage 1)"
FIC=$(intra "$(after "$SCv3")" --job-name=full_Cft_v3 \
            private_inference/run_private_clusters_repeats.sh --repeats 5 Cft_v3)
echo "  Cft_v3 -> $FIC"

echo "== D_inter: the no-r models"
TAn=$(inter --job-name=pair_A_nor scripts/run_pair.sh pair_lux_s80_deg_nor pair_lux_s80_deg_nor)
TCn=$(inter "$(after "$TAn")" --job-name=pair_C_nor --time=03:00:00 \
            scripts/run_pair.sh pair_C_lux2olde_k7_s80_deg_nor pair_C_s80_deg_nor pair_lux_s80_deg_nor)
echo "  A / B no-r -> $TAn   C no-r (fine-tunes it) -> $TCn"

echo "== D_inter: test pairs (5 seeds each)"
QC=$(inter "$(after "$FIC")" --job-name=pairs_Cft --time=02:00:00 \
           scripts/run_sample_pairs_transfer.sh --repeats 5 Cft)
QBn=$(inter "$(after "$TAn")" --job-name=pairs_B_nor --time=02:00:00 \
            scripts/run_sample_pairs_transfer.sh --repeats 5 B_nor)
QCn=$(inter "$(after "$TCn" "$FIC")" --job-name=pairs_Cft_nor --time=02:00:00 \
            scripts/run_sample_pairs_transfer.sh --repeats 5 Cft_nor)
echo "  Cft -> $QC   B_nor -> $QBn   Cft_nor -> $QCn"

echo "== D_inter: full graph around D_intra Cft_v3 (stage 2)"
FC=$(inter "$(after "$QC")" --job-name=full_C_Cft \
           private_inference/run_private_blocks_repeats.sh --repeats 5 C Cft_v3)
echo "  C on Cft_v3 -> $FC"

cat <<'EOF'

All 13 submitted. Watch with:
    squeue -u $USER -o "%.10i %.14j %.9T %.10M %.12L %R"
Check early in the logs:
  D_intra Cft_*   "fine-tuning from .../transfer_A_v3_s80/..."  and edge marginal [0.877, 0.123]
  D_intra O       "Size of the input features 7 2 12"  (no condition)
  D_inter A no-r  "Size of the input features 9 2 ..." with y 3 wider than none (n_i, n_j, w_ij)
EOF
