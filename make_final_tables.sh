#!/usr/bin/env bash
# The final road-network tables and their Excel workbook, LOCAL, after
# `bash fetch_s80.sh final` (submit_final.sh's runs) has copied every sample:
#
#     bash make_final_tables.sh
#
#   D_intra  Luxembourg: Real | A v3
#            Oldenburg:  Real | B (no h) | B v3 | C (no h) | C v3 | BS1 | BS2 | DiGress (Olde 80/20)
#            C is A's model fine-tuned on Oldenburg (Cft_*); edge and density rows left out
#   D_inter  Luxembourg: Real | A
#            Oldenburg:  Real | B (no r) | B | C (no r) | C | BS1 | BS2
#            B / C read X^ from D_intra's generated supernodes; cross-edge rows left out
#   Full graph (Oldenburg, whole graph): B | C | BS1 | BS2, relative errors
#
# TSVs in results/*/tables/final/, the workbook at results/final_tables.xlsx.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$ROOT/results
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kmis

echo "== D_intra"
cd "$ROOT/D_intra/evaluation"
python repeat_summary.py --configs A_v3 --out-dir "$R/D_intra/tables/final"
python repeat_summary.py --configs B_nohop_centroid B_v3 Cft_nohop_centroid Cft_v3 BS1 BS2 O_uncond \
       --out-dir "$R/D_intra/tables/final"

echo "== D_inter"
cd "$ROOT/D_inter/evaluation"
python transfer_summary.py --configs A_deg --tag _final --out-dir "$R/D_inter/tables/final"
python transfer_summary.py --configs B_deg_nor B_deg Cft_deg_nor Cft_deg BS1 BS2 --tag _final \
       --out-dir "$R/D_inter/tables/final"

echo "== full graph"
cd "$ROOT/D_inter/private_inference"
python syn_repeats_summary.py --configs B_v3_s80-B_deg Cft_v3_s80-C_deg \
       --out-dir "$R/D_inter/tables/syn_final"
python -I syn_re_summary.py "$R/D_inter/tables/syn_final/syn_per_seed.tsv"

echo "== runtime and memory"
python cost_summary.py --final

echo "== workbook"
conda activate base                       # openpyxl
python -I "$ROOT/make_final_excel.py" "$R" "$R/final_tables.xlsx"
