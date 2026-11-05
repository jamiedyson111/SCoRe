#!/usr/bin/env bash
# The final tables with Vermont in place of Oldenburg, and their Excel workbook, LOCAL,
# after `bash fetch_s80.sh vermont` (submit_vermont.sh's runs):
#
#     bash make_vermont_tables.sh
#
#   D_intra  Luxembourg: Real | A v3                       (the Oldenburg study's, unchanged)
#            Vermont:    Real | B (no h) | B v3 | C (no h) | C v3 | BS1 | BS2 | DiGress (VT 80/20)
#   D_inter  Luxembourg: Real | A                          (unchanged)
#            Vermont:    Real | B (no r) | B | C (no r) | C | BS1 | BS2
#   Full graph (Vermont, whole graph): B | C | BS1 | BS2, relative errors
#
# Samples from results/samples/*/s80_verm, TSVs in results/*/tables/vermont and
# syn_vermont, the workbook at results/final_tables_vermont.xlsx.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
R=$ROOT/results
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kmis

echo "== D_intra"
cd "$ROOT/D_intra/evaluation"
python repeat_summary.py --tag lux2verm_k7_s80 --samples-dir "$R/samples/D_intra/s80_verm" \
       --configs B_nohop_centroid B_v3 Cft_nohop_centroid Cft_v3 BS1 BS2 O_uncond \
       --out-dir "$R/D_intra/tables/vermont"

echo "== D_inter"
cd "$ROOT/D_inter/evaluation"
python transfer_summary.py --graph vermont --samples-dir "$R/samples/D_inter/s80_verm" \
       --configs B_deg_nor B_deg Cft_deg_nor Cft_deg BS1 BS2 --tag _vermont \
       --out-dir "$R/D_inter/tables/vermont"

echo "== full graph"
cd "$ROOT/D_inter/private_inference"
python syn_repeats_summary.py --release "$R/private_inference/release_vermont_k7.pt" \
       --split-file "$ROOT/D_intra/data/splits/vermont_k7_s80.json" \
       --configs B_v3_s80-B_deg Cft_v3_s80-C_deg --out-dir "$R/D_inter/tables/syn_vermont"
python -I syn_re_summary.py "$R/D_inter/tables/syn_vermont/syn_per_seed.tsv"

echo "== runtime and memory"
python cost_summary.py --final --graph vermont

echo "== workbook and paste files"
conda activate base                       # openpyxl
python -I "$ROOT/make_final_excel.py" "$R" "$R/final_tables_vermont.xlsx" vermont
