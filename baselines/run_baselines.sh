#!/bin/bash
# BS1 (ER) and BS2 (SC-RG) on the D_intra and D_inter test sets, 5 runs each.
#
# Runs Huiping's four scripts in competitors/Baseline/ unchanged and renames their
# output to the names the scorers read. LOCAL, `kmis` env, about a minute:
#
#     bash baselines/run_baselines.sh                 # Oldenburg and Luxembourg
#     bash baselines/run_baselines.sh 5 vermont       # Vermont only, into s80_verm/
#
# Writes, for runs s0..s4 (her seeds 42..46):
#
#     results/samples/D_intra/s80/{BS1,BS2}_s<s>.txt      63 Oldenburg test supernodes
#     results/samples/D_intra/s80/{A_BS1,A_BS2}_s<s>.txt  1763 Luxembourg test supernodes
#     results/samples/D_inter/s80/{BS1,BS2}_s<s>.pt       187 Oldenburg test pairs
#     results/samples/D_inter/s80/{A_BS1,A_BS2}_s<s>.pt   3998 Luxembourg test pairs
#     results/samples/D_{intra,inter}/s80_verm/{BS1,BS2}_s<s>.*  1104 Vermont test
#                                                          supernodes, 2881 test pairs
#
# The full-graph BS1 / BS2 are built inside
# D_inter/private_inference/syn_repeats_summary.py from the same functions.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
N=${1:-5}
WHICH=${2:-}                 # "vermont": the Vermont row only
HP="$ROOT/competitors/Baseline"
INTRA_IN="$ROOT/D_intra/data/cluster_lux/raw"
INTER_IN="$ROOT/D_inter/data/pair_lux/raw"
INTRA_OUT="$ROOT/results/samples/D_intra/s80"
INTER_OUT="$ROOT/results/samples/D_inter/s80"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$INTRA_OUT" "$INTER_OUT"

# prefix   D_intra test file             D_inter test file
ROWS=(":pairs_B_lux2olde_k7_s80.pt:pairs_inter_B_lux2olde_k7_s80.pt"
      "A_:pairs_A_lux2olde_k7_s80.pt:pairs_inter_luxembourg_osm_k7_s80.pt")
if [ "$WHICH" = vermont ]; then
    ROWS=("VT_:pairs_B_lux2verm_k7_s80.pt:pairs_inter_B_lux2verm_k7_s80_deg.pt")
    INTRA_OUT="${INTRA_OUT}_verm"; INTER_OUT="${INTER_OUT}_verm"
    mkdir -p "$INTRA_OUT" "$INTER_OUT"
fi
for row in "${ROWS[@]}"; do
    IFS=: read -r P INTRA INTER <<< "$row"
    case "$P" in "") echo "== Oldenburg test set (B, C)" ;; A_) echo "== Luxembourg test set (A)" ;;
                 VT_) echo "== Vermont test set (B, C)" ;; esac
    "$PY" "$HP/ER_intra.py"    --input "$INTRA_IN/$INTRA" --output-dir "$TMP/er"   --num-samples "$N" --check | tail -1
    "$PY" "$HP/SC_RG_intra.py" --input "$INTRA_IN/$INTRA" --output-dir "$TMP/scrg" --num-samples "$N" --check | tail -1
    "$PY" "$HP/ER_inter.py"    --input "$INTER_IN/$INTER" --output-dir "$TMP/er_inter"   --num-samples "$N" --check | tail -1
    "$PY" "$HP/SC_RG_inter.py" --input "$INTER_IN/$INTER" --output-dir "$TMP/scrg_inter" --num-samples "$N" --check | tail -1
    [ "$P" = VT_ ] && P=""         # Vermont has its own folders, under Oldenburg's names
    for ((s = 0; s < N; s++)); do
        mv "$TMP/er/ER_samples_s$s.txt"            "$INTRA_OUT/${P}BS1_s$s.txt"
        mv "$TMP/scrg/scrg_samples_s$s.txt"        "$INTRA_OUT/${P}BS2_s$s.txt"
        mv "$TMP/er_inter/pair_samples_s$s.pt"     "$INTER_OUT/${P}BS1_s$s.pt"
        mv "$TMP/scrg_inter/pair_samples_s$s.pt"   "$INTER_OUT/${P}BS2_s$s.pt"
    done
done
echo "wrote ${N} runs of BS1 / BS2 to $INTRA_OUT and $INTER_OUT"
