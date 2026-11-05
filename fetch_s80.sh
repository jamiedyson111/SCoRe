#!/bin/bash
# Copy every s80 sample from BlueBEAR into results/samples/, named for the scorers.
# LOCAL, after D_intra/scripts/submit_s80.sh's jobs have finished (VPN on):
#
#     bash fetch_s80.sh             # everything
#     bash fetch_s80.sh uncond      # only configurations whose name contains "uncond"
#                                   # (plain DiGress); skips D_inter and the full graph
#     bash fetch_s80.sh deg         # only the internal-degree D_inter
#                                   # (D_inter/scripts/submit_inter_deg.sh)
#     bash fetch_s80.sh final       # only the runs submit_final.sh adds (fine-tuned
#                                   # D_intra C, DiGress on Oldenburg, D_inter no-r, Cft)
#     bash fetch_s80.sh vermont     # only submit_vermont.sh's runs, into the *_verm folders
#
#   D_intra  outputs/*/*-transfer_<CFG>_s80_sample_x5/conditional_samples_s<s>.txt
#              -> results/samples/D_intra/s80/<CFG>_s<s>.txt
#   D_inter  outputs/*/*-pairs_<CFG>_s80_sample_x5/pair_samples_s<s>.pt
#              -> results/samples/D_inter/s80/<CFG>_s<s>.pt
#   full     D_intra/private_inference/out/{clusters,blocks}_oldenburg_k7_*_s80*_s<s>.pt
#              -> results/samples/assembled/
#   costs    every job's cost_*.json (runtime, peak GPU memory; src/costlog.py)
#              -> results/costs/raw/, read by D_inter/private_inference/cost_summary.py
#
# The newest run directory of each configuration is taken. A missing one is
# reported and skipped, so this can be rerun as jobs finish.
set -uo pipefail

HOST=dysonjz@bluebear.bham.ac.uk
P=/rds/projects/c/chenhp-graphsyn
ROOT=$(cd "$(dirname "$0")" && pwd)
S=$ROOT/results/samples
C=$ROOT/results/costs/raw
mkdir -p "$S/D_intra/s80" "$S/D_inter/s80" "$S/assembled" "$C"
ONLY="${1:-}"                # optional filter on configuration names
FINAL=" Cft_v3 Cft_nohop_centroid O_uncond pair_lux_s80_deg_nor pair_C_s80_deg_nor Cft_deg B_deg_nor Cft_deg_nor "
want() {
    [ -z "$ONLY" ] && return 0
    [ "$ONLY" = final ] && { [[ "$FINAL" == *" $1 "* ]]; return; }
    [[ "$1" == *"$ONLY"* ]]
}
[ -n "$ONLY" ] && echo "(only configurations matching '$ONLY')"

newest() {                   # newest DIR GLOB -> the newest matching run directory
    ssh "$HOST" "ls -td $P/$1/outputs/*/*-$2/ 2>/dev/null | head -1"
}
cost() {                     # cost REMOTE_FILE LOCAL_NAME
    rsync -az "$HOST:$1" "$C/$2" 2>/dev/null || echo "  (no cost file $(basename "$1"))"
}

if [ "$ONLY" = vermont ]; then
    # Vermont's samples go to their own folders under Oldenburg's names, so the scorers
    # read them with --samples-dir ..._verm (and the lux2verm pairs files).
    mkdir -p "$S/D_intra/s80_verm" "$S/D_inter/s80_verm"
    echo "== training costs (Vermont)"
    for CFG in Cft_v3 Cft_nohop_centroid O_uncond; do
        D=$(newest D_intra "transfer_${CFG}_verm_s80")
        [ -n "$D" ] && cost "${D}cost_train.json" "train_D_intra_${CFG}_verm.json" || echo "  D_intra $CFG: not trained yet"
    done
    for NAME in pair_C_verm_s80_deg pair_C_verm_s80_deg_nor; do
        D=$(newest D_inter "$NAME")
        [ -n "$D" ] && cost "${D}cost_train.json" "train_D_inter_$NAME.json" || echo "  D_inter $NAME: not trained yet"
    done
    echo "== D_intra (Vermont)"
    for CFG in B_nohop_centroid B_v3 Cft_nohop_centroid Cft_v3 O_uncond; do
        D=$(newest D_intra "transfer_${CFG}_verm_s80_sample_x5")
        [ -n "$D" ] || { echo "  $CFG: not sampled yet"; continue; }
        for s in 0 1 2 3 4; do
            rsync -az "$HOST:${D}conditional_samples_s$s.txt" "$S/D_intra/s80_verm/${CFG}_s$s.txt" \
                || echo "  $CFG: seed $s missing in $D"
        done
        cost "${D}cost_sample.json" "sample_D_intra_${CFG}_verm.json"
        echo "  $CFG <- $D"
    done
    echo "== D_inter (Vermont)"
    for CFG in B_deg B_deg_nor Cft_deg Cft_deg_nor; do
        RUN="pairs_${CFG%%_*}_verm_s80_deg"; [[ $CFG == *_nor ]] && RUN="${RUN}_nor"
        D=$(newest D_inter "${RUN}_sample_x5")
        [ -n "$D" ] || { echo "  $CFG: not sampled yet"; continue; }
        for s in 0 1 2 3 4; do
            rsync -az "$HOST:${D}pair_samples_s$s.pt" "$S/D_inter/s80_verm/${CFG}_s$s.pt" \
                || echo "  $CFG: seed $s missing in $D"
        done
        cost "${D}cost_sample.json" "sample_D_inter_${CFG}_verm.json"
        echo "  $CFG <- $D"
    done
    echo "== full graph (Vermont)"
    O=$P/D_intra/private_inference/out
    rsync -az --include='clusters_vermont_k7_*_s80_s*.pt' --include='blocks_vermont_k7_*_s80-*_s*.pt' \
          --exclude='*' "$HOST:$O/" "$S/assembled/"
    ls "$S/assembled" | grep vermont | sed 's/^/  /'
    for CFG in B_v3 Cft_v3; do
        cost "$O/clusters_vermont_k7_${CFG}_s80_cost.json" "full_intra_${CFG}_s80_verm.json"
    done
    cost "$O/blocks_vermont_k7_B_v3_s80-B_deg_cost.json" "full_inter_B_v3_s80-B_deg_verm.json"
    cost "$O/blocks_vermont_k7_Cft_v3_s80-C_deg_cost.json" "full_inter_Cft_v3_s80-C_deg_verm.json"
    echo; echo "Then: bash make_vermont_tables.sh"
    exit 0
fi

echo "== training costs"
for CFG in A_v3 A_nohop_centroid C_v3 C_nohop_centroid A_uncond C_uncond \
           Cft_v3 Cft_nohop_centroid O_uncond; do
    want "$CFG" || continue
    D=$(newest D_intra "transfer_${CFG}_s80")
    [ -n "$D" ] && cost "${D}cost_train.json" "train_D_intra_$CFG.json" || echo "  D_intra $CFG: not trained yet"
done
for NAME in pair_lux_s80 pair_C_s80 pair_lux_s80_deg pair_C_s80_deg \
            pair_lux_s80_deg_nor pair_C_s80_deg_nor; do
    want "$NAME" || continue
    D=$(newest D_inter "$NAME")
    [ -n "$D" ] && cost "${D}cost_train.json" "train_D_inter_$NAME.json" || echo "  D_inter $NAME: not trained yet"
done

echo "== D_intra"
for CFG in B_v3 B_nohop_centroid C_v3 C_nohop_centroid A_v3 A_nohop_centroid \
           B_uncond C_uncond A_uncond \
           Cft_v3 Cft_nohop_centroid O_uncond; do          # *_uncond: plain DiGress
    want "$CFG" || continue
    D=$(newest D_intra "transfer_${CFG}_s80_sample_x5")
    [ -n "$D" ] || { echo "  $CFG: not sampled yet"; continue; }
    for s in 0 1 2 3 4; do
        rsync -az "$HOST:${D}conditional_samples_s$s.txt" "$S/D_intra/s80/${CFG}_s$s.txt" \
            || echo "  $CFG: seed $s missing in $D"
    done
    cost "${D}cost_sample.json" "sample_D_intra_$CFG.json"
    echo "  $CFG <- $D"
done

echo "== D_inter"
# local name -> run name. *_deg: the internal-degree D_inter; *_realX: with true X_i;
# *_nor: the no-r ablation; Cft*: C on D_intra's fine-tuned C.
for CFG in B C A B_deg C_deg A_deg B_deg_realX C_deg_realX Cft_deg B_deg_nor Cft_deg_nor; do
    want "$CFG" || continue
    RUN="pairs_${CFG%%_*}_s80"; [[ $CFG == *_deg* ]] && RUN="${RUN}_deg"
    [[ $CFG == *_realX ]] && RUN="${RUN}_realX"
    [[ $CFG == *_nor ]] && RUN="${RUN}_nor"
    D=$(newest D_inter "${RUN}_sample_x5")
    [ -n "$D" ] || { echo "  $CFG: not sampled yet"; continue; }
    for s in 0 1 2 3 4; do
        rsync -az "$HOST:${D}pair_samples_s$s.pt" "$S/D_inter/s80/${CFG}_s$s.pt" \
            || echo "  $CFG: seed $s missing in $D"
    done
    cost "${D}cost_sample.json" "sample_D_inter_$CFG.json"
    echo "  $CFG <- $D"
done

echo "== full graph"
if [ -z "$ONLY" ]; then
rsync -az --include='clusters_oldenburg_k7_*_s80_s*.pt' --include='blocks_oldenburg_k7_*_s80-*_s*.pt' \
      --exclude='*' "$HOST:$P/D_intra/private_inference/out/" "$S/assembled/"
ls "$S/assembled" | grep _s80 | sed 's/^/  /'
O=$P/D_intra/private_inference/out
for CFG in B C; do
    cost "$O/clusters_oldenburg_k7_${CFG}_v3_s80_cost.json" "full_intra_${CFG}_v3_s80.json"
    cost "$O/blocks_oldenburg_k7_${CFG}_v3_s80-${CFG}_cost.json" "full_inter_${CFG}_v3_s80-${CFG}.json"
done
elif [ "$ONLY" = final ]; then
rsync -az --include='clusters_oldenburg_k7_Cft_v3_s80_s*.pt' \
      --include='blocks_oldenburg_k7_Cft_v3_s80-*_deg_s*.pt' --exclude='*' \
      "$HOST:$P/D_intra/private_inference/out/" "$S/assembled/"
ls "$S/assembled" | grep Cft_v3 | sed 's/^/  /'
O=$P/D_intra/private_inference/out
cost "$O/clusters_oldenburg_k7_Cft_v3_s80_cost.json" "full_intra_Cft_v3_s80.json"
cost "$O/blocks_oldenburg_k7_Cft_v3_s80-C_deg_cost.json" "full_inter_Cft_v3_s80-C_deg.json"
elif [ "$ONLY" = deg ]; then
rsync -az --include='blocks_oldenburg_k7_*_s80-*_deg_s*.pt' --exclude='*' \
      "$HOST:$P/D_intra/private_inference/out/" "$S/assembled/"
ls "$S/assembled" | grep _deg_ | sed 's/^/  /'
O=$P/D_intra/private_inference/out
for CFG in B C; do
    cost "$O/blocks_oldenburg_k7_${CFG}_v3_s80-${CFG}_deg_cost.json" "full_inter_${CFG}_v3_s80-${CFG}_deg.json"
done
else
    echo "  skipped (filter '$ONLY')"
fi

echo "== peak device GPU memory of every job (gpu_monitor.sh)"
mkdir -p "$ROOT/results/costs/gpumem"
rsync -az "$HOST:$P/gpumem/" "$ROOT/results/costs/gpumem/" 2>/dev/null || echo "  none yet"

cat <<'EOF'

Then score (kmis env):
  cd D_inter/evaluation && python transfer_summary.py --configs B_deg C_deg B_deg_realX C_deg_realX BS1 BS2
  cd D_inter/evaluation && python transfer_summary.py --configs A_deg A_BS1 A_BS2
  cd D_inter/private_inference && python syn_repeats_summary.py --configs B_v3_s80-B_deg C_v3_s80-C_deg \
        --out-dir ../../results/D_inter/tables/syn_s80_deg
  cd D_intra/evaluation && python repeat_summary.py --configs B_uncond B_nohop_centroid B_v3 C_uncond C_nohop_centroid C_v3 BS1 BS2
  cd D_intra/evaluation && python repeat_summary.py --configs A_uncond A_nohop_centroid A_v3 A_BS1 A_BS2
  cd D_inter/evaluation && python transfer_summary.py --configs B C BS1 BS2
  cd D_inter/evaluation && python transfer_summary.py --configs A A_BS1 A_BS2
  cd D_inter/private_inference && python syn_repeats_summary.py
  cd D_inter/private_inference && python cost_summary.py             # runtime + GPU memory
  cd D_intra/extraction && python coarsening_stats.py                # coarsening + state sizes
EOF
