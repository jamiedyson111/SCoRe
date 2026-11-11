#!/bin/bash
# Copy the r-embedded D_inter samples (submit_sampling_nov.sh) from BlueBEAR, LOCAL:
#     bash fetch_remb.sh
#   test pairs  outputs/*/*-pairs_{B,Cft}[_verm]_s80_deg_remb_sample_x5/pair_samples_s<s>.pt
#               -> results/samples/D_inter/s80[_verm]/{B,Cft}_deg_remb_s<s>.pt
#   full graph  D_intra/private_inference/out/blocks_<graph>_k7_*-*_deg_remb_s<s>.pt
#               -> results/samples/assembled/
set -uo pipefail
HOST=dysonjz@bluebear.bham.ac.uk
P=/rds/projects/c/chenhp-graphsyn
ROOT=$(cd "$(dirname "$0")" && pwd)
S=$ROOT/results/samples
C=$ROOT/results/costs/raw
mkdir -p "$S/D_inter/s80" "$S/D_inter/s80_verm" "$S/assembled" "$C"
for g in olde verm; do
    sfx=$([ $g = verm ] && echo _verm || echo ""); dir=$S/D_inter/s80$sfx
    for c in B Cft; do
        D=$(ssh "$HOST" "ls -td $P/D_inter/outputs/*/*-pairs_${c}${sfx}_s80_deg_remb_sample_x5/ 2>/dev/null | head -1")
        [ -n "$D" ] || { echo "  $c $g: not sampled"; continue; }
        for s in 0 1 2 3 4; do rsync -az "$HOST:${D}pair_samples_s$s.pt" "$dir/${c}_deg_remb_s$s.pt" || echo "  $c $g seed $s missing"; done
        rsync -az "$HOST:${D}cost_sample.json" "$C/sample_D_inter_${c}_deg_remb${sfx}.json" 2>/dev/null
        echo "  test pairs $c $g <- $D"
    done
done
O=$P/D_intra/private_inference/out
rsync -az --include='blocks_*_k7_*_s80-*_deg_remb_s*.pt' --include='blocks_*_k7_*_deg_remb_cost.json' --exclude='*' "$HOST:$O/" "$S/assembled/"
ls "$S/assembled" | grep _remb | sed 's/^/  /'
