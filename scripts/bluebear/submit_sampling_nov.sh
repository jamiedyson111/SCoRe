#!/bin/bash
# Sampling for the models trained 5-6 November 2026. A SUBMITTER, not a job: run it
# once on the BlueBEAR login node:
#
#     bash submit_sampling_nov.sh
#
# 1. Road D_inter with the r EMBEDDING (pair_*_remb), 5 seeds, Oldenburg and Vermont:
#      test pairs   B (pair_lux_s80_deg_remb) and C (pair_C[_verm]_s80_deg_remb), X^ from
#                   D_intra B_v3 / Cft_v3 as before   -> runs pairs_{B,Cft}[_verm]_s80_deg_remb_*
#      full graph   B on B_v3, C on Cft_v3            -> blocks_<graph>_k7_*-{B,C}_deg_remb_s*.pt
#    The existing scripts with --model-sfx _remb; the raw-r samples are untouched.
# 2. DBLP -> Collab D_intra v3, 5 seeds on the 1,787 Collab test supernodes:
#      B  the DBLP A model (dintra_A_v3_dblp2coll) on cluster_B_dblp2coll_k3_s80
#      C  the fine-tune (dintra_Cft_v3_dblp2coll) on cluster_Cft_dblp2coll_k3_s80
#    Batches of 8 on an 80 GB card: supernodes reach 380 nodes.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
for f in $P/D_inter/scripts/run_sample_pairs_transfer.sh $P/D_inter/private_inference/run_private_blocks_repeats.sh; do
    grep -q model-sfx "$f" || { echo "ERROR: upload $f (no --model-sfx)" >&2; exit 1; }
done

GPU=(--account=chenhp-graphsyn --qos=bbgpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --parsable)

echo "== road D_inter, r embedded"
cd $P/D_inter
for g in oldenburg vermont; do
    case $g in oldenburg) T1=03:00:00; T2=03:00:00 ;; vermont) T1=10:00:00; T2=16:00:00 ;; esac
    for c in B Cft; do
        j=$(sbatch "${GPU[@]}" --time=$T1 --job-name=remb_pairs_${c}_${g:0:4} \
            --output=$P/D_inter/logs/remb_pairs_${c}_${g:0:4}_%j.out --error=$P/D_inter/logs/remb_pairs_${c}_${g:0:4}_%j.err \
            scripts/run_sample_pairs_transfer.sh --graph $g --repeats 5 --model-sfx _remb $c)
        echo "  test pairs $c $g -> $j"
    done
    for c in B C; do
        ic=$([ $c = B ] && echo B_v3 || echo Cft_v3)
        j=$(sbatch "${GPU[@]}" --time=$T2 --job-name=remb_full_${c}_${g:0:4} \
            --output=$P/D_inter/logs/remb_full_${c}_${g:0:4}_%j.out --error=$P/D_inter/logs/remb_full_${c}_${g:0:4}_%j.err \
            private_inference/run_private_blocks_repeats.sh --graph $g --repeats 5 --model-sfx _remb $c $ic)
        echo "  full graph $c ($ic) $g -> $j"
    done
done

echo "== DBLP -> Collab D_intra"
R=$P/D_intra
mkdir -p $R/logs/jobs
for c in B Cft; do
    model=$([ $c = B ] && echo dintra_A_v3_dblp2coll || echo dintra_Cft_v3_dblp2coll)
    rundir=$(ls -td $R/outputs/*/*-$model/ | head -1)
    ckpt=$(ls -t ${rundir%/}/checkpoints/$model/last*.ckpt | head -1)
    [ -f "$ckpt" ] || { echo "ERROR: no checkpoint for $model" >&2; exit 1; }
    cat > $R/logs/jobs/dc_sample_$c.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $R/scripts/setup.sh
export PYTHONPATH=$R:\${PYTHONPATH:-}
echo "checkpoint: $ckpt"
cd $R/src
python3 sample_conditional.py +sample_batch=8 dataset=cluster_${c}_dblp2coll_k3_s80 +experiment=cluster_lux \\
    dataset.mark_centroid=true model.hop_embed_dim=1 +repeats=5 \\
    general.name=dc_${c}_v3_dblp2coll_sample_x5 general.gpus=1 general.wandb=disabled \\
    "general.test_only='$ckpt'"
EOF
    j=$(cd $R && sbatch "${GPU[@]}" --gres=gpu:a100_80:1 --mem=64G --time=12:00:00 --job-name=dc_sample_$c \
        --output=$R/logs/dc_sample_${c}_%j.out --error=$R/logs/dc_sample_${c}_%j.err $R/logs/jobs/dc_sample_$c.sh)
    echo "  Collab test supernodes, $c ($model) -> $j"
done
