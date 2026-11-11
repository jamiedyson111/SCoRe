#!/bin/bash
# YouTube D_inter on ONE GPU (Huiping, 9 Nov 2026): reduced training sets, LiveJournal the
# same size as DBLP's (103,850 pairs), YouTube 80,000; selection stratified on the cross
# density w_ij/(n_i+n_j) and pair size, favouring unused supernodes (make_stratified_pairs.py).
# A SUBMITTER: run once on the BlueBEAR login node, AFTER submit_collab_lanes.sh.
#
#   yt_build    CPU  dataprep/make_stratified_pairs.py -> pairs_inter_{A,B,C}_live2yout_k4_s80_strat_deg.pt,
#                    the 5 x 1,000 YouTube test groups, the distribution checks
#   yt_inter_A  GPU  train A on LiveJournal, 2 epochs (DBLP's A settings otherwise)
#   yt_inter_C  GPU  fine-tune A on YouTube, 2 epochs, lr 5e-5 (as dc_inter_C)
# The GPU jobs follow yt_grp_Cft in the YouTube lane (one job at a time). Sampling the test
# groups is submitted once the epoch times are known.
set -euo pipefail
P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
LY=$P/coarsen_compare_expr/output/livejournal_youtube
mkdir -p $I/logs/jobs
cat > $I/logs/jobs/yt_build.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
cd $I/dataprep
python3 make_stratified_pairs.py --public livejournal --public-npz $LY/com_livejournal_csr_kmis_cb_v4_k4_b500_l10.npz \\
    --private youtube --private-npz $LY/com_youtube_csr_kmis_cb_v4_k4_b500_l10.npz --n-public 103850 --n-private 80000
mkdir -p $P/competitor_data/youtube
cp $I/data/pair_lux/raw/live2yout_k4_s80_strat_test_groups_d_inter.json $P/competitor_data/youtube/test_groups_d_inter.json
EOF

# train NAME DATASET EPOCHS LR FINETUNE_FROM(- for none)
train() {
    cat > $I/logs/jobs/$1.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
export PYTHONPATH=$I:\${PYTHONPATH:-}
FT=()
if [ "$5" != "-" ]; then
    CK=\$(ls -t $I/outputs/*/*-$5/checkpoints/$5/last*.ckpt 2>/dev/null | head -1 || true)
    [ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $5" >&2; exit 1; }
    echo "fine-tuning from \$CK"
    FT=("general.finetune_from='\$CK'")
fi
cd $I/src
python3 main.py general.name=$1 general.wandb=disabled \${FT[@]+"\${FT[@]}"} dataset=$2 +experiment=pair_lux \\
    model.r_embed_dim=1 general.check_val_every_n_epochs=1 general.samples_to_generate=32 \\
    general.final_model_samples_to_generate=32 train.n_epochs=$3 train.batch_size=2 train.lr=$4
EOF
}
train dinter_A_live2yout_remb pair_A_live2yout_k4_s80_strat_deg 2 0.0002 -
train dinter_C_live2yout_remb pair_C_live2yout_k4_s80_strat_deg 2 0.00005 dinter_A_live2yout_remb

YC=$(squeue -u $USER -h -n yt_grp_Cft -o %i)
GPU=(--parsable --account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=128G)
b=$(cd $I && sbatch --parsable --account=chenhp-graphsyn --qos=bbdefault --nodes=1 --ntasks=1 --cpus-per-task=4 \
    --mem=200G --time=08:00:00 --job-name=yt_build --output=$I/logs/yt_build_%j.out \
    --error=$I/logs/yt_build_%j.err $I/logs/jobs/yt_build.sh)
a=$(cd $I && sbatch "${GPU[@]}" --time=1-12:00:00 --nice=0 --dependency=afterok:$b${YC:+,afterany:$YC} \
    --job-name=yt_inter_A --output=$I/logs/yt_inter_A_%j.out --error=$I/logs/yt_inter_A_%j.err \
    $I/logs/jobs/dinter_A_live2yout_remb.sh)
c=$(cd $I && sbatch "${GPU[@]}" --time=1-00:00:00 --nice=0 --dependency=afterok:$a \
    --job-name=yt_inter_C --output=$I/logs/yt_inter_C_%j.out --error=$I/logs/yt_inter_C_%j.err \
    $I/logs/jobs/dinter_C_live2yout_remb.sh)
echo "yt_build $b; yt_inter_A $a (after $b${YC:+, $YC}); yt_inter_C $c (after $a)"
squeue -u $USER -o "%.10i %.15j %.8T %.6y %E"
