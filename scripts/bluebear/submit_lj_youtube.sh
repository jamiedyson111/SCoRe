#!/bin/bash
# LiveJournal (G_s) -> YouTube (G_t), k = 4: D_intra, built and trained on BlueBEAR.
# A SUBMITTER, not a job: run it once on the BlueBEAR login node:
#
#     bash submit_lj_youtube.sh
#
# Coarsening: Huiping's CB-KMIS (kmis_cb_v4_l10: k = 4, capacity b = 500, small-group
# merge l = 10), coarsen_compare_expr/output/livejournal_youtube/. Built from her
# partition (assignment + centroids), never re-coarsened.
#
#   lj_build    CPU: extraction/make_partition_datasets.py (A / B / C, with h, no h,
#               unconditioned) then make_finetune_datasets.py (Cft, O), every supernode
#               with n >= 3, 80 / 20 by supernode; written to data/cluster_lux/raw/ and
#               the split to data/splits/{livejournal,youtube}_k4_s80.json
#   lj_intra_A  D_intra A v3 on LiveJournal (~98,900 train supernodes, up to 500 nodes)
#   lj_intra_C  A fine-tuned on YouTube train (~44,100), tested on YouTube's 20%
#
# Budget as DBLP -> Collab: each model sees about as many training examples as its road
# run (Luxembourg A 1.41M -> 14 epochs; Vermont C 442k -> 10 epochs). Batch 6 on an
# 80 GB card: from DBLP's measured 27.6 GB at 8 x 296^2, a batch padded to 500 nodes
# needs ~59 GB. Learning rates are the road runs' (2e-4; fine-tune 5e-5). Logging
# samples cut to 64 at the end, as their size makes the default slow.
# Each job waits (afterok) for the one before.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
R=$P/D_intra
SRC=$P/coarsen_compare_expr/output/livejournal_youtube
LJ=$SRC/com_livejournal_csr_kmis_cb_v4_k4_b500_l10.npz
YT=$SRC/com_youtube_csr_kmis_cb_v4_k4_b500_l10.npz
TAG=live2yout_k4_s80
for f in "$LJ" "$YT" $R/extraction/make_partition_datasets.py $R/extraction/make_finetune_datasets.py \
         $R/configs/dataset/cluster_{A,Cft}_${TAG}.yaml; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
mkdir -p $R/logs/jobs

cat > $R/logs/jobs/lj_build.sh <<EOF
#!/bin/bash
set -euo pipefail
. $R/scripts/setup.sh
cd $R/extraction
python3 -I -c "import sys, runpy; sys.path.insert(0, '$R/extraction'); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \\
    make_partition_datasets.py --public livejournal --public-npz $LJ --private youtube --private-npz $YT
python3 -I -c "import sys, runpy; sys.path.insert(0, '$R/extraction'); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')" \\
    make_finetune_datasets.py --tag $TAG
ls -la $R/data/cluster_lux/raw/ | grep $TAG
EOF

# train NAME FINETUNE_RUN PYTHON_ARGS... -> a training job script
train() {
    local name=$1 ft=$2; shift 2
    cat <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
ROOT=$R
. \$ROOT/scripts/setup.sh
export PYTHONPATH=\$ROOT:\${PYTHONPATH:-}
FT=()
if [ "$ft" != "-" ]; then
    CK=\$(ls -t \$ROOT/outputs/*/*-$ft/checkpoints/$ft/last*.ckpt 2>/dev/null | head -1 || true)
    [ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $ft" >&2; exit 1; }
    echo "fine-tuning from \$CK"
    FT=("general.finetune_from='\$CK'")
fi
cd \$ROOT/src
python3 main.py +experiment=cluster_lux general.name=$name general.wandb=disabled \\
    dataset.mark_centroid=true model.hop_embed_dim=1 \${FT[@]+"\${FT[@]}"} $*
EOF
}
V="general.check_val_every_n_epochs=5 general.sample_every_val=4 general.final_model_samples_to_generate=64"
train dintra_A_v3_live2yout - dataset=cluster_A_${TAG} train.n_epochs=14 train.batch_size=6 $V \
    > $R/logs/jobs/lj_intra_A.sh
train dintra_Cft_v3_live2yout dintra_A_v3_live2yout dataset=cluster_Cft_${TAG} train.n_epochs=10 \
    train.batch_size=6 train.lr=0.00005 $V > $R/logs/jobs/lj_intra_C.sh

SB=(--account=chenhp-graphsyn --nodes=1 --ntasks=1 --parsable)
cd $R
B=$(sbatch "${SB[@]}" --qos=bbdefault --cpus-per-task=8 --mem=96G --time=08:00:00 --job-name=lj_build \
    --output=$R/logs/lj_build_%j.out --error=$R/logs/lj_build_%j.err $R/logs/jobs/lj_build.sh)
A=$(sbatch "${SB[@]}" --qos=bbgpu --gres=gpu:a100_80:1 --cpus-per-task=4 --mem=64G --time=24:00:00 \
    --dependency=afterok:$B --job-name=lj_intra_A \
    --output=$R/logs/lj_intra_A_%j.out --error=$R/logs/lj_intra_A_%j.err $R/logs/jobs/lj_intra_A.sh)
C=$(sbatch "${SB[@]}" --qos=bbgpu --gres=gpu:a100_80:1 --cpus-per-task=4 --mem=64G --time=10:00:00 \
    --dependency=afterok:$A --job-name=lj_intra_C \
    --output=$R/logs/lj_intra_C_%j.out --error=$R/logs/lj_intra_C_%j.err $R/logs/jobs/lj_intra_C.sh)
echo "build -> $B   D_intra A (LiveJournal) -> $A   D_intra C (YouTube fine-tune) -> $C"
