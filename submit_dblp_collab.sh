#!/bin/bash
# DBLP (G_s) -> ogbl-collab (G_t), k = 3: the main models, as one chain of SLURM
# jobs. A SUBMITTER, not a job: run it once on the BlueBEAR login node:
#
#     bash /rds/projects/c/chenhp-graphsyn/submit_dblp_collab.sh
#
# Datasets (built locally by D_intra/extraction/make_partition_datasets.py,
# make_finetune_datasets.py --tag dblp2coll_k3_s80 and D_inter/dataprep/
# make_partition_pairs.py --r-edges 0 1 2 4 8): every supernode with n >= 3 and every
# connected pair, no size cap; D_inter's r bins are 0, 1, 2-3, 4-7, >=8.
#
#   D_intra  A v3   train DBLP (12,064 supernodes, up to 296 nodes)
#            C v3   A fine-tuned on Collab train (7,147, up to 380 nodes)
#   D_inter  A      train DBLP pairs (103,851, n_i + n_j up to 528)
#            C      A fine-tuned on Collab train pairs (66,136, up to 601)
#
# TRAINING BUDGET: each model sees about as many training examples as its road run
# (Luxembourg A; Vermont C, the fine-tune closest in size to Collab's):
#   D_intra A  1.41M = 117 epochs x 12,064     C  442k = 62 epochs x 7,147
#   D_inter A  1.24M =  12 epochs x 103,851    C  507k =  8 epochs x 66,136
# BATCHES: memory grows with batch x (largest graph in the batch)^2. Scaled from the
# road runs' peak allocation (D_intra 13.8 GB at 64 x 79^2, D_inter 19.9 GB at
# 32 x 122^2), a batch padded to the largest graph needs about 24 GB (D_intra A,
# batch 8), 30 GB (D_intra C, 6), 23 GB (D_inter A, 2) and 30 GB (D_inter C, 2).
# Measured (November 2026): D_intra A at batch 8 peaks at 27.6 GB, 15% above its
# estimate, at 228 s an epoch. So the A jobs (~27 GB) take any A100 and the C jobs
# (~35 GB, too tight for 40) ask for an 80 GB card.
# Learning rates are the road runs': 2e-4 for A, 5e-5 for a fine-tune.
# D_inter validates every 4 epochs (the best-checkpoint callback watches val NLL)
# and generates 32 logging samples instead of 128 / 512: they are for the log only,
# and at these sizes the defaults would cost hours.
#
# Each C waits (afterok) for its A and fine-tunes A's newest last.ckpt. When the
# queue is empty, the checkpoints are under D_intra/outputs/*-dintra_*_dblp2coll/
# and D_inter/outputs/*-dinter_*_dblp2coll/.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
for f in $P/D_intra/data/cluster_lux/raw/pairs_{A,Cft}_dblp2coll_k3_s80.pt \
         $P/D_inter/data/pair_lux/raw/pairs_inter_{A,C}_dblp2coll_k3_s80_deg.pt \
         $P/D_intra/configs/dataset/cluster_{A,Cft}_dblp2coll_k3_s80.yaml \
         $P/D_inter/configs/dataset/pair_{A,C}_dblp2coll_k3_s80_deg.yaml; do
    [ -f "$f" ] || { echo "ERROR: $f is missing -- upload it first" >&2; exit 1; }
done

SB=(--account=chenhp-graphsyn --qos=bbgpu --nodes=1 --ntasks=1
    --cpus-per-task=4 --mem=64G --parsable)

# body PROJECT NAME FINETUNE_RUN PYTHON_ARGS... -> the job's shell script
body() {
    local proj=$1 name=$2 ft=$3; shift 3
    cat <<EOF
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
ROOT=$P/$proj
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
python3 main.py general.name=$name general.wandb=disabled \${FT[@]+"\${FT[@]}"} $*
EOF
}
# submit PROJECT JOBNAME GPU TIME DEPENDENCY NAME FINETUNE PYTHON_ARGS...
# GPU: gpu:1 (any A100) or gpu:a100_80:1 (an 80 GB card).
# Each job is written to logs/jobs/<JOBNAME>.sh and submitted as a file: BlueBEAR
# cancels (as root, at launch, with no log) the same body passed with --wrap.
submit() {
    local proj=$1 job=$2 gpu=$3 time=$4 dep=$5; shift 5
    local d=(); [ -n "$dep" ] && d=(--dependency=afterok:$dep)
    local script="$P/$proj/logs/jobs/$job.sh"
    mkdir -p "$P/$proj/logs/jobs"
    { echo '#!/bin/bash'; body "$proj" "$@"; } > "$script"
    (cd "$P/$proj" && sbatch "${SB[@]}" --gres="$gpu" ${d[@]+"${d[@]}"} --job-name="$job" --time="$time" \
        --output="$P/$proj/logs/${job}_%j.out" --error="$P/$proj/logs/${job}_%j.err" "$script")
}

V3="dataset.mark_centroid=true model.hop_embed_dim=1"
INTER="general.check_val_every_n_epochs=4 general.samples_to_generate=32 general.final_model_samples_to_generate=32"

echo "== D_intra"
IA=$(submit D_intra dc_intra_A gpu:1 12:00:00 "" dintra_A_v3_dblp2coll - \
     dataset=cluster_A_dblp2coll_k3_s80 +experiment=cluster_lux $V3 \
     train.n_epochs=117 train.batch_size=8 general.check_val_every_n_epochs=20 general.sample_every_val=4)
IC=$(submit D_intra dc_intra_C gpu:a100_80:1 08:00:00 "$IA" dintra_Cft_v3_dblp2coll dintra_A_v3_dblp2coll \
     dataset=cluster_Cft_dblp2coll_k3_s80 +experiment=cluster_lux $V3 \
     train.n_epochs=62 train.batch_size=6 train.lr=0.00005 general.check_val_every_n_epochs=20 general.sample_every_val=4)
echo "  A v3 -> $IA   C v3 (after A) -> $IC"

echo "== D_inter"
XA=$(submit D_inter dc_inter_A gpu:1 48:00:00 "" dinter_A_dblp2coll - \
     dataset=pair_A_dblp2coll_k3_s80_deg +experiment=pair_lux $INTER \
     train.n_epochs=12 train.batch_size=2)
XC=$(submit D_inter dc_inter_C gpu:a100_80:1 30:00:00 "$XA" dinter_C_dblp2coll dinter_A_dblp2coll \
     dataset=pair_C_dblp2coll_k3_s80_deg +experiment=pair_lux $INTER \
     train.n_epochs=8 train.batch_size=2 train.lr=0.00005)
echo "  A -> $XA   C (after A) -> $XC"

cat <<'EOF'

4 jobs submitted (2 run at a time; each C after its A). Watch with:
    squeue -u $USER -o "%.10i %.14j %.9T %.10M %.12L %R"
Check early in the logs (D_*/logs/dc_*):
  D_intra A  "Size of the input features 8 2 ..." (centroid node type, h embedded)
  D_inter A  13 condition columns; the first epoch's time decides whether 12 fit in 48 h
EOF
