#!/bin/bash
# Huiping's evaluation protocol (9 November 2026): fixed, non-overlapping test groups,
# the same for every method, saved in competitor_data/<graph>/test_groups_*.json.
# A SUBMITTER, not a job: run it once on the BlueBEAR login node:
#
#     bash submit_collab_eval.sh
#
#   coll_build_B  CPU  pairs_inter_B_dblp2coll_k3_s80_deg.pt from the A and C files
#                      (dataprep/make_b_from_a_c.py; checked identical to the local build)
#   coll_pairs_B  GPU  D_inter B (the DBLP model) on the 5 x 1,000 Collab test pairs
#   coll_pairs_C  GPU  D_inter C (fine-tuned on Collab) on the same pairs, after dc_inter_C
#                      Context: GENERATED internal structure for the test supernodes (D_intra
#                      B / C, seed 0), the REAL one for training supernodes. One sample per
#                      pair; the 5 groups give the mean and std. Packed batches.
#   yt_grp_B, yt_grp_Cft  GPU  D_intra B / C on the 5 x 1,000 YouTube test supernodes
#                      (lower priority: Collab first)
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
R=$P/D_intra
G=$P/competitor_data
for f in $I/dataprep/make_b_from_a_c.py $G/collab/test_groups_d_inter.json $G/youtube/test_groups_d_intra.json; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
grep -q gen_samples $I/src/sample_pairs_conditional.py || { echo "ERROR: upload sample_pairs_conditional.py" >&2; exit 1; }
grep -q subset_file $R/src/sample_conditional.py || { echo "ERROR: upload sample_conditional.py" >&2; exit 1; }
mkdir -p $I/logs/jobs $R/logs/jobs

DB=$(ls -td $R/outputs/*/*-dc_sample_B/ | head -1)conditional_samples_s0.txt
DC=$(ls -td $R/outputs/*/*-dc_sample_Cft/ | head -1)conditional_samples_s0.txt
[ -f "$DB" ] && [ -f "$DC" ] || { echo "ERROR: Collab D_intra samples missing" >&2; exit 1; }

jobhead() {
    cat <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $1/scripts/setup.sh
export PYTHONPATH=$1:\${PYTHONPATH:-}
EOF
}

{ jobhead $I; echo "cd $I/dataprep && python3 make_b_from_a_c.py --tag dblp2coll_k3_s80_deg"; } > $I/logs/jobs/coll_build_B.sh

# pairs NAME MODEL DATASET D_INTRA_SAMPLES
pairs() {
    { jobhead $I; cat <<EOF
CK=\$(ls -t $I/outputs/*/*-$2/checkpoints/$2/last*.ckpt 2>/dev/null | head -1 || true)
[ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $2" >&2; exit 1; }
echo "checkpoint: \$CK"
cd $I/src
python3 sample_pairs_conditional.py dataset=$3 +experiment=pair_lux general.gpus=1 general.wandb=disabled \\
    general.name=$1 +repeats=1 +sample_batch=128 +pair_budget=5760000 \\
    +subset_file=$G/collab/test_groups_d_inter.json +gen_samples=$4 "general.test_only='\$CK'"
EOF
    } > $I/logs/jobs/$1.sh
}
pairs coll_pairs_B dinter_A_dblp2coll_remb pair_B_dblp2coll_k3_s80_deg "$DB"
pairs coll_pairs_C dinter_C_dblp2coll_remb pair_C_dblp2coll_k3_s80_deg "$DC"

# intra NAME MODEL DATASET
intra() {
    { jobhead $R; cat <<EOF
CK=\$(ls -t $R/outputs/*/*-$2/checkpoints/$2/last*.ckpt 2>/dev/null | head -1 || true)
[ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $2" >&2; exit 1; }
echo "checkpoint: \$CK"
cd $R/src
python3 sample_conditional.py dataset=$3 +experiment=cluster_lux dataset.mark_centroid=true model.hop_embed_dim=1 \\
    general.gpus=1 general.wandb=disabled general.name=$1 +repeats=1 +sort_by_size=true +sample_batch=8 \\
    +subset_file=$G/youtube/test_groups_d_intra.json "general.test_only='\$CK'"
EOF
    } > $R/logs/jobs/$1.sh
}
intra yt_grp_B dintra_A_v3_live2yout cluster_B_live2yout_k4_s80
intra yt_grp_Cft dintra_Cft_v3_live2yout cluster_Cft_live2yout_k4_s80

GPU=(--account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --parsable)
b=$(cd $I && sbatch --parsable --account=chenhp-graphsyn --qos=bbdefault --nodes=1 --ntasks=1 --cpus-per-task=2 \
    --mem=96G --time=02:00:00 --job-name=coll_build_B --output=$I/logs/coll_build_B_%j.out \
    --error=$I/logs/coll_build_B_%j.err $I/logs/jobs/coll_build_B.sh)
CI=$(squeue -u $USER -h -n dc_inter_C -o %i)
pb=$(cd $I && sbatch "${GPU[@]}" --time=1-00:00:00 --nice=0 --dependency=afterok:$b --job-name=coll_pairs_B \
    --output=$I/logs/coll_pairs_B_%j.out --error=$I/logs/coll_pairs_B_%j.err $I/logs/jobs/coll_pairs_B.sh)
pc=$(cd $I && sbatch "${GPU[@]}" --time=1-00:00:00 --nice=0 ${CI:+--dependency=afterok:$CI} --job-name=coll_pairs_C \
    --output=$I/logs/coll_pairs_C_%j.out --error=$I/logs/coll_pairs_C_%j.err $I/logs/jobs/coll_pairs_C.sh)
yb=$(cd $R && sbatch "${GPU[@]}" --time=06:00:00 --nice=100 --job-name=yt_grp_B \
    --output=$R/logs/yt_grp_B_%j.out --error=$R/logs/yt_grp_B_%j.err $R/logs/jobs/yt_grp_B.sh)
yc=$(cd $R && sbatch "${GPU[@]}" --time=06:00:00 --nice=100 --job-name=yt_grp_Cft \
    --output=$R/logs/yt_grp_Cft_%j.out --error=$R/logs/yt_grp_Cft_%j.err $R/logs/jobs/yt_grp_Cft.sh)
echo "build B $b; Collab test pairs B $pb (after $b), C $pc (after ${CI:-nothing}); YouTube D_intra B $yb, C $yc"
squeue -u $USER -o "%.10i %.15j %.8T %.6y %R"
