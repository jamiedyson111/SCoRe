#!/bin/bash
# Job priority order (November 2026):
#   1. road networks (Oldenburg, Vermont)   nice 0
#   2. DBLP -> Collab                       nice 50
#   3. LiveJournal -> YouTube D_intra       nice 100
# A SUBMITTER, not a job: run it once on the BlueBEAR login node:
#
#     bash submit_priority_nov.sh
#
# Nice only orders MY jobs against each other (2 GPUs at a time); a lower-priority job
# still runs whenever no higher one is eligible, so no GPU sits idle.
#
#   road     the 8 r-embedded D_inter sampling jobs (already queued, nice 0)
#            NEW Ours-Target D_intra on Oldenburg and Vermont: v3 trained from scratch
#                on the target's 80% alone, the gradient steps of plain DiGress-target
#                (O_uncond), then 5 seeds on the test supernodes
#   Collab   dc_inter_C (queued); dc_sample_B / dc_sample_Cft RESUBMITTED sorted by
#                size (~5x less padding; same samples' distribution, scored by id)
#            NEW Ours-Target and plain DiGress (O_uncond) on Collab, 1.41M examples
#                each (the road budget: 197 epochs x 7,147 train supernodes, batch 8),
#                then 5 seeds on the 1,787 test supernodes
#   YouTube  lj_intra_C (queued); NEW sampling of B (LiveJournal A) and C (fine-tune)
#                on a fixed random 2,000 of the 11,016 YouTube test supernodes, 5 seeds
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
R=$P/D_intra
grep -q sort_by_size $R/src/sample_conditional.py || { echo "ERROR: upload sample_conditional.py" >&2; exit 1; }
for f in $R/extraction/make_target_datasets.py \
         $R/configs/dataset/cluster_T_{lux2olde_k7,lux2verm_k7,dblp2coll_k3}_s80.yaml \
         $R/data/cluster_lux/raw/pairs_{B,Cft}_dblp2coll_k3_s80.pt $R/data/cluster_lux/raw/pairs_O_dblp2coll_k3_s80_uncond.pt \
         $R/data/cluster_lux/raw/pairs_{B,Cft}_live2yout_k4_s80.pt; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
mkdir -p $R/logs/jobs

echo "== Target datasets (built here from the Cft files)"
(. $R/scripts/setup.sh >/dev/null && cd $R/extraction && \
 python3 make_target_datasets.py --tag lux2olde_k7_s80 lux2verm_k7_s80 dblp2coll_k3_s80)

# job NAME BODY -> writes logs/jobs/NAME.sh
job() {
    cat > $R/logs/jobs/$1.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $R/scripts/setup.sh
export PYTHONPATH=$R:\${PYTHONPATH:-}
cd $R/src
$2
EOF
}
# train NAME EXTRA_ARGS
train() { job "$1" "python3 main.py +experiment=cluster_lux general.name=$1 general.wandb=disabled $2"; }
# sample NAME MODEL EXTRA_ARGS: the newest last*.ckpt of MODEL, found when the job starts
sample() {
    job "$1" "CK=\$(ls -t $R/outputs/*/*-$2/checkpoints/$2/last*.ckpt 2>/dev/null | head -1 || true)
[ -n \"\$CK\" ] || { echo \"ERROR: no last*.ckpt for $2\" >&2; exit 1; }
echo \"checkpoint: \$CK\"
python3 sample_conditional.py +experiment=cluster_lux general.name=$1 general.gpus=1 general.wandb=disabled \\
    +repeats=5 +sort_by_size=true $3 \"general.test_only='\$CK'\""
}
# submit NAME NICE TIME GPU_KIND [DEPENDENCY] -> job id
submit() {
    local gres=gpu:1 mem=32G dep=()
    [ "$4" = a100_80 ] && { gres=gpu:a100_80:1; mem=64G; }
    [ -n "${5:-}" ] && dep=(--dependency=afterok:$5)
    (cd $R && sbatch --parsable --account=chenhp-graphsyn --qos=bbgpu --gres=$gres --nodes=1 --ntasks=1 \
        --cpus-per-task=4 --mem=$mem --time=$3 --nice=$2 ${dep[@]+"${dep[@]}"} --job-name=$1 \
        --output=$R/logs/$1_%j.out --error=$R/logs/$1_%j.err $R/logs/jobs/$1.sh)
}
V3="dataset.mark_centroid=true model.hop_embed_dim=1"
VAL="general.check_val_every_n_epochs=20 general.sample_every_val=4 general.final_model_samples_to_generate=64"

echo "== 1. road (nice 0)"
train dintra_T_v3_olde "dataset=cluster_T_lux2olde_k7_s80 $V3 train.n_epochs=5500 general.check_val_every_n_epochs=250 general.sample_every_val=4"
train dintra_T_v3_verm "dataset=cluster_T_lux2verm_k7_s80 $V3 train.n_epochs=320 general.check_val_every_n_epochs=20 general.sample_every_val=4"
sample tg_sample_olde dintra_T_v3_olde "dataset=cluster_T_lux2olde_k7_s80 $V3"
sample tg_sample_verm dintra_T_v3_verm "dataset=cluster_T_lux2verm_k7_s80 $V3 +sample_batch=128"
a=$(submit dintra_T_v3_olde 0 08:00:00 any);  b=$(submit tg_sample_olde 0 02:00:00 any $a)
c=$(submit dintra_T_v3_verm 0 10:00:00 any);  d=$(submit tg_sample_verm 0 04:00:00 any $c)
echo "  Target Oldenburg train $a, sample $b   Target Vermont train $c, sample $d"

echo "== 2. Collab (nice 50)"
for j in $(squeue -u $USER -h -n dc_sample_B,dc_sample_Cft -t PD -o %i); do scancel $j; echo "  cancelled unsorted $j"; done
scontrol update jobid=$(squeue -u $USER -h -n dc_inter_C -o %i) nice=50 && echo "  dc_inter_C nice 50"
sample dc_sample_B   dintra_A_v3_dblp2coll   "dataset=cluster_B_dblp2coll_k3_s80 $V3 +sample_batch=8"
sample dc_sample_Cft dintra_Cft_v3_dblp2coll "dataset=cluster_Cft_dblp2coll_k3_s80 $V3 +sample_batch=8"
train dintra_T_v3_coll "dataset=cluster_T_dblp2coll_k3_s80 $V3 train.n_epochs=197 train.batch_size=8 $VAL"
train dintra_O_coll    "dataset=cluster_O_dblp2coll_k3_s80_uncond train.n_epochs=197 train.batch_size=8 $VAL"
sample tg_sample_coll dintra_T_v3_coll "dataset=cluster_T_dblp2coll_k3_s80 $V3 +sample_batch=8"
sample O_sample_coll  dintra_O_coll    "dataset=cluster_O_dblp2coll_k3_s80_uncond +sample_batch=8"
e=$(submit dc_sample_B 50 12:00:00 a100_80); f=$(submit dc_sample_Cft 50 12:00:00 a100_80)
g=$(submit dintra_T_v3_coll 50 16:00:00 a100_80); h=$(submit tg_sample_coll 50 12:00:00 a100_80 $g)
i=$(submit dintra_O_coll 50 16:00:00 a100_80);    k=$(submit O_sample_coll 50 12:00:00 a100_80 $i)
echo "  sample B $e, C $f   Target train $g, sample $h   DiGress train $i, sample $k"

echo "== 3. YouTube D_intra (nice 100)"
LJA=$(squeue -u $USER -h -n lj_intra_A -o %i); LJC=$(squeue -u $USER -h -n lj_intra_C -o %i)
scontrol update jobid=$LJC nice=100 && echo "  lj_intra_C nice 100"
sample yt_sample_B   dintra_A_v3_live2yout   "dataset=cluster_B_live2yout_k4_s80 $V3 +sample_batch=8 +subset=2000"
sample yt_sample_Cft dintra_Cft_v3_live2yout "dataset=cluster_Cft_live2yout_k4_s80 $V3 +sample_batch=8 +subset=2000"
m=$(submit yt_sample_B 100 20:00:00 a100_80 $LJA); n=$(submit yt_sample_Cft 100 20:00:00 a100_80 $LJC)
echo "  YouTube B $m (after $LJA), C $n (after $LJC)"

echo; squeue -u $USER -o "%.10i %.20j %.8T %.6y %.12l %R"
