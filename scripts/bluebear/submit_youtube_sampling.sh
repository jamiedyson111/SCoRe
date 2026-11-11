#!/bin/bash
# YouTube D_inter sampling on the 5 x 1,000 test groups (competitor_data/youtube/test_groups_d_inter.json),
# with the Collab context rule: D_intra's GENERATED structure for test supernodes, REAL otherwise.
# A SUBMITTER: run once on the BlueBEAR login node, after submit_youtube_inter.sh.
#
#   YouTube lane: yt_inter_C -> yt_extra_B -> yt_extra_Cft -> yt_pairs_B (groups 1-5) -> yt_pairs_C_0 (groups 1-3)
#   Collab lane:  coll_g1_C_1 -> yt_pairs_C_1 (groups 4-5)
#
# yt_extra_*: D_intra B / Cft on the 1,543 test supernodes touched by the D_inter test pairs
# that the D_intra groups (yt_grp_*) did not cover; merged with yt_grp_*'s samples into
# data/wholegraph/yt_gen_{B,Cft}.txt for +gen_samples.
set -euo pipefail
P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
R=$P/D_intra
W=$I/data/wholegraph
G=$P/competitor_data/youtube
id() { squeue -u $USER -h -n "$1" -o %i; }
mkdir -p $W $I/logs/jobs $R/logs/jobs

# the extra supernodes, and the C halves of the D_inter groups
python3 - <<EOF
import json
g = json.load(open("$G/test_groups_d_inter.json"))
te = set(json.load(open("$R/data/splits/youtube_k4_s80.json"))["test"])
have = {x for grp in json.load(open("$G/test_groups_d_intra.json"))["groups"] for x in grp}
need = sorted({x for grp in g["groups"] for p in grp for x in (p["sn_i"], p["sn_j"]) if x in te} - have)
json.dump({"graph": "youtube", "what": "test supernodes of the D_inter test pairs not in the D_intra groups",
           "groups": [need]}, open("$W/yt_extra_supernodes.json", "w"))
for name, part in (("yt_pairs_groups123", g["groups"][:3]), ("yt_pairs_groups45", g["groups"][3:])):
    json.dump(dict(g, groups=part), open(f"$W/{name}.json", "w"))
print(f"{len(need)} extra supernodes")
EOF

# intra NAME MODEL DATASET
intra() {
    cat > $R/logs/jobs/$1.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $R/scripts/setup.sh
export PYTHONPATH=$R:\${PYTHONPATH:-}
CK=\$(ls -t $R/outputs/*/*-$2/checkpoints/$2/last*.ckpt | head -1)
echo "checkpoint: \$CK"
cd $R/src
python3 sample_conditional.py dataset=$3 +experiment=cluster_lux dataset.mark_centroid=true model.hop_embed_dim=1 \\
    general.gpus=1 general.wandb=disabled general.name=$1 +repeats=1 +sort_by_size=true +sample_batch=8 \\
    +subset_file=$W/yt_extra_supernodes.json "general.test_only='\$CK'"
EOF
}
intra yt_extra_B dintra_A_v3_live2yout cluster_B_live2yout_k4_s80
intra yt_extra_Cft dintra_Cft_v3_live2yout cluster_Cft_live2yout_k4_s80

# pairs NAME MODEL DATASET D_INTRA_CFG SUBSET
pairs() {
    cat > $I/logs/jobs/$1.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
export PYTHONPATH=$I:\${PYTHONPATH:-}
GEN=$W/yt_gen_$4.txt
cat \$(ls -td $R/outputs/*/*-yt_grp_$4/ | head -1)conditional_samples.txt \\
    \$(ls -td $R/outputs/*/*-yt_extra_$4/ | head -1)conditional_samples.txt > \$GEN
echo "context: \$(grep -c '^supernode=' \$GEN) generated supernodes in \$GEN"
CK=\$(ls -t $I/outputs/*/*-$2/checkpoints/$2/last*.ckpt | head -1)
echo "checkpoint: \$CK"
cd $I/src
python3 sample_pairs_conditional.py dataset=$3 +experiment=pair_lux general.gpus=1 general.wandb=disabled \\
    general.name=$1 +repeats=1 +sample_batch=128 +pair_budget=5760000 \\
    +subset_file=$5 +gen_samples=\$GEN "general.test_only='\$CK'"
EOF
}
pairs yt_pairs_B dinter_A_live2yout_remb pair_B_live2yout_k4_s80_strat_deg B $G/test_groups_d_inter.json
pairs yt_pairs_C_0 dinter_C_live2yout_remb pair_C_live2yout_k4_s80_strat_deg Cft $W/yt_pairs_groups123.json
pairs yt_pairs_C_1 dinter_C_live2yout_remb pair_C_live2yout_k4_s80_strat_deg Cft $W/yt_pairs_groups45.json

A=$(id yt_inter_A); C=$(id yt_inter_C); G1=$(id coll_g1_C_1)
[ -n "$C" ] && [ -n "$G1" ] || { echo "ERROR: yt_inter_C or coll_g1_C_1 not in the queue" >&2; exit 1; }
GPU=(--parsable --account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G)
sub() { local dir=$1 name=$2 time=$3 dep=$4; (cd $dir && sbatch "${GPU[@]}" --time=$time --dependency=$dep \
    --job-name=$name --output=$dir/logs/${name}_%j.out --error=$dir/logs/${name}_%j.err $dir/logs/jobs/$name.sh); }
eb=$(sub $R yt_extra_B 03:00:00 afterany:$C)
ec=$(sub $R yt_extra_Cft 03:00:00 afterany:$eb)
pb=$(sub $I yt_pairs_B 1-12:00:00 afterok:$eb,afterany:$ec${A:+,afterok:$A})
c0=$(sub $I yt_pairs_C_0 1-06:00:00 afterok:$C,afterok:$ec,afterany:$pb)
c1=$(sub $I yt_pairs_C_1 1-06:00:00 afterok:$C,afterok:$ec,afterany:$G1)
echo "yt_extra_B $eb, yt_extra_Cft $ec, yt_pairs_B $pb, yt_pairs_C_0 $c0 (YouTube lane); yt_pairs_C_1 $c1 (after coll_g1_C_1)"
squeue -u $USER -o "%.10i %.15j %.8T %.10M %E"
