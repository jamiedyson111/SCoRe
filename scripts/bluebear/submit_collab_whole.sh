#!/bin/bash
# Whole-graph reconstruction of Collab (Huiping, 9 Nov 2026), one generation: D_inter on
# EVERY pair with at least one D_intra test supernode (34,768 pairs). 5,000 of them are the
# evaluation groups sampled by coll_pairs_{B,C}; this submits the other 29,768 as 6 shards
# (data/wholegraph/collab_rest_shard<k>.json, ~9 GPU-h each, from make_wholegraph_shards.py).
# A SUBMITTER: run once on the BlueBEAR login node:
#
#     bash submit_collab_whole.sh C     # config C (first)
#     bash submit_collab_whole.sh B     # B, only if GPU time allows
#
# Context exactly as coll_pairs_*: GENERATED internal structure for the test supernodes
# (D_intra seed 0 samples), REAL for the rest. C waits for dc_inter_C (its checkpoint).
# Priority nice 150: after coll_pairs_* (0) and the short YouTube D_intra groups (100).
set -euo pipefail
CFG=${1:?usage: submit_collab_whole.sh C|B}

P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
R=$P/D_intra
W=$I/data/wholegraph
case $CFG in
    C) MODEL=dinter_C_dblp2coll_remb; DS=pair_C_dblp2coll_k3_s80_deg; DI=dc_sample_Cft ;;
    B) MODEL=dinter_A_dblp2coll_remb; DS=pair_B_dblp2coll_k3_s80_deg; DI=dc_sample_B ;;
    *) echo "config must be C or B" >&2; exit 1 ;;
esac
GEN=$(ls -td $R/outputs/*/*-$DI/ | head -1)conditional_samples_s0.txt
[ -f "$GEN" ] || { echo "ERROR: $GEN missing" >&2; exit 1; }
mkdir -p $I/logs/jobs

DEP=""
if [ "$CFG" = C ]; then
    CI=$(squeue -u $USER -h -n dc_inter_C -o %i)
    DEP=${CI:+--dependency=afterok:$CI}
fi
for k in 0 1 2 3 4 5; do
    S=$W/collab_rest_shard$k.json
    [ -f "$S" ] || { echo "ERROR: $S missing" >&2; exit 1; }
    N=coll_whole_${CFG}_$k
    cat > $I/logs/jobs/$N.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
export PYTHONPATH=$I:\${PYTHONPATH:-}
CK=\$(ls -t $I/outputs/*/*-$MODEL/checkpoints/$MODEL/last*.ckpt 2>/dev/null | head -1 || true)
[ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $MODEL" >&2; exit 1; }
echo "checkpoint: \$CK"
cd $I/src
python3 sample_pairs_conditional.py dataset=$DS +experiment=pair_lux general.gpus=1 general.wandb=disabled \\
    general.name=$N +repeats=1 +sample_batch=128 +pair_budget=5760000 \\
    +subset_file=$S +gen_samples=$GEN "general.test_only='\$CK'"
EOF
    j=$(cd $I && sbatch --parsable --account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 \
        --ntasks=1 --cpus-per-task=4 --mem=64G --time=1-00:00:00 --nice=150 $DEP --job-name=$N \
        --output=$I/logs/${N}_%j.out --error=$I/logs/${N}_%j.err $I/logs/jobs/$N.sh)
    echo "$N $j"
done
squeue -u $USER -o "%.10i %.15j %.8T %.6y %R"
