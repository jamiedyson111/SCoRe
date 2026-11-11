#!/bin/bash
# One GPU for Collab, one for YouTube (Huiping, 9 Nov 2026). Slurm does not pin jobs to a
# GPU, so the lanes are made with dependencies: each lane runs one job at a time, in order.
#
#   Collab lane:  coll_pairs_B (running) -> coll_pairs_C -> coll_g1_C_0 -> coll_g1_C_1
#                 -> coll_whole_C_0 ... coll_whole_C_5 (all-test version, lowest priority)
#   YouTube lane: yt_grp_B -> yt_grp_Cft -> (YouTube D_inter, submitted later)
#
# coll_g1_C_*: whole graph with only D_intra group 1's 357 test supernodes generated
# (D_intra C, seed 0) and EVERY other supernode original; D_inter C on all 7,917 pairs
# touching group 1 (data/wholegraph/collab_group1_shard*.json, make_wholegraph_shards.py).
# A SUBMITTER: run once on the BlueBEAR login node.
set -euo pipefail
P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
R=$P/D_intra
W=$I/data/wholegraph
id() { squeue -u $USER -h -n "$1" -o %i; }

# context: D_intra C's samples for group 1's supernodes only
GEN=$(ls -td $R/outputs/*/*-dc_sample_Cft/ | head -1)conditional_samples_s0.txt
G1=$W/dc_sample_Cft_group1_s0.txt
python3 - "$GEN" "$W/collab_group1_supernodes.json" "$G1" <<'EOF'
import json, sys
src, keep, dst = sys.argv[1], set(json.load(open(sys.argv[2]))["supernodes"]), sys.argv[3]
blocks = open(src).read().split("\n\n")
out = [b for b in blocks if "supernode=" in b and int(b.split("supernode=")[1].split()[0]) in keep]
open(dst, "w").write("\n\n".join(out) + "\n\n")
print(f"{dst}: {len(out)} of {len(keep)} group-1 supernodes")
if len(out) != len(keep):
    sys.exit("ERROR: some group-1 supernodes have no sample")
EOF

CI=$(sacct -n -X -j 61089925 -o JobID | head -1 | tr -d ' ')     # dc_inter_C
PB=$(id coll_pairs_B); PC=$(id coll_pairs_C); YB=$(id yt_grp_B); YC=$(id yt_grp_Cft)
scontrol update JobId=$PC Dependency=afterok:$CI,afterany:$PB
scontrol update JobId=$YC Dependency=afterany:$YB

prev=$PC
for k in 0 1; do
    N=coll_g1_C_$k
    cat > $I/logs/jobs/$N.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
export PYTHONPATH=$I:\${PYTHONPATH:-}
CK=\$(ls -t $I/outputs/*/*-dinter_C_dblp2coll_remb/checkpoints/dinter_C_dblp2coll_remb/last*.ckpt | head -1)
echo "checkpoint: \$CK"
cd $I/src
python3 sample_pairs_conditional.py dataset=pair_C_dblp2coll_k3_s80_deg +experiment=pair_lux general.gpus=1 \\
    general.wandb=disabled general.name=$N +repeats=1 +sample_batch=128 +pair_budget=5760000 \\
    +subset_file=$W/collab_group1_shard$k.json +gen_samples=$G1 "general.test_only='\$CK'"
EOF
    prev=$(cd $I && sbatch --parsable --account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 \
        --ntasks=1 --cpus-per-task=4 --mem=64G --time=1-00:00:00 --nice=150 \
        --dependency=afterok:$CI,afterany:$prev --job-name=$N \
        --output=$I/logs/${N}_%j.out --error=$I/logs/${N}_%j.err $I/logs/jobs/$N.sh)
    echo "$N $prev"
done
for k in 0 1 2 3 4 5; do
    j=$(id coll_whole_C_$k)
    scontrol update JobId=$j Dependency=afterok:$CI,afterany:$prev
    prev=$j
done
squeue -u $USER -o "%.10i %.15j %.8T %.6y %E"
