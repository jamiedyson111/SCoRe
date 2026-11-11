#!/bin/bash
# YouTube D_inter on 5 x 500 (10 Nov 2026): the full 5 x 1,000 is too expensive
# (pairs with a 500-node supernode take 60% of the GPU time).
# Each half: within a test group, pairs ranked by w_ij / (n_i + n_j), 5 equal bins of 200,
# 100 drawn at random from each (seed 0), so every half keeps its group's density mix.
# Saved for the competitors as competitor_data/youtube/test_groups_d_inter_500.json.
#
# Cancels yt_pairs_B, yt_pairs_C_0, yt_pairs_C_1 and coll_g1_C_0/1, then submits:
#   YouTube lane: yt_half_B -> yt_half_C            (5 x 500 each)
#   Collab lane:  coll_g1_C_0 -> coll_g1_C_1         (unchanged, after coll_pairs_C)
# Every job now passes +partial_file (saved every 10 min): a failed or timed-out job is
# resubmitted with the same script and resumes where it stopped.
# A SUBMITTER: run once on the BlueBEAR login node.
set -euo pipefail
P=/rds/projects/c/chenhp-graphsyn
I=$P/D_inter
R=$P/D_intra
W=$I/data/wholegraph
G=$P/competitor_data/youtube
id() { squeue -u $USER -h -n "$1" -o %i; }
grep -q partial_file $I/src/sample_pairs_conditional.py || { echo "ERROR: upload sample_pairs_conditional.py" >&2; exit 1; }
mkdir -p $W/partial

# the stratified halves (needs torch for the pairs file: the project env)
( . $I/scripts/setup.sh >/dev/null 2>&1; python3 - <<EOF
import json, random, torch
blob = torch.load("$I/data/pair_lux/raw/pairs_inter_C_live2yout_k4_s80_strat_deg.pt", map_location="cpu", weights_only=False)
test = blob["splits"][2]
g = json.load(open("$G/test_groups_d_inter.json"))
rng = random.Random(0)
halves = []
for grp in g["groups"]:
    for p in grp:
        r = test[p["idx"]]
        assert (int(r["sn_i"]), int(r["sn_j"])) == (p["sn_i"], p["sn_j"])
    ranked = sorted(grp, key=lambda p: (test[p["idx"]]["w"] / (test[p["idx"]]["n_i"] + test[p["idx"]]["n_j"]), rng.random()))
    half = []
    for b in range(5):
        half += rng.sample(ranked[b * 200:(b + 1) * 200], 100)
    halves.append(sorted(half, key=lambda p: p["idx"]))
out = dict(g, what="D_inter test pairs, 5 x 500: half of each test_groups_d_inter.json group, "
                   "stratified on w_ij/(n_i+n_j) (5 bins of 200, 100 from each), seed 0",
           half_of="test_groups_d_inter.json", groups=halves)
json.dump(out, open("$G/test_groups_d_inter_500.json", "w"))
json.dump(out, open("$W/yt_pairs_halves.json", "w"))
size = lambda ps: sum((test[p["idx"]]["n_i"] + test[p["idx"]]["n_j"]) ** 2 for p in ps) / len(ps)
print("halves:", [len(h) for h in halves], "| mean (n_i+n_j)^2 half %.0f vs full %.0f"
      % (size([p for h in halves for p in h]), size([p for x in g["groups"] for p in x])))
EOF
)

for j in yt_pairs_B yt_pairs_C_0 yt_pairs_C_1 coll_g1_C_0 coll_g1_C_1; do
    k=$(id $j); [ -n "$k" ] && scancel $k && echo "cancelled $j $k"
done

# pairs NAME MODEL DATASET SUBSET GEN
pairs() {
    cat > $I/logs/jobs/$1.sh <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
. $I/scripts/setup.sh
export PYTHONPATH=$I:\${PYTHONPATH:-}
CK=\$(ls -t $I/outputs/*/*-$2/checkpoints/$2/last*.ckpt | head -1)
echo "checkpoint: \$CK"
cd $I/src
python3 sample_pairs_conditional.py dataset=$3 +experiment=pair_lux general.gpus=1 general.wandb=disabled \\
    general.name=$1 +repeats=1 +sample_batch=128 +pair_budget=5760000 \\
    +subset_file=$4 +gen_samples=$5 +partial_file=$W/partial/$1.pt "general.test_only='\$CK'"
EOF
}
for c in B C; do
    [ $c = B ] && M=dinter_A_live2yout_remb || M=dinter_C_live2yout_remb
    [ $c = B ] && D=B || D=Cft
    [ -s $W/yt_gen_$D.txt ] || { echo "ERROR: $W/yt_gen_$D.txt missing" >&2; exit 1; }
    pairs yt_half_$c $M pair_${c}_live2yout_k4_s80_strat_deg $W/yt_pairs_halves.json $W/yt_gen_$D.txt
done
G1=$W/dc_sample_Cft_group1_s0.txt
for k in 0 1; do
    pairs coll_g1_C_$k dinter_C_dblp2coll_remb pair_C_dblp2coll_k3_s80_deg $W/collab_group1_shard$k.json $G1
done

GPU=(--parsable --account=chenhp-graphsyn --qos=bbgpu --gres=gpu:a100_80:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G)
sub() { (cd $I && sbatch "${GPU[@]}" --time=$2 ${3:+--dependency=$3} --job-name=$1 \
    --output=$I/logs/${1}_%j.out --error=$I/logs/${1}_%j.err $I/logs/jobs/$1.sh); }
PC=$(id coll_pairs_C)
hb=$(sub yt_half_B 20:00:00)
hc=$(sub yt_half_C 20:00:00 afterany:$hb)
g0=$(sub coll_g1_C_0 1-00:00:00 ${PC:+afterany:$PC})
g1=$(sub coll_g1_C_1 1-00:00:00 afterany:$g0)
echo "yt_half_B $hb -> yt_half_C $hc;  coll_g1_C_0 $g0 (after ${PC:-nothing}) -> coll_g1_C_1 $g1"
squeue -u $USER -o "%.10i %.15j %.8T %.10M %E"
