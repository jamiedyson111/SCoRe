#!/bin/bash
#SBATCH --job-name=higen_sample
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_sample_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_sample_%j.err

# Five HiGen graphs of 6105 nodes from a trained snapshot, exported as
# competitors/out/HiGen/graph_s<i>.npz.
#
#   sbatch competitors/higen/run_sample.sh [model_snapshot_XXXXXXX.pth]   # default: newest

set -euo pipefail
BASE=/rds/projects/c/chenhp-graphsyn
REPO=$BASE/competitors/HiGen_main
SAVE=$REPO/exp/higen/LuxRegions2/-run    # HiGen names it "<hp>-<tag>"
module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate "$BASE/envs/higen"
export PYTHONUNBUFFERED=1
cd "$REPO"

MODEL="${1:-$(basename "$(ls "$SAVE"/model_snapshot_*.pth | sort | tail -1)")}"
echo "snapshot: $SAVE/$MODEL"
python - "$SAVE" "$MODEL" <<'PY'
import sys, yaml
save, model = sys.argv[1:]
c = yaml.load(open("configs/higen_lux.yaml"), Loader=yaml.FullLoader)
c["test"].update(test_model_dir=save, test_model_name=model)
yaml.dump(c, open("configs/higen_lux_test.yaml", "w"), sort_keys=False)
PY
python main.py -c configs/higen_lux_test.yaml -t --tag sample --hp "" --model_name "$MODEL"
OUTF=$(ls -t "$SAVE"/-sample/gen_graphs_epoch_*.pickle.xz | head -1)
python "$BASE/competitors/higen/export.py" "$OUTF" "$BASE/competitors/out/HiGen"
