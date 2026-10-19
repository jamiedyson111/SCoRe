#!/bin/bash
#SBATCH --job-name=higen_train
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_train_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_train_%j.err

# Train HiGen on the Luxembourg regions. RESUMABLE: resubmit after a timeout and
# it continues from the newest snapshot (one every 10 epochs) until max_epoch.
#
#   sbatch competitors/higen/run_train.sh        # from /rds/projects/c/chenhp-graphsyn
#
# The first job also builds the 2-level Louvain hierarchies of the 120 regions
# (cached under data/LuxRegions2/), which takes a while before epoch 1.

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

# A resume config, when there is something to resume from: the repo resumes
# through train.is_resume / resume_dir / resume_model / resume_epoch.
LAST=$(ls "$SAVE"/model_snapshot_*.pth 2>/dev/null | sort | tail -1 || true)
CFG=configs/higen_lux.yaml
if [ -n "$LAST" ]; then
    EPOCH=$(basename "$LAST" .pth | sed 's/model_snapshot_0*//')
    echo "resuming from $LAST (epoch $EPOCH)"
    python - "$CFG" "$SAVE" "$(basename "$LAST")" "$EPOCH" <<'PY'
import sys, yaml
cfg, save, model, epoch = sys.argv[1:]
c = yaml.load(open(cfg), Loader=yaml.FullLoader)
c["train"].update(is_resume=True, resume_dir=save, resume_model=model, resume_epoch=int(epoch))
yaml.dump(c, open("configs/higen_lux_resume.yaml", "w"), sort_keys=False)
PY
    CFG=configs/higen_lux_resume.yaml
fi
# exp_dir/<dataset>/<hp>-<tag>: --tag run pins it to $SAVE for every job.
python main.py -c "$CFG" --tag run --hp "" 2>&1 | tee -a "$BASE/competitors/logs/higen_train.log"
