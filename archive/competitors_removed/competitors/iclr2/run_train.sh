#!/bin/bash
#SBATCH --job-name=iclr2_train
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/competitors/logs/iclr2_train_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/competitors/logs/iclr2_train_%j.err

# Train ICLR [2] on the Luxembourg regions. RESUMABLE: submit it again after a
# timeout and it carries on from the newest checkpoint (one every 5k steps, apply_patch.py).
#
#   sbatch competitors/iclr2/run_train.sh            # from /rds/projects/c/chenhp-graphsyn
#
# It stops by itself at training.num_steps (200k). The log prints step_time every
# 500 steps -- multiply by 200k for the total, and resubmit until done.

set -euo pipefail
BASE=/rds/projects/c/chenhp-graphsyn
REPO=$BASE/competitors/graph-generation
RUN=$REPO/outputs/lux_regions
module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3        # as D_intra/scripts/setup.sh
eval "$(conda shell.bash hook)"
conda activate "$BASE/envs/graph-generation"
. "$BASE/competitors/gpu_monitor.sh"    # peak GPU memory + wall clock (Section VI-E)
export PYTHONUNBUFFERED=1
cd "$REPO"

# Done already? The repo stops only when step == num_steps exactly, so resuming
# from the final checkpoint would train past it forever. A chained job that
# starts after training has finished must exit here instead.
if [ -f "$RUN/checkpoints/step_200000.pt" ]; then
    echo "training already finished (step_200000.pt exists) -- nothing to do"
    exit 0
fi
RESUME=False
ls "$RUN"/checkpoints/step_*.pt >/dev/null 2>&1 && RESUME=True
echo "resume: $RESUME"
python main.py +experiment=lux_regions hydra.run.dir="$RUN" training.resume=$RESUME
