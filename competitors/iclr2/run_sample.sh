#!/bin/bash
#SBATCH --job-name=iclr2_sample
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/competitors/logs/iclr2_sample_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/competitors/logs/iclr2_sample_%j.err

# Five ICLR [2] graphs of Oldenburg's size (6105 nodes) from a trained checkpoint,
# via the repo's own test mode, then exported as competitors/out/ICLR2/graph_s<i>.npz.
#
#   sbatch competitors/iclr2/run_sample.sh [STEP]      # STEP default: newest checkpoint

set -euo pipefail
BASE=/rds/projects/c/chenhp-graphsyn
REPO=$BASE/competitors/graph-generation
RUN=$REPO/outputs/lux_regions
STEP="${1:-$(ls "$RUN"/checkpoints/step_*.pt | sed 's/.*step_\([0-9]*\)\.pt/\1/' | sort -n | tail -1)}"
module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3        # as D_intra/scripts/setup.sh
eval "$(conda shell.bash hook)"
conda activate "$BASE/envs/graph-generation"
export PYTHONUNBUFFERED=1
cd "$REPO"
echo "checkpoint step: $STEP"

# Same run directory, so resume finds the checkpoint; the olde5 dataset only
# changes the test list (5 x 6105-node targets). Validation is off, so there is no
# per-beta score to choose by: one EMA beta, 0.999, the middle of the repo's
# [0.99, 0.999, 0.9999, 1]. skip_metrics (apply_patch.py) returns the graphs
# without the repo's metric suite, which at this size takes hours.
BETA=0.999
echo "EMA beta: $BETA"
python main.py +experiment=lux_regions hydra.run.dir="$RUN" dataset.name=lux_regions_olde5 \
    training.resume="$STEP" testing=True "ema.betas=[$BETA]" skip_metrics=true
python "$BASE/competitors/iclr2/export.py" "$RUN/test/step_$STEP.pkl" "$BASE/competitors/out/ICLR2" "$BETA"
