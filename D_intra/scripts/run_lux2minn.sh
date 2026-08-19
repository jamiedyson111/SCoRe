#!/bin/bash
#SBATCH --job-name=lux2minn
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/lux2minn_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/lux2minn_%j.err

# Public -> private transfer using the model that is ALREADY TRAINED.
# No retraining: this loads the existing Luxembourg checkpoint (epoch 59) and
# generates a subgraph for each of the 78 Minnesota supernodes.
#
#   sbatch scripts/run_lux2minn.sh
#   sbatch scripts/run_lux2minn.sh /abs/path/to/epoch=N.ckpt
#
# Complements run_transfer.sh B: that retrains and tests on 40 Minnesota
# supernodes (31 are reserved for configuration C's training), whereas this
# tests on all 78 and so has better statistical power for the "does it transfer
# at all" question.
#
# COMPATIBILITY. Minnesota is coarsened at the SAME k=7 as training, and its
# conditions are standardised with the LUXEMBOURG training statistics. Both are
# required: k had zero variance during training, so its input weight received no
# gradient and is untrained -- feeding a different k would push a non-zero value
# through an essentially random weight. Matching k keeps that dimension at 0.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")

DEFAULT_CKPT="$ROOT/outputs/2026-07-28/09-26-25-cluster_lux/checkpoints/cluster_lux/epoch=59.ckpt"
CKPT="${1:-$DEFAULT_CKPT}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    echo "Available:" >&2
    ls -1 "$ROOT"/outputs/*/*cluster_lux*/checkpoints/cluster_lux/*.ckpt 2>/dev/null >&2 \
        || echo "  (none)" >&2
    exit 1
fi
echo "checkpoint: $CKPT"

# setup.sh sits at $ROOT/scripts/ or $ROOT/ depending on how the tree was
# uploaded. Try both, and say so clearly instead of dying on `set -e`.
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi

# AFTER setup.sh, never before: setup.sh starts with `module purge`, which wipes
# PYTHONPATH. `from src import utils` in main.py needs the project root on the
# path, and src is NOT pip-installed in the BlueBEAR env.
#
# Deliberately not `pip install -e .` -- that would bind `src` to whichever
# folder it was installed from, so one experiment would silently import the
# other's model code. Per-job PYTHONPATH keeps each tree importing its own.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}
cd "$ROOT/src"

# The '=' in epoch=N.ckpt breaks hydra's override grammar unless the value
# carries its own quotes; shell quotes are stripped before hydra sees them.
python3 sample_conditional.py \
    dataset=lux2minn \
    +experiment=cluster_lux \
    general.name=lux2minn \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'"

echo
echo "Done. Output:"
echo "  ls -t $ROOT/outputs/*/*lux2minn*/conditional_samples.txt | head -1"
