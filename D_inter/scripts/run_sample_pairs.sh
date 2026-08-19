#!/bin/bash
#SBATCH --job-name=pair_sample
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_sample_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_sample_%j.err

# Sample cross blocks from a trained D_inter checkpoint, on the TEST split.
#
#   sbatch scripts/run_sample_pairs.sh /abs/path/to/epoch=179.ckpt
#
# Pass the checkpoint that minimised val/cross_CE, not last.ckpt. The pair_lux
# run overfits: train E_CE keeps falling long after val/cross_CE has turned
# around, so a late checkpoint is a worse model that looks better on the
# training loss.
#
# Writes generated_samples1.txt into the ORIGINAL run directory (main.py chdirs
# to whatever sits above `checkpoints/` in the path you give it), and prints:
#
#   Test cross-block CE          <- the number to quote
#   [test] W_out-degree ...      <- with its uniform-placement baseline beside it

set -euo pipefail

CKPT="${1:-}"
if [ -z "$CKPT" ]; then
    echo "usage: sbatch scripts/run_sample_pairs.sh /abs/path/to/epoch=NNN.ckpt" >&2
    exit 1
fi
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT" >&2
    exit 1
fi

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# Submitting from anywhere else -- your home directory, say -- leaves ROOT
# pointing at something unrelated (/rds/homes/d) and every later step fails
# with a confusing message. Fall back to the known install rather than guess.
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

# setup.sh sits at $ROOT/scripts/ or $ROOT/ depending on how the tree was
# uploaded. Try both, and say so clearly instead of dying on `set -e`.
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi

# AFTER setup.sh, never before: setup.sh starts with `module purge`, which wipes
# PYTHONPATH. See the note in run_pair.sh.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}
cd "$ROOT/src"

echo "checkpoint: $CKPT"
echo

python3 main.py \
    dataset=pair_lux \
    +experiment=pair_lux \
    general.name=pair_lux_test \
    general.wandb=disabled \
    general.test_only="'$CKPT'"
# The inner single quotes are LITERAL and reach Hydra, which strips them itself.
# Without them Hydra's override grammar hits the second '=' in "epoch=179.ckpt"
# and dies with "mismatched input '=' expecting <EOF>" -- shell quoting alone
# does not help, since the problem is on Hydra's side of argv.
