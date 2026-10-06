#!/bin/bash
#SBATCH --job-name=priv_blocks
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_%j.err

# STAGE 2 of private inference: generate one cross block per connected pair.
#
#   sbatch private_inference/run_private_blocks.sh CKPT RELEASE CLUSTERS [OUT]
#
# CLUSTERS is stage 1's output. Passing clusters_<graph>_k<k>_true.pt instead --
# the real internal blocks stage 0 writes next to the release -- makes this the
# validation run: it isolates this stage from the D_intra checkpoint, so a bad
# number points here rather than at stage 1.
#
# Runtime scales with the number of connected pairs times the 500 reverse steps.
# Oldenburg k=7 has 552 pairs (~18 batches); Luxembourg has 10902 (~341 batches),
# which is why the wall clock is 4h rather than the 2h used for test sampling.

set -euo pipefail

CKPT="${1:-}"
RELEASE="${2:-}"
THIRD="${3:-}"
OUT="${4:-}"
# Optional 5th argument: score only N randomly chosen pairs. Validation only --
# the omitted pairs would be empty blocks if the result were assembled.
LIMIT="${5:-}"

if [ -z "$CKPT" ] || [ -z "$RELEASE" ] || [ -z "$THIRD" ]; then
    echo "usage: sbatch private_inference/run_private_blocks.sh \\" >&2
    echo "           /abs/path/to/epoch=NNN.ckpt /abs/path/to/release_<g>_k<k>.pt \\" >&2
    echo "           /abs/path/to/clusters_<g>_k<k>{,_true}.pt [OUT.pt]" >&2
    exit 1
fi
for f in "$CKPT" "$RELEASE"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f" >&2; exit 1; }
done

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
# Submitting from inside a subdirectory would make SLURM_SUBMIT_DIR that
# subdirectory, not the project root. Step up one level if src/ is not there.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# Submitting from anywhere else -- your home directory, say -- leaves ROOT
# pointing at something unrelated and every later step fails confusingly.
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi

# AFTER setup.sh, never before: setup.sh starts with `module purge`, which wipes
# PYTHONPATH.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}

[ -f "$THIRD" ] || { echo "ERROR: clusters file not found: $THIRD" >&2; exit 1; }
CONTEXT_ARGS="+clusters=$THIRD"
echo "context: $THIRD"
[ -n "$OUT" ] && CONTEXT_ARGS="$CONTEXT_ARGS +out=$OUT"
[ -n "$LIMIT" ] && CONTEXT_ARGS="$CONTEXT_ARGS +limit=$LIMIT"

echo "checkpoint: $CKPT"
echo "release:    $RELEASE"
echo

cd "$ROOT/private_inference"

# shellcheck disable=SC2086
# V1 by default: the releases publish the 6-name condition, so a V2 checkpoint
# needs DATASET=pair_lux_v2 *and* a release that publishes r, or the cond_names
# assert in sample_private_blocks.py fires.
DATASET="${DATASET:-pair_lux}"

python3 sample_private_blocks.py \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name=private_blocks \
    general.wandb=disabled \
    general.test_only="'$CKPT'" \
    +release="$RELEASE" \
    $CONTEXT_ARGS
# The inner single quotes on test_only are LITERAL and reach Hydra, which strips
# them itself. Without them Hydra's override grammar hits the second '=' in
# "epoch=179.ckpt" and dies with "mismatched input '=' expecting <EOF>" -- shell
# quoting alone does not help, the problem is on Hydra's side of argv.
