#!/bin/bash
#SBATCH --job-name=priv_clusters
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/priv_clusters_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/priv_clusters_%j.err

# STAGE 1 of private inference: generate one internal block per supernode.
#
#   sbatch private_inference/run_private_clusters.sh CKPT RELEASE [DATASET] [OUT]
#
# DATASET is the hydra dataset name whose blob carries the standardisation the
# checkpoint was trained under. It must be the dataset the checkpoint was
# actually trained on, or the conditions are scaled wrongly.
#
# The default is cluster_B_lux2olde_k7 -- configuration B, trained on Luxembourg
# and applied to Oldenburg, which is the public -> private direction this stage
# exists to demonstrate. It used to be cluster_lux; that is now WRONG for any
# current checkpoint. cluster_lux is the legacy 3-dim condition [n, rho_in, k],
# while A/B/C carry the 10-dim [n, rho_in, k, h1..h7] added on 2026-08-06, and
# its pairs file also predates the k-MIS assignment fix.
#
# Mismatching the two is at least loud: the model's y-MLP expects width 22 and a
# cluster_lux blob supplies 15, so it dies with a shape error on the first batch.
# Use configuration C's dataset instead only if you deliberately want the model
# that saw target-graph training data -- configuration C rather than B.
#
# Runtime scales with the supernode count times the 500 reverse steps.
# Oldenburg k=7 has 317 supernodes (~10 batches); Luxembourg has 8815 (~276).

set -euo pipefail

CKPT="${1:-}"
RELEASE="${2:-}"
DATASET="${3:-cluster_B_lux2olde_k7}"
OUT="${4:-}"

if [ -z "$CKPT" ] || [ -z "$RELEASE" ]; then
    echo "usage: sbatch private_inference/run_private_clusters.sh \\" >&2
    echo "           /abs/path/to/epoch=NNN.ckpt /abs/path/to/release_<g>_k<k>.pt \\" >&2
    echo "           [dataset_name] [OUT.pt]" >&2
    exit 1
fi
for f in "$CKPT" "$RELEASE"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f" >&2; exit 1; }
done

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_intra}"
# Submitting from inside a subdirectory would make SLURM_SUBMIT_DIR that
# subdirectory, not the project root. Step up one level if src/ is not there.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# Submitting from anywhere else -- your home directory, say -- leaves ROOT
# pointing at something unrelated and every later step fails confusingly.
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
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

EXTRA=""
[ -n "$OUT" ] && EXTRA="+out=$OUT"

echo "checkpoint: $CKPT"
echo "release:    $RELEASE"
echo "dataset:    $DATASET"
echo

cd "$ROOT/private_inference"

# shellcheck disable=SC2086
python3 sample_private_clusters.py \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    general.name=private_clusters \
    general.wandb=disabled \
    general.test_only="'$CKPT'" \
    +release="$RELEASE" \
    $EXTRA
# The inner single quotes on test_only are LITERAL and reach Hydra, which strips
# them itself. Without them Hydra's override grammar hits the second '=' in
# "epoch=NNN.ckpt" and dies with "mismatched input '=' expecting <EOF>".
