#!/bin/bash
#SBATCH --job-name=priv_blocks_x
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_x_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_x_%j.err

# STAGE 2, repeated: one cross block per connected pair of the private release,
# around stage 1's generated internal blocks, N seeded times.
#
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 C C_v3
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3 /abs/epoch=N.ckpt
#
# The first CFG is D_inter's configuration, the second the D_intra one whose
# clusters_<graph>_k<k>_<ICFG>_s<seed>.pt (from D_intra's
# run_private_clusters_repeats.sh) are the context. Seed s reads clusters seed s.
# Output, beside the release in D_intra/private_inference/out/:
#     blocks_<graph>_k<k>_<ICFG>-<CFG>_s<seed>.pt
# scored locally by syn_repeats_summary.py.
#
# CHECKPOINTS are those of the transfer table (scripts/run_sample_pairs_transfer.sh):
#   B  dataset pair_B_lux2olde_k7, the V1 pair_lux model, epoch=179
#   C  dataset pair_C_lux2olde_k7, the newest pair_C_olde run at its recorded best
#
# Submit after stage 1, with --dependency=afterok:<stage 1 job>.
# Oldenburg k=7: 552 pairs, 18 batches, roughly 15-25 min a seed.

set -euo pipefail

REPEATS=1
if [ "${1:-}" = "--repeats" ]; then
    REPEATS="${2:?--repeats needs a count}"
    shift 2
fi
CFG="${1:-}"
ICFG="${2:-}"
CKPT="${3:-}"

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
RELEASE="$(dirname "$ROOT")/D_intra/private_inference/out/release_oldenburg_k7.pt"

if [ -z "$ICFG" ]; then
    echo "usage: sbatch private_inference/run_private_blocks_repeats.sh [--repeats N] {B|C} ICFG [CKPT]" >&2
    exit 1
fi
V1_CKPT="$ROOT/outputs/2026-08-06/16-57-54-pair_lux/checkpoints/pair_lux/epoch=179.ckpt"
case "$CFG" in
    B) DATASET=pair_B_lux2olde_k7; : "${CKPT:=$V1_CKPT}" ;;
    C) DATASET=pair_C_lux2olde_k7 ;;
    *) echo "usage: sbatch private_inference/run_private_blocks_repeats.sh [--repeats N] {B|C} ICFG [CKPT]" >&2
       exit 1 ;;
esac

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}

if [ -z "$CKPT" ]; then
    # C: as run_sample_pairs_transfer.sh -- the newest pair_C_olde run, at the
    # checkpoint its monitored ModelCheckpoint recorded as best.
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-pair_C_olde/ 2>/dev/null | head -1 || true)
    [ -n "$RUNDIR" ] || { echo "ERROR: no pair_C_olde run under $ROOT/outputs" >&2; exit 1; }
    LAST=$(ls -t "${RUNDIR%/}"/checkpoints/pair_C_olde/last*.ckpt 2>/dev/null | head -1 || true)
    CKPT=$(cd "$ROOT/src" && python3 - "$LAST" <<'PY'
import graph_tool  # noqa: F401  before torch.load, see sample_private_blocks.py
import sys, torch
sys.path.insert(0, ".")
ck = torch.load(sys.argv[1], map_location='cpu')
scored = {}
for k, v in ck.get('callbacks', {}).items():
    if 'ModelCheckpoint' in k:
        scored.update({p: float(s) for p, s in (v.get('best_k_models') or {}).items()})
print(min(scored, key=scored.get) if scored else '')
PY
)
    # Lightning stores these paths relative to the run directory.
    [ -n "$CKPT" ] && [ "${CKPT#/}" = "$CKPT" ] && CKPT="${RUNDIR%/}/$CKPT"
    echo "run dir:    $RUNDIR"
fi
for f in "$CKPT" "$RELEASE"; do
    [ -n "$f" ] && [ -f "$f" ] || { echo "ERROR: not found: ${f:-<none>}" >&2; exit 1; }
done

TAG="${ICFG}-${CFG}"
echo "D_inter: $CFG  dataset=$DATASET    context: D_intra $ICFG    repeats=$REPEATS"
echo "checkpoint: $CKPT"
echo "release:    $RELEASE"
echo "output:     blocks_oldenburg_k7_${TAG}_s<seed>.pt"
cd "$ROOT/private_inference"

python3 sample_private_blocks.py \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name="private_blocks_${TAG}" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'" \
    +release="$RELEASE" \
    +clusters_tag="$ICFG" \
    +tag="$TAG" \
    +repeats="$REPEATS"
