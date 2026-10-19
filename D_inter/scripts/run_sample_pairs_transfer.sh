#!/bin/bash
#SBATCH --job-name=pair_transfer
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_transfer_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_transfer_%j.err

# One cross block per TEST pair of a transfer configuration, N seeded times.
#
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 B
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 C
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 A
#   sbatch scripts/run_sample_pairs_transfer.sh B /abs/path/epoch=N.ckpt    # explicit ckpt
#
#   A  dataset pair_lux             test: 1091 Luxembourg pairs
#   B  dataset pair_B_lux2olde_k7   test:  276 Oldenburg pairs
#   C  dataset pair_C_lux2olde_k7   test:  276 Oldenburg pairs (the same ones)
#
# CHECKPOINTS. A and B share one model: B's training data is pair_lux's, so the
# V1 pair_lux checkpoint is the B model too. It is epoch=179 of the 2026-08-06
# run, the val/cross_CE minimum (configs/experiment/pair_lux.yaml). C is the
# newest pair_C_olde run, at the checkpoint its ModelCheckpoint recorded as best
# (lowest val/epoch_NLL), read from last*.ckpt rather than guessed from names.
#
# --repeats is an ARGUMENT, not an env var: BlueBEAR does not pass the submitting
# shell's environment into the job. Output: outputs/<date>/<time>-pairs_<CFG>_
# sample_x<N>/pair_samples_s<seed>.pt, scored by evaluation/transfer_summary.py.
#
# Runtime: about 1 min per batch of 32 pairs per seed. B or C: 9 batches, ~10-15
# min a seed. A: 35 smaller batches, ~25-35 min a seed. 4h covers A at 5 seeds;
# run_pairs_repeats.sh asks for 2h for B and C.

set -euo pipefail

REPEATS=1
if [ "${1:-}" = "--repeats" ]; then
    REPEATS="${2:?--repeats needs a count}"
    shift 2
fi
CFG="${1:-}"
CKPT="${2:-}"

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter

V1_CKPT="$ROOT/outputs/2026-08-06/16-57-54-pair_lux/checkpoints/pair_lux/epoch=179.ckpt"
case "$CFG" in
    A) DATASET=pair_lux;           : "${CKPT:=$V1_CKPT}" ;;
    B) DATASET=pair_B_lux2olde_k7; : "${CKPT:=$V1_CKPT}" ;;
    C) DATASET=pair_C_lux2olde_k7 ;;
    *) echo "usage: sbatch scripts/run_sample_pairs_transfer.sh [--repeats N] {A|B|C} [CKPT]" >&2
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
    # C: the newest pair_C_olde run, at the best checkpoint it recorded itself.
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-pair_C_olde/ 2>/dev/null | head -1 || true)
    if [ -z "$RUNDIR" ]; then
        echo "ERROR: no pair_C_olde run under $ROOT/outputs -- train C first:" >&2
        echo "       sbatch scripts/run_pair.sh pair_C_lux2olde_k7 pair_C_olde" >&2
        exit 1
    fi
    LAST=$(ls -t "${RUNDIR%/}"/checkpoints/pair_C_olde/last*.ckpt 2>/dev/null | head -1 || true)
    # Unpickling a checkpoint imports the model code (src/analysis/...), so this
    # runs from src/ with graph_tool first, exactly as the samplers do. main.py
    # has two ModelCheckpoint callbacks; only the monitored one has scores, and
    # its best_k_models is read directly so a missing best_model_path cannot
    # silently select nothing.
    CKPT=$(cd "$ROOT/src" && python3 - "$LAST" <<'EOF'
import graph_tool  # noqa: F401  before torch.load, see sample_private_blocks.py
import sys, torch
sys.path.insert(0, ".")
ck = torch.load(sys.argv[1], map_location='cpu')
scored = {}
for k, v in ck.get('callbacks', {}).items():
    if 'ModelCheckpoint' in k:
        scored.update({p: float(s) for p, s in (v.get('best_k_models') or {}).items()})
print(min(scored, key=scored.get) if scored else '')
EOF
)
    # Lightning stores these paths relative to the run directory.
    [ -n "$CKPT" ] && [ "${CKPT#/}" = "$CKPT" ] && CKPT="${RUNDIR%/}/$CKPT"
    echo "run dir:    $RUNDIR"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: ${CKPT:-<none>}" >&2
    exit 1
fi

echo "configuration: $CFG  dataset=$DATASET  repeats=$REPEATS"
echo "checkpoint:    $CKPT"
cd "$ROOT/src"

# The '=' in epoch=N.ckpt needs hydra's own quotes inside the shell's.
python3 sample_pairs_conditional.py \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name="pairs_${CFG}_sample_x${REPEATS}" \
    general.gpus=1 \
    general.wandb=disabled \
    +repeats="$REPEATS" \
    "general.test_only='$CKPT'"

echo
echo "Done. Output:"
echo "  ls -td $ROOT/outputs/*/*-pairs_${CFG}_sample_x${REPEATS}/ | head -1"
