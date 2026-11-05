#!/bin/bash
#SBATCH --job-name=pair_lux
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_lux_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_lux_%j.err

set -euo pipefail

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

# Conditional inter-cluster model D_inter.
# 8176 train / 1635 val / 1091 test supernode PAIRS from Luxembourg at k=7.
# Condition c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k], standardised on train (V1,
# 6 dims -- the default, and what the private-inference releases supply).
# DATASET=pair_lux_v2 switches to the 11-dim V2 condition, which appends the
# r_{i->j} out-degree histogram raw; read configs/dataset/pair_lux_v2.yaml first,
# it is nearly redundant at k=7 and blocks private inference until stage 0
# publishes r.
#
# Only the cross block B_ij is generated: L_i, L_j and the sides are observed
# context, held fixed through training and through every reverse step.
#
# Longer wall clock than D_intra: combined graphs are ~2x the nodes, so E is ~4x
# the entries, and there are more examples.
#
# Run scripts/run_pair_debug.sh first and confirm all four checks listed in it.
#
# WATCH val/cross_CE, not val/epoch_NLL. And read the W_out-degree line against
# the uniform-placement baseline printed beside it -- the cross block averages
# ~1.1 edges in ~240 slots, so predicting "no edge" everywhere already scores a
# low CE without having learned anything about WHERE the edges go.

# Dataset and run name are ARGUMENTS:
#
#   sbatch scripts/run_pair.sh                                   # pair_lux (V1) = A
#   sbatch scripts/run_pair.sh pair_C_lux2olde_k7 pair_C_olde    # transfer C
#   sbatch scripts/run_pair.sh pair_lux_v2                       # the V2 condition
#
# Not environment variables: BlueBEAR's job-submit plugin does not carry the
# submitting shell's environment into the job, so `DATASET=... sbatch` silently
# trained on the default. The env var is still read as a fallback for runs
# launched some other way. NAME sets the run directory and checkpoint folder.
#
# FINE-TUNING (config C, the internal-degree D_inter). A third argument starts
# from another model's weights instead of from scratch:
#
#   sbatch scripts/run_pair.sh pair_C_lux2olde_k7_s80_deg pair_C_s80_deg pair_lux_s80_deg
#
# It is a checkpoint file, or a RUN NAME resolved here, when the job starts, to
# that run's newest last*.ckpt -- so C can be queued behind A with afterok before
# A's checkpoint exists. Fine-tuning runs FT_EPOCHS epochs at FT_LR (A: 180 epochs
# at 2e-4 over 6904 pairs = ~39k steps; C: 100 epochs over 362 pairs = ~1.2k steps
# at a quarter of the rate, so the Oldenburg pairs adapt the model without
# overwriting it). No validation split, so the final checkpoint is the C model.
DATASET="${1:-${DATASET:-pair_lux}}"
NAME="${2:-pair_lux}"
FINETUNE="${3:-}"
FT_EPOCHS=100
FT_LR=0.00005
echo "dataset=$DATASET  name=$NAME"

EXTRA=()
if [ -n "$FINETUNE" ]; then
    CKPT="$FINETUNE"
    if [ ! -f "$CKPT" ]; then
        RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$FINETUNE"/ 2>/dev/null | head -1 || true)
        CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$FINETUNE"/last*.ckpt 2>/dev/null | head -1 || true)
    fi
    [ -n "$CKPT" ] && [ -f "$CKPT" ] || {
        echo "ERROR: no checkpoint for '$FINETUNE' (a file, or a run name under $ROOT/outputs)" >&2
        exit 1; }
    echo "fine-tuning from $CKPT: $FT_EPOCHS epochs at lr $FT_LR"
    EXTRA=("general.finetune_from='$CKPT'" "train.n_epochs=$FT_EPOCHS" "train.lr=$FT_LR")
fi

python3 main.py \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name="$NAME" \
    general.wandb=disabled \
    ${EXTRA[@]+"${EXTRA[@]}"}
