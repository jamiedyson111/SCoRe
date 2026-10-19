#!/bin/bash
#SBATCH --job-name=coarse_debug
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/coarse_debug_%j.out
#SBATCH --error=logs/coarse_debug_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

# --- Environment ---
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
# --- Run from src/ so Hydra finds ../configs ---
cd "$ROOT/src"

# Smoke test for the K-MIS coarse-graph dataset. Runs ONE train batch and ONE
# val batch, then stops -- about a minute. Confirms the uploaded data loads and
# the dimensions line up, before committing a long allocation.
#
# BOTH overrides are required, same as planar:
#   dataset=coarse_sbm      -> selects the coarse_sbm branch in main.py.
#                              Without it, main.py falls back to config.yaml's
#                              qm9 default and silently trains on molecules.
#   +experiment=coarse_sbm  -> n_layers=6, diffusion_steps=500, batch_size=32,
#                              extra_features='all', lambda_train=[5,0].
#
# general.name=debug is what triggers fast_dev_run. It also forces batch_size=2
# and SKIPS the sampling/test phase -- that is expected for a smoke test.
#
# Sanity check the .out:
#   "Size of the input features"  ->  7 2 12
#   X_CE                          ->  0.000   (one dummy node class: nothing to learn)
#   E_CE                          ->  around 0.5
#   node marginal ~[1.], edge marginal 2 values
# If you see 4-node/5-edge marginals or "Dataset smiles were found", it is on
# QM9 -- kill it, the dataset= override did not take.
python3 main.py \
    dataset=coarse_sbm \
    +experiment=coarse_sbm \
    general.name=debug \
    general.wandb=offline \
    general.gpus=0