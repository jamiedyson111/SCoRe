#!/bin/bash
#SBATCH --job-name=digress_planar
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/planar_%j.out
#SBATCH --error=logs/planar_%j.err
 
set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# --- Memory fragmentation guard (same fix that resolved the SBM OOM) ---
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
 
# --- Flush stdout to the .out file promptly instead of block-buffering ---
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
 
# Reproduce the DiGress paper's planar setup.
#
# BOTH overrides are required:
#   dataset=planar      -> selects the untyped SPECTRE/planar branch in main.py
#                          (main.py dispatches on dataset.name; WITHOUT this it
#                          falls back to the config.yaml default of qm9 and
#                          silently trains on molecules).
#   +experiment=planar  -> applies the paper hyperparameters:
#                          n_layers=10, diffusion_steps=1000, batch_size=64,
#                          extra_features='all', lambda_train=[5,0], the paper's
#                          hidden_mlp_dims / hidden_dims, check_val_every_n_epochs=100,
#                          sample_every_val=4, final_model_samples_to_generate=40.
#                          (experiment/planar.yaml does NOT set the dataset, so
#                           it cannot switch it on its own.)
#
# We override ONLY wandb (offline for BlueBEAR) and the run name.
#
# Sanity check the .out once it starts: node marginal should be ~[1.] and the
# edge marginal 2 values. If you see 4-node/5-edge marginals or "Dataset smiles
# were found", it's on QM9 again -- kill it.
#
# If this OOMs on the 40GB A100, add:  train.batch_size=32
# (batch size does not change the reported Deg/Clus MMDs, only speed/memory.)
python3 main.py \
    dataset=planar \
    +experiment=planar \
    general.name=planar_repro \
    general.wandb=offline
 