#!/bin/bash
#SBATCH --job-name=cluster_debug
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cluster_debug_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cluster_debug_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_intra}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
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

# Smoke test for the conditional D_intra model. No GPU requested on purpose --
# fast_dev_run is one train batch and one val batch, which is seconds of CPU, and
# a CPU job backfills into the queue immediately instead of waiting for a GPU.
#
# THE LINE TO CHECK:  "Size of the input features 7 2 15"
#
#   15 = 3 condition + 1 time + 11 spectral.
#   If it says 12, the condition is NOT reaching the model and you would be
#   training an unconditional model without any error being raised. Stop and
#   check that data/cluster_lux/raw/pairs_luxembourg_osm_k7.pt uploaded correctly.
#
# Also expect:
#   node marginal  tensor([1.])            one dummy node class
#   edge marginal  [0.8768, 0.1232]        sparse, tree-like subgraphs
#                                          (was [0.9158, 0.0842] before the
#                                          2026-08-06 k-MIS fix, when 15.8% of
#                                          clusters were disconnected)
#   X_CE           0.000                   nothing to learn with one node class
#   E_CE           around 0.3, falling     the only loss that means anything here
python3 main.py \
    dataset=cluster_lux \
    +experiment=cluster_lux \
    general.name=debug \
    general.gpus=0 \
    general.wandb=disabled
