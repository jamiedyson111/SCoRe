#!/bin/bash
#SBATCH --job-name=cluster_lux
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cluster_lux_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cluster_lux_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_intra}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
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

# Conditional intra-cluster model D_intra.
# 6611 train / 1322 val / 882 test supernodes from Luxembourg at k=7, median 11
# nodes. Condition c_i = [n, rho_in, k], standardised on the train split.
#
# --gres=gpu:1 with no a100 pin: these subgraphs are tiny (max 71 nodes), so any
# GPU on the cluster is ample and a generic request queues far sooner.
#
# Run scripts/run_cluster_debug.sh first and confirm "input features 7 2 15".
python3 main.py \
    dataset=cluster_lux \
    +experiment=cluster_lux \
    general.name=cluster_lux \
    general.wandb=disabled
