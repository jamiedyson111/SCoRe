#!/bin/bash
#SBATCH --job-name=pair_debug
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_debug_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_debug_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# Submitting from anywhere else -- your home directory, say -- leaves ROOT
# pointing at something unrelated (/rds/homes/d) and every later step fails
# with a confusing message. Fall back to the known install rather than guess.
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
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

# Smoke test for the conditional D_inter model. No GPU on purpose -- fast_dev_run
# is one train batch and one val batch, seconds of CPU, and a CPU job backfills
# into the queue immediately instead of waiting for a GPU.
#
# THE LINES TO CHECK. All four confirmed on BlueBEAR under V2 on 2026-08-18,
# and the three below the dims line are identical under V1.
#
#   Size of the input features 8 2 18      <- V1, the default
#                              8 2 23      <- V2, i.e. DATASET=pair_lux_v2
#       8  = 2 side classes + 6 extra features
#       2  = edge classes
#       18 = 6 condition + 1 time + 11 spectral;  23 = 11 + 1 + 11.
#       Only the condition term moves, so this line is how you confirm WHICH
#       condition you are actually training on -- check it against $DATASET.
#       If it says 23 under pair_lux you are reading the V2 raw file, and vice
#       versa: the two raw files differ only in the width of y, and nothing else
#       in the run will tell you which one loaded.
#       If it says 12, the condition is NOT reaching the model at all and you
#       would be training an unconditional model with no error raised.
#
#   node marginal  tensor([0.5002, 0.4998])
#       The two sides of a pair. Must be ~50/50 -- a skew here means the
#       orientation randomisation in make_pair_blocks.py did not take effect.
#
#   edge marginal  tensor([0.9955, 0.0045])
#       CROSS BLOCK ONLY. If this comes out nearer [0.97, 0.03] it is the
#       whole-graph marginal, meaning PairGraphDataModule.edge_counts is not
#       being used -- the limit distribution would then be wrong and sampling
#       would degrade while training still looked fine.
#
#   Val Atom type KL 0.00
#       X is observed, not generated, so it must contribute nothing to the NLL.
#       Anything nonzero means the cross restriction is not being applied.
#
# Also expect:
#   Val cross-block CE   ~0.24 at init, falling      <- the number that matters
#   X_CE                 small and falling to ~0     trivial: X_t == X
#   E_CE                 cross block only, ~0.4 at init

# V1 by default. For the 11-dim V2 condition:  DATASET=pair_lux_v2 sbatch <this>
DATASET="${DATASET:-pair_lux}"

python3 main.py \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name=debug \
    general.gpus=0 \
    general.wandb=disabled
