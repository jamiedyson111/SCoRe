#!/bin/bash
#SBATCH --job-name=eval_clusters
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/eval_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/eval_%j.err

# Compare the generated cluster subgraphs against the real held-out ones.
#
#   sbatch scripts/run_evaluate.sh                      # newest conditional_samples.txt
#   sbatch scripts/run_evaluate.sh /abs/path/to/conditional_samples.txt
#
# No GPU: this is numpy/networkx only, a couple of minutes single-core. bbshort
# so it backfills into the queue straight away.

set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
PAIRS="$ROOT/data/cluster_lux/raw/pairs_luxembourg_osm_k7.pt"

GEN="${1:-}"
if [ -z "$GEN" ]; then
    GEN=$(ls -t "$ROOT"/outputs/*/*cond_sample*/conditional_samples.txt 2>/dev/null | head -1 || true)
    if [ -z "$GEN" ]; then
        echo "ERROR: no conditional_samples.txt found under $ROOT/outputs/*/*cond_sample*/" >&2
        echo "Run scripts/run_sample_conditional.sh first, or pass a path explicitly." >&2
        exit 1
    fi
fi

for f in "$GEN" "$PAIRS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: not found: $f" >&2
        exit 1
    fi
done

echo "generated: $GEN"
echo "pairs:     $PAIRS"
echo "graphs in file: $(grep -c '^N=' "$GEN")"
echo

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
cd "$ROOT"

python3 evaluation/evaluate_clusters.py --generated "$GEN" --pairs "$PAIRS"
