#!/bin/bash
#SBATCH --job-name=eval_pairs
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbshort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/eval_pairs_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/eval_pairs_%j.err

# Score generated cross blocks against the real held-out ones, independently of
# the in-training metric.
#
#   sbatch scripts/run_evaluate_pairs.sh                        # newest generated_samples*.txt
#   sbatch scripts/run_evaluate_pairs.sh /abs/path/to/generated_samples1.txt
#
# Reports W_1 over out-degree with TWO baselines beside it. Read them: the cross
# block averages ~1.1 edges in ~240 slots, so uniform random placement already
# scores well on out-degree and emitting nothing scores badly. W_out-degree here
# is mostly a check on edge COUNT calibration, not on placement.
#
# No GPU: numpy/networkx only, a couple of minutes single-core.

set -euo pipefail

export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
# Submitting from anywhere else -- your home directory, say -- leaves ROOT
# pointing at something unrelated (/rds/homes/d) and every later step fails
# with a confusing message. Fall back to the known install rather than guess.
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
# Must be the same blob the model trained on -- evaluate_pairs.py takes its
# reference test records straight from blob['splits'][2]. Both files are on disk
# and differ only in the width of y, so pointing at the wrong one compares
# against a different condition in silence.
#   V1 (default):  matches dataset=pair_lux
#   V2:            PAIRS=.../pairs_inter_luxembourg_osm_k7_v2.pt sbatch <this>
PAIRS="${PAIRS:-$ROOT/data/pair_lux/raw/pairs_inter_luxembourg_osm_k7.pt}"

GEN="${1:-}"
if [ -z "$GEN" ]; then
    GEN=$(ls -t "$ROOT"/outputs/*/*/generated_samples*.txt 2>/dev/null | head -1 || true)
    if [ -z "$GEN" ]; then
        echo "ERROR: no generated_samples*.txt found under $ROOT/outputs/*/*/" >&2
        echo "Run scripts/run_sample_pairs.sh first, or pass a path explicitly." >&2
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

if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi

# AFTER setup.sh, never before: setup.sh starts with `module purge`.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}
cd "$ROOT"

python3 evaluation/evaluate_pairs.py --generated "$GEN" --pairs "$PAIRS" --split test
