#!/usr/bin/env bash
# Every D_inter results table, in one command, with no arguments:
#
#     ./make_tables.sh                          # all of them
#     ./make_tables.sh pair_stats               # just one, or several by name
#     ./make_tables.sh --list                   # what the names are
#     ./make_tables.sh path/to/generated.txt    # score a specific samples file
#
# With no samples file given, the newest generated_samples*.txt is used, looked
# for in the project root and then under outputs/*/*/.
#
# Each table is printed and also written under results/tables/. pair_stats
# additionally leaves its LaTeX, per-pair CSVs and plots there.
#
# The `kmis` env is activated here -- do NOT conda activate first. Override the
# env name with KMIS_ENV=... if yours is called something else.
#
# WHICH PAIRS FILE. The blob must be the one the model trained on: the two on
# disk differ only in the width of y, and pointing at the wrong one compares
# against a different condition in silence. v1 is the default (dataset=pair_lux);
# for v2, PAIRS=data/pair_lux/raw/pairs_inter_luxembourg_osm_k7_v2.pt ./make_tables.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=$ROOT/results/tables
PAIRS=${PAIRS:-$ROOT/data/pair_lux/raw/pairs_inter_luxembourg_osm_k7.pt}
SPLIT=${SPLIT:-test}

ALL="pair_stats degree_w1"

DESC_pair_stats="cross-block structural table vs real, with uniform/zero baselines"
DESC_degree_w1="the out-degree W_1 metric alone, with its three reference points"

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    for s in $ALL; do
        d=DESC_$s
        printf '  %-12s %s\n' "$s" "${!d}"
    done
    exit 0
fi

# A leading path argument is the samples file, not a table name.
GEN=${GEN:-}
if [ $# -gt 0 ] && [ -f "$1" ]; then GEN=$1; shift; fi

# ---- env ------------------------------------------------------------------
# Evaluation is the kmis side of the split (numpy/scipy/torch), not digress.
# LOCALLY that is the `kmis` env. ON BLUEBEAR there is no kmis env, and none is
# needed: these scripts import only numpy/scipy/networkx/torch, no torch_sparse,
# so the `digress` env setup.sh loads runs them and gives identical numbers.
ENV_NAME=${KMIS_ENV:-kmis}
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    hook=""
    for base in "${CONDA_EXE:-/nonexistent}" "$HOME/anaconda3/bin/conda" \
                "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
        h=${base%/bin/conda}/etc/profile.d/conda.sh
        if [ -f "$h" ]; then hook=$h; break; fi
    done
    # conda.sh, conda activate and setup.sh are none of them written to survive
    # `set -u`.
    set +u
    activated=0
    if [ -n "$hook" ]; then
        . "$hook"
        conda activate "$ENV_NAME" 2>/dev/null && activated=1
    fi
    # The cluster path: conda arrives with the modules, so there is no hook to
    # find until setup.sh has run `module load`.
    if [ $activated -eq 0 ]; then
        for s in "$ROOT/scripts/setup.sh" "$ROOT/setup.sh"; do
            if [ -f "$s" ] && command -v module >/dev/null 2>&1; then
                . "$s"; activated=1; break
            fi
        done
    fi
    set -u
    if [ $activated -eq 0 ]; then
        echo "ERROR: could not activate an environment for the tables." >&2
        echo "       Locally: '$ENV_NAME' was not found. Activate it yourself and" >&2
        echo "       re-run, or name the env you want with KMIS_ENV=<name>." >&2
        echo "       On BlueBEAR: source setup.sh first, then re-run." >&2
        exit 1
    fi
fi

# Cheap and worth it: the alternative is a traceback several minutes in, or --
# worse -- a run in whatever python happened to be on PATH.
python -c "import numpy, scipy, networkx, torch" 2>/dev/null || {
    echo "ERROR: numpy/scipy/networkx/torch not importable in $(which python)." >&2
    exit 1
}

# ---- inputs ---------------------------------------------------------------
if [ -z "$GEN" ]; then
    GEN=$(ls -t "$ROOT"/generated_samples*.txt "$ROOT"/outputs/*/*/generated_samples*.txt \
          2>/dev/null | head -1 || true)
fi
if [ -z "$GEN" ] || [ ! -f "$GEN" ]; then
    echo "ERROR: no generated_samples*.txt in $ROOT/ or $ROOT/outputs/*/*/." >&2
    echo "       Sample first (scripts/run_sample_pairs.sh), or pass a path." >&2
    exit 1
fi
[ -f "$PAIRS" ] || { echo "ERROR: pairs file not found: $PAIRS" >&2; exit 1; }

# Absolute from here on: the tables are run from $ROOT, so a relative path given
# from anywhere else would resolve against the wrong directory.
GEN=$(cd "$(dirname "$GEN")" && pwd)/$(basename "$GEN")
PAIRS=$(cd "$(dirname "$PAIRS")" && pwd)/$(basename "$PAIRS")

mkdir -p "$OUT"
printf 'generated: %s  (%s graphs)\npairs:     %s\nsplit:     %s\n' \
    "$GEN" "$(grep -c '^N=' "$GEN")" "$PAIRS" "$SPLIT"

# ---- helpers --------------------------------------------------------------
table () {  # table <name> <description> <command...>
    local name=$1 desc=$2; shift 2
    printf '\n%s\n%s -- %s\n%s\n' \
        "==================================================================" \
        "$name" "$desc" \
        "=================================================================="
    ( cd "$ROOT" && "$@" ) | tee "$OUT/$name.txt"
    printf '   -> results/tables/%s.txt\n' "$name"
}

run_section () {
case $1 in
pair_stats)
    # --plots needs a WORKING matplotlib; without one pair_stats.py dies at the
    # very end, after the table has already been written, so ask for plots only
    # if it can actually be imported.
    #
    # Probe in the SAME ORDER pair_stats.py imports. This failure is load-order
    # dependent: on BlueBEAR, torch pulls in the system libstdc++ first, and PIL
    # (reached through matplotlib.colors) then wants a newer GLIBCXX than that
    # copy has. A bare `import matplotlib` in a fresh process gets the conda
    # libstdc++ and succeeds -- which is exactly how the crash got past the
    # first version of this check.
    local plots=()
    python -c "import numpy, torch, scipy
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot" 2>/dev/null \
        && plots=(--plots) \
        || echo "   note: matplotlib unusable here -- table and CSVs only, no plots"
    table pair_stats "$DESC_pair_stats" \
        python evaluation/pair_stats.py --generated "$GEN" --pairs "$PAIRS" \
            --split "$SPLIT" --out-dir "$OUT/pair_stats" "${plots[@]+"${plots[@]}"}"
    printf '   -> results/tables/pair_stats/{stats_table.tex,stats_table_simple.tex,*.csv}\n' ;;
degree_w1)
    table degree_w1 "$DESC_degree_w1" \
        python evaluation/evaluate_pairs.py --generated "$GEN" --pairs "$PAIRS" \
            --split "$SPLIT" ;;
*)
    echo "unknown table '$1' -- try: $ALL" >&2; exit 2 ;;
esac
}

sections=("$@")
[ "${#sections[@]}" -eq 0 ] && sections=($ALL)
for s in "${sections[@]}"; do run_section "$s"; done

printf '\nall tables in %s\n' "$OUT"
