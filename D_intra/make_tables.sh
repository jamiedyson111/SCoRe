#!/usr/bin/env bash
# Every D_intra results table, in one command, with no arguments:
#
#     ./make_tables.sh                    # all of them
#     ./make_tables.sh intra_k7           # just one, or several by name
#     ./make_tables.sh --list             # what the names are
#
# Each table is printed and also written to results/tables/oldenburg/<name>.txt,
# with its LaTeX block split out into <name>.tex ready to \input. Samples are
# read from results/samples/oldenburg/ (run_olde_transfer.sh says how to fill
# it). The Minnesota tables and samples one level up are left as they were:
# supernode ids are reused across graphs, so a Minnesota samples file would
# match Oldenburg test records by id and be scored against the wrong clusters.
#
# The `kmis` env is activated here -- do NOT conda activate first. Override the
# env name with KMIS_ENV=... if yours is called something else.
#
# A table whose samples are missing is skipped with a note rather than failing,
# so this is safe to run on a partial set of results.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SAMPLES=$ROOT/results/samples/oldenburg
PAIRS_DIR=$ROOT/data/cluster_lux/raw
OUT=$ROOT/results/tables/oldenburg
TAG=lux2olde_k7

ALL="intra_k7 transfer_k7 ablation_A"

DESC_intra_k7="structural stats, Oldenburg k=7 test split, B/C x hops/no-hops"
DESC_transfer_k7="realism / transfer / condition tables, A B C at k=7"
DESC_ablation_A="paired hop-histogram ablation on configuration A"

if [ "${1:-}" = "--list" ] || [ "${1:-}" = "-l" ]; then
    for s in $ALL; do
        d=DESC_$s
        printf '  %-12s %s\n' "$s" "${!d}"
    done
    exit 0
fi

# ---- env ------------------------------------------------------------------
# Evaluation is the kmis side of the split (torch 1.11 + scipy); the digress env
# cannot import evaluate_clusters' dependencies.
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

mkdir -p "$OUT"

# ---- helpers --------------------------------------------------------------
have () {   # every named samples file present?
    for f in "$@"; do [ -f "$SAMPLES/$f" ] || return 1; done
}

skip () { printf '\n-- %s: SKIPPED, missing %s\n' "$1" "$2"; }

table () {  # table <name> <description> <command...>
    local name=$1 desc=$2; shift 2
    printf '\n%s\n%s -- %s\n%s\n' \
        "==================================================================" \
        "$name" "$desc" \
        "=================================================================="
    # cd: make_intra_table.py and friends import evaluate_clusters as a sibling.
    ( cd "$ROOT/evaluation" && "$@" ) | tee "$OUT/$name.txt"
    # Every script prints its LaTeX last, behind a '% ---- LaTeX' banner.
    awk '/^% ---- LaTeX/{f=1} f' "$OUT/$name.txt" > "$OUT/$name.tex"
    [ -s "$OUT/$name.tex" ] || rm -f "$OUT/$name.tex"
    printf '   -> results/tables/oldenburg/%s.txt%s\n' "$name" \
        "$([ -f "$OUT/$name.tex" ] && echo ", results/tables/oldenburg/$name.tex")"
}

run_section () {
case $1 in
intra_k7)
    have B.txt B_nohop.txt C.txt C_nohop.txt \
        || { skip intra_k7 "results/samples/oldenburg/{B,C}{,_nohop}.txt"; return; }
    table intra_k7 "$DESC_intra_k7" \
        python make_intra_table.py --tag $TAG --pairs-dir "$PAIRS_DIR" \
            --b "$SAMPLES/B.txt"             --b-nohop "$SAMPLES/B_nohop.txt" \
            --c "$SAMPLES/C.txt"             --c-nohop "$SAMPLES/C_nohop.txt" \
            --latex ;;
transfer_k7)
    have A.txt B.txt C.txt \
        || { skip transfer_k7 "results/samples/oldenburg/{A,B,C}.txt"; return; }
    table transfer_k7 "$DESC_transfer_k7" \
        python compare_transfer.py --tag $TAG --pairs-dir "$PAIRS_DIR" \
            --a "$SAMPLES/A.txt" --b "$SAMPLES/B.txt" --c "$SAMPLES/C.txt" \
            --latex ;;
ablation_A)
    # A, not B/C: 882 held-out supernodes against 160, and the ablation needs the
    # power. See the module docstring of compare_ablation.py.
    have A.txt A_nohop.txt \
        || { skip ablation_A "results/samples/oldenburg/A{,_nohop}.txt"; return; }
    table ablation_A "$DESC_ablation_A" \
        python compare_ablation.py --cfg A --tag $TAG \
            --pairs-dir "$PAIRS_DIR" \
            --gen-full "$SAMPLES/A.txt" --gen-abl "$SAMPLES/A_nohop.txt" \
            --latex ;;
*)
    echo "unknown table '$1' -- try: $ALL" >&2; exit 2 ;;
esac
}

sections=("$@")
[ "${#sections[@]}" -eq 0 ] && sections=($ALL)
for s in "${sections[@]}"; do run_section "$s"; done

printf '\nall tables in %s\n' "$OUT"
