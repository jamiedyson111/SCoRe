#!/bin/bash
# Push the shared DiGress code into each experiment folder, so that D_intra/ and
# D_inter/ are each a COMPLETE, RUNNABLE tree you can upload as-is:
#
#     ./sync.sh                 -> updates both
#     ./sync.sh D_inter         -> updates one
#
#     rsync -avz D_inter/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_inter/
#
# There is no _upload/ staging step any more. The folder you edit is the folder
# you upload.
#
# THE ONE RULE: the shared model code has a single source of truth in
# DiGress_upload/. The copies inside D_intra/ and D_inter/ are generated, and
# this script OVERWRITES them. Edit shared code in DiGress_upload/ and re-run
# this; edit experiment code (datasets/*_dataset.py, the pair_lux and cluster_lux
# configs, data/, scripts/, evaluation/, dataprep/) directly in its own folder,
# where nothing will touch it.
#
# Run with --check to see whether any generated copy has drifted from the master
# before it gets overwritten.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
SHARED=$ROOT/DiGress_upload

CHECK=0
EXPS=()
for arg in "$@"; do
  case "$arg" in
    --check)         CHECK=1 ;;
    D_intra|D_inter) EXPS+=("$arg") ;;
    *) echo "usage: ./sync.sh [--check] [D_intra|D_inter]" >&2; exit 1 ;;
  esac
done
[ ${#EXPS[@]} -eq 0 ] && EXPS=(D_intra D_inter)

# README.md is the experiment's own and must survive. run_*.sh and setup.sh are
# placed into scripts/ below rather than at the root, so every script lives in
# one place.
EX="--exclude=__pycache__ --exclude=*.pyc --exclude=.gitkeep --exclude=README.md
    --exclude=run_*.sh --exclude=setup.sh --exclude=logs/"

for EXP in "${EXPS[@]}"; do
  OUT=$ROOT/$EXP
  [ -d "$OUT" ] || { echo "no such experiment folder: $OUT" >&2; exit 1; }

  if [ $CHECK -eq 1 ]; then
    echo "=== $EXP: generated files that differ from DiGress_upload/ ==="
    diff -rq $EX "$SHARED/src" "$OUT/src" 2>/dev/null \
      | grep -v "^Only in $OUT" || true
    diff -rq $EX "$SHARED/configs" "$OUT/configs" 2>/dev/null \
      | grep -v "^Only in $OUT" || true
    echo
    continue
  fi

  # 1. the shared codebase, laid into the experiment folder
  rsync -a $EX "$SHARED/" "$OUT/"

  # 2. every script in one place, including setup.sh -- the run scripts look for
  #    it in scripts/ first, then the root
  mkdir -p "$OUT/scripts"
  cp "$SHARED"/run_*.sh "$OUT/scripts/" 2>/dev/null || true
  cp "$SHARED"/setup.sh "$OUT/scripts/"
  cp "$SHARED"/setup.sh "$OUT/"

  # 3. SLURM fails immediately if the --output directory is missing
  mkdir -p "$OUT/logs"

  # 4. orca is the orbit-metric helper. Without +x the run crashes at the very
  #    END, after training has finished.
  chmod +x "$OUT"/*.sh "$OUT"/scripts/*.sh 2>/dev/null || true
  chmod +x "$OUT/src/analysis/orca/orca"

  n=$(find "$OUT" -type f -not -path "*__pycache__*" -not -name .gitkeep | wc -l)
  echo "synced $EXP  ($n files, $(du -sh "$OUT" | cut -f1))"
  echo "   upload:  rsync -avz $ROOT/$EXP/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/$EXP/"
  echo "   then:    cd /rds/projects/c/chenhp-graphsyn/$EXP && sbatch scripts/<name>.sh"
done
