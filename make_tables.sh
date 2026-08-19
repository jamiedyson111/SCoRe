#!/usr/bin/env bash
# Every results table in the repo, in one command, with no arguments:
#
#     ./make_tables.sh              # D_intra and D_inter
#     ./make_tables.sh D_intra      # one project
#
# Each project's own make_tables.sh does the work and writes into its
# results/tables/; run those directly to pick individual tables or to point at a
# different samples file. Both need the `kmis` env, which they activate
# themselves.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECTS=("$@")
[ "${#PROJECTS[@]}" -eq 0 ] && PROJECTS=(D_intra D_inter)

status=0
for p in "${PROJECTS[@]}"; do
    printf '\n##################################################################\n'
    printf '# %s\n' "$p"
    printf '##################################################################\n'
    if [ ! -x "$ROOT/$p/make_tables.sh" ]; then
        echo "no $p/make_tables.sh -- skipped" >&2
        continue
    fi
    # One project failing must not hide the other's tables.
    "$ROOT/$p/make_tables.sh" || { echo "$p FAILED" >&2; status=1; }
done
exit $status
