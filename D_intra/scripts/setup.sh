module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate /rds/projects/c/chenhp-graphsyn/envs/digress

# --- orca: make sure the orbit-counting binary is executable ----------------
# spectre_utils.orca() shells out to src/analysis/orca/orca. Only the SAMPLING
# metrics use it, and on a training run those first fire at epoch 29 -- so a
# missing execute bit does NOT fail fast. It kills the job twelve minutes in,
# after training has been going perfectly, and the traceback lands in .err while
# .out just stops mid-line at "Computing orbit stats...". That is what happened
# to jobs 50840373 / 50855159 / 50856021 on 2026-08-11:
#     PermissionError: [Errno 13] Permission denied: .../src/analysis/orca/orca
#
# The execute bit does not survive the trip from Windows: the working copy under
# \\wsl.localhost is 644 and `rsync -a` preserves that faithfully. sync.sh does
# chmod the copy it lays down, but only when it is re-run after the file moves,
# which is exactly how this slipped through. Doing it here covers every job that
# sources setup.sh, however the tree got uploaded.
#
# This WARNS rather than exits, because setup.sh is shared with jobs that never
# sample (extraction, evaluation) and those should not die over a metric they
# will not compute. Either way the line appears in the first seconds of the log.
_setup_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
for _cand in "$_setup_dir" "$_setup_dir/.."; do   # setup.sh sits at $ROOT/ or $ROOT/scripts/
    [ -d "$_cand/src" ] || continue
    _orca="$_cand/src/analysis/orca/orca"
    if [ ! -f "$_orca" ]; then
        echo "WARNING: orca binary missing at $_orca" >&2
        echo "         sampling metrics will crash at the first sampling epoch" >&2
    elif [ ! -x "$_orca" ]; then
        chmod +x "$_orca" 2>/dev/null || true
        if [ -x "$_orca" ]; then
            echo "setup.sh: orca was not executable -- fixed"
        else
            echo "WARNING: $_orca is not executable and chmod failed" >&2
            echo "         sampling metrics WILL crash at the first sampling epoch" >&2
        fi
    fi
    break
done
unset _setup_dir _cand _orca
