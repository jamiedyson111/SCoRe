# Peak GPU memory and wall clock of a SLURM job, measured the same way for D_intra
# and D_inter (Section VI-E). SOURCE it from a job script (setup.sh does, for D_intra and D_inter):
#
#     . /path/to/gpu_monitor.sh
#
# It polls nvidia-smi every 0.5 s in the background (device memory in use, so
# the CUDA context and the allocator's cache are included, unlike torch's own
# counter in src/costlog.py) and, when the job's shell exits, writes one line to
#
#     /rds/projects/c/chenhp-graphsyn/gpumem/<jobid>_<jobname>.tsv
#     jobid  jobname  gpu  peak_mib  wall_s  exit  command
#
# Does nothing outside a SLURM GPU job, and never fails the job.

if [ -n "${SLURM_JOB_ID:-}" ] && command -v nvidia-smi >/dev/null 2>&1 && [ -z "${_GPUMON_PID:-}" ]; then
    _GPUMON_DIR=${GPUMON_DIR:-/rds/projects/c/chenhp-graphsyn/gpumem}
    mkdir -p "$_GPUMON_DIR" 2>/dev/null || true
    _GPUMON_LOG=$(mktemp "${TMPDIR:-/tmp}/gpumem_${SLURM_JOB_ID}_XXXX" 2>/dev/null || echo "/tmp/gpumem_$SLURM_JOB_ID")
    _GPUMON_T0=$(date +%s)
    _GPUMON_ARGS="$*"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$_GPUMON_LOG" 2>/dev/null &
    _GPUMON_PID=$!
    _gpumon_stop() {
        local rc=$?
        kill "$_GPUMON_PID" 2>/dev/null || true
        local peak gpu
        peak=$(sort -n "$_GPUMON_LOG" 2>/dev/null | tail -1)
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$SLURM_JOB_ID" "${SLURM_JOB_NAME:-}" "$gpu" \
            "${peak:-}" "$(( $(date +%s) - _GPUMON_T0 ))" "$rc" "$0 ${_GPUMON_ARGS:-}" \
            > "$_GPUMON_DIR/${SLURM_JOB_ID}_${SLURM_JOB_NAME:-job}.tsv" 2>/dev/null || true
        echo "gpu_monitor: peak device memory ${peak:-?} MiB on ${gpu:-?}, wall $(( $(date +%s) - _GPUMON_T0 )) s"
        rm -f "$_GPUMON_LOG" 2>/dev/null || true
        return $rc
    }
    trap _gpumon_stop EXIT
fi
