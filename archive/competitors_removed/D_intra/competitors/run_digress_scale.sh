#!/bin/bash
#SBATCH --job-name=digress_scale
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/digress_scale_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/digress_scale_%j.err

# Competitor DiGress [14]: memory and time of one-shot denoising at doubling n,
# up to Oldenburg's 6105 nodes. See digress_scale.py.
#
#   cd /rds/projects/c/chenhp-graphsyn/D_intra && sbatch competitors/run_digress_scale.sh

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_intra}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 "$ROOT/competitors/digress_scale.py"
