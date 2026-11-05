#!/bin/bash
#SBATCH --job-name=higen_prep
#SBATCH --account=chenhp-graphsyn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_prep_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/competitors/logs/higen_prep_%j.err

# CPU only: build HiGen's hierarchies of the 120 regions before training. See
# preprocess.py. Submit the training chain with --dependency=afterok:<this job>.

set -euo pipefail
BASE=/rds/projects/c/chenhp-graphsyn
module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"
conda activate "$BASE/envs/higen"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python "$BASE/competitors/higen/preprocess.py" --repo "$BASE/competitors/HiGen_main" --workers 32
