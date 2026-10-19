#!/bin/bash
# One-off setup of ICLR [2] (Bergmeister et al., iterative local expansion) on
# BlueBEAR. Run on a LOGIN node (it needs the internet), from this directory:
#
#   bash setup_bluebear.sh
#
# Clones the repo next to D_intra/D_inter, creates its conda env from the repo's
# own environment.yaml, compiles orca, and installs the Luxembourg data + config.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=/rds/projects/c/chenhp-graphsyn
REPO=$BASE/competitors/graph-generation

module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3        # as D_intra/scripts/setup.sh

if [ ! -d "$REPO" ]; then
    git clone https://github.com/AndreasBergmeister/graph-generation "$REPO"
fi
cd "$REPO"
eval "$(conda shell.bash hook)"
# A prefix env beside D_intra's digress env, so jobs activate it by path.
if [ ! -d "$BASE/envs/graph-generation" ]; then
    conda env create -f environment.yaml -p "$BASE/envs/graph-generation"
fi
conda activate "$BASE/envs/graph-generation"
( cd util/orca && g++ -O2 -std=c++11 -o orca orca.cpp )

cp "$HERE/lux_regions.yaml" config/experiment/lux_regions.yaml
python "$HERE/apply_patch.py" --repo "$REPO"
python "$HERE/convert_data.py" --repo "$REPO"
python -c "import graph_generation, torch; print('ok, torch', torch.__version__)"
echo "ICLR [2] ready in $REPO"
