#!/bin/bash
# One-off setup of HiGen [5] on BlueBEAR. Run on a LOGIN node (needs internet),
# from this directory:
#
#   bash setup_bluebear.sh
#
# The repo pins torch 1.10/CUDA 10.2, which has no kernels for BlueBEAR's A100s;
# its README names PyTorch 1.13.1 + PyG 2.2, so this builds that on CUDA 11.7.
# pyyaml stays at 5.4.1: HiGen calls yaml.load without a Loader, which 6.x rejects.
# numpy stays below 1.24 for networkx 2.3 (np.int).

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=/rds/projects/c/chenhp-graphsyn
REPO=$BASE/competitors/HiGen_main
ENV=$BASE/envs/higen

module purge; module load bluebear
module load bear-apps/2024a/live
module load Miniforge3/25.3.0-3        # as D_intra/scripts/setup.sh
eval "$(conda shell.bash hook)"

[ -d "$REPO" ] || git clone https://github.com/Karami-m/HiGen_main "$REPO"
if [ ! -d "$ENV" ]; then
    conda create -y -p "$ENV" python=3.8
    conda activate "$ENV"
    pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
    pip install torch-geometric==2.2.0
    pip install "networkx==2.3" pyemd tqdm tensorboardX scipy "pyyaml==5.4.1" easydict \
                matplotlib python-igraph python-louvain
else
    conda activate "$ENV"
fi
# networkx 2.3 (which HiGen's code needs) uses np.int, removed in numpy 1.24, and
# pip pulls a newer numpy in with the other packages. Pinned after them, so it
# also repairs an env built before this line existed (2026-10-13).
pip install "numpy<1.24"

cd "$REPO"
python "$HERE/apply_patch.py" --repo "$REPO"
cp "$HERE/higen_lux.yaml" configs/higen_lux.yaml
python "$HERE/convert_data.py" --repo "$REPO"
( cd utils/orca 2>/dev/null && g++ -O2 -std=c++11 -o orca orca.cpp ) || true
python -c "import torch, torch_geometric; print('ok, torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "HiGen ready in $REPO"
