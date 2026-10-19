#!/bin/bash
#SBATCH --job-name=priv_clusters_x
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/priv_clusters_x_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/priv_clusters_x_%j.err

# STAGE 1, repeated: one internal block per supernode of the private release,
# N seeded times, for one D_intra configuration.
#
#   sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3
#   sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 C_v3
#   sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3 /abs/epoch=N.ckpt
#
# Writes private_inference/out/clusters_<graph>_k<k>_<CFG>_s<seed>.pt, which
# D_inter's run_private_blocks_repeats.sh reads as its context.
#
# CFG and the checkpoint are resolved EXACTLY as scripts/run_sample_conditional.sh
# resolves them -- same dataset, same v3 switches, same run directory (newest)
# and checkpoint (oldest top-5 in it) -- so the assembled graph is built from the
# models the D_intra repeat tables score. Keep the two in step.
#
# The release is private_inference/out/release_oldenburg_k7.pt (stage 0,
# built locally by make_private_release.py and copied up).
#
# Oldenburg k=7: 317 supernodes, 10 batches x 500 steps, a few minutes a seed.

set -euo pipefail

REPEATS=1
if [ "${1:-}" = "--repeats" ]; then
    REPEATS="${2:?--repeats needs a count}"
    shift 2
fi
CFG="${1:-}"
CKPT="${2:-}"

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_intra}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
RELEASE="$ROOT/private_inference/out/release_oldenburg_k7.pt"

V3=()
SFX=""
case "$CFG" in
    *_v3)       SFX=_v3;       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) SFX=_centroid; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   SFX=_hopmlp;   V3=(model.hop_embed_dim=1) ;;
esac
BASE="${CFG%$SFX}"
case "$BASE" in
    B)       DATASET=cluster_B_lux2olde_k7;       NAME=transfer_B ;;
    C)       DATASET=cluster_C_lux2olde_k7;       NAME=transfer_C_olde ;;
    B_nohop) DATASET=cluster_B_lux2olde_k7_nohop; NAME=transfer_B_nohop ;;
    C_nohop) DATASET=cluster_C_lux2olde_k7_nohop; NAME=transfer_C_olde_nohop ;;
    *) echo "usage: sbatch private_inference/run_private_clusters_repeats.sh [--repeats N] CFG [CKPT]" >&2
       echo "  CFG = {B|C}[_nohop], optionally + _v3|_centroid|_hopmlp  (e.g. B_v3)" >&2
       exit 1 ;;
esac
NAME="$NAME$SFX"

if [ -z "$CKPT" ]; then
    # As run_sample_conditional.sh: the newest run directory of this name, then
    # the oldest surviving epoch checkpoint inside it.
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
    [ -n "$RUNDIR" ] || { echo "ERROR: no run directory for $NAME under $ROOT/outputs" >&2; exit 1; }
    CKPT=$(ls -tr "${RUNDIR%/}"/checkpoints/"$NAME"/epoch=*.ckpt 2>/dev/null | head -1 || true)
    echo "run dir:    $RUNDIR"
fi
for f in "$CKPT" "$RELEASE"; do
    [ -n "$f" ] && [ -f "$f" ] || { echo "ERROR: not found: ${f:-<none>}" >&2; exit 1; }
done

echo "configuration: $CFG  dataset=$DATASET  repeats=$REPEATS"
echo "checkpoint:    $CKPT"
echo "release:       $RELEASE"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi
# AFTER setup.sh, which starts with `module purge` and wipes PYTHONPATH.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}
cd "$ROOT/private_inference"

# The '=' in epoch=N.ckpt needs hydra's own quotes inside the shell's.
python3 sample_private_clusters.py \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    general.name="private_clusters_${CFG}" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'" \
    +release="$RELEASE" \
    +tag="$CFG" \
    +repeats="$REPEATS"
