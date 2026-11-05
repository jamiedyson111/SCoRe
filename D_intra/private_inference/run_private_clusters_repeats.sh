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
#   sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 Cft_v3   # C fine-tuned
#   sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3 /abs/epoch=N.ckpt
#
# Writes private_inference/out/clusters_<graph>_k<k>_<CFG>_s80_s<seed>.pt, which
# D_inter's run_private_blocks_repeats.sh reads as its context.
#
# CFG and the checkpoint are resolved EXACTLY as scripts/run_sample_conditional.sh
# resolves them -- same dataset, same v3 switches, the newest run directory
# and its final checkpoint (last*.ckpt) -- so the assembled graph is built from the
# models the D_intra repeat tables score. Keep the two in step.
#
# The release is private_inference/out/release_oldenburg_k7.pt (stage 0,
# built locally by make_private_release.py and copied up).
#
# Oldenburg k=7: 317 supernodes, 10 batches x 500 steps, a few minutes a seed.

set -euo pipefail

# --graph G (first argument): the private graph. oldenburg (default) or vermont;
# Vermont's datasets are tagged lux2verm and its own runs carry _verm in their name,
# so they never collide with Oldenburg's. The Luxembourg models (A, and B's) are shared.
GRAPH=oldenburg
if [ "${1:-}" = "--graph" ]; then GRAPH="${2:?--graph needs a name}"; shift 2; fi
case "$GRAPH" in
    oldenburg) GTAG=lux2olde; GSFX=""; SB=() ;;
    vermont)   GTAG=lux2verm; GSFX=_verm; SB=(+sample_batch=128) ;;   # 17x the items: bigger batches
    *) echo "ERROR: unknown --graph $GRAPH (oldenburg | vermont)" >&2; exit 1 ;;
esac
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
RELEASE="$ROOT/private_inference/out/release_${GRAPH}_k7.pt"

V3=()
SFX=""
case "$CFG" in
    *_v3)       SFX=_v3;       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) SFX=_centroid; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   SFX=_hopmlp;   V3=(model.hop_embed_dim=1) ;;
esac
BASE="${CFG%$SFX}"
# The 80/20 supernode-split models (data/splits/), at their FINAL checkpoint (no
# validation split). B uses the Luxembourg model, transfer_A*_s80.
NOHOP=""; [[ "$BASE" == *_nohop ]] && NOHOP=_nohop
case "${BASE%_nohop}" in
    B) DATASET=cluster_B_${GTAG}_k7_s80$NOHOP; NAME=transfer_A$NOHOP ;;
    C) DATASET=cluster_C_lux2olde_k7_s80$NOHOP; NAME=transfer_C$NOHOP ;;
    # C fine-tuned from A on Oldenburg train (scripts/run_transfer.sh Cft_v3)
    Cft) DATASET=cluster_Cft_${GTAG}_k7_s80$NOHOP; NAME=transfer_Cft$NOHOP ;;
    *) echo "usage: sbatch private_inference/run_private_clusters_repeats.sh [--repeats N] CFG [CKPT]" >&2
       echo "  CFG = {B|C|Cft}[_nohop], optionally + _v3|_centroid|_hopmlp  (e.g. B_v3)" >&2
       exit 1 ;;
esac
NAME="$NAME${SFX}"; [ "${BASE%_nohop}" = Cft ] && NAME="$NAME$GSFX"
NAME="${NAME}_s80"
TAG="${CFG}_s80"                 # output files clusters_<graph>_k<k>_<TAG>_s<seed>.pt

if [ -z "$CKPT" ]; then
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
    [ -n "$RUNDIR" ] || { echo "ERROR: no run directory for $NAME under $ROOT/outputs" >&2; exit 1; }
    CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$NAME"/last*.ckpt 2>/dev/null | head -1 || true)
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
    ${SB[@]+"${SB[@]}"} \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    general.name="private_clusters_${TAG}" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'" \
    +release="$RELEASE" \
    +tag="$TAG" \
    +repeats="$REPEATS"
