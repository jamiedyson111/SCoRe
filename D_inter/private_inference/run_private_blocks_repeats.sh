#!/bin/bash
#SBATCH --job-name=priv_blocks_x
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_x_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/priv_blocks_x_%j.err

# STAGE 2, repeated: one cross block per connected pair of the private release,
# around stage 1's generated internal blocks, N seeded times.
#
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 C C_v3
#   sbatch private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3 /abs/epoch=N.ckpt
#
# The first CFG is D_inter's configuration, the second the D_intra one whose
# clusters_<graph>_k<k>_<ICFG>_s<seed>.pt (from D_intra's
# run_private_clusters_repeats.sh) are the context. Seed s reads clusters seed s.
# Output, beside the release in D_intra/private_inference/out/:
#     blocks_<graph>_k<k>_<ICFG>_s80-<CFG>_deg_s<seed>.pt
# scored locally by syn_repeats_summary.py. The internal-degree D_inter: X^_i, X^_j
# are computed from the generated clusters, the condition [n_i, n_j, w_ij, r_{i->j},
# r_{j->i}] from the release, which must be one make_private_release.py wrote with
# r_ij / r_ji (November 2026 or later).
#
# CHECKPOINTS are those of the transfer table (scripts/run_sample_pairs_transfer.sh),
# each at its final checkpoint:
#   B  dataset pair_B_lux2olde_k7_s80_deg, the newest pair_lux_s80_deg run
#   C  dataset pair_C_lux2olde_k7_s80_deg, the newest pair_C_s80_deg run
# ICFG is given without the split suffix (B_v3); files are tagged <ICFG>_s80.
#
# Submit after stage 1, with --dependency=afterok:<stage 1 job>.
# Oldenburg k=7: 552 pairs, 18 batches, roughly 15-25 min a seed.

set -euo pipefail

# --graph G (first argument): the private graph, oldenburg (default) or vermont.
# Vermont's datasets are tagged lux2verm and its own models / runs carry _verm.
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
# --model-sfx SFX: sample the models trained under <name>SFX (e.g. _remb, the r-embedded
# condition) and tag the output with SFX too, so it never overwrites the originals'.
MSFX=""
if [ "${1:-}" = "--model-sfx" ]; then MSFX="${2:?--model-sfx needs a suffix}"; shift 2; fi
CFG="${1:-}"
ICFG="${2:-}"
CKPT="${3:-}"

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter
RELEASE="$(dirname "$ROOT")/D_intra/private_inference/out/release_${GRAPH}_k7.pt"

if [ -z "$ICFG" ]; then
    echo "usage: sbatch private_inference/run_private_blocks_repeats.sh [--repeats N] {B|C} ICFG [CKPT]" >&2
    exit 1
fi
# The 80/20 supernode-split models (October 2026), at their FINAL checkpoint.
case "$CFG" in
    B) DATASET=pair_B_${GTAG}_k7_s80_deg; MODEL=pair_lux_s80_deg ;;     # B's model is A's
    C) DATASET=pair_C_${GTAG}_k7_s80_deg; MODEL=pair_C${GSFX}_s80_deg ;;
    *) echo "usage: sbatch private_inference/run_private_blocks_repeats.sh [--repeats N] {B|C} ICFG [CKPT]" >&2
       exit 1 ;;
esac
ICFG="${ICFG}_s80"           # the clusters tag run_private_clusters_repeats.sh wrote
MODEL="$MODEL$MSFX"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}

if [ -z "$CKPT" ]; then
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$MODEL"/ 2>/dev/null | head -1 || true)
    [ -n "$RUNDIR" ] || { echo "ERROR: no $MODEL run under $ROOT/outputs" >&2; exit 1; }
    CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$MODEL"/last*.ckpt 2>/dev/null | head -1 || true)
    echo "run dir:    $RUNDIR"
fi
for f in "$CKPT" "$RELEASE"; do
    [ -n "$f" ] && [ -f "$f" ] || { echo "ERROR: not found: ${f:-<none>}" >&2; exit 1; }
done

TAG="${ICFG}-${CFG}_deg${MSFX}"
echo "D_inter: $CFG  dataset=$DATASET    context: D_intra $ICFG    repeats=$REPEATS"
echo "checkpoint: $CKPT"
echo "release:    $RELEASE"
echo "output:     blocks_oldenburg_k7_${TAG}_s<seed>.pt"
python3 -c "import torch, sys; p = torch.load(sys.argv[1], weights_only=False)['pairs'][0]; sys.exit('r_ij' not in p)" "$RELEASE" || {
    echo "ERROR: $RELEASE has no r_ij / r_ji -- upload the one make_private_release.py now writes" >&2
    exit 1; }
cd "$ROOT/private_inference"

python3 sample_private_blocks.py \
    ${SB[@]+"${SB[@]}"} \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name="private_blocks_${TAG}" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'" \
    +release="$RELEASE" \
    +clusters_tag="$ICFG" \
    +tag="$TAG" \
    +repeats="$REPEATS"
