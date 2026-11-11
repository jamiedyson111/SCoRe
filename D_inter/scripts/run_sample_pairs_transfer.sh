#!/bin/bash
#SBATCH --job-name=pair_transfer
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_transfer_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_inter/logs/pair_transfer_%j.err

# One cross block per TEST pair of a configuration, N seeded times. The internal-
# degree D_inter (the paper's Section B): the model sees node features X_i, X_j.
#
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 A
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 B            # generated X^
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 C
#   sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 --real-x B   # true X (ablation)
#   sbatch scripts/run_sample_pairs_transfer.sh B /abs/path/last.ckpt    # explicit ckpt
#
# 80/20 SUPERNODE SPLIT shared with D_intra, no validation split:
#
#   A  dataset pair_lux_s80_deg             test: Luxembourg pairs touching a test supernode
#   B  dataset pair_B_lux2olde_k7_s80_deg   test: Oldenburg pairs touching a test supernode
#   C  dataset pair_C_lux2olde_k7_s80_deg   test: the same Oldenburg pairs
#   Cft  as C, but X^ from D_intra's fine-tuned C (Cft_v3) instead of its joint C
#   <CFG>_nor  the no-r ablation (condition [n_i, n_j, w_ij]), e.g. B_nor, Cft_nor
#
# Models, each at its FINAL checkpoint (last*.ckpt; training runs a fixed length):
#   A, B  the newest pair_lux_s80_deg run (B's training data is A's)
#   C     the newest pair_C_s80_deg run (A's model fine-tuned on Oldenburg)
#
# CONTEXT. B and C read GENERATED X^_i, X^_j: seed s takes the internal blocks of
# D_intra's whole-graph run D_intra/private_inference/out/clusters_oldenburg_k7_
# <CFG>_v3_s80_s<s>.pt (run_private_clusters_repeats.sh --repeats 5 <CFG>_v3), so
# those must exist. --real-x uses the true X_i, X_j instead, measuring the
# train--inference shift; its run is named ..._realX. A always uses true X: D_intra
# has no generated L^ for Luxembourg's training supernodes.
#
# --repeats is an ARGUMENT, not an env var: BlueBEAR does not pass the submitting
# shell's environment into the job. Output: outputs/<date>/<time>-pairs_<CFG>_
# s80_deg[_realX]_sample_x<N>/pair_samples_s<seed>.pt, scored by
# evaluation/transfer_summary.py.
#
# Runtime: about 1 min per batch of 32 pairs per seed. B or C (187 pairs): a few
# minutes a seed. A (3998 pairs): about 2 h a seed -- 4 h covers 2 seeds, so give A
# --time=12:00:00 for 5 (run_pairs_repeats.sh does).

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
REAL_X=0
if [ "${1:-}" = "--real-x" ]; then
    REAL_X=1
    shift
fi
# --model-sfx SFX: sample the models trained under <name>SFX (e.g. _remb, the r-embedded
# condition) and name the run with SFX too, so it never overwrites the originals' samples.
MSFX=""
if [ "${1:-}" = "--model-sfx" ]; then MSFX="${2:?--model-sfx needs a suffix}"; shift 2; fi
CFG="${1:-}"
CKPT="${2:-}"

ROOT="${SLURM_SUBMIT_DIR:-/rds/projects/c/chenhp-graphsyn/D_inter}"
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
[ -d "$ROOT/src" ] || ROOT=/rds/projects/c/chenhp-graphsyn/D_inter

# _nor: the no-r ablation, condition [n_i, n_j, w_ij] only (pair_*_deg_nor datasets,
# dataprep/make_nor_pairs.py; models pair_lux_s80_deg_nor and pair_C_s80_deg_nor).
NOR=""
[[ "$CFG" == *_nor ]] && NOR=_nor
case "${CFG%_nor}" in
    A)   DATASET=pair_lux_s80_deg;           MODEL=pair_lux_s80_deg; REAL_X=1 ;;
    B)   DATASET=pair_B_${GTAG}_k7_s80_deg; MODEL=pair_lux_s80_deg ;;
    C)   DATASET=pair_C_${GTAG}_k7_s80_deg; MODEL=pair_C${GSFX}_s80_deg ;;
    # Cft: the same D_inter C model, given X^ from D_intra's FINE-TUNED C (Cft_v3)
    # rather than its jointly trained C (C_v3) -- both halves of C fine-tuned.
    Cft) DATASET=pair_C_${GTAG}_k7_s80_deg; MODEL=pair_C${GSFX}_s80_deg ;;
    *) echo "usage: sbatch scripts/run_sample_pairs_transfer.sh [--repeats N] [--real-x] {A|B|C|Cft}[_nor] [CKPT]" >&2
       exit 1 ;;
esac
DATASET="$DATASET$NOR"
MODEL="$MODEL$NOR$MSFX"
[ "${CFG%_nor}" != A ] || [ -z "$GSFX" ] || { echo "ERROR: A is Luxembourg's; --graph does not apply" >&2; exit 1; }
RUN="pairs_${CFG%_nor}${GSFX}_s80_deg$NOR$MSFX"
CONTEXT=()
if [ "$REAL_X" = 1 ]; then
    [ "${CFG%_nor}" = A ] || RUN="${RUN}_realX"
else
    CDIR="$(dirname "$ROOT")/D_intra/private_inference/out"
    CTAG="${CFG%_nor}_v3_s80"
    for ((s = 0; s < REPEATS; s++)); do
        f="$CDIR/clusters_${GRAPH}_k7_${CTAG}_s$s.pt"
        [ -f "$f" ] || { echo "ERROR: $f is missing -- run D_intra's" \
            "run_private_clusters_repeats.sh --graph $GRAPH --repeats $REPEATS ${CFG%_nor}_v3 first, or pass --real-x" >&2; exit 1; }
    done
    CONTEXT=(+clusters_dir="$CDIR" +clusters_tag="$CTAG")
fi

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
    if [ -z "$RUNDIR" ]; then
        echo "ERROR: no $MODEL run under $ROOT/outputs -- train it first:" >&2
        echo "       see scripts/submit_inter_deg.sh" >&2
        exit 1
    fi
    CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$MODEL"/last*.ckpt 2>/dev/null | head -1 || true)
    echo "run dir:    $RUNDIR"
fi
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: ${CKPT:-<none>}" >&2
    exit 1
fi

echo "configuration: $CFG  dataset=$DATASET  repeats=$REPEATS  run=$RUN"
echo "checkpoint:    $CKPT"
cd "$ROOT/src"

# The '=' in epoch=N.ckpt needs hydra's own quotes inside the shell's.
python3 sample_pairs_conditional.py \
    ${SB[@]+"${SB[@]}"} \
    dataset="$DATASET" \
    +experiment=pair_lux \
    general.name="${RUN}_sample_x${REPEATS}" \
    general.gpus=1 \
    general.wandb=disabled \
    +repeats="$REPEATS" \
    ${CONTEXT[@]+"${CONTEXT[@]}"} \
    "general.test_only='$CKPT'"

echo
echo "Done. Output:"
echo "  ls -td $ROOT/outputs/*/*-${RUN}_sample_x${REPEATS}/ | head -1"
