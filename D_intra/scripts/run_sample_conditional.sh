#!/bin/bash
#SBATCH --job-name=cond_sample
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cond_sample_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/cond_sample_%j.err

# Generate a subgraph for every supernode in the test split, conditioned on its
# c_i vector. This is the "feed in a supernode, get back its cluster" step -- it
# is SEPARATE from training and needs a trained checkpoint. Use it to redo the
# sampling of an already-trained configuration without paying for training again.
#
#   sbatch scripts/run_sample_conditional.sh --repeats 5 B_v3      # 5 seeded runs, one job
#   sbatch scripts/run_sample_conditional.sh --repeats 5 C_v3
#   sbatch scripts/run_sample_conditional.sh B_v3 /abs/path/last.ckpt   # explicit checkpoint
#
# --repeats N samples the test split N times (seeds 0..N-1) from the one loaded
# checkpoint and writes conditional_samples_s<seed>.txt for each, into a run
# directory suffixed _x<N>. evaluation/repeat_summary.py turns them into mean / std.
# It is an ARGUMENT, not an environment variable: BlueBEAR's job-submit plugin
# does not carry the submitting shell's environment into the job.
#
# The configuration fixes both the dataset (whose test split is sampled) and the
# model (the newest transfer_<A|C>..._s80 run, at its final checkpoint):
#
#   A   Luxembourg model, Luxembourg test supernodes (1763)
#   B   Luxembourg model, Oldenburg test supernodes (63)
#   C   Luxembourg + Oldenburg model, the same 63
#   Cft A's model fine-tuned on the 254 Oldenburg training supernodes, the same 63
#   O_uncond  plain DiGress trained on Oldenburg alone (254), the same 63
#
# with any of _nohop (condition [n, rho_in, k]) or _uncond (no condition: plain
# DiGress, size n_i only) and _v3 / _centroid / _hopmlp.
# Split: 80/20 by supernode (data/splits/), no validation split.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")

# DATASET fixes which conditions are fed in; NAME fixes which checkpoint is used.
# They are separate variables because B deliberately pairs the Oldenburg dataset
# with the checkpoint trained before Oldenburg existed -- see B below.
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
# Version 3 suffix (_v3 | _centroid | _hopmlp), as in run_transfer.sh. The
# checkpoint comes from the suffixed run and the dataset gets the same switches,
# so B_v3 resamples the transfer_B_v3 model on Oldenburg.
V3=()
SFX=""
case "$CFG" in
    *_v3)       SFX=_v3;       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) SFX=_centroid; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   SFX=_hopmlp;   V3=(model.hop_embed_dim=1) ;;
esac
CFG="${CFG%$SFX}"
# 80/20 SUPERNODE SPLIT (October 2026, data/splits/), no validation split. There
# are two trained models per variant: the Luxembourg model (run name
# transfer_A*_s80) and C (transfer_C*_s80). A and B both use the Luxembourg model
# -- B's training data IS A's -- and differ only in the test set: A tests on
# Luxembourg's held-out supernodes, B on Oldenburg's.
BASE="${CFG%_nohop}"
BASE="${BASE%_uncond}"
NOHOP="${CFG#$BASE}"                    # "", "_nohop" or "_uncond" (no condition)
case "$BASE" in
    A|B) MODEL=A ;;
    C)   MODEL=C ;;
    Cft) MODEL=Cft ;;                   # C fine-tuned from A on Oldenburg train
    O)   MODEL=O ;;                     # O_uncond: plain DiGress on Oldenburg alone
    *) echo "usage: sbatch scripts/run_sample_conditional.sh [--repeats N] CFG [CKPT]" >&2
       echo "  CFG = {A|B|C|Cft}[_nohop|_uncond], optionally + _v3|_centroid|_hopmlp; or O_uncond" >&2
       exit 1 ;;
esac
DATASET="cluster_${BASE}_${GTAG}_k7_s80${NOHOP}"
# A (and B) use the shared Luxembourg model; Cft and O were trained on the private graph.
MSFX=""; case "$MODEL" in Cft|O) MSFX="$GSFX" ;; esac
NAME="transfer_${MODEL}${NOHOP}${SFX}${MSFX}_s80"
CFG="$CFG$SFX"
CKPT="${2:-}"

if [ -z "$CKPT" ]; then
    # The newest run directory of the model, at its FINAL checkpoint: there is no
    # validation split to choose an epoch by, so training runs a fixed length
    # and the last epoch is the model. "-$NAME/" with nothing after it matches the
    # training run, not the *_sample directories.
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
    if [ -z "$RUNDIR" ]; then
        echo "ERROR: no run directory for $NAME under $ROOT/outputs -- train it first:" >&2
        echo "       sbatch scripts/run_transfer.sh ${MODEL}${NOHOP}${SFX}" >&2
        exit 1
    fi
    CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$NAME"/last*.ckpt 2>/dev/null | head -1 || true)
    echo "run dir:    $RUNDIR"
fi

if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: ${CKPT:-<none>}" >&2
    echo >&2
    echo "Available checkpoints for $NAME:" >&2
    ls -1 "$ROOT"/outputs/*/*-"$NAME"/checkpoints/"$NAME"/*.ckpt 2>/dev/null >&2 \
        || echo "  (none found under $ROOT/outputs/*/*-$NAME/checkpoints/)" >&2
    exit 1
fi

echo "configuration: $CFG  ->  dataset=$DATASET  (checkpoint from $NAME)"
echo "checkpoint: $CKPT"

# setup.sh sits at $ROOT/scripts/ or $ROOT/ depending on how the tree was
# uploaded. Try both, and say so clearly instead of dying on `set -e`.
if   [ -f "$ROOT/scripts/setup.sh" ]; then . "$ROOT/scripts/setup.sh"
elif [ -f "$ROOT/setup.sh" ];         then . "$ROOT/setup.sh"
else
    echo "ERROR: setup.sh found in neither $ROOT/scripts/ nor $ROOT/" >&2
    exit 1
fi

# AFTER setup.sh, never before: setup.sh starts with `module purge`, which wipes
# PYTHONPATH. `from src import utils` in main.py needs the project root on the
# path, and src is NOT pip-installed in the BlueBEAR env.
#
# Deliberately not `pip install -e .` -- that would bind `src` to whichever
# folder it was installed from, so one experiment would silently import the
# other's model code. Per-job PYTHONPATH keeps each tree importing its own.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
python3 -c "import src" 2>/dev/null || {
    echo "ERROR: 'src' not importable. ROOT=$ROOT PYTHONPATH=$PYTHONPATH" >&2
    exit 1
}
cd "$ROOT/src"

# dataset= and +experiment= are both required, exactly as for training: main.py
# needs the dataset name to select the right branch and to rebuild dataset_infos,
# which the checkpoint load depends on.
# The checkpoint filename contains '=' (epoch=59.ckpt), which hydra's override
# grammar parses as a second key/value separator:
#     mismatched input '=' expecting <EOF>
# Shell quotes do not help -- bash strips them before hydra sees the argument.
# Hydra needs its OWN quotes inside the value, hence the '...' nested in "...".
# From $CFG, not $NAME, and tagged _olde: B sampled on Oldenburg shares its
# checkpoint (hence its NAME) with B sampled on Minnesota, and the two must not be
# confused when the newest conditional_samples.txt is collected.
SAMPLE_NAME="transfer_${CFG}${GSFX}_s80_sample"
REP_ARGS=()
if [ "$REPEATS" -gt 1 ]; then
    SAMPLE_NAME="${SAMPLE_NAME}_x${REPEATS}"
    REP_ARGS=("+repeats=$REPEATS")
fi
python3 sample_conditional.py \
    ${SB[@]+"${SB[@]}"} \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    ${REP_ARGS[@]+"${REP_ARGS[@]}"} \
    general.name="$SAMPLE_NAME" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'"

echo
echo "Done. Find the output with:"
echo "  ls -td $ROOT/outputs/*/*-${SAMPLE_NAME}/ | head -1"
