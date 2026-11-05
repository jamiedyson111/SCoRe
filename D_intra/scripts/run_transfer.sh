#!/bin/bash
#SBATCH --job-name=transfer
#SBATCH --account=chenhp-graphsyn
#SBATCH --qos=bbgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/rds/projects/c/chenhp-graphsyn/D_intra/logs/transfer_%j.out
#SBATCH --error=/rds/projects/c/chenhp-graphsyn/D_intra/logs/transfer_%j.err

# Train and sample ONE configuration of the D_intra transfer study.
#
#   sbatch scripts/run_transfer.sh A_v3   the Luxembourg model (A, and B's model)
#   sbatch scripts/run_transfer.sh C_v3   Luxembourg + Oldenburg -> test Oldenburg
#   sbatch scripts/run_transfer.sh A_uncond   plain DiGress, no condition (baseline)
#   sbatch scripts/run_transfer.sh C_uncond
#
# Only C needs training for Oldenburg. A and B never see the private graph during
# training: A's pairs file is byte-identical to the Minnesota one, and B's train
# and val splits and standardisation are too (checked 2026-10-05) -- only B's
# TEST split changed. Their existing checkpoints are the Oldenburg models, so
# re-sample them with run_sample_conditional.sh rather than paying 6 GPU hours
# to reproduce them. scripts/run_olde_transfer.sh submits exactly that.
#
# Trains, then generates one subgraph per test supernode from the best
# checkpoint, in a single job.
#
# n_epochs is 200, not 1000: the k=7 Luxembourg run reached its best validation
# NLL at epoch 59 and never improved across 460 further epochs, so 1000 wastes
# roughly eight hours of GPU time per configuration.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")
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
CFG="${1:-}"

# Version 3 (method section V A 1) b): a suffix on any configuration below.
#   _v3        both changes: centroid node type + hop embedding MLP_h (d_h = 1)
#   _centroid  the centroid node type alone
#   _hopmlp    the hop embedding alone
# Same datasets, splits and settings as the run without the suffix, so the two
# are directly comparable. NAME keeps the suffix, so each gets its own run
# directory and checkpoints.
V3=()
BASE="$CFG"
SFX=""
case "$CFG" in
    *_v3)       SFX=_v3;       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) SFX=_centroid; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   SFX=_hopmlp;   V3=(model.hop_embed_dim=1) ;;
esac
BASE="${CFG%$SFX}"
FINETUNE=""
EPOCHS=200
LR=()
VAL_EVERY=10

# 80/20 SUPERNODE SPLIT (October 2026, data/splits/), no validation split: the
# model trains for a fixed 200 epochs and its FINAL checkpoint is the model.
# Two models per variant: A (Luxembourg; B uses this one -- B's training data is
# A's exactly) and C (Luxembourg + Oldenburg). B is not trained.
case "$BASE" in
    A) DATASET=cluster_A_lux2olde_k7_s80 ;;
    C) DATASET=cluster_C_lux2olde_k7_s80 ;;
    A_nohop) DATASET=cluster_A_lux2olde_k7_s80_nohop ;;
    C_nohop) DATASET=cluster_C_lux2olde_k7_s80_nohop ;;
    # Plain DiGress, one supernode at a time: no condition at all (y has width 0).
    # Same architecture, data, split and epochs; only c_i is removed.
    A_uncond) DATASET=cluster_A_lux2olde_k7_s80_uncond ;;
    C_uncond) DATASET=cluster_C_lux2olde_k7_s80_uncond ;;
    # C as a FINE-TUNE (extraction/make_finetune_datasets.py): the newest run of the
    # matching A model (same variant and suffix), trained further on the 254
    # Oldenburg training supernodes only, keeping A's standardisation and marginal.
    Cft)       DATASET=cluster_Cft_${GTAG}_k7_s80 ;       FINETUNE="transfer_A${SFX}_s80" ;;
    Cft_nohop) DATASET=cluster_Cft_${GTAG}_k7_s80_nohop ; FINETUNE="transfer_A_nohop${SFX}_s80" ;;
    # Plain DiGress on Oldenburg alone, from scratch: 254 train / 63 test supernodes.
    O_uncond)  DATASET=cluster_O_${GTAG}_k7_s80_uncond ;;
    B|B_nohop|B_uncond)
       echo "B is not trained: its training data is A's, so A's model is B's model." >&2
       echo "Train ${CFG/#B/A}, then sample B with scripts/run_sample_conditional.sh $CFG" >&2
       exit 1 ;;
    *) echo "usage: sbatch scripts/run_transfer.sh {A|C|Cft}[_nohop|_uncond] | O_uncond" >&2
       echo "       optionally suffixed _v3, _centroid or _hopmlp" >&2
       exit 1 ;;
esac
if [ ${#V3[@]} -gt 0 ] && [[ "$BASE" == *_nohop || "$BASE" == *_uncond ]] && [[ "${V3[*]}" == *hop_embed* ]]; then
    echo "ERROR: $CFG embeds h, but the ${BASE#*_} condition has no h to embed" >&2
    exit 1
fi

# Only the private-graph models (Cft*, O_uncond) are graph-specific; A and C are the
# Luxembourg / Oldenburg runs and keep their names.
case "$BASE" in Cft|Cft_nohop|O_uncond) ;; *) [ -z "$GSFX" ] || { echo "ERROR: --graph $GRAPH only applies to Cft* and O_uncond" >&2; exit 1; } ;; esac
NAME="transfer_${CFG}${GSFX}_s80"
echo "configuration $CFG  ->  dataset=$DATASET  name=$NAME  ${V3[*]:-}"

# Fine-tune (Cft*): start from the A run's newest last*.ckpt, resolved now so the
# job fails at once if A is missing. 100 epochs at lr 5e-5 (A: 200 at 2e-4), the
# same schedule as D_inter's C.
FT_ARGS=()
if [ -n "$FINETUNE" ]; then
    FTDIR=$(ls -td "$ROOT"/outputs/*/*-"$FINETUNE"/ 2>/dev/null | head -1 || true)
    FTCKPT=$(ls -t "${FTDIR%/}"/checkpoints/"$FINETUNE"/last*.ckpt 2>/dev/null | head -1 || true)
    if [ -z "$FTCKPT" ]; then
        echo "ERROR: no last*.ckpt for $FINETUNE to fine-tune from -- train ${FINETUNE#transfer_} first" >&2
        exit 1
    fi
    EPOCHS=100
    LR=(train.lr=0.00005)
    FT_ARGS=("general.finetune_from='$FTCKPT'")
    echo "fine-tuning from: $FTCKPT"
fi
# Plain DiGress on Oldenburg alone has 254 training graphs, 4 batches an epoch: 200
# epochs would be ~800 steps against A's ~22k (111 batches x 200). Train it for
# about A's number of gradient steps instead, validating rarely.
if [ "$BASE" = O_uncond ]; then
    case "$GRAPH" in
        oldenburg) EPOCHS=5500; VAL_EVERY=250 ;;   # 254 graphs, 4 batches an epoch
        vermont)   EPOCHS=320;  VAL_EVERY=20 ;;    # 4418 graphs, 70 batches an epoch
    esac
fi

# THE LINE TO CHECK in this job's log:  "Size of the input features 7 2 NN"
#
#   A,   B,   C          NN = 22 = 10 condition + 1 time + 11 spectral
#                                  10 = n, rho_in, k, h1..h7
#   *_nohop              NN = 15 =  3 condition + 1 time + 11 spectral
#                                   3 = n, rho_in, k
#   *_uncond             NN = 12 =  0 condition + 1 time + 11 spectral
#                                   plain DiGress: 12 is CORRECT here, and only here
#
# So 15 is CORRECT for the ablation and WRONG for A/B/C -- on a plain A/B/C run
# it means either the rebuilt pairs_*.pt did not upload, or
# data/cluster_lux/processed/ still holds a cache keyed on the same filename.
# PyG reuses that silently, with no error raised. The _nohop datasets have
# distinct filenames precisely so they get their own cache entry.
# If it says 12 on anything but *_uncond, the condition is not reaching the model.
#
# Version 3 changes the first and last numbers:
#   _centroid   8 2 22    X gains the centroid class (2 classes + 6 extra)
#   _hopmlp     7 2 16    h1..h7 become one embedded number: 4 + 1 + 11
#   _v3         8 2 16
#   *_nohop_centroid  8 2 15   no-hop condition, centroid marked: the
#                              baseline that differs from _v3 in h_i alone
#
# Also expect the edge marginal:
#   A, B   [0.8768, 0.1232]     Luxembourg train
#   C      [0.9009, 0.0991]     Luxembourg + oversampled Oldenburg
# NLLs are not comparable between C and A/B because their marginals differ.

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

# The orca execute bit is handled by setup.sh, sourced above -- look for
# "orca was not executable -- fixed" near the top of this job's log. Without it
# the run dies at epoch 29, at the first sampling validation, not at startup.

cd "$ROOT/src"

python3 main.py \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    ${FT_ARGS[@]+"${FT_ARGS[@]}"} \
    ${LR[@]+"${LR[@]}"} \
    general.name="$NAME" \
    general.wandb=disabled \
    train.n_epochs=$EPOCHS \
    general.check_val_every_n_epochs=$VAL_EVERY \
    general.sample_every_val=4

# --- generate from the FINAL checkpoint ------------------------------------
# No validation split, so no epoch is chosen: the last epoch is the model. The
# newest run directory of this name, so a crashed earlier run is never used.
RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
if [ -z "$RUNDIR" ]; then
    echo "ERROR: no run directory for $NAME under $ROOT/outputs -- training failed?" >&2
    exit 1
fi
CKPT=$(ls -t "${RUNDIR%/}"/checkpoints/"$NAME"/last*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ]; then
    echo "ERROR: no last*.ckpt in $RUNDIR -- training may have failed" >&2
    exit 1
fi
echo
echo "run dir:       $RUNDIR"
echo "sampling from: $CKPT"

# The '=' in epoch=N.ckpt breaks hydra's override grammar unless the value
# carries its own quotes; shell quotes are stripped before hydra sees them.
python3 sample_conditional.py \
    ${SB[@]+"${SB[@]}"} \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    general.name="${NAME}_sample" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'"

echo
echo "Done. Output:"
echo "  ls -t $ROOT/outputs/*/*${NAME}_sample*/conditional_samples.txt | head -1"
