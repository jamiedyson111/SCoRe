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
#   sbatch scripts/run_transfer.sh A      train Luxembourg -> test Luxembourg
#   sbatch scripts/run_transfer.sh B      train Luxembourg -> test Oldenburg
#   sbatch scripts/run_transfer.sh C      train Lux + Olde -> test Oldenburg
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
case "$CFG" in
    *_v3)       BASE="${CFG%_v3}";       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) BASE="${CFG%_centroid}"; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   BASE="${CFG%_hopmlp}";   V3=(model.hop_embed_dim=1) ;;
esac

case "$BASE" in
    A) DATASET=cluster_A_lux2olde_k7 ;;
    B) DATASET=cluster_B_lux2olde_k7 ;;
    C) DATASET=cluster_C_lux2olde_k7 ;;
    # The no-hop ablation: identical splits and seed, condition truncated to
    # [n, rho_in, k]. NAME picks up the suffix, so these get their own run
    # directories, checkpoints and sample dumps and cannot collide with A/B/C.
    A_nohop) DATASET=cluster_A_lux2olde_k7_nohop ;;
    B_nohop) DATASET=cluster_B_lux2olde_k7_nohop ;;
    C_nohop) DATASET=cluster_C_lux2olde_k7_nohop ;;
    *) echo "usage: sbatch scripts/run_transfer.sh {A|B|C}[_nohop]" >&2
       echo "       optionally suffixed _v3, _centroid or _hopmlp" >&2
       exit 1 ;;
esac
if [ ${#V3[@]} -gt 0 ] && [[ "$BASE" == *_nohop ]] && [[ "${V3[*]}" == *hop_embed* ]]; then
    echo "ERROR: $CFG embeds h, but the _nohop condition has no h to embed" >&2
    exit 1
fi

NAME="transfer_$CFG"
# C is the one model whose training set contains the private graph, so the
# Oldenburg C gets its own name. Under plain transfer_C, the newest-run lookup in
# run_sample_conditional.sh would fall back to the Minnesota-trained C until this
# one existed, and score it on Oldenburg without complaint.
[[ "$BASE" == C* ]] && NAME="transfer_${CFG/#C/C_olde}"
echo "configuration $CFG  ->  dataset=$DATASET  name=$NAME  ${V3[*]:-}"

# THE LINE TO CHECK in this job's log:  "Size of the input features 7 2 NN"
#
#   A,   B,   C          NN = 22 = 10 condition + 1 time + 11 spectral
#                                  10 = n, rho_in, k, h1..h7
#   *_nohop              NN = 15 =  3 condition + 1 time + 11 spectral
#                                   3 = n, rho_in, k
#
# So 15 is CORRECT for the ablation and WRONG for A/B/C -- on a plain A/B/C run
# it means either the rebuilt pairs_*.pt did not upload, or
# data/cluster_lux/processed/ still holds a cache keyed on the same filename.
# PyG reuses that silently, with no error raised. The _nohop datasets have
# distinct filenames precisely so they get their own cache entry.
# If it says 12 the condition is not reaching the model at all.
#
# Version 3 changes the first and last numbers:
#   _centroid   8 2 22    X gains the centroid class (2 classes + 6 extra)
#   _hopmlp     7 2 16    h1..h7 become one embedded number: 4 + 1 + 11
#   _v3         8 2 16
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
    general.name="$NAME" \
    general.wandb=disabled \
    train.n_epochs=200 \
    general.check_val_every_n_epochs=10 \
    general.sample_every_val=4

# --- generate from the best checkpoint -------------------------------------
# ModelCheckpoint keeps the top-5 by validation NLL as epoch=N.ckpt, written into
# THIS run's own hydra directory. Two separate things matter:
#
# 1. Restrict to the NEWEST run. The old glob was
#        outputs/*/*$NAME*/checkpoints/$NAME/epoch=*.ckpt
#    which matches every dated run of this configuration ever made, and
#    `ls -tr | head -1` then takes the oldest file across all of them. On
#    2026-08-12 that silently sampled A and B from epoch=9.ckpt of the 11 August
#    jobs that CRASHED at epoch 29, throwing away the 200-epoch runs that had
#    just finished. The bug is invisible in the output -- you get plausible
#    graphs from an undertrained model.
#    The trailing "-$NAME/" with no wildcard after it is deliberate: it matches
#    the hydra run directory (TIME-transfer_A) and NOT the TIME-transfer_A_sample
#    directory that the sampling step below creates.
#
# 2. Within that run, prefer the OLDEST checkpoint. Validation converges by about
#    epoch 10 and then wanders, so the earliest surviving top-5 checkpoint is as
#    good as any and avoids the winner's curse of picking the single lowest point
#    on a noisy curve. last.ckpt is deliberately avoided.
RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
if [ -z "$RUNDIR" ]; then
    echo "ERROR: no run directory for $NAME under $ROOT/outputs -- training failed?" >&2
    exit 1
fi
CKPT=$(ls -tr "${RUNDIR%/}"/checkpoints/"$NAME"/epoch=*.ckpt 2>/dev/null | head -1 || true)
if [ -z "$CKPT" ]; then
    echo "ERROR: no checkpoint in $RUNDIR -- training may have failed" >&2
    exit 1
fi
echo
echo "run dir:       $RUNDIR"
echo "sampling from: $CKPT"

# The '=' in epoch=N.ckpt breaks hydra's override grammar unless the value
# carries its own quotes; shell quotes are stripped before hydra sees them.
python3 sample_conditional.py \
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
