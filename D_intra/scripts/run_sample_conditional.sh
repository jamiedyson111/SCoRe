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
#   sbatch scripts/run_sample_conditional.sh B                     # newest transfer_B run
#   sbatch scripts/run_sample_conditional.sh B /abs/path/epoch=N.ckpt
#   sbatch scripts/run_sample_conditional.sh B_k4                  # Minnesota at k=4
#   sbatch scripts/run_sample_conditional.sh legacy                # the old cluster_lux model
#
# The configuration name is required because it fixes BOTH the checkpoint and the
# dataset, and those two must agree on CONDITION WIDTH or the run dies with a
# shape error on the first batch:
#
#   A/B/C and *_k4          10-dim [n, rho_in, k, h1..h7]   width 22
#   *_nohop                  3-dim [n, rho_in, k]           width 15
#   legacy cluster_lux       3-dim, and predates the k-MIS fix
#
# The k=4 datasets keep the 7-wide hop histogram on purpose -- a k=4 cluster has
# provably empty rings 5..7 -- so they are width-compatible with the k=7
# checkpoints and no retraining is needed to feed them in.
#
# Up to 882 test supernodes x 500 denoising steps. 2h is generous; the k=4
# transfer splits are 114 supernodes and finish in minutes.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONUNBUFFERED=1

ROOT=/rds/projects/c/chenhp-graphsyn/D_intra
# Submitting from inside scripts/ would make SLURM_SUBMIT_DIR the scripts dir,
# not the project root. Step up one level if src/ is not where ROOT says.
[ -d "$ROOT/src" ] || ROOT=$(dirname "$ROOT")

# DATASET fixes which conditions are fed in; NAME fixes which checkpoint is used.
# They are separate variables because the k=4 configurations deliberately pair a
# NEW dataset with an EXISTING checkpoint -- see B_k4 below. SAMPLE_NAME is
# derived from $CFG rather than $NAME so two datasets sharing one checkpoint
# still write to different run directories instead of overwriting each other.
CFG="${1:-}"
# Version 3 suffix (_v3 | _centroid | _hopmlp), as in run_transfer.sh. The
# checkpoint comes from the suffixed run and the dataset gets the same switches,
# so B_k4_v3 resamples the transfer_B_v3 model on Minnesota at k=4.
V3=()
SFX=""
case "$CFG" in
    *_v3)       SFX=_v3;       V3=(dataset.mark_centroid=true model.hop_embed_dim=1) ;;
    *_centroid) SFX=_centroid; V3=(dataset.mark_centroid=true) ;;
    *_hopmlp)   SFX=_hopmlp;   V3=(model.hop_embed_dim=1) ;;
esac
CFG="${CFG%$SFX}"
case "$CFG" in
    A|B|C)  DATASET="cluster_${CFG}_lux2minn_k7";       NAME="transfer_$CFG" ;;
    A_nohop|B_nohop|C_nohop)
            DATASET="cluster_${CFG%_nohop}_lux2minn_k7_nohop"
            NAME="transfer_$CFG" ;;

    # Minnesota at k=4, Luxembourg still at k=7.
    #
    # B_k4 points a NEW dataset at the EXISTING transfer_B checkpoint, and that is
    # correct rather than a shortcut. B trains and validates on Luxembourg only,
    # so changing the private graph's coarsening changes B's TEST split and
    # nothing else -- same training records, same standardisation statistics
    # (computed from the training split), same weights. Retraining would produce
    # the same model up to seed noise and cost 6 GPU hours. C is different: its
    # training set now contains k=4 clusters, so C_k4 must be trained with
    # run_transfer.sh before it can be sampled here.
    B_k4)       DATASET="cluster_B_lux2minn_k7k4";       NAME="transfer_B" ;;
    B_k4_nohop) DATASET="cluster_B_lux2minn_k7k4_nohop"; NAME="transfer_B_nohop" ;;
    C_k4)       DATASET="cluster_C_lux2minn_k7k4";       NAME="transfer_C_k4" ;;
    C_k4_nohop) DATASET="cluster_C_lux2minn_k7k4_nohop"; NAME="transfer_C_k4_nohop" ;;

    legacy) DATASET="cluster_lux";                       NAME="cluster_lux" ;;
    *) echo "usage: sbatch scripts/run_sample_conditional.sh CFG [CKPT]" >&2
       echo "  CFG = {A|B|C}[_nohop] | {B|C}_k4[_nohop] | legacy, optionally + _v3|_centroid|_hopmlp" >&2
       exit 1 ;;
esac

NAME="$NAME$SFX"
CFG="$CFG$SFX"
CKPT="${2:-}"

if [ -z "$CKPT" ]; then
    # Take the NEWEST run directory for this configuration, then the OLDEST
    # checkpoint inside it. Both halves matter, and the first half is why this
    # script exists in its current form:
    #
    # A configuration can have several dated run directories -- the 11 August
    # jobs crashed at epoch 29 and were resubmitted on the 12th. Globbing
    # outputs/*/*$NAME*/ across all of them and taking `ls -tr | head -1` reaches
    # into the OLDEST run, which is how transfer_A and transfer_B came to be
    # sampled from epoch=9.ckpt of the crashed jobs while their finished
    # 200-epoch runs sat unused. It fails silently: plausible graphs, undertrained
    # model. Same fix as run_transfer.sh.
    #
    # "-$NAME/" with nothing after it matches the hydra run directory and not the
    # *_sample directories that this script and run_transfer.sh create.
    #
    # Within the chosen run, oldest-first: validation converges by about epoch 10
    # and then wanders (A ranges 88-106 after that), so the earliest surviving
    # top-5 checkpoint is as defensible as any and avoids picking the single
    # lowest point on a noisy curve. last.ckpt is deliberately avoided.
    RUNDIR=$(ls -td "$ROOT"/outputs/*/*-"$NAME"/ 2>/dev/null | head -1 || true)
    if [ -z "$RUNDIR" ]; then
        echo "ERROR: no run directory for $NAME under $ROOT/outputs" >&2
        exit 1
    fi
    CKPT=$(ls -tr "${RUNDIR%/}"/checkpoints/"$NAME"/epoch=*.ckpt 2>/dev/null | head -1 || true)
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
# From $CFG, not $NAME: B and B_k4 share a checkpoint (hence a NAME) but must not
# share a run directory, or the second run silently overwrites the first's
# conditional_samples.txt and you compare a file against itself.
SAMPLE_NAME="transfer_${CFG}_sample"
python3 sample_conditional.py \
    dataset="$DATASET" \
    +experiment=cluster_lux \
    ${V3[@]+"${V3[@]}"} \
    general.name="$SAMPLE_NAME" \
    general.gpus=1 \
    general.wandb=disabled \
    "general.test_only='$CKPT'"

echo
echo "Done. Find the output with:"
echo "  ls -t $ROOT/outputs/*/*${SAMPLE_NAME}*/conditional_samples.txt | head -1"
