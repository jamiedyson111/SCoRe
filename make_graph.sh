#!/usr/bin/env bash
# Build ONE synthetic graph out of both models, and score it. One command,
# including the two stages that have to run on BlueBEAR.
#
#     ./make_graph.sh                    # the real thing, end to end
#     ./make_graph.sh --real             # local self-test, no cluster, ~12 s
#     ./make_graph.sh --dry-run          # print every remote command, run none
#
# D_intra generates the inside of each supernode, D_inter the edges between
# them, and this stitches the blocks into a single L^syn over the private
# graph's own node set -- then puts it beside G^priv in a table.
#
#     stage 0  make_private_release.py     local, CPU    the coarse release
#     stage 1  sample_private_clusters.py  BlueBEAR GPU  one L_i per supernode
#     stage 2  sample_private_blocks.py    BlueBEAR GPU  one B_ij per pair
#     stage 3  assemble_syn.py             local, CPU    L^syn + the table
#
# Stages 0 and 3 run here, in the `kmis` env; stages 1 and 2 are submitted to
# BlueBEAR, waited on, and their output pulled back. Everything already done is
# skipped, so re-running after an interruption costs nothing.
#
# --real replaces the generated blocks with the private graph's OWN blocks. No
# checkpoint, no GPU, no cluster, and it must reproduce G^priv edge for edge --
# so it is both a demonstration of the pipeline and the test that the assembly
# is sound. Run it first; if it fails, no generated result is worth reading.
#
# ssh is multiplexed over one connection, so a password is asked for at most
# once. Set BLUEBEAR_HOST / BLUEBEAR_ROOT to point somewhere else.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GRAPH=oldenburg
K=                                     # blank -> per-graph default, see below
REAL=0
FORCE=0
LOCAL=0
DRY=0
WAIT=1
DATASET=                               # blank -> derived from K, see below
DATASET_INTER=pair_lux                 # the D_inter config stage 2 samples under
CKPT_INTRA=${CKPT_INTRA:-}             # blank -> discover the newest on BlueBEAR
CKPT_INTER=${CKPT_INTER:-}
HOST=${BLUEBEAR_HOST:-dysonjz@bluebear.bham.ac.uk}
RROOT=${BLUEBEAR_ROOT:-/rds/projects/c/chenhp-graphsyn}
POLL=${POLL:-60}                       # seconds between squeue checks
MAX_WAIT=${MAX_WAIT:-21600}            # 6 h; stage 1 on Luxembourg asks for 4

usage () { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; }

while [ $# -gt 0 ]; do
    case $1 in
        --graph)       GRAPH=$2; shift 2 ;;
        --k)           K=$2; shift 2 ;;
        --dataset)       DATASET=$2; shift 2 ;;
        --dataset-inter) DATASET_INTER=$2; shift 2 ;;
        --ckpt-intra)  CKPT_INTRA=$2; shift 2 ;;
        --ckpt-inter)  CKPT_INTER=$2; shift 2 ;;
        --real)        REAL=1; shift ;;
        --local)       LOCAL=1; shift ;;   # never touch the cluster
        --dry-run)     DRY=1; shift ;;
        --no-wait)     WAIT=0; shift ;;    # submit and return; re-run to collect
        --force)       FORCE=1; shift ;;   # rebuild the release
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# Oldenburg is the PRIVATE graph and is coarsened at k=7, the same locality as
# Luxembourg, the public one: its supernodes come out at median 15 (Luxembourg
# 11) with a hop histogram close to Luxembourg's, so no per-graph k is needed.
# Passing --k overrides this.
[ -z "$K" ] && K=7
# The D_intra dataset carries the standardisation the checkpoint was trained
# under. Any other private graph has to name its dataset explicitly -- e.g. the
# old Minnesota run was --graph minnesota --k 4 --dataset cluster_B_lux2minn_k7k4.
if [ -z "$DATASET" ]; then
    case "$GRAPH:$K" in
        oldenburg:7) DATASET=cluster_B_lux2olde_k7 ;;
        *) echo "ERROR: no default D_intra dataset for --graph $GRAPH --k $K;" >&2
           echo "       pass one with --dataset." >&2
           exit 2 ;;
    esac
fi

PI_INTRA=$ROOT/D_intra/private_inference
PI_INTER=$ROOT/D_inter/private_inference
# Stages 1 and 2 both default to writing beside the release, so one directory
# holds the whole chain and nothing is copied between the two projects.
WORK=$ROOT/results/private_inference      # local; BlueBEAR keeps its own private_inference/out
RWORK=$RROOT/D_intra/private_inference/out
RELEASE=$WORK/release_${GRAPH}_k${K}.pt
CLUSTERS=$WORK/clusters_${GRAPH}_k${K}.pt
BLOCKS=$WORK/blocks_${GRAPH}_k${K}.pt
if [ $REAL -eq 1 ]; then
    OUT=$WORK/selftest_${GRAPH}_k${K}
else
    OUT=$WORK/syn_${GRAPH}_k${K}
fi

step () { printf '\n=== %s ===\n' "$1"; }
say  () { printf '    %s\n' "$1"; }

# ---- env ------------------------------------------------------------------
ENV_NAME=${KMIS_ENV:-kmis}
if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
    hook=""
    for base in "${CONDA_EXE:-/nonexistent}" "$HOME/anaconda3/bin/conda" \
                "$HOME/miniconda3/bin/conda" /opt/conda/bin/conda; do
        h=${base%/bin/conda}/etc/profile.d/conda.sh
        if [ -f "$h" ]; then hook=$h; break; fi
    done
    set +u
    activated=0
    if [ -n "$hook" ]; then
        . "$hook"
        conda activate "$ENV_NAME" 2>/dev/null && activated=1
    fi
    set -u
    if [ $activated -eq 0 ]; then
        echo "ERROR: could not activate the '$ENV_NAME' env." >&2
        echo "       Activate it yourself and re-run, or set KMIS_ENV=<name>." >&2
        exit 1
    fi
fi
# torch_sparse is what separates kmis from digress, and without it the failure
# lands deep inside the extraction import rather than here.
python -c "import torch_sparse, h5py" 2>/dev/null || {
    echo "ERROR: the coarsening stack (torch_sparse, h5py) is not importable in" >&2
    echo "       $(which python). Stages 0 and 3 need the kmis env, which is why" >&2
    echo "       this script runs HERE and only submits stages 1-2 to BlueBEAR." >&2
    exit 1
}

# ---- ssh, over a single multiplexed connection ----------------------------
SSH_DIR=$(mktemp -d)
trap 'ssh -O exit -o ControlPath="$SSH_DIR/ctl" "$HOST" 2>/dev/null || true; rm -rf "$SSH_DIR"' EXIT
SSH=(ssh -o ControlMaster=auto -o ControlPath="$SSH_DIR/ctl" -o ControlPersist=10m
     -o ConnectTimeout=15)

rsh () {   # run a command on BlueBEAR, or just show it under --dry-run
    if [ $DRY -eq 1 ]; then printf '    [dry-run] ssh %s %s\n' "$HOST" "$1" >&2; return 0; fi
    "${SSH[@]}" "$HOST" "$1"
}
rsh_quiet () { "${SSH[@]}" "$HOST" "$1" 2>/dev/null; }   # for tests, never dry

# ---- stage 0: the coarse release ------------------------------------------
step "stage 0  release  ($GRAPH, k=$K)"
if [ -f "$RELEASE" ] && [ $FORCE -eq 0 ]; then
    say "already built: ${RELEASE#$ROOT/}   (--force to rebuild)"
else
    # --k-hist stays 7 at every k: the models were trained on a 7-column hop
    # histogram, and rings above k are provably empty rather than padded.
    ( cd "$PI_INTRA" && python make_private_release.py --graph "$GRAPH" --k "$K" --k-hist 7 )
fi

# ---- stages 1 and 2: the two models ---------------------------------------
if [ $REAL -eq 1 ]; then
    step "stages 1-2  skipped (--real: the private graph's own blocks)"
    ASSEMBLE=(--real)
elif [ -f "$CLUSTERS" ] && [ -f "$BLOCKS" ]; then
    step "stages 1-2  already here"
    say "${CLUSTERS#$ROOT/}"
    say "${BLOCKS#$ROOT/}"
    ASSEMBLE=(--clusters "$CLUSTERS" --blocks "$BLOCKS")
elif [ $LOCAL -eq 1 ]; then
    echo "ERROR: --local, but the generated blocks are not here yet." >&2
    echo "       Drop --local to run stages 1-2 on BlueBEAR, or use --real." >&2
    exit 1
else
    step "stages 1-2  on BlueBEAR ($HOST)"

    if [ $DRY -eq 0 ]; then
        rsh_quiet true || {
            echo "ERROR: cannot reach $HOST over ssh." >&2
            echo "       Check the connection, or --real for the local self-test." >&2
            exit 1
        }
    fi

    # --- push the release ---------------------------------------------------
    if [ $DRY -eq 1 ]; then
        say "[dry-run] rsync -az ${WORK#$ROOT/}/ $HOST:$RWORK/"
    else
        rsh "mkdir -p $RWORK"
        push=("$RELEASE")
        # The true blocks let stage 2 be validated without stage 1; they are
        # small and stage 0 writes them beside the release, so send them too.
        [ -f "$WORK/clusters_${GRAPH}_k${K}_true.pt" ] &&
            push+=("$WORK/clusters_${GRAPH}_k${K}_true.pt")
        rsync -az -e "ssh -o ControlPath=$SSH_DIR/ctl" "${push[@]}" "$HOST:$RWORK/"
        say "release pushed to $RWORK/"
    fi

    # --- checkpoints --------------------------------------------------------
    # NEWEST IS NOT ENOUGH. Several configurations have more than one dated run
    # -- the 11 August jobs crashed at epoch 29 and were resubmitted -- and,
    # worse, runs of the SAME name can be trained on different conditions. The
    # 2026-08-18 pair_lux run is V2 (11-dim c_ij, y width 23) while private
    # inference must use V1 (6-dim, width 18), because the release publishes six
    # numbers per pair and cannot supply r. Picking it produced a size mismatch
    # in load_state_dict several minutes into a GPU job.
    #
    # So match on what actually distinguishes them: the pairs_file the run was
    # trained on has to be the one the dataset stage 1/2 will run with. `name`
    # does not work -- pair_lux and pair_lux_v2 both declare `name: pair_lux`,
    # because src/main.py dispatches on it.
    pairs_file_of () {   # the pairs_file a local dataset config names
        sed -n "s/^pairs_file:[[:space:]]*'\?\([^']*\)'\?[[:space:]]*$/\1/p" "$1" | head -1
    }
    # Walk run directories newest first and take the first trained on that file.
    remote_pick () {     # remote_pick <outputs glob> <ckpt subdir> <pairs file>
        rsh_quiet "for d in \$(ls -dt $1 2>/dev/null); do
                       d=\${d%/}
                       grep -qE \"pairs_file:[[:space:]]*'?$3'?[[:space:]]*\\\$\" \"\$d/.hydra/config.yaml\" 2>/dev/null || continue
                       c=\$(ls -t \"\$d/checkpoints/$2\"/epoch=*.ckpt 2>/dev/null | head -1)
                       [ -n \"\$c\" ] && { echo \"\$c\"; break; }
                   done"
    }
    WANT_INTRA=$(pairs_file_of "$ROOT/D_intra/configs/dataset/${DATASET}.yaml")
    WANT_INTER=$(pairs_file_of "$ROOT/D_inter/configs/dataset/${DATASET_INTER}.yaml")
    # B IS NOT RETRAINED FOR A NEW PRIVATE GRAPH, so requiring a run trained on
    # the lux2olde file would reject the only correct checkpoint. B trains and
    # validates on Luxembourg alone; swapping or recoarsening the private graph
    # changes its TEST split and nothing else. Verified against the files:
    #   lux2minn_k7 vs lux2minn_k7k4 (2026-08-18) and
    #   lux2minn_k7 vs lux2olde_k7   (2026-10-05):
    # identical cond_names, cond_mean and cond_std, identical train and val
    # records, 6611/1322 either way -- only the test split differs. So a run
    # trained on pairs_B_lux2minn_k7.pt is a valid source for either dataset,
    # and only for that reason is the match widened here.
    ALT_INTRA=${WANT_INTRA/lux2olde/lux2minn}
    ALT_INTRA=${ALT_INTRA/k7k4/k7}
    if [ "$ALT_INTRA" != "$WANT_INTRA" ]; then
        PAT_INTRA="($WANT_INTRA|$ALT_INTRA)"
    else
        PAT_INTRA="$WANT_INTRA"
    fi
    if [ $DRY -eq 1 ]; then
        say "[dry-run] would take the newest run trained on the right pairs file:"
        say "  D_intra: $PAT_INTRA  (dataset=$DATASET)"
        say "  D_inter: $WANT_INTER  (dataset=$DATASET_INTER)"
        : "${CKPT_INTRA:=<newest transfer_B checkpoint>}"
        : "${CKPT_INTER:=<newest pair_lux checkpoint>}"
        DEP_SHOWN="--dependency=afterok:<stage 1 job>"
    fi
    if [ -z "$CKPT_INTRA" ] && [ $DRY -eq 0 ]; then
        CKPT_INTRA=$(remote_pick "$RROOT/D_intra/outputs/*/*/" transfer_B "$PAT_INTRA" || true)
    fi
    if [ -z "$CKPT_INTER" ] && [ $DRY -eq 0 ]; then
        CKPT_INTER=$(remote_pick "$RROOT/D_inter/outputs/*/*/" pair_lux "$WANT_INTER" || true)
    fi
    if [ $DRY -eq 0 ]; then
        for pair in "D_intra:$CKPT_INTRA" "D_inter:$CKPT_INTER"; do
            if [ -z "${pair#*:}" ]; then
                proj=${pair%%:*}
                want=$WANT_INTRA; [ "$proj" = D_inter ] && want=$WANT_INTER
                echo "ERROR: no $proj checkpoint on BlueBEAR was trained on $want," >&2
                echo "       which is what this run's dataset requires. A checkpoint" >&2
                echo "       trained on anything else has a different condition width" >&2
                echo "       and dies in load_state_dict minutes into the job." >&2
                echo "       Pass one explicitly with --ckpt-$(echo "$proj" | cut -d_ -f2)" >&2
                echo "       if you are sure, or check: ssh $HOST \\" >&2
                echo "         grep pairs_file $RROOT/$proj/outputs/*/*/.hydra/config.yaml" >&2
                exit 1
            fi
        done
        say "D_intra ckpt: $CKPT_INTRA"
        say "            trained on $WANT_INTRA${ALT_INTRA:+ (or $ALT_INTRA)}"
        say "D_inter ckpt: $CKPT_INTER"
        say "            trained on $WANT_INTER"
    fi

    # --- submit -------------------------------------------------------------
    JIDS=""
    DEP=${DEP_SHOWN:-}
    if [ $DRY -eq 1 ] || ! rsh_quiet "test -f $RWORK/clusters_${GRAPH}_k${K}.pt"; then
        CMD1="cd $RROOT/D_intra && sbatch --parsable private_inference/run_private_clusters.sh '$CKPT_INTRA' $RWORK/release_${GRAPH}_k${K}.pt $DATASET"
        J1=$(rsh "$CMD1")
        [ $DRY -eq 0 ] && { say "stage 1 submitted: job $J1"; JIDS=$J1; DEP="--dependency=afterok:$J1"; }
    else
        say "stage 1 already done on BlueBEAR, skipping"
    fi
    if [ $DRY -eq 1 ] || ! rsh_quiet "test -f $RWORK/blocks_${GRAPH}_k${K}.pt"; then
        CMD2="cd $RROOT/D_inter && DATASET=$DATASET_INTER sbatch --parsable $DEP private_inference/run_private_blocks.sh '$CKPT_INTER' $RWORK/release_${GRAPH}_k${K}.pt $RWORK/clusters_${GRAPH}_k${K}.pt"
        J2=$(rsh "$CMD2")
        [ $DRY -eq 0 ] && { say "stage 2 submitted: job $J2"; JIDS="${JIDS:+$JIDS,}$J2"; }
    else
        say "stage 2 already done on BlueBEAR, skipping"
    fi

    if [ $DRY -eq 1 ]; then
        printf '\n(dry run: nothing was submitted)\n'
        exit 0
    fi

    if [ -n "$JIDS" ] && [ $WAIT -eq 0 ]; then
        cat <<EOF

    submitted: $JIDS  (--no-wait, so not waiting)
    Re-run this script when they finish and it will collect and score.
    Cancel with:  ssh $HOST scancel $JIDS
EOF
        exit 0
    fi

    # --- wait ---------------------------------------------------------------
    if [ -n "$JIDS" ]; then
        say "waiting on $JIDS  (checking every ${POLL}s; Ctrl-C is safe, the jobs keep running)"
        waited=0
        while :; do
            # A failed ssh means unknown, not done -- keep waiting rather than
            # concluding the jobs finished and reporting missing files.
            left=$(rsh_quiet "squeue -h -j $JIDS -o %i 2>/dev/null | wc -l") || left=-1
            [ "${left:-0}" -eq 0 ] && break
            if [ $waited -ge $MAX_WAIT ]; then
                echo "ERROR: still queued after $((MAX_WAIT / 60)) min. Jobs: $JIDS" >&2
                echo "       They keep running -- re-run this script to collect." >&2
                exit 1
            fi
            sleep "$POLL"
            waited=$((waited + POLL))
        done
        say "jobs finished after $((waited / 60)) min"
    fi

    # --- collect ------------------------------------------------------------
    # A job that failed leaves no file, and a stage 2 whose dependency never
    # cleared leaves none either. Say which, and show the tail of its log.
    for f in clusters blocks; do
        if ! rsh_quiet "test -f $RWORK/${f}_${GRAPH}_k${K}.pt"; then
            echo "ERROR: BlueBEAR produced no ${f}_${GRAPH}_k${K}.pt." >&2
            proj=D_intra; [ "$f" = blocks ] && proj=D_inter
            echo "--- tail of the newest $proj log ---" >&2
            rsh_quiet "tail -25 \$(ls -t $RROOT/$proj/logs/priv_*.err $RROOT/$proj/logs/priv_*.out 2>/dev/null | head -1)" >&2 || true
            exit 1
        fi
    done
    # One file per call: rsync globs a remote path with its own matcher, which
    # knows * but not {a,b} -- brace expansion is the shell's, and the remote
    # path never reaches one. The connection is multiplexed, so two calls cost
    # nothing.
    for f in clusters blocks; do
        rsync -az -e "ssh -o ControlPath=$SSH_DIR/ctl" \
              "$HOST:$RWORK/${f}_${GRAPH}_k${K}.pt" "$WORK/"
    done
    say "pulled back: ${CLUSTERS#$ROOT/}, ${BLOCKS#$ROOT/}"
    ASSEMBLE=(--clusters "$CLUSTERS" --blocks "$BLOCKS")
fi

# ---- stage 3: assemble and score ------------------------------------------
step "stage 3  assemble and score"
( cd "$PI_INTER" && python assemble_syn.py --release "$RELEASE" \
      "${ASSEMBLE[@]}" --out-dir "$OUT" )

cat <<EOF

  ${OUT#$ROOT/}/
    syn_summary.txt          the table above
    syn_table.tex            the same as LaTeX
    L_syn_${GRAPH}_k${K}.pt        the assembled graph
    degree_distribution.csv  synthetic vs real, per degree
EOF
