#!/bin/bash
# D_inter with the r EMBEDDING on the road networks: the condition becomes
# c_ij = [n_i, n_j, w_ij, MLP_r(r_ij), MLP_r(r_ji)] (model.r_embed_dim=1; one MLP for
# both directions, src/diffusion_model_discrete.py). A SUBMITTER, not a job: run it
# once on the BlueBEAR login node:
#
#     bash submit_road_remb.sh
#
# The three road D_inter models that use r, retrained with the same data and
# schedule as the originals (scripts/run_pair.sh), under new names so the originals
# are kept:
#
#   pair_lux_s80_deg_remb      A: Luxembourg train pairs, 180 epochs, batch 32, lr 2e-4
#                              (also the B model, for Oldenburg and Vermont)
#   pair_C_s80_deg_remb        C Oldenburg: A fine-tuned on the Oldenburg training
#                              pairs, 100 epochs at lr 5e-5          (after A)
#   pair_C_verm_s80_deg_remb   C Vermont: the same on Vermont        (after A)
#
# The no-r models (_nor) do not use r and are not retrained. Each C fine-tunes A's
# newest last.ckpt and waits (afterok) for A. Jobs are written to logs/jobs/*.sh and
# submitted as files: BlueBEAR cancels the same body passed with --wrap.
set -euo pipefail

P=/rds/projects/c/chenhp-graphsyn
R=$P/D_inter
for f in $R/data/pair_lux/raw/pairs_inter_luxembourg_osm_k7_s80_deg.pt \
         $R/data/pair_lux/raw/pairs_inter_C_lux2{olde,verm}_k7_s80_deg.pt \
         $R/configs/dataset/pair_lux_s80_deg.yaml \
         $R/configs/dataset/pair_C_lux2{olde,verm}_k7_s80_deg.yaml; do
    [ -f "$f" ] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
grep -q r_embed_dim $R/src/diffusion_model_discrete.py || {
    echo "ERROR: upload D_inter/src first (no r embedding in diffusion_model_discrete.py)" >&2; exit 1; }

SB=(--account=chenhp-graphsyn --qos=bbgpu --gres=gpu:1 --nodes=1 --ntasks=1
    --cpus-per-task=4 --mem=32G --parsable)

# job NAME FINETUNE_RUN PYTHON_ARGS... -> the job's shell script
job() {
    local name=$1 ft=$2; shift 2
    cat <<EOF
#!/bin/bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 PYTHONUNBUFFERED=1
ROOT=$R
. \$ROOT/scripts/setup.sh
export PYTHONPATH=\$ROOT:\${PYTHONPATH:-}
FT=()
if [ "$ft" != "-" ]; then
    CK=\$(ls -t \$ROOT/outputs/*/*-$ft/checkpoints/$ft/last*.ckpt 2>/dev/null | head -1 || true)
    [ -n "\$CK" ] || { echo "ERROR: no last*.ckpt for $ft" >&2; exit 1; }
    echo "fine-tuning from \$CK"
    FT=("general.finetune_from='\$CK'")
fi
cd \$ROOT/src
python3 main.py +experiment=pair_lux general.name=$name general.wandb=disabled model.r_embed_dim=1 \${FT[@]+"\${FT[@]}"} $*
EOF
}
# submit JOBNAME TIME DEPENDENCY NAME FINETUNE PYTHON_ARGS...
submit() {
    local jn=$1 time=$2 dep=$3; shift 3
    local d=(); [ -n "$dep" ] && d=(--dependency=afterok:$dep)
    mkdir -p $R/logs/jobs
    job "$@" > $R/logs/jobs/$jn.sh
    (cd $R && sbatch "${SB[@]}" ${d[@]+"${d[@]}"} --job-name=$jn --time=$time \
        --output=$R/logs/${jn}_%j.out --error=$R/logs/${jn}_%j.err $R/logs/jobs/$jn.sh)
}

A=$(submit remb_A 08:00:00 "" pair_lux_s80_deg_remb - dataset=pair_lux_s80_deg)
CO=$(submit remb_C_olde 03:00:00 "$A" pair_C_s80_deg_remb pair_lux_s80_deg_remb \
     dataset=pair_C_lux2olde_k7_s80_deg train.n_epochs=100 train.lr=0.00005)
CV=$(submit remb_C_verm 06:00:00 "$A" pair_C_verm_s80_deg_remb pair_lux_s80_deg_remb \
     dataset=pair_C_lux2verm_k7_s80_deg train.n_epochs=100 train.lr=0.00005)
echo "A -> $A   C Oldenburg -> $CO   C Vermont -> $CV   (each C after A)"
echo 'Check A'"'"'s log (D_inter/logs/remb_A_*.out): "Size of the input features 9 2 25" is the'
echo 'dataset; the network itself sees y of width 17 once r is embedded.'
