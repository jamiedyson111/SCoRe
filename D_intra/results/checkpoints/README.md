# Checkpoints — not in this folder

The three trained D_intra models live only on BlueBEAR. They are the one part
of D_intra that is not reproducible from this folder without a GPU run.

Best epoch per configuration, from `results/logs/`:

| Config | Best val NLL | Epoch | Validation graphs |
|---|---|---|---|
| A | 122.20 | 179 | 1147 |
| B | 118.73 | 139 | 1147 |
| C | 320.97 | 29 | 7 |

Find them:

```bash
ls -lh /rds/projects/c/chenhp-graphsyn/DiGress/outputs/*/*transfer_*/checkpoints/cluster_lux/*.ckpt
```

Copy them here, renamed per configuration:

```bash
for c in A B C; do
  scp "USER@bluebear.bham.ac.uk:$(ssh USER@bluebear.bham.ac.uk \
    "ls -t /rds/projects/c/chenhp-graphsyn/DiGress/outputs/*/*transfer_${c}/checkpoints/cluster_lux/epoch=*.ckpt | head -1")" \
    ./${c}.ckpt
done
```

Also worth taking, for exact reproducibility of each run's resolved config:

```bash
scp USER@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/DiGress/outputs/*/*transfer_A/.hydra/config.yaml ./A_config.yaml
```

With a checkpoint present you can resample without retraining:

```bash
cd ../../model/src
python3 sample_conditional.py \
    dataset=cluster_A_lux2minn_k7 \
    +experiment=cluster_lux \
    general.name=resample_A \
    general.gpus=0 \
    general.wandb=disabled \
    "general.test_only='../../results/checkpoints/A.ckpt'"
```

The quoting matters: the `=` in `epoch=N.ckpt` breaks hydra's override grammar
unless the value carries its own quotes, and shell quotes are stripped before
hydra sees them.
