# Checkpoints — not in this folder

The trained D_intra models live only on BlueBEAR. They are the one part of
D_intra that is not reproducible from this folder without a GPU run.

## Where they are

```
/rds/projects/c/chenhp-graphsyn/D_intra/outputs/<date>/<time>-transfer_<CFG>/
    checkpoints/transfer_<CFG>/epoch=N.ckpt
    .hydra/config.yaml                       the run's resolved config
```

```bash
ls -lh /rds/projects/c/chenhp-graphsyn/D_intra/outputs/*/*transfer_*/checkpoints/transfer_*/*.ckpt
```

## Picking one: NOT the newest file

`ModelCheckpoint` runs with `monitor='val/epoch_NLL', save_top_k=5`, so the five
`epoch=*.ckpt` files in a run are the five **best** epochs, written in the order
they were found. The newest file is therefore whichever good epoch happened to
be found last, not the best one — `ls -t | head -1` picks the wrong checkpoint
and does it silently.

This has already cost one end-to-end run: stage 1 sampled from `epoch=199`
(val NLL 89.58) when `epoch=99` (86.65) was sitting in the same directory.

Pick by score instead. Every checkpoint stores the monitor's own record:

```bash
python3 - <<'PY'
import torch, glob
for f in sorted(glob.glob('checkpoints/transfer_B/epoch=*.ckpt')):
    cb = torch.load(f, map_location='cpu')['callbacks']
    for v in cb.values():
        if isinstance(v, dict) and 'best_model_score' in v:
            print(f"{f}  best_model_path={v['best_model_path']}")
            break
PY
```

or read the validation curve out of the run's log in `results/logs/`.

## Best epoch per configuration

**Not recorded here any more.** The table that used to sit here (A 179,
B 139, C 29) predated the 2026-08-12 retrain and was being copied into write-ups
after it had stopped being true. Only one figure has been re-verified since:

| Config | Epoch | Val NLL | Note |
|---|---|---|---|
| B | 99 | 86.65 | the checkpoint to use for private inference |

Re-derive the rest from `results/logs/` or the checkpoints themselves rather
than trusting a table that no run updates.

## Copying one down

```bash
CKPT=/rds/projects/c/chenhp-graphsyn/D_intra/outputs/2026-08-12/09-40-31-transfer_B/checkpoints/transfer_B/epoch=99.ckpt
scp "dysonjz@bluebear.bham.ac.uk:$CKPT" ./B.ckpt
scp dysonjz@bluebear.bham.ac.uk:$(dirname $(dirname $(dirname $CKPT)))/.hydra/config.yaml ./B_config.yaml
```

## Resampling from one

```bash
sbatch scripts/run_sample_conditional.sh B                      # newest transfer_B run
sbatch scripts/run_sample_conditional.sh B /abs/path/epoch=N.ckpt
sbatch scripts/run_sample_conditional.sh B_k4                   # Minnesota at k=4
```

By hand, from `D_intra/src`:

```bash
cd src
python3 sample_conditional.py \
    dataset=cluster_B_lux2minn_k7 \
    +experiment=cluster_lux \
    general.name=resample_B \
    general.gpus=0 \
    general.wandb=disabled \
    "general.test_only='/abs/path/to/epoch=99.ckpt'"
```

The quoting matters: the `=` in `epoch=N.ckpt` breaks hydra's override grammar
unless the value carries its own quotes, and shell quotes are stripped before
hydra sees them.
