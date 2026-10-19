# D_inter scripts (BlueBEAR)

SLURM job scripts. Submit from the project root,
`/rds/projects/c/chenhp-graphsyn/D_inter`. Configurations: **A** (train Luxembourg
→ test Luxembourg), **B** (train Luxembourg → test Oldenburg; the same model as A),
**C** (train Luxembourg + oversampled Oldenburg → test Oldenburg).

| script | what | example |
|---|---|---|
| `run_pair.sh` | **train** D_inter on a dataset | `sbatch scripts/run_pair.sh pair_C_lux2olde_k7 pair_C_olde` |
| `run_pair_debug.sh` | a short debug training run | `sbatch scripts/run_pair_debug.sh` |
| `run_sample_pairs_transfer.sh` | **sample** one cross block per test pair, N seeds | `sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 B` |
| `run_pairs_repeats.sh` | submit the above for several configurations | `bash scripts/run_pairs_repeats.sh 5 B C` |
| `run_sample_pairs.sh` | the original sampler (random test conditions, with replacement) | `sbatch scripts/run_sample_pairs.sh /abs/epoch=179.ckpt` |
| `run_evaluate_pairs.sh` | score the original sampler's output | `sbatch scripts/run_evaluate_pairs.sh` |
| `setup.sh` | sourced by every job: modules, the `digress` env, orca permissions | -- |

Arguments, not environment variables: BlueBEAR does not pass the submitting
shell's environment into the job.

## Checkpoints

- **A and B:** the V1 `pair_lux` model,
  `outputs/2026-08-06/16-57-54-pair_lux/checkpoints/pair_lux/epoch=179.ckpt`.
- **C:** the newest `pair_C_olde` run, at the checkpoint its ModelCheckpoint
  recorded as best (`epoch=219`).

## Outputs

`run_sample_pairs_transfer.sh` writes
`outputs/<date>/<time>-pairs_<CFG>_sample_x<N>/pair_samples_s0.pt … _s<N-1>.pt`.
Copy them to the local `results/samples/D_inter/transfer/<CFG>_s<seed>.pt` and score with
`evaluation/transfer_summary.py`.
