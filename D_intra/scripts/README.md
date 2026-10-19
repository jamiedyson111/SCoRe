# D_intra scripts (BlueBEAR)

SLURM job scripts. Submit from the project root,
`/rds/projects/c/chenhp-graphsyn/D_intra`. Configurations: **A** (train Luxembourg
→ test Luxembourg), **B** (train Luxembourg → test Oldenburg), **C** (train
Luxembourg + Oldenburg → test Oldenburg), each optionally `_nohop`, and suffixed
`_v3` (centroid node type + hop embedding), `_centroid` or `_hopmlp`.

| script | what | example |
|---|---|---|
| `run_transfer.sh` | **train** one configuration (200 epochs), then sample its test split | `sbatch scripts/run_transfer.sh B_v3` |
| `run_all_transfer.sh` | train several configurations as a chain, each after the previous succeeds | `bash scripts/run_all_transfer.sh A_v3 B_v3 C_v3` |
| `run_olde_transfer.sh` | everything for the Oldenburg transfer tables (train C, resample A/B) | `bash scripts/run_olde_transfer.sh _v3` |
| `run_sample_conditional.sh` | **sample** a trained configuration's test split, N seeds | `sbatch scripts/run_sample_conditional.sh --repeats 5 B_v3` |
| `run_olde_repeats.sh` | submit `run_sample_conditional.sh --repeats N` for several configurations | `bash scripts/run_olde_repeats.sh 5 B_v3 C_v3` |
| `setup.sh` | sourced by every job: modules, the `digress` env, orca permissions | -- |

`--repeats` must be an argument, not an environment variable: BlueBEAR does not
pass the submitting shell's environment into the job.

## Outputs

Each job writes a hydra run directory, `outputs/<date>/<time>-<name>/`:
checkpoints under `checkpoints/<name>/epoch=N.ckpt`, and samples as
`conditional_samples.txt` (one run) or `conditional_samples_s0.txt … _s4.txt`
(`--repeats 5`, in a directory ending `_x5`). Copy repeated samples to the local
`results/samples/D_intra/oldenburg/repeats/<CFG>_s<seed>.txt` and score them with
`evaluation/repeat_summary.py`.

The checkpoint used for a configuration is the newest run directory of that name,
and the oldest surviving top-5 checkpoint in it.
