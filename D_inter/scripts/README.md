# D_inter scripts (BlueBEAR)

SLURM job scripts. Submit from the project root,
`/rds/projects/c/chenhp-graphsyn/D_inter`. Configurations: **A** (train Luxembourg
→ test Luxembourg), **B** (train Luxembourg → test Oldenburg; the same model as A),
**C** (the Luxembourg model fine-tuned on the Oldenburg training pairs → test
Oldenburg). These are the internal-degree D_inter (`*_deg`: node features X_i =
normalised internal degree instead of L_i, c_ij = [n_i, n_j, w_ij, r_{i->j},
r_{j->i}]); see the top-level D_inter README.

| script | what | example |
|---|---|---|
| `submit_inter_deg.sh` | **everything** for the internal-degree D_inter: train A, fine-tune C, test pairs (generated and true X), full graph | `bash scripts/submit_inter_deg.sh` |
| `run_pair.sh` | **train** D_inter on a dataset; a 3rd argument fine-tunes from a run or checkpoint | `sbatch scripts/run_pair.sh pair_C_lux2olde_k7_s80_deg pair_C_s80_deg pair_lux_s80_deg` |
| `run_pair_debug.sh` | a short debug training run | `sbatch scripts/run_pair_debug.sh` |
| `run_sample_pairs_transfer.sh` | **sample** one cross block per test pair, N seeds (B, C: generated X^; `--real-x`: true X) | `sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 B` |
| `run_pairs_repeats.sh` | submit the above for several configurations | `bash scripts/run_pairs_repeats.sh 5 B C` |
| `run_sample_pairs.sh` | the original sampler (random test conditions, with replacement) | `sbatch scripts/run_sample_pairs.sh /abs/epoch=179.ckpt` |
| `run_evaluate_pairs.sh` | score the original sampler's output | `sbatch scripts/run_evaluate_pairs.sh` |
| `setup.sh` | sourced by every job: modules, the `digress` env, orca permissions | -- |

Arguments, not environment variables: BlueBEAR does not pass the submitting
shell's environment into the job.

## Checkpoints

No validation split: A trains a fixed 180 epochs, C fine-tunes it for 100 at
lr 5e-5, and each final checkpoint (`last.ckpt`) is used.

- **A and B:** the newest `pair_lux_s80_deg` run (B's training data is A's).
- **C:** the newest `pair_C_s80_deg` run.

## Outputs

`run_sample_pairs_transfer.sh` writes
`outputs/<date>/<time>-pairs_<CFG>_s80_deg[_realX]_sample_x<N>/pair_samples_s0.pt … _s<N-1>.pt`.
`bash fetch_s80.sh deg` (local) copies them to `results/samples/D_inter/s80/<CFG>_deg[_realX]_s<seed>.pt`;
score with `evaluation/transfer_summary.py --configs B_deg C_deg B_deg_realX C_deg_realX BS1 BS2`.
