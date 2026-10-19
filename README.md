# DP-Graph-syn

Generate a synthetic version of a private graph from a coarse, privacy-safe
release. The graph is coarsened with **k-MIS** into supernodes; only per-supernode
and per-pair summaries are released; two conditional diffusion models (both
DiGress) rebuild the fine graph from them:

- **D_intra** generates the inside of each supernode, from c_i = [n, ρ_in, k, h1..hk].
- **D_inter** generates the edges between two supernodes, from
  c_ij = [n_i, ρ_i, n_j, ρ_j, w_ij, k] and the two insides.

The blocks are then assembled into one graph on the private graph's own nodes.

**Public graph:** Luxembourg (114,599 nodes). **Private graph:** Oldenburg (6,105
nodes). **k = 7** throughout.

## The full pipeline

| # | step | runs on | where | command |
|---|---|---|---|---|
| 1 | **k-MIS + D_intra datasets** | local, `kmis` env | `D_intra/extraction/` | `python make_transfer_datasets.py` |
| 2 | **Train D_intra** | BlueBEAR GPU | `D_intra/` | `sbatch scripts/run_transfer.sh B_v3` |
| 3 | **Evaluate D_intra** | BlueBEAR GPU → local | `D_intra/` | `sbatch scripts/run_sample_conditional.sh --repeats 5 B_v3`, then `evaluation/repeat_summary.py` |
| 4 | **k-MIS + D_inter datasets** | local, `kmis` env | `D_inter/dataprep/` | `python make_pair_blocks.py`, `python make_transfer_pairs.py` |
| 5 | **Train D_inter** | BlueBEAR GPU | `D_inter/` | `sbatch scripts/run_pair.sh pair_C_lux2olde_k7 pair_C_olde` |
| 6 | **Evaluate D_inter** | BlueBEAR GPU → local | `D_inter/` | `bash scripts/run_pairs_repeats.sh 5 B C`, then `evaluation/transfer_summary.py` |
| 7 | **Assemble the full graph** | local → BlueBEAR GPU | `D_intra/private_inference/`, `D_inter/private_inference/` | release → D_intra on every supernode → D_inter on every pair |
| 8 | **Evaluate the full graph** | local | `D_inter/private_inference/` | `python syn_repeats_summary.py` |

Steps 1-3 are in [D_intra/README.md](D_intra/README.md), steps 4-8 in
[D_inter/README.md](D_inter/README.md), each with exact commands, inputs, outputs
and run times.

## Configurations

| | trained on | tested on |
|---|---|---|
| **A** | Luxembourg | Luxembourg (in-distribution reference) |
| **B** | Luxembourg | Oldenburg (transfer, no private training data) |
| **C** | Luxembourg + 40% of Oldenburg | Oldenburg (the same test supernodes / pairs as B) |

Oldenburg is split 40 / 10 / 50 (train / val / test) so that B and C are scored
on the same held-out half. The whole-graph evaluation (steps 7-8) uses all of
Oldenburg; it is held out for B, not for C.

## Repository

```
D_intra/        D_intra: code, configs, datasets, job scripts. Uploaded to BlueBEAR.
D_inter/        D_inter: code, configs, datasets, job scripts, assembly. Uploaded to BlueBEAR.
results/        everything produced: tables, samples, checkpoints, the release (local only)
baselines/      BS1 (random) and BS2 (Chung-Lu) baselines
competitors/    ICLR [2], HiGen [5] wrappers; Huiping's ER / SC_RG (Baseline/)
archive/        old and unused files, kept for reference
k-mis-pool/     the k-MIS coarsening code (Bacciu et al.)
graph-coarsening/  Loukas coarsening library, for reference
make_graph.sh   steps 7-8 for ONE graph, end to end, including the BlueBEAR jobs
make_tables.sh  the single-run tables of both projects
```

Every folder has a README.

## Environments

| env | where | used for |
|---|---|---|
| `kmis` | local | k-MIS coarsening, datasets (steps 1, 4), the release, all scoring |
| `digress` | BlueBEAR, `/rds/projects/c/chenhp-graphsyn/envs/digress` | training and sampling (steps 2, 3, 5, 6, 7) |

Every BlueBEAR job sources its project's `scripts/setup.sh`, which loads the
modules and activates `digress`. Arguments are passed on the command line, never
as environment variables: BlueBEAR does not carry the submitting shell's
environment into the job.

## Uploading to BlueBEAR

```bash
rsync -avz --exclude __pycache__ --exclude 'data/*/processed' D_intra/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_intra/
rsync -avz --exclude __pycache__ --exclude 'data/*/processed' D_inter/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_inter/
```

- `data/*/processed` is excluded: it is the PyTorch Geometric cache, rebuilt on
  BlueBEAR from `data/*/raw`, and a locally built one may not match its torch.
- **Never add `--delete`.** BlueBEAR holds things that are not in the local copy
  -- trained checkpoints and samples (`outputs/`), logs, its own caches and
  `private_inference/out/` -- and `--delete` would remove all of them.
