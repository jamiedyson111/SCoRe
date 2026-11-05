# DP-Graph-syn

Generate a synthetic version of a private graph from a coarse, privacy-safe
release. The graph is coarsened with **k-MIS** into supernodes; only per-supernode
and per-pair summaries are released; two conditional diffusion models (both
DiGress) rebuild the fine graph from them:

- **D_intra** generates the inside of each supernode, from c_i = [n, ρ_in, k, h1..hk].
- **D_inter** generates the edges between two supernodes, from
  c_ij = [n_i, n_j, w_ij, r_i→j, r_j→i] (sizes, number of cross edges, and the
  fraction of each side's nodes with 0, 1, 2, 3, 4+ cross edges) and each node's
  normalised internal degree X_i = d_in(v) / (n_i − 1).

The blocks are then assembled into one graph on the private graph's own nodes.

**Public graph:** Luxembourg (114,599 nodes). **Private graph:** Oldenburg (6,105
nodes). **k = 7** throughout.

## The full pipeline

| # | step | runs on | where | command |
|---|---|---|---|---|
| 1 | **k-MIS + D_intra datasets** | local, `kmis` env | `D_intra/extraction/` | `python make_transfer_datasets.py` |
| 2 | **Train D_intra** | BlueBEAR GPU | `D_intra/` | `sbatch scripts/run_transfer.sh A_v3` (and `C_v3`, `*_nohop_centroid`) |
| 3 | **Evaluate D_intra** | BlueBEAR GPU → local | `D_intra/` | `bash scripts/run_olde_repeats.sh 5 B_v3 C_v3`, then `evaluation/repeat_summary.py` |
| 4 | **k-MIS + D_inter datasets** | local, `kmis` env | `D_inter/dataprep/` | `python make_pair_blocks.py`, `python make_transfer_pairs.py` |
| 5 | **Train D_inter** | BlueBEAR GPU | `D_inter/` | `sbatch scripts/run_pair.sh pair_lux_s80 pair_lux_s80` (and C) |
| 6 | **Evaluate D_inter** | BlueBEAR GPU → local | `D_inter/` | `bash scripts/run_pairs_repeats.sh 5 B C`, then `evaluation/transfer_summary.py` |
| 7 | **Assemble the full graph** | local → BlueBEAR GPU | `D_intra/private_inference/`, `D_inter/private_inference/` | release → D_intra on every supernode → D_inter on every pair |
| 8 | **Evaluate the full graph** | local | `D_inter/private_inference/` | `python syn_repeats_summary.py` (whole graph + held-out test part) |
| 9 | **Costs** | local | `D_intra/extraction/`, `D_inter/private_inference/` | `python coarsening_stats.py`, `python cost_summary.py` (runtime and GPU memory recorded by every job) |

**All of it at once:** `bash /rds/projects/c/chenhp-graphsyn/D_intra/scripts/submit_s80.sh` on
BlueBEAR submits steps 2, 3, 5, 6 and 7 as one dependency chain; `bash fetch_s80.sh`
locally copies every output and cost record back into `results/`.

Steps 1-3 are in [D_intra/README.md](D_intra/README.md), steps 4-8 in
[D_inter/README.md](D_inter/README.md), each with exact commands, inputs, outputs
and run times.

## Split and configurations

**80 / 20 train / test by supernode, no validation split, the same supernodes for
D_intra and D_inter.** D_intra's step 1 shuffles each graph's supernodes and writes
the assignment to `D_intra/data/splits/<graph>_k7_s80.json`; D_inter trains only
on pairs whose two supernodes are both training supernodes, and tests on every
pair touching a test supernode. Models train for a fixed number of epochs and
the final checkpoint is used.

| | trained on | tested on |
|---|---|---|
| **A** | Luxembourg (80% of supernodes) | Luxembourg's held-out 20% |
| **B** | the same -- **A's model** | Oldenburg's held-out 20% (63 supernodes, 187 pairs) |
| **C** | **fine-tuning**: A's model, trained further on Oldenburg's 80% only (100 epochs, lr 5e-5; Oldenburg's conditions standardised with Luxembourg's statistics, A's edge marginal kept) | the same Oldenburg 20% as B |

(An earlier C trained one model jointly on Luxembourg + Oldenburg; its runs are
the `C_v3` / `pair_C_s80` ones. The final tables use the fine-tuned C, `Cft_*`.)

**The final tables** -- D_intra and D_inter on each graph, the full graph as
relative errors, runtime and memory -- come from `bash submit_final.sh` on
BlueBEAR, then `bash fetch_s80.sh final` and `bash make_final_tables.sh` locally
(→ `results/final_tables.xlsx`).

The full graph (steps 7-8) is generated entirely by the models and scored twice:
as a whole (all 6105 Oldenburg nodes; held out for B, ~80% training supernodes
for C) and as its **test part**, the subgraph on the 63 held-out supernodes, which
contains no training supernode for either.

## Running on another pair of graphs (e.g. DBLP → Collab)

The method is the same for any public graph G_s and private graph G_t:

| | trained on | then fine-tuned on | tested on |
|---|---|---|---|
| **A** | G_s train (80% of its supernodes) | -- | G_s test (20%) |
| **B** | the same model as A | -- | G_t test (20%) |
| **C** | the same model as A | G_t train (80%), no G_s data | the same G_t test as B |

What the datasets must satisfy:

1. **One coarsening, one k, one split.** Coarsen each graph once with k-MIS (the
   same k for G_s and G_t, and for D_intra and D_inter), then split its
   **supernodes** 80 / 20 with a fixed seed. Write the split to
   `D_intra/data/splits/<graph>_k<k>_s80.json`; everything else reads it. B and C
   share G_t's test supernodes exactly.
2. **D_inter's pairs follow the supernode split.** A pair is a training pair only
   if both supernodes are training supernodes, and a test pair if it touches any
   test supernode.
3. **Standardise with G_s's statistics.** B's and C's conditions use A's
   `cond_mean` / `cond_std` (computed on G_s's training split), and C's files carry
   A's edge marginal (`edge_marginal`), so the fine-tune starts from A's model as is.
4. **Supernode size.** DiGress works on dense n × n matrices: supernodes with
   n_i < 3 are dropped, and D_inter drops pairs with n_i + n_j > 128. Dense or
   social graphs give giant supernodes at large k (DBLP at k = 7: largest 296,322
   nodes; at k = 2: median 5, 95th percentile 40, largest 3,166), so use a small k
   and cap the supernode size for D_intra, reporting how many supernodes and pairs
   the cap drops. Road networks are fine at k = 7 (largest supernode: 79 nodes on Oldenburg, 71 on Luxembourg).
5. **The graphs** are undirected and simple (duplicate edges merged, self-loops
   dropped); `coarsening_comparison/load_graph.py` reads `.txt`, `.mtx` and `.mat`
   this way.

The record formats, which the builders write and the training code reads:

- D_intra (`D_intra/extraction/README.md`): one record per supernode,
  `{sn, graph, n, nodes, centre, edge_index, y_raw, y}`, `edge_index` in local ids
  with both directions of every edge, `y_raw` = [n, rho_in, k, h1..hk].
- D_inter (`D_inter/dataprep/README.md`): one record per connected pair,
  `{sn_i, sn_j, n_i, n_j, edge_index, w, r, r_ji, rho_i, rho_j, y_raw}`, where
  `edge_index` is the pair's combined graph (L_i, L_j and the cross block B_ij; local
  ids 0..n_i-1 are side i) and `y_raw` = [n_i, n_j, w_ij, r_i→j (5), r_j→i (5)].
- Each file: `splits` = [train, val, test] (val = 256 training records, logging
  only), `cond_names`, `cond_mean`, `cond_std`, `cond_constant`, and for C
  `edge_marginal`.

The builders to follow: `D_intra/extraction/make_transfer_datasets.py` (A / B / C)
and `make_finetune_datasets.py` (C as a fine-tune); `D_inter/dataprep/
make_pair_blocks.py` (A), `make_transfer_pairs.py` (B, C). They currently name
Luxembourg and Oldenburg in the code (`kmis_pairs.GRAPHS`, `NAME` / `K` in
`make_pair_blocks.py`, the Oldenburg loader in `make_transfer_pairs.py`), so a new
pair of graphs means registering the graphs there or passing them in.

## Repository

```
D_intra/        D_intra: code, configs, datasets, job scripts. Uploaded to BlueBEAR.
D_inter/        D_inter: code, configs, datasets, job scripts, assembly. Uploaded to BlueBEAR.
results/        everything produced: tables, samples, checkpoints, the release (local only)
baselines/      BS1 (Huiping's ER) and BS2 (her SC-RG): runner and full-graph version
competitors/    Baseline/: Huiping's ER (BS1) and SC-RG (BS2) scripts
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
