# SCoRe: Structure-Aware and Transferable Coarse-to-Fine Graph Reconstruction

Code for SCoRe. A graph is partitioned into connected supernodes of bounded size, and a
compact structural synopsis of it is extracted. Two conditional discrete diffusion models
(both built on DiGress) then rebuild the fine graph from the synopsis, one block at a time:

- **D_intra** generates the internal edges of each supernode V_i from
  c_i = [n_i, rho_i, K, h_i]: size, internal density, the coarsening parameter K and the
  centroid-relative hop histogram h_i (the fraction of non-centroid vertices at each
  distance 1..K from the centroid). The centroid is marked by a fixed node class.
- **D_inter** generates the cross edges B_ij of each connected pair from
  c_ij = [n_i, n_j, w_ij, r_i->j, r_j->i]: both sizes, the number of cross edges, and the
  directional cross-degree histograms (the fraction of each side's vertices with 0, 1, 2,
  3, 4+ cross edges). Each vertex also carries its normalised internal degree, taken from
  the generated internal structure at inference.

The blocks are assembled into the synthetic graph. Models are trained on a source graph
G_s and applied to the synopsis S_K(G_t) of a target graph G_t, either directly
(source-only transfer) or after fine-tuning on part of the target (target adaptation).

## Transfer tasks

| Source -> target | K | Partition | Cross-degree bins |
|---|---|---|---|
| Luxembourg -> Oldenburg | 7 | degree-ranked K-MIS (`k-mis-pool`) | 0, 1, 2, 3, 4+ |
| Luxembourg -> Vermont | 7 | degree-ranked K-MIS (`k-mis-pool`) | 0, 1, 2, 3, 4+ |
| DBLP -> ogbl-collab | 3 | size-aware K-MIS (`.npz` partition) | 0, 1, 2-3, 4-7, 8+ |
| LiveJournal -> YouTube | 4 | CB-KMIS, b = 500, l = 10 (`.npz` partition) | 0, 1, 2, 3, 4+ |

All graphs are undirected and simple (edges symmetrised, self-loops and repeated edges
removed, every connected component kept). Supernodes with fewer than three vertices are
not modelled.

## Configurations

Supernodes are split 80 / 20 into training and held-out sets (seed 0); a pair is a
training pair if both its supernodes are training supernodes and a test pair if it
touches a held-out supernode. The same split serves D_intra and D_inter.

| | Trained on | Evaluated on | Paper |
|---|---|---|---|
| **A** | G_s (80% of its supernodes) | G_s held-out 20% | source evaluation |
| **B** | the same model as A | G_t held-out 20% | source-only transfer (ZS) |
| **C** | A fine-tuned on G_t's 80% (lr 5e-5, A's condition statistics and edge marginal) | the same G_t 20% | target adaptation (FT) |
| **T** | G_t's 80% only, no source pretraining | the same G_t 20% | target-only reference |
| **O** | plain DiGress on G_t's 80%, no condition | the same G_t 20% | DiGress baseline |

ER and SC-RG (`competitors/Baseline/`, labelled BS1 and BS2 in the tables) are the
non-learning references: ER places the exact edge counts uniformly; SC-RG builds hop
layers from h_i and a V_i-side degree sequence from r_i->j.

**Evaluation.** Road networks: five sampling seeds of each trained model, mean and
standard deviation across seeds. ogbl-collab and YouTube: one generation scored on five
fixed, non-overlapping groups of held-out instances (`results/competitor_data/<graph>/
test_groups_*.json`), mean and standard deviation across groups. The whole graph is
generated in full for the road networks; for ogbl-collab it is a partial
reconstruction (one group of 357 held-out supernodes and the 7,917 pairs touching them
are generated, everything else kept), and for YouTube it is not run.

### Names in the code

The code predates the paper's terminology and keeps its original names:

| Code | Paper |
|---|---|
| public graph | source graph G_s |
| private graph | target graph G_t |
| release (`make_private_release.py`) | target synopsis S_K(G_t) |
| `private_inference/` | whole-graph reconstruction |
| `v3` | D_intra with the hop embedding and centroid class |
| `remb` | D_inter with the cross-degree embedding |
| `Cft` | configuration C (fine-tuned) |
| BS1 / BS2 | ER / SC-RG |

## Road networks: the full pipeline

| # | step | runs on | where | command |
|---|---|---|---|---|
| 1 | **k-MIS + D_intra datasets** | local, `kmis` env | `D_intra/extraction/` | `python make_transfer_datasets.py` |
| 2 | **Train D_intra** | BlueBEAR GPU | `D_intra/` | `sbatch scripts/run_transfer.sh A_v3` (and `C_v3`, `*_nohop_centroid`) |
| 3 | **Evaluate D_intra** | BlueBEAR GPU → local | `D_intra/` | `bash scripts/run_olde_repeats.sh 5 B_v3 C_v3`, then `evaluation/repeat_summary.py` |
| 4 | **k-MIS + D_inter datasets** | local, `kmis` env | `D_inter/dataprep/` | `python make_pair_blocks.py`, `python make_transfer_pairs.py` |
| 5 | **Train D_inter** | BlueBEAR GPU | `D_inter/` | `sbatch scripts/run_pair.sh pair_lux_s80 pair_lux_s80` (and C) |
| 6 | **Evaluate D_inter** | BlueBEAR GPU → local | `D_inter/` | `bash scripts/run_pairs_repeats.sh 5 B C`, then `evaluation/transfer_summary.py` |
| 7 | **Assemble the full graph** | local → BlueBEAR GPU | `D_intra/private_inference/`, `D_inter/private_inference/` | target synopsis -> D_intra on every supernode -> D_inter on every pair |
| 8 | **Evaluate the full graph** | local | `D_inter/private_inference/` | `python syn_repeats_summary.py` (whole graph + held-out test part) |
| 9 | **Costs** | local | `D_intra/extraction/`, `D_inter/private_inference/` | `python coarsening_stats.py`, `python cost_summary.py` (runtime and GPU memory recorded by every job) |

**All of it at once:** `bash /rds/projects/c/chenhp-graphsyn/D_intra/scripts/submit_s80.sh` on
BlueBEAR submits steps 2, 3, 5, 6 and 7 as one dependency chain; `bash fetch_s80.sh`
locally copies every output and cost record back into `results/`.

Steps 1-3 are in [D_intra/README.md](D_intra/README.md), steps 4-8 in
[D_inter/README.md](D_inter/README.md), each with exact commands, inputs, outputs
and run times.

### Road-network runs

The held-out split is 63 Oldenburg supernodes and 187 pairs (Vermont: 1,104 supernodes,
2,881 pairs). Models train for a fixed number of epochs and the final checkpoint is
used. An earlier C trained one model jointly on Luxembourg + Oldenburg; its runs are the
`C_v3` / `pair_C_s80` ones. The tables use the fine-tuned C, `Cft_*`.

**The final tables** (D_intra and D_inter on each graph, the whole graph as relative
errors, runtime and memory) come from `bash submit_final.sh` on BlueBEAR, then
`bash fetch_s80.sh final` and `bash make_final_tables.sh` locally
(-> `results/final_tables.xlsx`).

The whole graph (steps 7-8) is generated entirely by the models and scored twice: as a
whole, and as its **test part**, the subgraph on the held-out supernodes, which contains
no training supernode for either B or C.

## Running on another pair of graphs

The method is the same for any source graph G_s and target graph G_t (configurations A, B
and C above). What the datasets must satisfy:

1. **One coarsening, one K, one split.** Coarsen each graph once (the
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
   n_i < 3 are dropped; the road-network D_inter datasets drop pairs with
   n_i + n_j > 128, and the YouTube D_inter training set pairs above 800. Dense or
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

## Collaboration and social graphs

DBLP -> ogbl-collab (K = 3) and LiveJournal -> YouTube (K = 4) use partitions computed
elsewhere and read from `.npz` files (fine graph, assignment, centroids):

- `D_intra/extraction/make_partition_datasets.py` and `D_inter/dataprep/make_partition_pairs.py`
  build the A / B / C datasets; `D_inter/dataprep/make_stratified_pairs.py` builds the reduced
  LiveJournal / YouTube D_inter training sets (stratified on w_ij / (n_i + n_j) and pair size).
- `baselines/make_test_groups.py` draws the fixed, non-overlapping test groups
  (`results/competitor_data/<graph>/test_groups_*.json`) shared with the competing methods.
- `D_intra/src/sample_conditional.py` and `D_inter/src/sample_pairs_conditional.py` sample them
  (`+subset_file`, `+gen_samples`, `+partial_file`); `repeat_summary.py` and
  `transfer_summary.py` score them per group (`--groups`); `baselines/group_baselines.py`
  produces ER and SC-RG on the same groups.
- `D_inter/private_inference/partial_whole_graph.py` assembles and scores the partial
  whole-graph reconstruction of Collab; `spectrum_only.py` computes the spectrum metric of
  large graphs by eigenvalue counting.
- The BlueBEAR jobs are in `scripts/bluebear/`.

## Repository

```
D_intra/        D_intra: code, configs, datasets, job scripts. Uploaded to BlueBEAR.
D_inter/        D_inter: code, configs, datasets, job scripts, assembly. Uploaded to BlueBEAR.
results/        tables, run costs and the fixed test groups (samples and checkpoints stay local)
baselines/      BS1 (Huiping's ER) and BS2 (her SC-RG): runners, full-graph and test-group versions
competitors/    Baseline/: Huiping's ER (BS1) and SC-RG (BS2) scripts
coarsening_comparison/  coarsening statistics of the candidate graphs
scripts/bluebear/       job submitters for the collaboration and social-graph experiments
archive/        old and unused files, kept for reference (not tracked)
k-mis-pool/     the k-MIS coarsening code (Bacciu et al.)
graph-coarsening/  Loukas coarsening library, for reference
make_graph.sh   steps 7-8 for ONE graph, end to end, including the BlueBEAR jobs
make_tables.sh  the single-run tables of both projects
make_final_tables.sh, make_vermont_tables.sh   the repeated-run tables (Oldenburg, Vermont)
fetch_s80.sh, submit_final.sh                  fetch results from / submit the final runs on BlueBEAR
```

Every folder has a README.

## Environments

| env | where | used for |
|---|---|---|
| `kmis` | local | coarsening, datasets (steps 1, 4), the target synopsis, all scoring |
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
