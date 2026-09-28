# D_intra — conditional intra-cluster diffusion

`D_intra` generates the internal structure of a single coarse supernode given a
summary vector `c_i = [n_i, rho_in_i, k, h_i]`. It is DiGress (Vignac et al.,
ICLR 2023) made conditional by direct conditioning on the graph-level `y`
channel.

`h_i` is the **hop histogram**: `h_i(r) = |{v in c_i \ s_i : d_G(v, s_i) = r}| /
(n_i - 1)` for `r = 1..k`, the proportion of non-centroid members at each
shortest-path distance from the centroid. It expands to `k = 7` columns, so
`c_i` is 10-dimensional.

It is also the only entry that carries information independent of cluster size:
`n_i` is *imposed* at sampling time, `k` is constant, and `rho_in_i` is
degenerate with `n_i` (R² 0.997 on Luxembourg) because every cluster is a
near-tree once connectivity is enforced. Without `h_i` there is nowhere the
claim "the model responds to the condition" can be demonstrated.

This folder is **self-contained**: it holds the model code and everything else
needed to run, so it can be uploaded to BlueBEAR as-is. The shared model code has
its single source of truth in `../DiGress_upload/` and is copied in by
`../sync.sh` — edit it there, not here.

> **The k-MIS assignment bug was fixed on 2026-08-06.** Every `.pt` in `data/`
> and every result below predates it and describes clusters built by the old
> rule. See "Rebuilding after the k-MIS fix".

## Layout

```
extraction/   graph -> (G_i, c_i) pairs.  Runs locally in the `kmis` env.
  kmis_pairs.py             k-MIS coarsening + induced-subgraph extraction
  make_transfer_datasets.py builds configurations A / B / C
  kmis/  benchmark/         coarsening code and graph loaders
  datasets/                 Luxembourg (DIMACS10) + Minnesota (Gleich)

src/          the model. Everything except the two files below is a synced copy
              of ../DiGress_upload/src and will be OVERWRITTEN by ../sync.sh
  datasets/cluster_dataset.py   the dataset class
  sample_conditional.py         drives sample_batch from a checkpoint

configs/      the cluster_* files are D_intra's; the rest are synced copies
  dataset/cluster_{A,B,C}_lux2minn_k7.yaml   the live configs, 10-dim condition
  dataset/cluster_{B,C}_lux2minn_k7k4.yaml   Minnesota at k=4, Luxembourg at k=7
                                             (no A: A is Luxembourg-only and its
                                             records are identical to the k7 A)
  dataset/cluster_lux.yaml, lux2minn.yaml    LEGACY, 3-dim condition -- see below
  experiment/cluster_lux.yaml                shared by all of them

data/cluster_lux/raw/   the pairs_*.pt dataset files
evaluation/             evaluate_clusters.py, compare_transfer.py
  make_intra_table.py   the generated-vs-real structural table, one column per
                        configuration, plain text and LaTeX
results/                samples/, logs/, checkpoints/ (checkpoints EMPTY)
scripts/                SLURM jobs
```

## The three configurations

| | Train | Test | Purpose |
|---|---|---|---|
| A | Luxembourg | Luxembourg | in-distribution reference |
| B | Luxembourg | Minnesota | zero-shot public -> private |
| C | Luxembourg + Minnesota | Minnesota (same as B) | does a little private data help |

Both graphs are coarsened at **k = 7** by default.

B and C are scored on identical supernodes in identical order, so the comparison
is paired.

### The Minnesota k = 4 variant

`--k-private 4` recoarsens the private graph alone, leaving Luxembourg at k = 7.
The point is that **k = 7 is the wrong locality for Minnesota**: Minnesota is the
denser graph (mean degree 2.50 vs 2.09), so the same k gives supernodes more than
twice the size the model trained on.

| Minnesota at | supernodes (n≥3) | median size | shared B/C test split |
|---|---|---|---|
| k = 7 | 94 | 28 | 47 |
| k = 4 | 226 | **11** | **114** |

Luxembourg at k = 7 has median size 11. So k = 4 puts the private clusters on the
same size distribution the model was trained on, removing the size-shift confound
listed under "Known issues", and it more than doubles the test split — the k = 7
B/C comparison was underpowered at 48 supernodes (see `compare_ablation.py`).

Two things make this work without touching the architecture:

- **The hop histogram keeps its 7 columns.** A cluster built at k = 4 has
  provably empty rings 5–7 (nearest-centre assignment puts every member within 4
  hops of its centre), so `sum(h)` is still exactly 1 and the input width stays
  22. Verified: `max(h5..h7) = 0.000000` across all 226 clusters. This is why the
  existing checkpoints load unchanged.
- **B needs no retraining.** B trains and validates on Luxembourg only, so the
  private graph's k changes B's *test* split and nothing else. The existing
  `transfer_B` checkpoint is the k = 4 B model. Only C is a new model.

What does change, and must be reported:

- `h`'s distribution shifts hard. Minnesota k = 4 has mean
  `h = [0.32, 0.40, 0.22, 0.06, 0, 0, 0]` against Luxembourg's
  `[0.23, 0.22, 0.23, 0.17, 0.09, 0.04, 0.01]`; standardised on Luxembourg
  statistics, `h2` sits ~3.2σ above the training mean.
- `k` is reported truthfully per graph (7 for Luxembourg, 4 for Minnesota). In C
  that column now has variance and is standardised normally. In **B** it does
  not: `standardise()` leaves a zero-variance feature on its raw scale, so B
  receives `(4 − 7)/1 = −3` where training was always 0. That is the honest
  encoding of "the granularity changed", but B's numbers should not be read as
  if the condition were in-distribution.

```bash
cd extraction && python make_transfer_datasets.py --k-public 7 --k-private 4
cd extraction && python make_transfer_datasets.py --k-public 7 --k-private 4 --no-hops
```

Then, on BlueBEAR — samples B from the existing checkpoints, trains C:

```bash
bash scripts/run_k4_transfer.sh
```

## Version 3: centroid node type and hop embedding

Two changes from the method section (V A 1) b). Both are switches, off by
default, so earlier runs and checkpoints are unaffected.

- **Centroid node type** (`dataset.mark_centroid=true`). `X` goes from one dummy
  class to two: `[0, 1]` for the centroid, `[1, 0]` for every other member. `h`
  is measured from the centroid, which the model could not otherwise identify.
  `X` is observed, not generated: it is held clean while noising, left out of the
  NLL, and imposed at every sampling step with node 0 as the centroid. Only `E`
  is diffused. No `.pt` rebuild is needed; every record already carries
  `centre`. The processed cache gets a `_centroid` suffix so it cannot collide
  with the 1-class one.
- **Hop embedding** (`model.hop_embed_dim=1`). `h1..h7` pass through
  `MLP_h: R^7 -> R^1` (hidden width `model.hop_embed_hidden`, 32) before `y`
  reaches the transformer, so `c_i = [n, rho_in, k, e_h]` is 4 dims, not 10.
  `MLP_h` trains end to end with the denoising loss. The stored `y` is unchanged.

Same datasets, splits and settings as A/B/C, so the results compare directly:

```bash
bash scripts/run_all_transfer.sh A_v3 B_v3 C_v3
sbatch scripts/run_sample_conditional.sh B_k4_v3    # B_v3 on Minnesota at k=4
```

`_centroid` and `_hopmlp` in place of `_v3` run each change on its own. Expected
log line: `Size of the input features 8 2 16` for `_v3` (`8 2 22` centroid only,
`7 2 16` hop embedding only).

## Running it

Extraction and evaluation need the `kmis` env (torch 1.11 + torch_sparse).
Training needs `digress` (torch 2.0.1).

**1. Build the datasets** — writes `pairs_{A,B,C}_<tag>.pt` into `data/cluster_lux/raw/`:

```bash
cd extraction && conda activate kmis
python make_transfer_datasets.py
```

**2. Refresh the shared code** (only needed after editing `../DiGress_upload/`):

```bash
cd .. && ./sync.sh D_intra
```

**3. Upload and run:**

```bash
rsync -avz D_intra/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_intra/
```

All three, in order, one command:

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra && bash scripts/run_all_transfer.sh
```

Note `bash`, not `sbatch` — this is a submitter that runs on the login node. It
queues three chained jobs (`--dependency=afterok`), so B starts only if A
succeeded and C only if B did. Each keeps its own 6h allocation, which one
18-hour job would not be granted. Pass a subset in order if you need one:
`bash scripts/run_all_transfer.sh B C`.

Or submit a single configuration directly:

```bash
sbatch scripts/run_transfer.sh A
```

Submitting all three independently is faster and is fine **once each config has
been processed at least once**. On a fresh upload it is not: all three share
`data/cluster_lux/` as their PyG root, and while the per-config caches do not
collide, `pre_filter.pt` and `pre_transform.pt` have the same name for every
dataset under that root, so simultaneous first runs can race on them.

Each job trains, then samples the test split to `conditional_samples.txt`.

If a chained job fails, the ones behind it never run — they sit in
`DependencyNeverSatisfied` and need cancelling by hand.

**4. Produce the tables** — one command, no arguments, no `conda activate`
(the script activates `kmis` itself):

```bash
./make_tables.sh
```

That runs all five tables over `results/samples/` and writes each to
`results/tables/<name>.txt`, with its LaTeX block split out as
`results/tables/<name>.tex`:

| name | what |
|---|---|
| `intra_k7` | structural stats, Minnesota k=7 split, B/C x hops/no-hops |
| `intra_k4` | structural stats, Minnesota k=4 split, B/C x hops/no-hops |
| `transfer_k7` | realism / transfer / condition tables, A B C at k=7 |
| `transfer_k4` | the same, B and C at k=4 (A is Luxembourg-only) |
| `ablation_A` | paired hop-histogram ablation on configuration A |

A table whose samples are missing is skipped with a note, so it is safe to run on
a partial set of results. `./make_tables.sh --list` prints the names;
`./make_tables.sh intra_k4 ablation_A` runs just those. `../make_tables.sh` does
this folder and D_inter together.

Tables are a LOCAL step -- the samples come off BlueBEAR, the tables are built
here. If you do want them on the cluster, the script works there unchanged:
there is no `kmis` env on BlueBEAR and none is needed, since these scripts import
only numpy/scipy/networkx/torch, so it falls back to sourcing `setup.sh` and runs
in `digress`, giving identical numbers.

The underlying scripts still take their own flags if you need a non-standard
comparison:

```bash
cd evaluation && conda activate kmis
python compare_transfer.py --a ../results/samples/A.txt \
    --b ../results/samples/B.txt --c ../results/samples/C.txt --latex
```

## Rebuilding after the k-MIS fix

`_cluster_k_mis` used to assign each node to the lowest-*ranked* centre within k
hops rather than the *nearest* one, which left 15.8% of induced subgraphs
disconnected. Fixed in `extraction/kmis/reduce.py`. Consequences:

| | before | after |
|---|---|---|
| Luxembourg clusters (n>=3) | 7,650 | 8,815 |
| Minnesota clusters (n>=3) | 78 | 94 |
| nodes assigned differently | — | 23.5% Lux, 35.1% Minn |
| real connectivity | 84.2% | 100% |

So: rebuild the datasets, retrain all three, redo every table.

**Delete the processed cache first.** The filenames do not change, so PyG will
silently reuse the stale processed copies and you will train on the old clusters
with no error raised:

```bash
rm -rf data/cluster_lux/processed/
```

Confirm the fix with `python check_kmis_assignment.py` in `k-mis-pool/` — expect
`ASSIGNMENT differs on 0/... nodes` and `0 disconnected`.

## Legacy: `dataset=cluster_lux`

`cluster_lux.yaml` and `lux2minn.yaml` point at `pairs_luxembourg_osm_k7.pt` and
`pairs_lux2minn_k7.pt`, which are still on the **3-dim** condition
`[n, rho_in, k]`. `make_transfer_datasets.py` does not write them — their builder
(`make_pairs.py`) lives in `k-mis-pool/`, not here, and carries its own duplicate
`COND` list and its own copy of `hop_histogram`.

`scripts/run_cluster.sh` and `run_cluster_debug.sh` target `cluster_lux`, so
their `7 2 15` checks are correct as written. Do not run a model trained on
A/B/C against `cluster_lux`, or vice versa: the widths do not match and the
failure is a shape error at the first batch, not a silent one.

Use `scripts/run_transfer.sh {A|B|C}` for everything current.

## Sanity check on every run

Grep any training log for:

```
Size of the input features 7 2 22
```

| dataset | expected | condition |
|---|---|---|
| `cluster_{A,B,C}_lux2minn_k7` | **`7 2 22`** | 10 = `n, rho_in, k, h1..h7` |
| `cluster_{B,C}_lux2minn_k7k4` | **`7 2 22`** | unchanged — `h` stays 7 wide at k=4 |
| `*_nohop` | `7 2 15` | 3 = `n, rho_in, k` |
| `cluster_lux` (legacy) | `7 2 15` | 3 = `n, rho_in, k` |

22 = 10 condition + 1 time + 11 spectral.

- **If a transfer run says 15**, you are training on the old 3-dim condition:
  either the rebuilt `pairs_*.pt` did not upload, or `data/cluster_lux/processed/`
  still holds a cache keyed on the same filename. PyG reuses that silently.
- **If it says 12**, the condition is not reaching the model at all and you are
  training an unconditional model with no error raised.

Also expect the edge marginal — `[0.8768, 0.1232]` for A and B,
`[0.9083, 0.0917]` for C. The two differ, so **NLL is not comparable between C
and A/B**. And `X_CE 0.000` (one node class, nothing to learn) with `E_CE`
around 0.3 and falling — the only loss that means anything here.

## Results as of 2026-08-05 (pre-fix)

Configuration A, 766 held-out Luxembourg clusters:

| | generated | real | random |
|---|---|---|---|
| connected % | 56.5 | 85.4 | — |
| acyclic % | 75.3 | 88.6 | — |
| edges (mean) | 13.22 | 13.62 | 13.22 |
| density (mean) | 0.2120 | 0.2124 | 0.2120 |
| degree dist. | 0.0544 | 0 | 0.599 |
| spectrum | 0.0862 | 0 | 0.144 |

Transfer, same 40 Minnesota supernodes: C beats B on 5 of 6 measures (edge-count
MAE 2.98 vs 6.53, degree distribution 0.163 vs 0.304). C is also the only
configuration that beats the `m = n-1` baseline.

Training: A best val NLL 122.20 @ epoch 179; B 118.73 @ 139; C 320.97 @ 29.
A and B converge by epoch ~10 — 200 epochs is roughly 10x more than needed.
C validates on only 7 graphs and its curve swings 321–1114, so its checkpoint
selection is unreliable and its numbers are not reproducible by retraining.

## Known issues

- Coarsening Luxembourg at k=7 gives 8,815 supernodes; the source paper's
  Table 2 reports 10,929. Eight orderings tried, none reproduces it. **Not**
  explained by the assignment bug — centre selection was always correct.
- C oversamples 31 Minnesota clusters x62 and validates on 7 graphs. At k = 4
  this improves to 90 distinct clusters x24 and 22 validation graphs.
- Cluster size shift is confounded with graph shift (Minnesota median 28 nodes
  vs Luxembourg 11) — intrinsic to fixed-k transfer, and **the reason for the
  k = 4 variant above**, which brings the private median to 11.
- `rho_in ~ 2/n` on Luxembourg because clusters are near-trees, so a partial
  correlation controlling for `n` linearly is a false positive. Control on
  `2/n`. Not yet re-measured after the k-MIS fix.
