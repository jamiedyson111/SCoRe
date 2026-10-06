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
  datasets/                 Luxembourg (DIMACS10), Oldenburg (Brinkhoff edge list),
                            Minnesota (Gleich, the old private graph)

src/          the model. Everything except the two files below is a synced copy
              of ../DiGress_upload/src and will be OVERWRITTEN by ../sync.sh
  datasets/cluster_dataset.py   the dataset class
  sample_conditional.py         drives sample_batch from a checkpoint

configs/      the cluster_* files are D_intra's; the rest are synced copies
  dataset/cluster_{A,B,C}_lux2olde_k7.yaml   the live configs, 10-dim condition
  dataset/cluster_*_lux2minn_*.yaml          Minnesota, the old private graph;
                                             kept for the results already reported
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
| B | Luxembourg | Oldenburg | zero-shot public -> private |
| C | Luxembourg + Oldenburg | Oldenburg (same as B) | does a little private data help |

Both graphs are coarsened at **k = 7** by default.

B and C are scored on identical supernodes in identical order, so the comparison
is paired.

### The private graph: Oldenburg

The private graph was Minnesota until 2026-10-05 and is now **Oldenburg**
(Brinkhoff's road network, 6,105 nodes and 7,029 edges, mean degree 2.30), a
graph over twice Minnesota's size. It is not on SuiteSparse, so `kmis_pairs.py`
reads it from the edge list in
`extraction/datasets/Brinkhoff/oldenburg/raw/OL.cedge`.

Both graphs are coarsened at **k = 7**, and unlike Minnesota no per-graph k is
needed:

| private graph | k | supernodes (n≥3) | median size | mean `h1..h7` | shared B/C test split |
|---|---|---|---|---|---|
| Luxembourg (public) | 7 | 8,815 | 11 | .23 .22 .23 .17 .09 .04 .01 | — |
| **Oldenburg** | **7** | **317** | **15** | **.20 .20 .23 .20 .11 .04 .01** | **160** |
| Minnesota (old) | 4 | 226 | 11 | .32 .40 .22 .06 0 0 0 | 114 |
| Minnesota (old) | 7 | 94 | 28 | — | 47 |

At k = 7 Oldenburg's hop histogram is close to Luxembourg's and the `k` column
is in-distribution for B, so the two caveats the Minnesota k = 4 variant carried
(`h2` ~3.2σ off the training mean, and B seeing `k = −3` where training was
always 0) do not arise. The size shift remains, but it is 15 vs 11 rather than
28 vs 11.

**Only C needs training.** A never sees the private graph, and
`pairs_A_lux2olde_k7.pt` is byte-identical to `pairs_A_lux2minn_k7.pt`. B trains
and validates on Luxembourg alone: its train and val records and its
standardisation statistics are identical too, and only its test split changed.
So the existing `transfer_A*` / `transfer_B*` checkpoints, and the A samples, are
already the Oldenburg ones. C's training set contains Oldenburg clusters (126
distinct, oversampled x17 to 24% of 8,753; validates on 31 graphs), so it is a
new model, run under the name `transfer_C_olde*` so that it cannot be confused
with the Minnesota-trained `transfer_C*`.

```bash
cd extraction && python make_transfer_datasets.py
cd extraction && python make_transfer_datasets.py --no-hops
```

Then, on BlueBEAR — samples B from the existing checkpoints, trains C:

```bash
bash scripts/run_olde_transfer.sh            # B, B_nohop sampled; C, C_nohop trained
bash scripts/run_olde_transfer.sh _v3        # B_v3 sampled; C_v3 trained
```

The Minnesota datasets, configs (`*_lux2minn_*`) and results are kept, and its
k = 4 run is still reproducible with
`make_transfer_datasets.py --private minnesota --k-private 4`. The scripts no
longer wire it up; see git history before 2026-10-05 for the k = 4 entries.

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
bash scripts/run_olde_transfer.sh _v3    # cheaper: samples B_v3, trains only C_v3
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

That runs all three tables over `results/samples/oldenburg/` and writes each to
`results/tables/oldenburg/<name>.txt`, with its LaTeX block split out as
`<name>.tex`. The Minnesota samples and tables one level up are left as they
were: supernode ids are reused across graphs, so a Minnesota samples file would
match Oldenburg test records by id and be scored against the wrong clusters.

| name | what |
|---|---|
| `intra_k7` | structural stats, Oldenburg k=7 split, B/C x hops/no-hops |
| `transfer_k7` | realism / transfer / condition tables, A B C at k=7 |
| `ablation_A` | paired hop-histogram ablation on configuration A |

A table whose samples are missing is skipped with a note, so it is safe to run on
a partial set of results. `./make_tables.sh --list` prints the names;
`./make_tables.sh intra_k7 ablation_A` runs just those. `../make_tables.sh` does
this folder and D_inter together.

**Repeated sampling (median / std).** Each table above is one random draw.
To report sampling noise, re-sample every Oldenburg configuration 5 times from
its existing checkpoint (no training; ~20 min a job) on BlueBEAR:

```bash
bash scripts/run_olde_repeats.sh                  # B_nohop, B_v3, C_nohop, C_v3 x seeds 0..4
bash scripts/run_olde_repeats.sh 5 A_nohop A_v3   # A, on Luxembourg (own sheet)
```

Copy each job's `conditional_samples_s<seed>.txt` to
`results/samples/oldenburg/repeats/<CFG>_s<seed>.txt`, then:

```bash
cd evaluation && conda activate kmis && python repeat_summary.py
```

That writes `results/tables/oldenburg/repeats_summary.tsv` (one row per metric,
a median and a std column per configuration) and `repeats_per_seed.tsv` (every
run's value). Both are tab-separated, so they paste into Excel as cells. The std
is across runs of ONE checkpoint, so it is sampling noise, not training noise.

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
| `cluster_{A,B,C}_lux2olde_k7` | **`7 2 22`** | 10 = `n, rho_in, k, h1..h7` |
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
- C oversamples 126 Oldenburg clusters x17 and validates on 31 graphs (Minnesota
  was 31 clusters x62 and 7 graphs at k = 7, 90 x24 and 22 at k = 4).
- Cluster size shift is confounded with graph shift — intrinsic to fixed-k
  transfer. Oldenburg's median is 15 against Luxembourg's 11; Minnesota's was 28
  at k = 7, which is why it needed the k = 4 variant.
- `rho_in ~ 2/n` on Luxembourg because clusters are near-trees, so a partial
  correlation controlling for `n` linearly is a false positive. Control on
  `2/n`. Not yet re-measured after the k-MIS fix.
