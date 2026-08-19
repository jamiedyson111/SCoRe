# D_intra — notes to self

Written 2026-08-05. Everything here was checked against the code or the data,
not remembered.

## What it is, in one line

A conditional discrete diffusion model that generates the inside of one coarse
supernode, given three numbers describing it: `c_i = [n, rho_in, k]`.

It is DiGress with the condition routed into the existing `y` channel. No new
architecture.

## The pipeline

```
Luxembourg graph
   -> k-MIS coarsening at k=7            8,815 supernodes
   -> induced subgraph per supernode     8,815 kept (n >= 3, none dropped)
   -> (G_i, c_i) pairs                   c_i = [n, rho_in, k]
   -> split 75/15/10                     6611 / 1322 / 882
   -> DiGress trains on (G_i | c_i)
   -> sample: give c_i, get a graph back
```

## The condition's journey — six stops

| # | Where | What happens |
|---|---|---|
| 1 | `kmis_pairs.py` | `[n, rho_in, k, h1..h7]` computed by plain arithmetic -> `y_raw` |
| 2 | `standardise()` | `(x - mean) / std` using TRAIN stats only -> `y` |
| 3 | `cluster_dataset.py` | becomes `Data(y=...)`, shape (1,10) |
| 4 | `forward()` | `hstack` with 11 structural features + timestep `t` -> width 22 |
| 5 | `mlp_in_y` | 2-layer MLP 22 -> 128 -> 64. **This is "the embedding"** |
| 6 | FiLM | 4 Linears per layer x 6 layers = ~400k params |

Standardisation numbers (Luxembourg train, configurations A and B):

```
          mean      std
n        12.9307   6.5281
rho_in    0.1890   0.0820
k         7.0      1.0     <- fake; real std is 0, replaced to avoid /0
h1        0.2296   0.0709
h2        0.2252   0.0554
h3        0.2300   0.0540
h4        0.1680   0.0695
h5        0.0893   0.0683
h6        0.0448   0.0526
h7        0.0131   0.0292
```

C differs (it includes oversampled Minnesota): mean `n` 16.71, `rho_in` 0.1646.

**std = typical distance from the mean.** Needed because raw `n` (4-71) would
swamp raw `rho_in` (0-0.5) purely through units.

**`sum(h) = 1` exactly** (verified: max deviation 1.19e-07, float32 precision).
Every member is within `k` hops of its assigned centre — guaranteed by
nearest-centre assignment plus k-maximality — so the 7 rings partition the
`n_i - 1` non-centroid nodes. The 7 columns therefore carry 6 degrees of
freedom, and `k` is doubly redundant: constant, *and* recoverable as `len(h)`.
Free correctness invariant: if any record's `h` stops summing to 1, the k-MIS
assignment fix has regressed.

Minnesota is standardised with **Luxembourg's** mean/std. Deliberate — the
model only understands that scale.

## FiLM in one box

```
FiLM(h) = gamma * h + beta        gamma, beta produced from y by learned Linears
```

Code: `newX = yx1 + (yx2 + 1) * weighted_V`

- **feature-wise**: each of 256 features gets its own scale and shift
- **linear**: the operation is a straight line in h
- **modulation**: adjusts an existing signal, doesn't add content
- Perez et al., AAAI 2018 — originally so a *question* could reconfigure a
  vision network. Same idea: the condition reprograms rather than informs.

**The `+1` matters.** At init weights ~ 0, so gamma ~ 1 and beta ~ 0 — FiLM is
the identity and conditioning must be *learned*. Nothing forces the model to
use the condition. That is exactly why the 29 July checkpoint scored 0.067.

## Why y is protected

- never noised — you generate the graph, not the condition
- never a loss target — `true_y = data.y[:, :0]`, and `lambda_train: [5, 0]`
- never overwritten in the reverse loop — otherwise it drifts into something
  the sampler invented
- `y` also carries the timestep `t`, so it is the model's clock as well

## Numbers worth remembering

| | |
|---|---|
| diffusion steps | 500 |
| transformer layers | 6 |
| input widths | `7 2 22` (X, E, y) — grep any log to confirm. Legacy `cluster_lux` is still `7 2 15` |
| y path | 22 -> 128 -> 64, then 64 -> 256 x 4 x 6 |
| condition reads per generated graph | 6 layers x 500 steps = 3,000 |
| Lux clusters | median 11, range 4-71, 100% connected |
| Minn clusters | median 23 — same k, denser graph |

## Results

A (Lux->Lux): density 0.2120 vs real 0.2124; degree dist 0.0544 vs ER floor
0.599. Connected 56.5 vs real 85.4 — under-connects, the main weakness.

Transfer: C beats B on 5 of 6. Edge-count MAE 2.98 vs 6.53.
**C is the only config that beats the naive `m = n-1` baseline.**

Training: A 122.20 @ ep179, B 118.73 @ ep139, C 320.97 @ ep29.

## Things that are easy to forget

**rho_in ~ 2/n on Luxembourg** (R2 = 0.93) because clusters are near-trees.
So controlling the partial correlation on `n` linearly gives 0.933 — a false
positive. Must control on `2/n`. Correct answer for the old model: 0.067.
Minnesota breaks the degeneracy (R2 = -0.15), which is why it is the only
place conditioning can be demonstrated.

**k-MIS used to assign by lowest RANK, not nearest centre. FIXED 2026-08-06.**
`_cluster_k_mis` flooded minimum rank over k rounds and discarded the distance,
so a lower-ranked centre 7 hops away beat a nearer one. 23.5% of Luxembourg
nodes went to the wrong centre and **15.8% of induced subgraphs came out
disconnected** — the path to the centre left the cluster.

The fix propagates the lexicographic key `(hop, rank)` packed as `hop*n + rank`,
one hop costing exactly `n`. Now 0% disconnected, and that is a theorem rather
than a measurement: the shortest path to your nearest centre is provably inside
your own cluster. So connectivity is a constraint the coarsening guarantees, and
100% is the number the model should be hitting — not 84.2%.

**Selection was never wrong.** The parallel flood in `sample.py` provably equals
the sequential greedy in the paper's pseudocode — verified, 0 disagreeing nodes
on both graphs. So the 8,815 vs 10,929 discrepancy has some other cause and
cannot be blamed on this.

Everything dated before 2026-08-06 — all three `.pt` files, all three
checkpoints, all three tables — describes the old clusters.
Lux 7,650 -> 8,815 clusters, Minnesota 78 -> 94, median size 12 -> 11.
Regression test: `k-mis-pool/check_kmis_assignment.py`.

**n is forced at sampling.** So the model matching `n` proves nothing. `k` is
constant. And post-fix `rho_in` is degenerate with `n` — R2 on `(1, n, 1/n)` is
**0.997** on Luxembourg, **0.989** on Minnesota, because every cluster is a
near-tree once connectivity is enforced, so `rho_in ~ 2/n`. The pre-fix scatter
that made Minnesota usable was an artefact of the bug.

So of the original three, **none** could carry evidence. That is why `h` was
added on 2026-08-06. R2 on size, A train / B test:

```
        Lux    Minn
n      1.000   1.000    imposed
rho_in 0.997   0.989    degenerate
h1     0.712   0.521
h2     0.440   0.510
h3     0.281   0.327
h4     0.449   0.236
h5     0.371   0.428
h6     0.264   0.382
h7     0.144   0.301
```

`h` is now the only place the conditioning claim can be demonstrated. Test it
the same way: partial correlation of a generated shape statistic against the
requested `h`, controlling for `n`.

**A and B are the same model trained twice** (identical train + val splits).
Gives a free seed-variance estimate: ~3% on val NLL.

**Everything converges by epoch ~10.** 200 epochs was ~10x more than needed,
which makes cross-validation cheap.

**C's val curve is noise** — 7 validation graphs, swinging 321-1114. Its
checkpoint was a lottery. C still won 5 of 6, which makes the result stronger,
not weaker. But C's numbers are NOT reproducible by retraining — keep the
checkpoint.

## Minnesota at k=4 (added 2026-08-14)

`make_transfer_datasets.py --k-public 7 --k-private 4`. Luxembourg stays at 7.

Why: at k=7 Minnesota gives 94 clusters of median size 28 against Luxembourg's
median 11, so "transfer" was partly just "generate graphs 2.5x bigger than
anything in training". At k=4 it gives 226 clusters of **median 11** — the same
size distribution — and the shared B/C test split goes 47 -> 114.

Three things to remember:

**`h` stays 7 wide.** `extract_pairs(..., k_hist=7)` decouples histogram width
from coarsening k. Rings 5..7 of a k=4 cluster are provably empty, so sum(h) is
still 1 (checked: max h5..h7 = 0.000000 over all 226) and the input width is
still 22. This is the whole reason no architecture change is needed and the k=7
checkpoints still load.

**B does not need retraining.** B trains and validates on Luxembourg only. The
private k changes B's TEST split and nothing else — same records, same
standardisation stats, same weights. `run_sample_conditional.sh B_k4` points the
new dataset at the existing `transfer_B` checkpoint. Only C is a new model.
Verified: `pairs_A_lux2minn_k7k4.pt` is record-for-record identical to
`pairs_A_lux2minn_k7.pt`, which is the same argument for A.

**`k` is truthful, and that costs B something.** 7 for Luxembourg, 4 for
Minnesota. In C the column has variance and standardises normally
(train range -1.749..0.572). In B it has zero training variance, so
`standardise()` leaves it raw and the test records arrive at **-3** where
training was always 0. Deliberate, per the design decision on 2026-08-14, but B's
result is not an in-distribution number. Same for `h2`, which lands ~3.2 sigma
above the Luxembourg training mean.

## Traps

- processed cache is keyed on the pairs **filename**. Edit a `.pt` without
  renaming and the stale processed copy is silently reused.
  `rm -rf data/cluster_lux/processed/`
- the `=` in `epoch=N.ckpt` breaks hydra unless the value carries its own
  quotes: `"general.test_only='$CKPT'"`. Shell quotes get stripped first.
- upstream `spectre_dataset.py` appends every graph **twice** (lines 86, 96).
  Do not copy from it.
- `.sh` files edited on Windows need `sed -i 's/\r$//'`.

## Commands

```bash
cd extraction  && python make_transfer_datasets.py      # kmis env
cd ..          && ./sync.sh D_intra                     # shared code into this folder
rsync -avz D_intra/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_intra/
sbatch scripts/run_transfer.sh A                                # on BlueBEAR

cd evaluation  && python compare_transfer.py --a ../results/samples/A.txt \
                    --b ../results/samples/B.txt --c ../results/samples/C.txt --latex
```

This folder is self-contained and uploads as-is. The shared model code still has
one source of truth in `../DiGress_upload/`; `../sync.sh` copies it in here and
into D_inter. So: edit shared model code in `DiGress_upload/` then re-run
`sync.sh`, and edit D_intra's own files (`src/datasets/cluster_dataset.py`,
`src/sample_conditional.py`, the `cluster_*` configs, `data/`, `scripts/`,
`evaluation/`, `extraction/`) directly here, where sync never touches them.
`./sync.sh --check` lists any synced file that has drifted.

## Open

- 8,815 supernodes vs the paper's 10,929. Eight orderings tried, none match.
- which 3 of Huiping's 6 condition params to drop — never confirmed
- D_inter spec is in Overleaf, not yet read
