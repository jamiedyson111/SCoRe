# Private inference -- D_inter half

Stages 2 and 3 of the four-stage pipeline. Stages 0 and 1 live in
`D_intra/private_inference/` and produce the two files this half consumes:

```
  release_<graph>_k<k>.pt     the coarse release        (stage 0, D_intra)
  clusters_<graph>_k<k>.pt    one L_i per supernode     (stage 1, D_intra)
              |
              v
  stage 2  sample_private_blocks.py    digress env, GPU
             --> blocks_<graph>_k<k>.pt        one B_ij per connected pair
  stage 3  assemble_syn.py             kmis env, local, CPU
             --> L_syn_<graph>_k<k>.pt + report + LaTeX table
```

`../../make_graph.sh` runs all four stages from one command, submitting stages 1
and 2 to BlueBEAR over ssh and waiting for them. The commands below are for
running one on its own; `--dry-run` prints them without submitting anything.

## Repeated runs: mean ± std of the assembled graph (B and C)

`make_graph.sh` builds one graph. For the reconstruction table with a spread,
each configuration is sampled N times end to end: seed `s` of stage 1 is the
context for seed `s` of stage 2, so every seed is a whole synthetic graph.

| config | D_intra | D_inter |
|---|---|---|
| `B_v3_s80-B` | `transfer_A_v3_s80` (Luxembourg only) | `pair_lux_s80` |
| `C_v3_s80-C` | `transfer_C_v3_s80` (Lux + Olde train) | `pair_C_s80` |

Each at its final checkpoint, the same models the D_intra and D_inter tables
score (80/20 supernode split, no validation). Every block is generated. The
graph is scored as a whole and as its **test part** -- the subgraph on the 63
held-out test supernodes -- which contains no training supernode for B or C.

On BlueBEAR (the release must already be in `D_intra/private_inference/out/`):

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra
JB=$(sbatch --parsable private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3)
JC=$(sbatch --parsable private_inference/run_private_clusters_repeats.sh --repeats 5 C_v3)
cd /rds/projects/c/chenhp-graphsyn/D_inter
sbatch --dependency=afterok:$JB private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3
sbatch --dependency=afterok:$JC private_inference/run_private_blocks_repeats.sh --repeats 5 C C_v3
```

Files land beside the release as `clusters_oldenburg_k7_<ICFG>_s80_s<seed>.pt` and
`blocks_oldenburg_k7_<ICFG>_s80-<CFG>_s<seed>.pt`. Copy them into the local
`results/samples/assembled/`, then:

```bash
python syn_repeats_summary.py      # -> results/D_inter/tables/syn_s80/syn_summary{,_test}.tsv
```

Rows are `assemble_syn.py`'s own (it imports `assemble`, `stats`, `topology`);
BS1 (Huiping's ER) and BS2 (her SC-RG) are redrawn per seed, on every supernode
and pair, by `baselines/full_graph.py`. Feeding the private graph's own blocks in as every
seed reproduces the Real column exactly (RE_GCC, degW, RE_APL all 0).

## Stage 2 -- cross blocks

```bash
sbatch private_inference/run_private_blocks.sh \
    /abs/path/checkpoints/pair_lux/epoch=179.ckpt \
    out/release_oldenburg_k7.pt \
    out/clusters_oldenburg_k7.pt
```

For each connected pair the model is handed the two internal blocks and the side
labels, and generates only the cross block. `sample_batch` re-imposes the observed
part after every reverse step, so `L_i` and `L_j` come back bit-identical.

Two things that are easy to get silently wrong, both handled explicitly:

- **The `given` encoding must match training entry for entry.**
  `PairDatasetInfos.sample_conditioning` in `src/datasets/pair_dataset.py` is the
  reference: no-edge is `[1,0]`, the diagonal is `[0,0]`, padding is `[0,0]`.
  `build_given` mirrors it. A mismatch produces plausible output and wrong numbers.
- **`rho_i` comes from the release, not from the generated `L_i`.** The brief
  defines `c_ij` from the released supernode conditions, and that is the scale the
  model was trained against. The realised density of a generated block drifts from
  it; the drift is reported, never fed back in.

### Validating stage 2 on its own

Pass the true internal blocks that stage 0 writes next to the release, instead of
stage 1's output:

```bash
sbatch private_inference/run_private_blocks.sh CKPT RELEASE \
    /rds/.../D_intra/private_inference/out/clusters_oldenburg_k7_true.pt
```

That isolates stage 2 from stage 1 entirely. With real context the run should
reproduce the public-run numbers -- around 96% of blocks with `|B_ij|` exactly
`w_ij` -- so a bad result points here rather than at the D_intra checkpoint.

Do this on **Luxembourg** before trusting any Minnesota figure. The Minnesota
run scored 71.0% (154/217), which is consistent both with distribution shift
(`w_ij` z=+2.61 against the Luxembourg training pairs) and with a feeding bug,
and the two look identical from the output alone. Luxembourg is in-distribution
for D_inter, so it separates them.

Stage 2 reads `n_i`, `rho_i`, `n_j`, `rho_j`, `w` and `k` from the release. It is
unaffected by the hop-histogram fields stage 0 added on 2026-08-12 for D_intra's
condition.

> **The newest `pair_lux` checkpoint is not automatically the right one.** The
> 2026-08-18 run is V2, and `general.name` is `pair_lux` for both, so nothing in
> the run directory's name distinguishes them. Loading a V2 checkpoint under
> `dataset=pair_lux` dies in `load_state_dict` with
> `mlp_in_y.0.weight: copying a param with shape [128, 23] ... current model is
> [128, 18]` -- the 5 extra columns are `r`. `make_graph.sh` now picks by the
> `pairs_file` recorded in each run's `.hydra/config.yaml` rather than by date,
> because `dataset.name` is `pair_lux` in both configs.

**This runs on V1, and that is the default.** `configs/dataset/pair_lux.yaml` is
the 6-dim condition, which is exactly what the releases publish. The 11-dim V2
condition exists as `dataset=pair_lux_v2` (`DATASET=pair_lux_v2 sbatch ...`) and
additionally reads `r`, the 5-bin out-degree profile, from each entry of
`release['pairs']`. Do not select it here until stage 0 emits `r`.

**`r` would be a new disclosure.** `n`, `rho` and `w` are all supernode- or
coarse-graph-level; `r_{i->j}` is a statistic of the very block being generated,
so stage 0 must publish 5 numbers per connected pair (10902 of them on
Luxembourg, 217 on Minnesota) describing the row-degree profile of the true
`B_ij`. There is no way to derive it on this side. Until
`D_intra/private_inference/make_private_release.py` emits it, the `cond_names`
assert above fires on every old release -- which is the intended behaviour, not a
bug to work around.

It is an ordinary clusters file, read like any other. Stage 2 deliberately imports
nothing from the extraction stack: `kmis_pairs` pulls in `h5py` and `torch_sparse`,
which the GPU environment does not have, so rebuilding the true blocks on the
cluster fails at import. Everything stage 2 needs arrives as a file.

Pairs larger than the combined size 128 seen in training are generated anyway and
warned about; the transformer is size-agnostic, so that cap was a memory decision,
not a model limit. Use `+max_combined=N` to skip them.

## Stage 3 -- assembly and scoring

```bash
python assemble_syn.py --release  out/release_oldenburg_k7.pt \
                       --clusters out/clusters_oldenburg_k7.pt \
                       --blocks   out/blocks_oldenburg_k7.pt \
                       --out-dir  out/syn_oldenburg_k7
```

Places each `L_i` on the diagonal and each `B_ij` plus its transpose off-diagonal,
zero for pairs not connected in the released coarse graph.

The table has three columns. **Random** assembles blocks that honour the release
and nothing else -- `round(rho_in * n(n-1)/2)` edges per supernode, `w_ij` per
pair, placed uniformly at random -- so it matches every published count exactly
and differs only in *where* the edges go. It is not a straw man but the score a
generator gets for using the condition perfectly and learning no structure, and
without it none of the rows below have a scale.

The table is the reconstruction table: nodes, edges, mean degree, density and
components, then GCC and APL, then three discrepancies against the real graph.

| row | definition |
|---|---|
| GCC | global clustering coefficient, `3 tau_triangle / tau_3` (transitivity, not the mean of the local coefficients) |
| APL | mean geodesic over reachable pairs |
| REGCC | `\|GCC(G) - GCC(G^)\| / GCC(G)` |
| degW | `W_1` between the two degree distributions |
| REAPL | `\|APL(G) - APL(G^)\| / APL(G)` |
| Clustering | MMD between the local clustering-coefficient histograms (100 bins on [0, 1], Gaussian-TV kernel, sigma 0.1), as DiGress (`graph_mmd.py`) |
| Orbit | MMD between ORCA 4-node graphlet-orbit profiles (mean count per vertex, Gaussian-TV kernel, sigma 30) |
| Spectrum | MMD between normalised-Laplacian eigenvalue histograms (200 bins on [0, 2], sigma 1). Up to 20,000 nodes from a dense eigendecomposition; above (Vermont) by counting the eigenvalues below each bin edge from a sparse symmetric factorisation (Sylvester's law of inertia, `graph_mmd.spectrum_hist_inertia`), ~30 s per Vermont graph. `spectrum_only.py` computes this row on its own; `--validate` checks it against the dense histogram on Oldenburg |

Counting and dense agree except for eigenvalues exactly 2 (one per bipartite component,
e.g. every tree): floating point puts about half of them just above 2, outside the
histogram's range, so the dense histogram drops them; counting keeps them. On Oldenburg
this changes no cell at 4 dp except BS1 (ER, with ~580 components): 0.0277 dense, 0.0286
counted.

APL is **exact** -- every reachable pair -- for graphs up to 6500 nodes, which
covers Oldenburg at 6105 (a ~300 MB distance matrix; the limit was 5000 until
Oldenburg replaced Minnesota, at 2642, as the private graph). Luxembourg's 114599 nodes would need a 1.3e10-entry
distance matrix, so there it is estimated from `--apl-sources` BFS sweeps (100 by
default) drawn from the same seed in every column, and the report says which was
used. Sampling costs about 2.5%: Minnesota's APL is 35.35 exactly and 36.28 from
64 sources.

> **REGCC explodes when the real graph is nearly triangle-free.** Luxembourg's
> GCC is 0.0015, so the Random column scores REGCC 65.4 -- a relative error
> against a denominator that is almost zero. That is the metric behaving as
> defined, not a bug, but it means REGCC is not comparable between graphs and is
> better read next to the raw GCC row than on its own.

Assembly preserves node identity, so `L^syn` and `G^priv` are graphs on the same
vertices. degW therefore needs no finite-sample correction, and edge-level
precision / recall / F1 / Jaccard is available and reported under ALSO MEASURED,
split into internal and cross edges, which attributes any loss to D_intra or to
D_inter rather than pooling both into one number.

Also measured but outside the table: isolated nodes, largest-component fraction,
reachable pairs, triangles, average local clustering and degree assortativity.
The reachable fraction has to be read beside APL -- a fragmented graph measures
distances only within its fragments and so reports a SHORTER APL, not a longer
one. `--no-baseline` drops the Random column for a plain two-column table.

> The Erdos-Renyi reference changed on 2026-08-18. Its upper-triangle index
> inversion dropped an `n` and returned column indices past the end of the graph,
> so `bincount` returned a longer array than the node count and the baseline was
> measured against a degree multi-set that did not exist. On Minnesota it read
> 1.3110; it is 0.6548. Every other number in this table is unaffected -- the bug
> was confined to that one baseline.

Node identity is preserved: supernode `i`'s local node `u` is written back to fine
node id `release['supernodes'][i]['nodes'][u]`, so `L^syn` has exactly the node set
of `G^priv`. That is what makes the degree comparison clean -- both multi-sets have
the same length, so unlike the public-run pair metrics there is no finite-sample
correction and a perfect result scores exactly 0.

### Assembly self-test

```bash
python assemble_syn.py --release out/release_oldenburg_k7.pt --real \
                       --out-dir out/selftest_oldenburg
```

Assembles from the private graph's own blocks. It must reproduce `G^priv` edge for
edge; the script exits non-zero if it does not. All three graphs pass:

| | Oldenburg k=7 | Minnesota k=7 | Luxembourg k=7 |
|---|---|---|---|
| nodes | 6105 | 2642 | 114599 |
| edges reproduced | 7029 / 7029 | 3303 / 3303 | 119666 / 119666 |
| degree `W_1` | 0.0000 | 0.0000 | 0.0000 |
| `\|B_ij\|` = `w_ij` | 552/552 | 217/217 | 10902/10902 |

Run this after any change to the assembly code. If it fails, the bug is in
`assemble_syn.py` and no amount of looking at model output will show that.
