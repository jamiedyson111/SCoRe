# D_intra: the inside of each supernode

A conditional DiGress model. Given one supernode's condition

    c_i = [n, rho_in, k, h1 .. h7]

(its size, internal density, the coarsening locality, and its hop histogram: the
fraction of members 1..7 hops from the k-MIS centre), it generates the
supernode's internal subgraph L_i. Steps 1-3 of the pipeline (see the top-level
README) live here, plus stages 0-1 of the full-graph assembly.

```
extraction/          step 1: k-MIS coarsening and the datasets       (local, kmis env)
configs/             hydra configs; dataset/cluster_{A,B,C}_lux2olde_k7[_nohop].yaml
data/cluster_lux/raw the datasets from step 1, read by training
src/                 DiGress + the conditional modifications (MODIFICATIONS.md)
scripts/             step 2-3 BlueBEAR jobs                          (digress env)
evaluation/          step 3 scoring                                   (local, kmis env)
private_inference/   step 7, stages 0-1: the release, and D_intra on every supernode
competitors/         the one-shot DiGress memory test
make_tables.sh       single-run tables
```

## Model variants

| suffix | what | condition |
|---|---|---|
| *(none)* | the base model | [n, rho_in, k, h1..h7] |
| `_nohop` | the ablation: no hop histogram | [n, rho_in, k] |
| `_v3` | **the main model**: centroid node type (node 0 is the centre, held fixed) + hop embedding (h → MLP_h → e_h) | [n, rho_in, k, e_h] |
| `_centroid` | the centroid node type alone | e.g. `B_nohop_centroid`: no hops, centroid marked -- the control for E_h |

Any variant combines with a configuration, e.g. `A_v3`, `B_nohop_centroid`, `C_v3`.

## Step 1: k-MIS + datasets (local)

```bash
cd D_intra/extraction
python make_transfer_datasets.py              # Luxembourg -> Oldenburg, k = 7
python make_transfer_datasets.py --no-hops    # the _nohop ablation
```

Coarsens both graphs with k-MIS and writes
`data/cluster_lux/raw/pairs_{A,B,C}_lux2olde_k7[_nohop].pt`: one record per
supernode (its subgraph, condition, centre), split into train / val / test.
Minutes. Upload `data/cluster_lux/raw/` to BlueBEAR before training.

| file | train / val | test |
|---|---|---|
| `pairs_A_…` | Luxembourg 75% / 15% | Luxembourg 10% (882 supernodes) |
| `pairs_B_…` | Luxembourg | Oldenburg 50% (160 supernodes) |
| `pairs_C_…` | Luxembourg + Oldenburg 40% (oversampled) / Oldenburg 10% | the same 160 |

## Step 2: train (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra
sbatch scripts/run_transfer.sh B_v3                    # one model
bash scripts/run_all_transfer.sh A_v3 B_v3 C_v3        # several, chained
```

200 epochs, up to 6 h per model (`n_epochs` is set in the script; validation
NLL stops improving long before). Checkpoints: `outputs/<date>/<time>-transfer_<CFG>/checkpoints/transfer_<CFG>/epoch=N.ckpt`.
A and B train on identical Luxembourg data; only C sees Oldenburg.

## Step 3: evaluate (BlueBEAR, then local)

Sample every test supernode five times from the trained model:

```bash
sbatch scripts/run_sample_conditional.sh --repeats 5 B_v3
bash scripts/run_olde_repeats.sh 5 B_v3 C_v3 B_nohop_centroid C_nohop_centroid   # several at once
```

Output: `outputs/<date>/<time>-transfer_<CFG>_olde_sample_x5/conditional_samples_s0.txt … _s4.txt`,
minutes per seed. Copy them to the local
`results/samples/D_intra/oldenburg/repeats/<CFG>_s<seed>.txt`, then:

```bash
cd D_intra/evaluation
python repeat_summary.py --configs B_nohop_centroid B_v3 C_nohop_centroid C_v3 BS1 BS2
python repeat_summary.py --configs A_nohop A_v3 A_BS1 A_BS2          # A, on Luxembourg
```

Writes `results/D_intra/tables/oldenburg/repeats_summary.tsv` (A: `_A.tsv`):
mean ± std over the 5 seeds of edges, density and degree (mean and std),
connected %, acyclic %, components, edge-count MAE, degree W1, spectrum distance
and the hop error E_h. See `evaluation/README.md`.

## Step 7, stages 0-1: the full graph (D_intra's part)

```bash
cd D_intra/private_inference
python make_private_release.py --graph oldenburg --k 7       # stage 0, local
```

Writes `results/private_inference/release_oldenburg_k7.pt`: n, rho_in and h for
all 317 supernodes, w for all 552 connected pairs. Copy it to BlueBEAR's
`D_intra/private_inference/out/`, then generate every supernode, 5 seeds:

```bash
sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3   # stage 1, BlueBEAR
```

→ `private_inference/out/clusters_oldenburg_k7_B_v3_s<seed>.pt`. D_inter takes
it from there (stages 2-3, see `D_inter/README.md`).

## Running another k

There is one k: the **k-MIS locality**, the minimum hop distance between two
supernode centres (k = 7 everywhere at the moment). It enters D_intra in three
places:

1. **The coarsening.** k decides how the graph is cut into supernodes: a larger k
   gives fewer, larger supernodes.
2. **One entry of the condition.** c_i = [n, rho_in, **k**, h1 .. hk]. k is a
   single number in the condition, beside n, rho_in and the hop histogram -- it is
   not the whole condition.
3. **The width of the hop histogram.** h has one bin per ring 1 .. k, so the
   condition has 3 + k entries: 10 at k = 7, 8 at k = 5.

Every dataset is built at one k, so within it the k entry is the same for every
supernode. What changes between k values is the supernodes themselves and the
number of hop bins, so a new k means new datasets and a newly trained model --
the code needs no change. For example, k = 5:

1. `python make_transfer_datasets.py --k 5` writes `pairs_{A,B,C}_lux2olde_k5.pt`.
2. Create `configs/dataset/cluster_{A,B,C}_lux2olde_k5.yaml` (copy the `_k7` ones,
   change `pairs_file`).
3. In `scripts/run_transfer.sh` and `scripts/run_sample_conditional.sh`, the
   configuration → dataset mapping names `lux2olde_k7`; change it to `lux2olde_k5`.
4. Score with `repeat_summary.py --tag lux2olde_k5`.
