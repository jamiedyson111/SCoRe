# D_intra: the inside of each supernode

A conditional DiGress model. Given one supernode's condition

    c_i = [n, rho_in, k, h1 .. h7]

(its size, internal density, the coarsening locality, and its hop histogram: the
fraction of members 1..7 hops from the k-MIS centre), it generates the
supernode's internal subgraph L_i. Steps 1-3 of the pipeline (see the top-level
README) live here, plus stages 0-1 of the full-graph assembly.

```
extraction/          step 1: k-MIS coarsening, the split, the datasets  (local, kmis env)
data/splits/         the supernode split shared with D_inter
configs/             hydra configs; dataset/cluster_{A,B,C}_lux2olde_k7_s80[_nohop].yaml
data/cluster_lux/raw the datasets from step 1, read by training
src/                 DiGress + the conditional modifications (MODIFICATIONS.md)
scripts/             step 2-3 BlueBEAR jobs                             (digress env)
evaluation/          step 3 scoring                                      (local, kmis env)
private_inference/   step 7, stages 0-1: the release, and D_intra on every supernode
```

## The split

**80 / 20 train / test, by supernode, no validation split**, the same supernodes
for D_intra and D_inter. Step 1 shuffles each graph's supernodes (seed 0), holds
out 20% as test, and writes the assignment to `data/splits/<graph>_k7_s80.json`;
D_inter's dataset builders read that file.

| | train | test |
|---|---|---|
| Luxembourg | 7052 supernodes | 1763 |
| Oldenburg | 254 | 63 |

| config | trained on | tested on | model |
|---|---|---|---|
| **A** | Luxembourg train | Luxembourg test (1763) | the Luxembourg model |
| **B** | Luxembourg train | Oldenburg test (63) | the **same** Luxembourg model |
| **C** | **joint training**: Luxembourg train + Oldenburg train together, from scratch (Oldenburg ×9, 24% of training) | Oldenburg test (63) | its own model |

**The final tables use C as a fine-tune (`Cft`)** rather than joint training: A's
model, trained further on the 254 Oldenburg training supernodes only (100 epochs, lr
5e-5, A's condition standardisation and edge marginal kept), as D_inter's C is. The
comparison adds **plain DiGress trained on Oldenburg alone (`O_uncond`)**: no
condition, 254 train / 63 test supernodes, each graph generated at its test
supernode's size, trained for about A's number of gradient steps (5,500 epochs).

| config | trained on | model |
|---|---|---|
| **Cft** | A's model fine-tuned on Oldenburg train (254) | `transfer_Cft[_nohop]_v3/_centroid_s80` |
| **O_uncond** | Oldenburg train (254), from scratch, no condition | `transfer_O_uncond_s80` |

Datasets: `python extraction/make_finetune_datasets.py` (after step 1) writes
`pairs_Cft_lux2olde_k7_s80[_nohop].pt` and `pairs_O_lux2olde_k7_s80_uncond.pt` from
the existing A / B / C files, so the split and records are theirs. Training, sampling
and the full graph run through `../submit_final.sh`; the tables through
`../make_final_tables.sh`.

Without a validation split, every model trains for a fixed **200 epochs** and its
**final checkpoint** (`last.ckpt`) is the model. The training code still expects a
"val" slot; it holds 256 *training* supernodes, used for logging only.

## Model variants

| suffix | what | condition |
|---|---|---|
| `_v3` | **the main model**: centroid node type (node 0 is the centre, held fixed) + hop embedding (h → MLP_h → e_h) | [n, rho_in, k, e_h] |
| `_nohop_centroid` | **the no-hop control**: the centroid is marked, but h is not given | [n, rho_in, k] |
| *(none)*, `_nohop`, `_centroid`, `_hopmlp` | the other ablations | |

`_v3` and `_nohop_centroid` differ only in h_i, so comparing them (E_h above all)
measures what the hop condition adds.

## Step 1: k-MIS, the split, datasets (local)

```bash
cd D_intra/extraction
python make_transfer_datasets.py              # hop condition
python make_transfer_datasets.py --no-hops    # the no-hop condition (same split)
```

Coarsens both graphs with k-MIS (k = 7), writes the split to `data/splits/` and the
datasets to `data/cluster_lux/raw/pairs_{A,B,C}_lux2olde_k7_s80[_nohop].pt`. Under
a minute. Upload `data/` to BlueBEAR before training. `--test-frac` changes the
split (default 0.2).

## Step 2: train (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra
sbatch scripts/run_transfer.sh A_v3                 # the Luxembourg model (A and B)
sbatch scripts/run_transfer.sh A_nohop_centroid
sbatch scripts/run_transfer.sh C_v3                 # Luxembourg + Oldenburg, trained jointly
sbatch scripts/run_transfer.sh C_nohop_centroid
```

200 epochs, up to 6 h each. Run directory:
`outputs/<date>/<time>-transfer_<CFG>_s80/`, model `checkpoints/transfer_<CFG>_s80/last.ckpt`.
B is not trained -- `run_transfer.sh B…` stops and says so.

## Step 3: evaluate (BlueBEAR, then local)

Sample every test supernode five times:

```bash
bash scripts/run_olde_repeats.sh 5 B_nohop_centroid B_v3 C_nohop_centroid C_v3   # Oldenburg
TIME=04:00:00 bash scripts/run_olde_repeats.sh 5 A_nohop_centroid A_v3          # Luxembourg, 1763 supernodes
```

Output: `outputs/<date>/<time>-transfer_<CFG>_s80_sample_x5/conditional_samples_s0.txt … _s4.txt`.
Copy them to the local `results/samples/D_intra/s80/<CFG>_s<seed>.txt`, then:

```bash
cd D_intra/evaluation
python repeat_summary.py --configs B_nohop_centroid B_v3 C_nohop_centroid C_v3 BS1 BS2
python repeat_summary.py --configs A_nohop_centroid A_v3 A_BS1 A_BS2
```

Writes `results/D_intra/tables/s80/repeats_summary.tsv` (A: `_A.tsv`): mean ± std
over the 5 seeds of edges, density and degree (mean and std), connected %,
acyclic %, components, edge-count MAE, degree W1, spectrum distance and the hop
error E_h. BS1 (Huiping's ER) and BS2 (her SC-RG) come from
`bash baselines/run_baselines.sh` (see `baselines/README.md`).

## Step 7, stages 0-1: the full graph (D_intra's part)

```bash
cd D_intra/private_inference
python make_private_release.py --graph oldenburg --k 7       # stage 0, local
```

Writes `results/private_inference/release_oldenburg_k7.pt`: n, rho_in and h for all
317 supernodes, w for all 552 connected pairs. Copy it to BlueBEAR's
`D_intra/private_inference/out/`, then generate every supernode, 5 seeds:

```bash
sbatch private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3   # stage 1
```

→ `private_inference/out/clusters_oldenburg_k7_B_v3_s80_s<seed>.pt`. Every block is
generated by the model, training supernodes included. D_inter takes it from there
(see `D_inter/README.md`).

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

1. `python make_transfer_datasets.py --k 5` writes `pairs_{A,B,C}_lux2olde_k5_s80.pt`
   and the split `data/splits/<graph>_k5_s80.json`.
2. Create `configs/dataset/cluster_{A,B,C}_lux2olde_k5_s80.yaml` (copy the `_k7_s80`
   ones, change `pairs_file`).
3. In `scripts/run_transfer.sh` and `scripts/run_sample_conditional.sh`, the
   configuration → dataset mapping names `lux2olde_k7_s80`; change it to `lux2olde_k5_s80`.
4. Score with `repeat_summary.py --tag lux2olde_k5_s80`.
