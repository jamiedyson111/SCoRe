# D_inter: the edges between two supernodes

A conditional DiGress model that generates the cross block B_ij of a connected
pair of supernodes (i, j), on the n_i + n_j nodes of the pair (local ids
0 .. n_i-1 are side i, n_i .. n_i+n_j-1 side j). Since November 2026 it follows
the paper's Section B (the **internal-degree D_inter**, files and runs `*_deg`):

    X_i, X_j  node features: x_v = d_in(v) / (n_i - 1), v's normalised degree inside
              its own supernode. They REPLACE L_i, L_j -- the model never sees the
              internal edges -- and are held fixed, never diffused.
    c_ij      [n_i, n_j, w_ij, r_{i->j}, r_{j->i}]       (w_ij = number of cross edges;
              r_{i->j} = fraction of V_i nodes with 0,1,2,3,4+ edges into V_j)

Only B_ij is noised, predicted and scored. X_i, X_j come from the true L_i, L_j in
training and from D_intra's GENERATED L^_i, L^_j at inference (configs B and C) --
the train--inference shift the paper evaluates; the `realX` runs repeat B and C
with the true X_i to measure it.

The earlier D_inter (runs without `_deg`) was shown the combined graph
H_ij = [[L_i, B_ij], [B_ijᵀ, L_j]], with the insides re-imposed after every
denoising step, under c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k], and its C was
trained jointly. Its results are the s80 tables without `_deg`.

Steps 4-6 of the pipeline (see the top-level README) live here, plus stages 2-3
of the full-graph assembly and its evaluation (steps 7-8).

```
dataprep/            step 4: supernode-pair datasets                    (local, kmis env)
configs/             hydra configs; dataset/pair_lux_s80.yaml, pair_{B,C}_lux2olde_k7_s80.yaml
data/pair_lux/raw    the datasets from step 4, read by training
src/                 DiGress + the conditional/inpainting modifications (MODIFICATIONS.md)
scripts/             step 5-6 BlueBEAR jobs                             (digress env)
evaluation/          step 6 scoring                                      (local, kmis env)
private_inference/   steps 7-8: D_inter on every pair, assembly, full-graph scoring
```

**The no-r ablation (`_nor`).** The same model with c_ij = [n_i, n_j, w_ij], built by
`dataprep/make_nor_pairs.py` from the `_deg` files (the r columns dropped, nothing
else changed): `pair_lux_s80_deg_nor` (A / B) and `pair_C_s80_deg_nor` (C, fine-tuned
from it). Compared with the `_deg` models it measures what r_{i->j}, r_{j->i} add.
**Cft** is D_inter's C given X^ from D_intra's fine-tuned C (`Cft_v3`) instead of its
joint C; the final tables use it. Both run through `../submit_final.sh`.

## The split

**80 / 20 by supernode, shared with D_intra, no validation split.** D_inter reads
D_intra's supernode assignment (`D_intra/data/splits/<graph>_k7_s80.json`), so both
models train on exactly the same supernodes:

- a **training pair** has **both** supernodes in the training set;
- a **test pair** has **at least one** test supernode -- so no test cross block,
  and no pair involving a test supernode, is ever trained on.

| | train pairs | test pairs |
|---|---|---|
| Luxembourg | 6904 | 3998 |
| Oldenburg | 362 | 187 |

| config | trained on | tested on | model |
|---|---|---|---|
| **A** | Luxembourg train pairs | Luxembourg test pairs (3998) | the Luxembourg model |
| **B** | Luxembourg train pairs | Oldenburg test pairs (187) | the **same** Luxembourg model |
| **C** | **fine-tuning**: the Luxembourg model, fine-tuned on the 362 Oldenburg train pairs (true X; 100 epochs at lr 5e-5, A's condition standardisation and noise marginal kept) | Oldenburg test pairs (187) | its own model |

Without a validation split, A trains for a fixed **180 epochs** (the
epoch at which the Luxembourg model's held-out cross-block loss bottomed out
before) and its **final checkpoint** is the model. The "val" slot holds 256
training pairs, used for logging only. Pairs above 128 combined nodes are left
out (3 on Oldenburg). Which supernode takes the i slot is randomised.

## Step 4: pair datasets (local)

Run after D_intra's step 1, which writes the split:

```bash
cd D_inter/dataprep
python make_pair_blocks.py        # Luxembourg pairs (A, and B's training data)
python make_transfer_pairs.py     # B and C: adds the Oldenburg pairs
```

Writes `data/pair_lux/raw/pairs_inter_luxembourg_osm_k7_s80_deg.pt` and
`pairs_inter_{B,C}_lux2olde_k7_s80_deg.pt` (under a minute). The records keep
the real H_ij (internal edges included) for evaluation and the baselines; the
dataset class reduces L_i to X_i. Their test pairs are the earlier files', in
the same order, so the BS1 / BS2 samples score against them unchanged. Upload
`data/` to BlueBEAR before training, and the new release
(`D_intra/private_inference/make_private_release.py`, which now publishes r_ij
and r_ji per pair) to `D_intra/private_inference/out/`.

## Step 5: train (BlueBEAR)

Everything (training, test pairs, full graph) is one submitter:

```bash
bash /rds/projects/c/chenhp-graphsyn/D_inter/scripts/submit_inter_deg.sh
```

or by hand:

```bash
cd /rds/projects/c/chenhp-graphsyn/D_inter
sbatch scripts/run_pair.sh pair_lux_s80_deg pair_lux_s80_deg                         # A and B
sbatch scripts/run_pair.sh pair_C_lux2olde_k7_s80_deg pair_C_s80_deg pair_lux_s80_deg # C, after A
```

The third argument makes C a fine-tune of that run's newest `last*.ckpt`. Model:
`outputs/<date>/<time>-<name>/checkpoints/<name>/last.ckpt`.

## Step 6: evaluate (BlueBEAR, then local)

One generated cross block per test pair, 5 seeds. B and C use the X^_i, X^_j of
D_intra's generated blocks (seed s reads D_intra's whole-graph
`clusters_oldenburg_k7_<CFG>_v3_s80_s<s>.pt`); `--real-x` and A use the true X_i:

```bash
sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 B            # 187 pairs, minutes
sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 --real-x B   # the shift ablation
sbatch --time=12:00:00 scripts/run_sample_pairs_transfer.sh --repeats 5 A   # 3998 pairs
```

Output: `outputs/<date>/<time>-pairs_<CFG>_s80_deg[_realX]_sample_x5/pair_samples_s0.pt … _s4.pt`.
`bash fetch_s80.sh deg` copies them to `results/samples/D_inter/s80/<CFG>_deg[_realX]_s<seed>.pt`; then:

```bash
cd D_inter/evaluation
python transfer_summary.py --configs B_deg C_deg B_deg_realX C_deg_realX BS1 BS2   # Oldenburg
python transfer_summary.py --configs A_deg A_BS1 A_BS2                             # Luxembourg
```

Writes `results/D_inter/tables/s80/pairs_summary_deg.tsv` (A: `_A_deg.tsv`): mean ± std
over the 5 seeds of cross edges, cross density and D_out (mean and std), degree
W1 and the out-degree-profile error E_r.

## Step 7, stages 2-3: assemble the full graph

D_intra's stage 1 generates every supernode; D_inter then generates the cross
block of **every** connected pair around those generated insides, seed for seed.
Nothing in the assembled graph is copied from the data:

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra && J=$(sbatch --parsable private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3)
cd /rds/projects/c/chenhp-graphsyn/D_inter && sbatch --dependency=afterok:$J private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3
```

(C: `C_v3` and `C C_v3`.) Output in BlueBEAR's `D_intra/private_inference/out/`:
`clusters_oldenburg_k7_B_v3_s80_s<seed>.pt` and
`blocks_oldenburg_k7_B_v3_s80-B_deg_s<seed>.pt` (the internal-degree D_inter;
`submit_inter_deg.sh` reuses the existing D_intra clusters and runs only this half). Copy both to the local
`results/samples/assembled/`. Stage 3, placing every L_i on the diagonal and every
B_ij off it, happens inside step 8.

## Step 8: evaluate the full graph (local)

```bash
cd D_inter/private_inference
python syn_repeats_summary.py                     # B_v3_s80-B and C_v3_s80-C (earlier D_inter)
python syn_repeats_summary.py --configs B_v3_s80-B_deg C_v3_s80-C_deg \
       --out-dir ../../results/D_inter/tables/syn_s80_deg            # internal-degree D_inter
```

Two tables, both from the same generated graphs:

- **whole graph** (`syn_summary.tsv`, `syn_table.tex`): all 6105 Oldenburg nodes.
  Every block is generated, but for C about 80% of the supernodes were training
  supernodes.
- **test part** (`syn_summary_test.tsv`, `syn_table_test.tex`): the subgraph on the
  63 held-out test supernodes -- their generated insides and the generated cross
  edges between them -- against the same subgraph of real Oldenburg (1152 nodes).
  No training supernode is in it.

Rows: nodes, edges (internal = D_intra's, cross = D_inter's), mean degree,
density, components, GCC, APL, and the errors RE_GCC, degW, RE_APL. BS1 (ER) / BS2 (SC-RG)
are built alongside on every supernode and pair (`baselines/full_graph.py`). Output: `results/D_inter/tables/syn_s80/`.

## Running another k

There is one k: the **k-MIS locality**, the same k as D_intra's (k = 7 at the
moment). In D_inter it does two things:

1. **The coarsening.** k decides the supernodes, and so which pairs exist and how
   large they are.
2. **Nothing else.** The internal-degree D_inter's condition [n_i, n_j, w_ij,
   r_{i->j}, r_{j->i}] has no k entry (the earlier one carried k as a constant
   sixth entry), and its width is 13 at any k.

Every dataset is built at one k; a new k
means new pair datasets and a newly trained model, with no code change. D_intra
and D_inter must use the same k. For example, k = 5:

1. Run D_intra's step 1 with `--k 5` first; it writes the k = 5 split.
2. In `dataprep/make_pair_blocks.py`, set `K = 5` at the top (that variable is
   this k) and run it; `make_transfer_pairs.py` follows the Luxembourg file's k.
3. Create `configs/dataset/pair_lux_k5_s80.yaml` naming the new file, and train
   with `sbatch scripts/run_pair.sh pair_lux_k5_s80 pair_lux_k5_s80`.
4. In `scripts/run_sample_pairs_transfer.sh`, configuration A's dataset and model
   are `pair_lux_s80`; change them to `pair_lux_k5_s80`.
