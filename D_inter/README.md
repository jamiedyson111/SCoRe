# D_inter: the edges between two supernodes

A conditional DiGress model that **inpaints** one block. For a connected pair of
supernodes (i, j) it is shown the combined graph

    H_ij = [ L_i    B_ij ]      local ids 0 .. n_i-1        are side i
           [ B_ijᵀ  L_j  ]      local ids n_i .. n_i+n_j-1  are side j

with the two insides L_i, L_j given (re-imposed after every denoising step) and
generates only the cross block B_ij, from the condition

    c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k]          (V1; w_ij = number of cross edges)

Steps 4-6 of the pipeline (see the top-level README) live here, plus stages 2-3
of the full-graph assembly and its evaluation (steps 7-8).

```
dataprep/            step 4: supernode-pair datasets                 (local, kmis env)
configs/             hydra configs; dataset/pair_lux.yaml (A, B), pair_{B,C}_lux2olde_k7.yaml
data/pair_lux/raw    the datasets from step 4, read by training
src/                 DiGress + the conditional/inpainting modifications (MODIFICATIONS.md)
scripts/             step 5-6 BlueBEAR jobs                          (digress env)
evaluation/          step 6 scoring                                   (local, kmis env)
private_inference/   steps 7-8: D_inter on every pair, assembly, full-graph scoring
make_tables.sh       the original Luxembourg table (Table V)
```

## Step 4: k-MIS + pair datasets (local)

```bash
cd D_inter/dataprep
python make_pair_blocks.py        # Luxembourg pairs (configuration A); K is set at the top
python make_transfer_pairs.py     # B and C: adds the Oldenburg pairs
```

Writes `data/pair_lux/raw/` (minutes to an hour). Upload it to BlueBEAR before training.

| file | train / val / test |
|---|---|
| `pairs_inter_luxembourg_osm_k7.pt` (V1) | 8176 / 1635 / 1091 Luxembourg pairs -- A and B train on this |
| `pairs_inter_luxembourg_osm_k7_v2.pt` | the same pairs with the 11-dim V2 condition (V1 + out-degree profile r); `make_pair_blocks.py` writes this, and V1 is its first 6 columns |
| `pairs_inter_B_lux2olde_k7.pt` | Luxembourg train / val, **276 Oldenburg test pairs** |
| `pairs_inter_C_lux2olde_k7.pt` | Luxembourg + Oldenburg train (×12), Oldenburg val, the same 276 test pairs |

Which supernode takes the i slot is randomised. Pairs above 128 combined nodes
are dropped from training (3 on Oldenburg).

## Step 5: train (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn/D_inter
sbatch scripts/run_pair.sh pair_lux pair_lux                       # A and B (one model)
sbatch scripts/run_pair.sh pair_C_lux2olde_k7 pair_C_olde          # C
```

250 epochs, 12 h limit (C: ~3 min per epoch, so it stops near epoch 242).
Checkpoints: `outputs/<date>/<time>-<name>/checkpoints/<name>/epoch=N.ckpt`. The
ones in use:

- **A and B:** `outputs/2026-08-06/16-57-54-pair_lux/checkpoints/pair_lux/epoch=179.ckpt`
  (the val cross-block loss minimum).
- **C:** the newest `pair_C_olde` run at the checkpoint it recorded as best
  (`epoch=219`). C over-fits: its cross-block loss was lowest near epoch 19.

## Step 6: evaluate (BlueBEAR, then local)

One generated cross block per test pair, 5 seeds, around the **real** L_i and L_j:

```bash
bash scripts/run_pairs_repeats.sh 5 B C          # ~1 h each
bash scripts/run_pairs_repeats.sh 5 A            # 1091 pairs, ~2-3 h
```

Output: `outputs/<date>/<time>-pairs_<CFG>_sample_x5/pair_samples_s0.pt … _s4.pt`.
Copy them to the local `results/samples/D_inter/transfer/<CFG>_s<seed>.pt`, then:

```bash
cd D_inter/evaluation
python transfer_summary.py --configs B C BS1 BS2        # Oldenburg
python transfer_summary.py --configs A A_BS1 A_BS2      # Luxembourg
```

Writes `results/D_inter/tables/transfer/pairs_summary.tsv` (A: `_A.tsv`): mean ±
std over the 5 seeds of cross edges, cross density and D_out (mean and std),
degree W1 and the out-degree-profile error E_r. See `evaluation/README.md`.

## Step 7, stages 2-3: assemble the full graph

After D_intra's stage 1 has generated every supernode
(`clusters_oldenburg_k7_B_v3_s<seed>.pt`, see `D_intra/README.md`), D_inter
generates the cross block of **every** connected pair (552 on Oldenburg) around
those generated insides, seed for seed:

```bash
cd /rds/projects/c/chenhp-graphsyn/D_intra && J=$(sbatch --parsable private_inference/run_private_clusters_repeats.sh --repeats 5 B_v3)
cd /rds/projects/c/chenhp-graphsyn/D_inter && sbatch --dependency=afterok:$J private_inference/run_private_blocks_repeats.sh --repeats 5 B B_v3
```

(C: `C_v3` and `C C_v3`.) Output, beside the release in BlueBEAR's
`D_intra/private_inference/out/`: `blocks_oldenburg_k7_B_v3-B_s<seed>.pt`. About
1 h for stage 1 and 1.5-2 h for stage 2. Copy the `clusters_*` and `blocks_*`
files to the local `results/samples/assembled/`.

Stage 3, placing every L_i on the diagonal and every B_ij off it to get one
6105-node graph on Oldenburg's own nodes, happens inside step 8.

For a single graph end to end (release, both BlueBEAR jobs, assembly, scoring)
there is also `../make_graph.sh` at the top of the project.

## Step 8: evaluate the full graph (local)

```bash
cd D_inter/private_inference
python syn_repeats_summary.py                     # B_v3-B and C_v3-C
```

Assembles each seed's graph and scores it against real Oldenburg: nodes, edges
(internal = D_intra's, cross = D_inter's), mean degree, density, components, GCC,
APL, and the errors RE_GCC, degW, RE_APL. BS1 / BS2 are built alongside, and
competitor graphs in `results/competitors/` (ICLR2, HiGen; DiGress as OOM) are
added as columns. Writes `results/D_inter/tables/syn/syn_sheet.tsv` (spreadsheet)
and `syn_table.tex`.

## Running another k

There is one k: the **k-MIS locality**, the same k as D_intra's (k = 7 at the
moment). In D_inter it does two things:

1. **The coarsening.** k decides the supernodes, and so which pairs exist and how
   large they are.
2. **One entry of the condition.** c_ij = [n_i, rho_i, n_j, rho_j, w_ij, **k**].
   k is a single number in the condition, beside the two sizes, the two densities
   and the cross-edge count -- not the whole condition. D_inter has no hop
   histogram, so its condition stays 6 entries at any k.

Every dataset is built at one k, so within it the k entry is constant; a new k
means new pair datasets and a newly trained model, with no code change. D_intra
and D_inter must use the same k. For example, k = 5:

1. In `dataprep/make_pair_blocks.py`, set `K = 5` at the top (that variable is
   this k); it writes the V2 file. Make the V1 file by keeping the first 6
   condition columns (as `make_transfer_pairs.py` does for B and C).
2. Create `configs/dataset/pair_lux_k5.yaml` naming that file, and train with
   `sbatch scripts/run_pair.sh pair_lux_k5 pair_lux_k5`.
3. In `scripts/run_sample_pairs_transfer.sh`, configuration A's dataset is
   `pair_lux`; change it to `pair_lux_k5`, and pass the new checkpoint as the
   second argument: `sbatch scripts/run_sample_pairs_transfer.sh --repeats 5 A /abs/epoch=N.ckpt`.
