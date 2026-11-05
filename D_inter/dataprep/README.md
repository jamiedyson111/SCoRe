# D_inter data preparation (local, `kmis` env)

Builds the supernode-pair records D_inter trains on. Each connected pair (i, j) of
a k-MIS coarsening is stored as one combined graph

    H_ij = [ L_i    B_ij ]      local ids 0 .. n_i-1        are side i
           [ B_ijᵀ  L_j  ]      local ids n_i .. n_i+n_j-1  are side j

with condition c_ij = [n_i, n_j, w_ij, r_{i->j}, r_{j->i}] (13 numbers; r is the
5-bin cross out-degree profile). Only B_ij is generated. The model never sees L_i,
L_j: the dataset class (`src/datasets/pair_dataset.py`) reduces them to the node
features x_v = d_in(v) / (n_i - 1), while the records keep the whole H_ij for the
evaluation and the baselines. Which supernode takes the i slot is randomised.

| file | what |
|---|---|
| `make_pair_blocks.py` | **Luxembourg pairs** (configuration A); also defines the shared-split helpers |
| `make_transfer_pairs.py` | **B and C datasets**: the Oldenburg pairs; B tests on them, C fine-tunes on them |

## Run

```bash
python make_pair_blocks.py          # Luxembourg, k = 7 (K at the top of the file); needs the split
python make_transfer_pairs.py       # needs the Luxembourg file
```

## Split

80 / 20 by supernode, **shared with D_intra**: both scripts read
`D_intra/data/splits/<graph>_k7_s80.json` (written by D_intra's
`make_transfer_datasets.py`, so run that first). A training pair has both
supernodes in training; a test pair touches at least one test supernode. No
validation split; the val slot holds 256 training pairs (logging only).

## Output (`../data/pair_lux/raw/`)

| file | train / test |
|---|---|
| `pairs_inter_luxembourg_osm_k7_s80_deg.pt` | 6904 / 3998 Luxembourg pairs -- A, and B's training data |
| `pairs_inter_B_lux2olde_k7_s80_deg.pt` | Luxembourg train, **187 Oldenburg test pairs** |
| `pairs_inter_C_lux2olde_k7_s80_deg.pt` | Oldenburg's **362 train pairs** (C fine-tunes A's model on them), the same 187 test pairs |

Each holds `splits` = [train, val, test] records `{sn_i, sn_j, n_i, n_j, w, r,
r_ji, rho_i, rho_j, edge_index, y_raw, y}` plus `cond_names`, `cond_mean`,
`cond_std`. n_i, n_j, w_ij are standardised on A's training split (in all three
files, so C's fine-tuning sees A's scale); the r columns stay raw. C's file also
carries `edge_marginal`, A's cross-block edge marginal, which training uses in
place of C's own so the noise model stays A's. The earlier files without `_deg`
(condition [n_i, rho_i, n_j, rho_j, w_ij, k], C joint) are the previous runs';
their records and test pairs are the same. Pairs above 128 combined nodes are dropped
(3 on Oldenburg). The training config `configs/dataset/<name>.yaml` names the file.
