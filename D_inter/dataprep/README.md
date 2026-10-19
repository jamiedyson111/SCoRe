# D_inter data preparation (local, `kmis` env)

Builds the supernode-pair records D_inter trains on. Each connected pair (i, j) of
a k-MIS coarsening is stored as one combined graph

    H_ij = [ L_i    B_ij ]      local ids 0 .. n_i-1        are side i
           [ B_ijᵀ  L_j  ]      local ids n_i .. n_i+n_j-1  are side j

with condition c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k] (V1). Only B_ij is
generated; L_i and L_j are given as context. Which supernode takes the i slot is
randomised.

| file | what |
|---|---|
| `make_pair_blocks.py` | **Luxembourg pairs** (configuration A); writes the V2 condition (V1 + the out-degree profile r) |
| `make_transfer_pairs.py` | **B and C datasets**: Oldenburg pairs, split 40/10/50, combined with the Luxembourg ones |

## Run

```bash
python make_pair_blocks.py          # Luxembourg, k = 7 (K is set at the top of the file)
python make_transfer_pairs.py       # needs the V1 Luxembourg file
```

## Output (`../data/pair_lux/raw/`)

| file | train / val / test |
|---|---|
| `pairs_inter_luxembourg_osm_k7.pt` (V1) | 8176 / 1635 / 1091 Luxembourg pairs |
| `pairs_inter_luxembourg_osm_k7_v2.pt` (V2) | the same pairs, 11-dim condition |
| `pairs_inter_B_lux2olde_k7.pt` | Luxembourg train/val, 276 Oldenburg test pairs |
| `pairs_inter_C_lux2olde_k7.pt` | Luxembourg + Oldenburg train (×12), Oldenburg val, the same 276 test pairs |

Each holds `splits` = [train, val, test] records `{sn_i, sn_j, n_i, n_j, w,
edge_index, y_raw, y}` plus `cond_names`, `cond_mean`, `cond_std`. The training
config `configs/dataset/<name>.yaml` names the file to use. Pairs above 128
combined nodes are dropped (3 on Oldenburg).
