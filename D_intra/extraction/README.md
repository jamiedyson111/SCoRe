# D_intra extraction: k-MIS coarsening and datasets (local, `kmis` env)

Coarsens a graph with k-MIS and turns every supernode into a training record
(its subgraph + condition c_i = [n, rho_in, k, h1..hk]).

| file | what |
|---|---|
| `make_transfer_datasets.py` | **builds the A / B / C datasets** for public → private transfer |
| `kmis_pairs.py` | the shared extraction: graph loading (`GRAPHS`, `load_graph`), coarsening, hop histogram, condition, standardisation |
| `kmis/`, `benchmark/` | the k-MIS coarsening code (Greedy ordering, maximal k-independent set, cluster assignment) |
| `datasets/` | the raw graphs (SuiteSparse Luxembourg, Minnesota; Brinkhoff `oldenburg/raw/OL.cedge`) |

## Run

```bash
python make_transfer_datasets.py                 # Luxembourg -> Oldenburg, k = 7, 80/20 split
python make_transfer_datasets.py --no-hops       # the no-hop ablation, condition [n, rho_in, k]
python make_transfer_datasets.py --k 5           # another locality (both graphs)
python make_transfer_datasets.py --k-public 7 --k-private 4    # differing localities
```

## Output

- `../data/splits/<graph>_k<k>_s80.json`: the **supernode split** -- 80% train, 20%
  test, seed 0, no validation -- shared with D_inter, whose builders read it.
- `../data/cluster_lux/raw/pairs_{A,B,C}_<tag>.pt`, tag `lux2olde_k7_s80` by default
  (`_nohop` appended for the ablation). Each holds `splits` = [train, val, test]
  records `{sn, n, nodes, centre, edge_index, y_raw, y}`, plus `cond_names`,
  `cond_mean`, `cond_std`. The val slot holds 256 TRAINING records, for logging only.

| | train | test |
|---|---|---|
| A | Luxembourg, 7052 supernodes | Luxembourg, 1763 |
| B | the same as A | Oldenburg, 63 |
| C | Luxembourg + Oldenburg's 254 (×9, 24% of training) | the same 63 |
| Cft | **C as a fine-tune** (the final tables): Oldenburg's 254 only, starting from A's model | the same 63 |

`python make_finetune_datasets.py` writes the Cft files (`pairs_Cft_<tag>[_nohop].pt`,
A's standardisation and `edge_marginal`) and plain DiGress on Oldenburg alone
(`pairs_O_<tag>_uncond.pt`) from the A / B / C files above.

Conditions are standardised with the training split's mean and std; k, constant
within a dataset, is left on its raw scale. `--test-frac` changes the split.
The script prints the config files to create, e.g.
`configs/dataset/cluster_A_<tag>.yaml`, which name the pairs file for training.
