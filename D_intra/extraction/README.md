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
python make_transfer_datasets.py                 # Luxembourg -> Oldenburg, k = 7
python make_transfer_datasets.py --no-hops       # the no-hop ablation, condition [n, rho_in, k]
python make_transfer_datasets.py --k 5           # another locality (both graphs)
python make_transfer_datasets.py --k-public 7 --k-private 4    # differing localities
```

## Output

`../data/cluster_lux/raw/pairs_{A,B,C}_<tag>.pt`, tag `lux2olde_k7` by default
(`_nohop` appended for the ablation). Each holds `splits` = [train, val, test]
records `{sn, n, nodes, centre, edge_index, y_raw, y}`, plus `cond_names`,
`cond_mean`, `cond_std`.

| | train / val | test |
|---|---|---|
| A | Luxembourg (75 / 15) | Luxembourg (10%, 882 supernodes) |
| B | Luxembourg | Oldenburg test split (50%, 160 supernodes) |
| C | Luxembourg + Oldenburg train split (40%) | the same 160 |

The script prints the config files to create, e.g.
`configs/dataset/cluster_A_<tag>.yaml`, which name the pairs file for training.
