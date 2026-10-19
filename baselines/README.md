# BS1 and BS2: uncoarsening baselines

Two rule-based baselines that see only the release. Every block gets exactly the
released edge count -- `round(rho_in * n(n-1)/2)` inside a supernode, `w_ij`
between a connected pair -- so they match the condition perfectly and differ from
D_intra / D_inter only in **where** the edges go.

| | placement |
|---|---|
| **BS1** (random) | every free slot of the block is equally likely |
| **BS2** (Chung-Lu) | each node draws an expected degree D_u from public Luxembourg's degree distribution; slot (u, v) is picked with probability ∝ D_u · D_v |

| file | what |
|---|---|
| `uncoarsen.py` | the two placement rules (`internal_block`, `cross_block`) |
| `make_baseline_samples.py` | writes BS1/BS2 samples for the D_intra and D_inter test sets |

## Run (local, `kmis` env, about a minute)

```bash
python make_baseline_samples.py          # 5 seeds
```

## Output

| file | test set | read by |
|---|---|---|
| `results/samples/D_intra/oldenburg/repeats/{BS1,BS2}_s<seed>.txt` | 160 Oldenburg supernodes | `D_intra/evaluation/repeat_summary.py` |
| `results/samples/D_intra/oldenburg/repeats/{A_BS1,A_BS2}_s<seed>.txt` | 882 Luxembourg supernodes | same, `--configs A_BS1 A_BS2` |
| `results/samples/D_inter/transfer/{BS1,BS2}_s<seed>.pt` | 276 Oldenburg pairs | `D_inter/evaluation/transfer_summary.py` |
| `results/samples/D_inter/transfer/{A_BS1,A_BS2}_s<seed>.pt` | 1091 Luxembourg pairs | same, `--configs A_BS1 A_BS2` |

The assembled-graph versions of BS1/BS2 are built inside
`D_inter/private_inference/syn_repeats_summary.py` from the same functions.

In the D_intra files node 0 is marked as the centroid (for BS2, the node with the
largest expected degree), so E_h is reported for the baselines too.
