# BS1 and BS2: Huiping's ER and SC-RG

The two baselines are Huiping's (`competitors/Baseline/`, scripts unchanged). Both
honour every released count exactly -- m_i inside a supernode, w_ij between a
connected pair -- and differ from D_intra / D_inter only in **where** the edges go.

| | D_intra block | D_inter block |
|---|---|---|
| **BS1** (ER, G(n,m)) | m_i edges uniformly among the n_i(n_i-1)/2 pairs | w_ij edges uniformly among the n_i n_j pairs |
| **BS2** (SC-RG) | hop layers from h_i around the centre, a random radial tree, then random edges only within or between adjacent layers up to m_i: E_h = 0 by construction | i-side out-degrees from r_{i->j}, random j-side neighbours, exactly w_ij edges: E_r = 0 by construction |

BS2's cross blocks read r_{i->j} from the real block. r is not in the release
and not in D_inter's (V1) condition, so BS2 sees a statistic D_inter is not given.
Only the i side is constrained: the datasets store r_{i->j}, not r_{j->i}.

| file | what |
|---|---|
| `run_baselines.sh` | runs her four scripts on the D_intra and D_inter test sets and renames the output for the scorers |
| `full_graph.py` | her generator functions on **every** supernode and pair of the release, for the assembled-graph table |

## Run (local, `kmis` env, about a minute)

```bash
bash baselines/run_baselines.sh          # 5 runs, her seeds 42..46
```

| file | test set | read by |
|---|---|---|
| `results/samples/D_intra/s80/{BS1,BS2}_s<s>.txt` | 63 Oldenburg test supernodes | `D_intra/evaluation/repeat_summary.py` |
| `results/samples/D_intra/s80/{A_BS1,A_BS2}_s<s>.txt` | 1763 Luxembourg test supernodes | same, `--configs A_BS1 A_BS2` |
| `results/samples/D_inter/s80/{BS1,BS2}_s<s>.pt` | 187 Oldenburg test pairs | `D_inter/evaluation/transfer_summary.py` |
| `results/samples/D_inter/s80/{A_BS1,A_BS2}_s<s>.pt` | 3998 Luxembourg test pairs | same, `--configs A_BS1 A_BS2` |

The assembled-graph BS1 / BS2 need no step of their own:
`D_inter/private_inference/syn_repeats_summary.py` builds them through
`full_graph.py` every time it runs.

The previous BS1 (uniform) / BS2 (Chung-Lu) code is in `archive/old_baselines/`.
