# D_intra evaluation (local, `kmis` env)

Scores generated supernode subgraphs against the real held-out ones.

| file | what |
|---|---|
| `repeat_summary.py` | **the main table**: every metric as mean ± std over 5 seeds, E_h included |
| `evaluate_clusters.py` | the sample parser (`parse_generated`), the hop histogram (`hop_hist`) and the per-graph metrics |
| `make_intra_table.py` | single-run table; also defines the shared statistics `repeat_summary.py` imports |
| `compare_transfer.py` | single-run A / B / C comparison table |
| `compare_ablation.py` | single-run full condition vs no-hop ablation |
| `_bootstrap_bc.py`, `_bootstrap_pair.py` | paired bootstrap tests between two runs on the same supernodes |

## Main table

```bash
python repeat_summary.py --configs B_nohop_centroid B_v3 C_nohop_centroid C_v3 BS1 BS2   # Oldenburg
python repeat_summary.py --configs A_nohop A_v3 A_BS1 A_BS2                             # Luxembourg (A)
```

- **Input:** `results/samples/D_intra/oldenburg/repeats/<CFG>_s0.txt … _s4.txt`, scored
  against the test split of `data/cluster_lux/raw/pairs_{A,B,C}_lux2olde_k7.pt`
  (160 Oldenburg supernodes; 882 Luxembourg for A).
- **Output:** `../results/D_intra/tables/oldenburg/repeats_summary.tsv` (A:
  `repeats_summary_A.tsv`), one "mean ± std" cell per configuration, and
  `repeats_per_seed*.tsv` with every run's raw value.
- A new method is added to `CONFIGS` at the top of `repeat_summary.py`, as
  `BS1`, `ER` and `SC_RG` are.

**Rows:** edges, density and degree (mean and std), connected %, acyclic %,
components, edge-count MAE, degree W1, spectrum distance, and the hop error

    E_h = 1/N_test · Σ_i Σ_{r=1..7} |ĥ_i(r) − h_i(r)|

measured from node 0, so only for samples whose `X:` row marks node 0 as the
centroid (v3, `*_centroid`, the baselines). The Real column's E_h is checked to be
0, which verifies the hop histogram against the stored condition.
