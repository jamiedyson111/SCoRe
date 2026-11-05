# D_intra evaluation (local, `kmis` env)

Scores generated supernode subgraphs against the real held-out ones.

| file | what |
|---|---|
| `repeat_summary.py` | **the main table**: every metric as mean ± std over 5 seeds, E_h included |
| `evaluate_clusters.py` | the sample parser (`parse_generated`), the hop histogram (`hop_hist`) and the per-graph metrics |
| `make_intra_table.py` | single-run table; also defines the shared statistics `repeat_summary.py` imports |
| `compare_transfer.py` | single-run A / B / C comparison table |
| `compare_ablation.py` | single-run full condition vs no-hop ablation |

## Main table

```bash
python repeat_summary.py --configs B_nohop_centroid B_v3 C_nohop_centroid C_v3 BS1 BS2   # Oldenburg
python repeat_summary.py --configs A_nohop_centroid A_v3 A_BS1 A_BS2                             # Luxembourg (A)
```

- **Input:** `results/samples/D_intra/s80/<CFG>_s0.txt … _s4.txt`, scored
  against the test split of `data/cluster_lux/raw/pairs_{A,B,C}_lux2olde_k7_s80.pt`
  (63 Oldenburg test supernodes; 1763 Luxembourg for A; 80/20 split by supernode).
- **Output:** `results/D_intra/tables/s80/repeats_summary.tsv` (A:
  `repeats_summary_A.tsv`), one "mean ± std" cell per configuration, and
  `repeats_per_seed*.tsv` with every run's raw value.
- A new method is added to `CONFIGS` at the top of `repeat_summary.py`, as
  `BS1` and `BS2` are. BS1 is Huiping's ER, BS2 her SC-RG, written by
  `bash baselines/run_baselines.sh`.

**Rows:** edges, density and degree (mean and std), connected %, acyclic %,
components, edge-count MAE, degree W1, spectrum distance, and the hop error

    E_h = 1/N_test · Σ_i Σ_{r=1..7} |ĥ_i(r) − h_i(r)|

measured from the node the `X:` row marks as the centroid, so only for samples
that mark exactly one (v3 and `*_centroid` mark node 0; BS1/BS2 mark the true
k-MIS centre where it sits in the block). The Real column's E_h is checked to be
0, which verifies the hop histogram against the stored condition.
