# D_inter evaluation (local, `kmis` env)

Scores generated cross blocks B_ij against the real ones.

| file | what |
|---|---|
| `transfer_summary.py` | **the main table**: mean ± std over 5 seeds, every block matched to its own test pair |
| `pair_stats.py` | per-block statistics (`real_pair_stats`, `summarise`), shared by the main table; also the original distributional table (Table V) |
| `evaluate_pairs.py` | condition fidelity of the original sampler's output (\|B_ij\| = w_ij) |
| `check_context_integrity.py` | checks sampling left the given L_i, L_j untouched |

## Main table

```bash
python transfer_summary.py --configs B C BS1 BS2        # Oldenburg (187 test pairs)
python transfer_summary.py --configs A A_BS1 A_BS2      # Luxembourg (3998 test pairs)
```

- **Input:** `results/samples/D_inter/s80/<CFG>_s0.pt … _s4.pt`, scored against the
  test split of `data/pair_lux/raw/pairs_inter_{B,C}_lux2olde_k7_s80.pt` (Oldenburg)
  or `pairs_inter_luxembourg_osm_k7_s80.pt` (A): every pair touching a test
  supernode. B and C are checked to be the same 187 pairs, and each block to
  match its test pair (sn_i, sn_j, n_i, n_j). The insides L_i, L_j are the real ones.
- **Output:** `results/D_inter/tables/s80/pairs_summary.tsv` (A:
  `pairs_summary_A.tsv`) and `pairs_per_seed*.tsv`.
- A new method is added to `PAIRS` and `LABELS` at the top of the script.

**Rows:** cross edges, cross density and D_out (mean and std), degree W1 (pooled
cross-degree, both sides), and the out-degree-profile error, in BOTH directions
(November 2026):

    E_r = 1/N_test · Σ_(i,j) ½ ( Σ_d |r̂_ij(d) − r_ij(d)| + Σ_d |r̂_ji(d) − r_ji(d)| )

where r_ij(d) is the fraction of V_i nodes with d cross edges into V_j, binned
0, 1, 2, 3, ≥4 for the road networks (DBLP / Collab: 0, 1, 2-3, 4-7, ≥8).
