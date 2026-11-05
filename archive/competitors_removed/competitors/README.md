# Competitors

Five competitors to D_intra + D_inter, all scored on the same assembled-graph
table against private Oldenburg (`D_inter/private_inference/syn_repeats_summary.py`),
and BS1/BS2 also on the D_intra and D_inter tables.

| | what | where | runs on |
|---|---|---|---|
| BS1 (ER) | Huiping's G(n,m): released counts, edges placed uniformly | `Baseline/ER_*.py`, run by `../baselines/run_baselines.sh` | local CPU |
| BS2 (SC-RG) | Huiping's synopsis-constrained random graph: counts + h_i (intra), + r_{i->j} (inter) | `Baseline/SC_RG_*.py`, same | local CPU |
| DiGress [14] | one-shot memory/time at doubling n up to 6105 | `../D_intra/competitors/` | 1 GPU job, minutes |
| ICLR [2] | Bergmeister et al., trained on Luxembourg regions | `iclr2/` | GPU, days |
| HiGen [5] | Karami, trained on Luxembourg regions | `higen/` | GPU, days |

**Protocol (as configuration B).** ICLR [2] and HiGen are unconditional
generators of whole graphs. Both train on the same 100 connected Luxembourg
regions of 2000-6000 nodes (`make_lux_regions.py` -> `data/lux_regions.pkl`; 20
more for validation) -- public data only -- and generate five graphs of
Oldenburg's 6105 nodes. Neither sees the release (rho_in, h, w_ij); both get only
the node count. HiGen cannot be told a size, so `higen/apply_patch.py` rescales
its sampled totals to 6105 nodes, keeping the training edges-per-node ratio.

Generated graphs go to `out/<METHOD>/graph_s<seed>.npz` (`n`, `edges`), which
`syn_repeats_summary.py` picks up as extra columns automatically; their
internal/cross edge rows are blank, since their nodes are not Oldenburg's.
DiGress appears as an `OOM` column when `digress_scale.tsv` records OOM at 6105.
