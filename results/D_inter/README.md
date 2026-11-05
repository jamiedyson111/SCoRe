# D_inter results

| folder / file | what |
|---|---|
| `tables/s80/pairs_summary.tsv` | **main D_inter table, current 80/20 split** (Oldenburg, 187 test pairs; `_A.tsv` for Luxembourg) |
| `tables/syn_s80/` | **full-graph tables, current split**: `syn_summary.tsv` (whole graph), `syn_summary_test.tsv` (held-out test part), `.tex` versions |
| `checkpoints/` | a local copy of a V2 `pair_lux` checkpoint; the trained models live on BlueBEAR |

`.tsv` files paste into Excel as cells. Rebuild them with
`evaluation/transfer_summary.py` and `private_inference/syn_repeats_summary.py`.
Samples are in `results/samples/D_inter/` and `results/samples/assembled/` at the top of the project.

Tables from earlier runs (the previous split, the original Luxembourg Table V) are in
`archive/old_tables/`.
