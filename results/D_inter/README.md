# D_inter results

| folder / file | what |
|---|---|
| `tables/transfer/pairs_summary.tsv` | **main D_inter table** (Oldenburg, 276 test pairs): B, C, BS1, BS2, mean ± std over 5 seeds |
| `tables/transfer/pairs_summary_A.tsv` | the same for A (Luxembourg, 1091 test pairs) |
| `tables/transfer/pairs_per_seed*.tsv` | every run's raw value |
| `tables/transfer/inter_sheet.tsv` | spreadsheet layout: Luxembourg and Oldenburg side by side |
| `tables/syn/` | **assembled-graph table**: D_intra + D_inter on all of Oldenburg (`syn_sheet.tsv` for the spreadsheet, `syn_table.tex`, `syn_summary.tsv`, `syn_per_seed.tsv`) |
| `tables/pair_stats*`, `tables/degree_w1.txt` | the original Luxembourg distributional table (Table V), from `make_tables.sh` |
| `checkpoints/` | a local copy of a V2 `pair_lux` checkpoint; the trained models live on BlueBEAR |

`.tsv` files paste into Excel as cells. Rebuild them with
`evaluation/transfer_summary.py` and `private_inference/syn_repeats_summary.py`.
Samples are in `results/samples/D_inter/` and `results/samples/assembled/` at the top of the project.
