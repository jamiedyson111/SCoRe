# D_intra results

| folder / file | what |
|---|---|
| `tables/oldenburg/repeats_summary.tsv` | **main D_intra table** (Oldenburg, B and C, baselines): mean ± std over 5 seeds |
| `tables/oldenburg/repeats_summary_A.tsv` | the same for A (Luxembourg) |
| `tables/oldenburg/repeats_per_seed*.tsv` | every run's raw value |
| `tables/oldenburg/intra_sheet.tsv` | spreadsheet layout: A, B, C with no-hops and v3, plus BS1/BS2 |
| `tables/oldenburg/hop/` | the hop-conditioning comparison: no-hops + centroid vs v3, with E_h (`hop_sheet.tsv`) |
| `tables/oldenburg/{intra_k7,transfer_k7,ablation_A}.{txt,tex}` | single-run tables from `make_tables.sh` |
| `checkpoints/` | a local copy of one checkpoint; the trained models live on BlueBEAR (see its README) |

`.tsv` files paste into Excel as cells. Rebuild the main tables with
`evaluation/repeat_summary.py`. Samples are in `results/samples/D_intra/` at the top of
the project; older Minnesota tables are in `archive/D_intra/`.
