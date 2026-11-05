# D_intra results

| folder / file | what |
|---|---|
| `tables/s80/repeats_summary.tsv` | **main D_intra table, current 80/20 split** (Oldenburg; `_A.tsv` for Luxembourg) |
| `checkpoints/` | a local copy of one checkpoint; the trained models live on BlueBEAR (see its README) |

`.tsv` files paste into Excel as cells. Rebuild the main tables with
`evaluation/repeat_summary.py`. Samples are in `results/samples/D_intra/` at the top of
the project; older Minnesota tables are in `archive/D_intra/`.

Tables from earlier runs (the previous split, the original Luxembourg Table V) are in
`archive/old_tables/`.
