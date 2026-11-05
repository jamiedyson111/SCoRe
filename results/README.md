# Results

Everything the project produces -- tables, samples, checkpoints, private-inference
files -- lives here, so `D_intra/` and `D_inter/` hold only the code and data
needed to train and run the models. Every script writes here by default.

| folder | what |
|---|---|
| `D_intra/tables/s80/` | **D_intra tables, current 80/20 split**: `repeats_summary.tsv` (Oldenburg), `repeats_summary_A.tsv` (Luxembourg) |
| `D_inter/tables/s80/` | **D_inter tables, current split**: `pairs_summary.tsv` (Oldenburg), `pairs_summary_A.tsv` (Luxembourg) |
| `D_inter/tables/syn_s80/` | **full-graph tables, current split**: `syn_summary.tsv` (whole graph), `syn_summary_test.tsv` (held-out test part), `.tex` versions |
| `D_intra/checkpoints/`, `D_inter/checkpoints/` | local copies of a few checkpoints; the trained models live on BlueBEAR |
| `samples/` | every generated sample (own README); `s80/` subfolders for the current split |
| `private_inference/` | the Oldenburg release (`release_oldenburg_k7.pt`) and its true blocks; also where `make_graph.sh` and `assemble_syn.py` write locally |
| `costs/` | runtime, GPU memory, coarsening and state-size tables (see `costs/README.md`) |

| written by | to |
|---|---|
| `D_intra/evaluation/repeat_summary.py` | `D_intra/tables/s80/` |
| `D_inter/evaluation/transfer_summary.py` | `D_inter/tables/s80/` |
| `D_inter/make_tables.sh`, `D_inter/evaluation/pair_stats.py` | `D_inter/tables/` |
| `D_inter/private_inference/syn_repeats_summary.py` | `D_inter/tables/syn_s80/` |
| `baselines/run_baselines.sh` (BS1 = ER, BS2 = SC-RG) | `samples/D_intra/s80/`, `samples/D_inter/s80/` |
| `D_intra/private_inference/make_private_release.py` | `private_inference/` |

`.tsv` files paste into Excel as cells. Samples, `.pt` and `.ckpt` files are
git-ignored; the tables are tracked.

Tables from earlier runs (the previous split, the original Luxembourg Table V) are in
`archive/old_tables/`.
