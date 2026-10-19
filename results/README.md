# Results

Everything the project produces -- tables, samples, checkpoints, private-inference
files -- lives here, so `D_intra/` and `D_inter/` hold only the code and data
needed to train and run the models. Every script writes here by default.

| folder | what |
|---|---|
| `D_intra/tables/oldenburg/` | **D_intra tables**: `repeats_summary.tsv` (Oldenburg, mean ± std), `repeats_summary_A.tsv` (Luxembourg), `intra_sheet.tsv` (spreadsheet), `hop/` (E_h comparison) |
| `D_inter/tables/transfer/` | **D_inter tables**: `pairs_summary.tsv` (Oldenburg), `pairs_summary_A.tsv` (Luxembourg), `inter_sheet.tsv` |
| `D_inter/tables/syn/` | **assembled-graph table**: `syn_sheet.tsv` (spreadsheet), `syn_table.tex`, `syn_summary.tsv` |
| `D_inter/tables/pair_stats*`, `degree_w1.txt` | the original Luxembourg D_inter table (Table V) |
| `D_intra/checkpoints/`, `D_inter/checkpoints/` | local copies of a few checkpoints; the trained models live on BlueBEAR |
| `samples/` | every generated sample (own README) |
| `private_inference/` | the Oldenburg release (`release_oldenburg_k7.pt`) and its true blocks; also where `make_graph.sh` and `assemble_syn.py` write locally |
| `competitors/` | competitor outputs: `digress_scale.tsv`, and `ICLR2/`, `HiGen/` graphs once copied from BlueBEAR |

| written by | to |
|---|---|
| `D_intra/evaluation/repeat_summary.py`, `D_intra/make_tables.sh` | `D_intra/tables/oldenburg/` |
| `D_inter/evaluation/transfer_summary.py` | `D_inter/tables/transfer/` |
| `D_inter/make_tables.sh`, `D_inter/evaluation/pair_stats.py` | `D_inter/tables/` |
| `D_inter/private_inference/syn_repeats_summary.py` | `D_inter/tables/syn/` |
| `baselines/make_baseline_samples.py` | `samples/D_intra/…`, `samples/D_inter/…` |
| `D_intra/private_inference/make_private_release.py` | `private_inference/` |

`.tsv` files paste into Excel as cells. Samples, `.pt` and `.ckpt` files are
git-ignored; the tables are tracked.
