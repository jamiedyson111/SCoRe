# D_inter — inter-supernode model

The second of the two models: the edges **between** supernodes, conditioned on
`c_ij`. D_intra fills in each supernode; D_inter connects them.

This folder is **self-contained** — it holds the model code and everything else
needed to run, so it uploads to BlueBEAR as-is:

```bash
rsync -avz D_inter/ dysonjz@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_inter/
```

```bash
cd /rds/projects/c/chenhp-graphsyn/D_inter && sbatch scripts/run_pair_debug.sh
```

The shared model code has a single source of truth in `../DiGress_upload/` and is
copied in by `../sync.sh`. Edit it there, not here. D_inter's own files —
`src/datasets/pair_dataset.py`, the `pair_lux` configs, `data/`, `scripts/`,
`evaluation/`, `dataprep/` — live here and sync never touches them.
`../sync.sh --check` lists any synced file that has drifted.

## What it does

For a pair of supernodes, everything is stored as one combined graph

```
H_ij = [ L_i    B_ij  ]
       [ B_ij^T L_j   ]
```

Only the cross block `B_ij` is generated. `L_i` and `L_j` are observed context:
never noised, never predicted, contributing to no loss. `X` is a 2-class **side**
indicator, so an entry of `E` is a cross entry exactly when its endpoints have
different node classes — that is how the model derives the mask, and it stays
exact at every step because `X` is never noised.

Condition `c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k]`, standardised on train.

## Layout

```
dataprep/make_pair_blocks.py   builds the dataset. Runs LOCALLY in the `kmis` env
dataprep/make_pairs.py         D_intra's generator, kept as reference
src/datasets/pair_dataset.py   dataset, datamodule, infos, W_out-degree metric
configs/dataset/pair_lux.yaml
configs/experiment/pair_lux.yaml
data/pair_lux/raw/             the .pt the dataset reads
evaluation/evaluate_pairs.py   scores generated_samples*.txt against real blocks
evaluation/pair_stats.py       the full structural table, plain text + LaTeX
make_tables.sh                 runs both of the above, no arguments
scripts/                       SLURM jobs; submit as `sbatch scripts/<name>.sh`
```

Everything else under `src/` and `configs/` is a synced copy of the shared code.

## Dataset

10,902 connected supernode pairs from Luxembourg at k=7, split by pair into
8176 / 1635 / 1091. Built from the **fixed** k-MIS coarsening (2026-08-06), so
100% of internal blocks are connected, over the same 8815 supernodes D_intra now
uses — the two models are consistent by construction.

Rebuild with:

```bash
cd dataprep && conda activate kmis && python make_pair_blocks.py
```

## Producing the tables

One command, no arguments, no `conda activate` — the script activates `kmis`
itself and scores the newest `generated_samples*.txt` it can find (project root,
then `outputs/*/*/`):

```bash
./make_tables.sh
```

Output lands in `results/tables/`: `pair_stats.txt` (the structural table against
real, with the uniform and zero baselines beside it), `degree_w1.txt` (the
out-degree metric alone with its three reference points), and, under
`results/tables/pair_stats/`, the LaTeX, the per-pair CSVs and the plots.

`./make_tables.sh --list` prints the table names, `./make_tables.sh pair_stats`
runs one, and a path argument scores a specific file:

```bash
./make_tables.sh outputs/2026-08-10/12-30-00-graph-tf-model/generated_samples1.txt
```

Two environment overrides matter. `SPLIT=val` scores the validation split, and
`PAIRS=` picks the blob — **it must be the one the model trained on**. Both files
are on disk and differ only in the width of `y`, so the wrong one compares
against a different condition in silence. v1 (`dataset=pair_lux`) is the default:

```bash
PAIRS=data/pair_lux/raw/pairs_inter_luxembourg_osm_k7_v2.pt ./make_tables.sh
```

`../make_tables.sh` does this folder and D_intra together.

Tables are a LOCAL step -- the samples come off BlueBEAR, the tables are built
here. If you do want them on the cluster, the script works there unchanged:
there is no `kmis` env on BlueBEAR and none is needed, since these scripts import
only numpy/scipy/networkx/torch, so it falls back to sourcing `setup.sh` and runs
in `digress`, giving identical numbers.

## Reading the numbers

The cross block is **very** sparse: mean density 0.0068, and **91.2% of pairs
have exactly one cross edge**. Two consequences:

- **`val/cross_CE` is the signal.** The constant-predictor floor is **0.02884
  nats** — what you score by ignoring the input and predicting the average
  density everywhere. It starts around 0.20, so *above* 0.0288 means worse than
  a constant. Below it means the model has learned where edges go.

- **`W_out-degree` is nearly blind here.** With one edge per pair the out-degree
  histogram is the same whichever node gets it: uniformly random placement
  scores 0.00017 against real, versus 0.0748 for emitting nothing. It measures
  edge-count calibration, not placement. Reported because the brief asks for it;
  the uniform baseline is printed beside it every time.

## Smoke test first

```bash
sbatch scripts/run_pair_debug.sh
```

Four checks, documented in that script's header — input features `8 2 18`, node
marginal `[0.5002, 0.4998]`, edge marginal `[0.9955, 0.0045]`, `Val Atom type KL
0.00`. The edge marginal is the one that fails silently: if it reads nearer
`[0.97, 0.03]` the whole-graph marginal is being used instead of the cross-block
one, the limit distribution is wrong, and sampling degrades while training still
looks healthy.

## Open

- `w_ij` is in the condition, so the model is told how many edges to place. The
  remaining task is only *where*. Consider dropping it if this proves trivial.
- Cross blocks are generated one pair at a time with nothing coordinating them,
  so a node's total degree across all its pairs is unconstrained.
- Private inference — assembling `L^syn` from D_intra's and D_inter's output —
  is not built.
