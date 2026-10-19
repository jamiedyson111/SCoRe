# DP-Graph-syn — repository map

Two models, D_intra and D_inter, each a conditional DiGress, plus everything
around them. Every folder below has its own README.

```
D_intra/            intra-cluster model: generate the inside of one supernode
                    given c_i = [n, rho_in, k, h1..h7]. Code and data only:
                    k-MIS extraction, training, sampling, evaluation, private
                    inference stages 0-1, the DiGress scaling test.

D_inter/            inter-cluster model: the cross block B_ij between two
                    supernodes, given c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k]
                    and both insides. Code and data only, plus private
                    inference stages 2-3 (assembly, whole-graph table).

baselines/          BS1 (random) and BS2 (Chung-Lu) uncoarsening baselines.
competitors/        ICLR [2] and HiGen [5] wrappers, their Luxembourg training
                    regions, and Huiping's ER / SC_RG baselines (Baseline/).

results/            everything produced: tables, samples, checkpoints, the
                    release, competitor outputs. Scripts write here by default.
archive/            old and stray files from the October clean-up (git-ignored).

k-mis-pool/         K-MIS coarsening (Bacciu et al.). Source of the clusters.
graph-coarsening/   Loukas coarsening library, for reference.

make_graph.sh       one assembled synthetic graph, end to end (stages 0-3).
make_tables.sh      every single-run results table.
sync.sh             copy the shared code into both projects.
```

## Why the split

`DiGress_upload/`, `D_intra/` and `D_inter/` all used to carry a byte-identical
copy of the DiGress source. Four copies, free to drift. Now the model code lives
once and each experiment holds only what is specific to it:

| | D_intra | D_inter |
|---|---|---|
| dataset class | `src/datasets/cluster_dataset.py` | `src/datasets/pair_dataset.py` (to write) |
| hydra configs | `configs/{dataset,experiment}/cluster_*.yaml` | `configs/.../pair_*.yaml` (to write) |
| data | `data/cluster_lux/raw/*.pt` | `data/pair_lux/raw/` (empty) |
| jobs | `scripts/*.sh` | `scripts/*.sh` |

## Running either one

DiGress needs a single flat root — `src/`, `configs/`, `data/`, `*.sh` all in
one place. `stage.sh` builds one by laying an experiment over the shared code:

```bash
./stage.sh D_intra          # -> _upload/D_intra/
rsync -av _upload/D_intra/ USER@bluebear.bham.ac.uk:/rds/projects/c/chenhp-graphsyn/D_intra/
```

`_upload/` is rebuilt from scratch each time, so it is never the thing you edit
and always safe to delete. Edit in `DiGress_upload/` or in the experiment folder,
then re-stage.

Job scripts assume the BlueBEAR root matches the experiment name
(`/rds/projects/c/chenhp-graphsyn/D_intra/`), not the old shared `DiGress/`.

## Every results table, one command

```bash
./make_tables.sh
```

No arguments, no `conda activate` — it activates `kmis` itself, runs
`D_intra/make_tables.sh` and `D_inter/make_tables.sh`, and leaves plain text plus
LaTeX in each project's `results/tables/`. Pass `D_intra` or `D_inter` to do one.
Each project's own script takes table names (`--list` to see them) and, for
D_inter, a samples file; see the two READMEs.

## The whole pipeline, one command

Both models exist to produce one graph: D_intra fills in the inside of each
supernode, D_inter joins them, and the blocks assemble into a single `L^syn` over
the private graph's own nodes.

```bash
./make_graph.sh --real
```

Runs it end to end in about ten seconds and prints the evaluation table. `--real`
uses the private graph's OWN blocks in place of generated ones: no checkpoint, no
GPU, no cluster. It must reproduce `G^priv` edge for edge, so it is both a
demonstration of the pipeline and the test that the assembly is sound -- run it
before trusting any generated result.

For the real thing, drop `--real`:

```bash
./make_graph.sh
```

The two sampling stages need a GPU and the trained checkpoints, so the script
does them on BlueBEAR itself: it pushes the release, picks the newest checkpoint
of each model, submits both jobs with the second chained on the first, waits,
pulls the output back and scores it. One ssh connection is multiplexed for the
whole run, so a password is asked for at most once. Everything already done is
skipped, so re-running after an interruption costs nothing and Ctrl-C is safe --
the jobs keep running and a later run collects them.

```bash
./make_graph.sh --dry-run     # print every remote command, run none
./make_graph.sh --no-wait     # submit and return; re-run later to collect
./make_graph.sh --local       # never touch the cluster
```

`BLUEBEAR_HOST` and `BLUEBEAR_ROOT` override where it connects;
`--ckpt-intra` / `--ckpt-inter` override the checkpoint discovery.

```bash
./make_graph.sh --graph luxembourg      # oldenburg is the default
```

Output lands in `D_inter/private_inference/out/{syn,selftest}_<graph>_k<k>/`:
the assembled graph, a plain-text table, the same table as LaTeX, and the degree
distributions as CSV. The table carries three columns -- Synthetic, Random and
Real -- where Random places the released edge counts uniformly at random, which
is the scale that makes the other two readable.

## Two conda envs

| env | torch | used for |
|---|---|---|
| `kmis` | 1.11 + torch_sparse | coarsening, extraction, evaluation |
| `digress` | 2.0.1 | training and sampling |

On BlueBEAR the digress env is at `/rds/projects/c/chenhp-graphsyn/envs/digress`,
outside every project folder, so wiping a project directory does not remove it.
