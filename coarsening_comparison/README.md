# Coarsening comparison: k-MIS vs Heavy Edge vs Variation Neighbourhoods

Everything for the initial coarsening comparison study, in one folder. Every
script reads the graph through `load_graph.py`, so all methods see exactly the
same graph, and every script takes the graph file as an argument.

| file | what |
|---|---|
| `heavy_edge_vn.py` | **Heavy Edge** and **Variation Neighbourhoods** coarsening (Loukas, JMLR 2019) + the Laplacian spectral error ε |
| `kmis_eps.py` | **k-MIS** coarsening for several k (degree and DivKSum rankings) + the same ε |
| `dataset_stats.py` | dataset table: \|V\|, \|E\|, density, min/max/mean/median degree, components |
| `coarsening_stats.py` | k-MIS coarsening statistics per k: #supernodes, n_i median / 95th pct / max, #super edges, #connected supernode pairs, m_i median / mean |
| `load_graph.py` | reads `.txt` edge lists (SNAP), `.mtx` (MatrixMarket, networkrepository / SuiteSparse) and `.mat` (SuiteSparse); every graph is made undirected |
| `convert_gt.py` | graph-tool `.gt` file (a Netzschleuder download, after `zstd -d`) → undirected `.mtx` |
| `convert_aminer.py` | AMiner DBLP-Citation-network (V12+ JSON Lines, zipped) → undirected `.mtx`, streamed |
| `to_excel.py` | `coarsening_stats.py`'s output → one row per k in the spreadsheet's column order |
| `graph_coarsening/` | Loukas's library (github.com/loukasa/graph-coarsening), with NumPy ≥ 1.24 fixes (`np.float`/`np.bool`/`np.Inf` → `float`/`bool`/`inf`) and `max_levels` 10 → 20 |
| `kmis/` | the k-MIS package (github.com/flandolfi/k-mis-pool), the same copy D_intra's datasets were built with |
| `data/facebook_combined.txt` | SNAP ego-Facebook (4,039 nodes, 88,234 edges), for a first test |

## The spectral error ε

For the first k eigenpairs (U_k, Λ_k) of the combinatorial Laplacian L = S Sᵀ
(S is the N × |E| gradient matrix) and a coarsening matrix C with Π = CᵀC,

    ε_k = | ‖ S Π U_k Λ_k^{-1/2} ‖₂ − 1 |        (restricted spectral approximation, Loukas 2019)

computed for k = 1 … kmax (kmax = 100; k = 0, the constant eigenvector, is skipped)
and reported as **eps_max** and **eps_median** over k.

- **Heavy Edge / Variation Neighbourhoods:** the library's own
  `coarsening_quality(G, C, kmax)["error_subspace"]`, in
  `graph_coarsening/coarsening_utils.py`, function `coarsening_quality`.
- **k-MIS:** the same formula, transcribed in numpy/scipy in `kmis_eps.py`, function
  `coarsening_quality`. C has row i = 1/√n_i on the members of supernode i
  (`build_C`), which is what the library's C is for its contractions.

**Compare at matched size.** kmax is capped at the coarse graph's size n, so when
n < 100 ε is measured on fewer eigenvectors. Choose `--rs` for Heavy Edge /
Variation Neighbourhoods so their n matches k-MIS's n at each k. For example,
k-MIS with k = 2 gives r = 0.998; use `--rs 0.998`.

## Environments

- `heavy_edge_vn.py`: pygsp, sortedcontainers, numpy, scipy, matplotlib.
- `kmis_eps.py`, `coarsening_stats.py`: torch, torch_sparse, torch_scatter,
  torch_geometric, numpy, scipy.
- `dataset_stats.py`: numpy, scipy.

On BlueBEAR, one virtual environment built on the `digress` env covers all four,
since it already has torch and torch_geometric. Set it up once, in this folder:

```bash
module purge; module load bluebear bear-apps/2024a/live Miniforge3/25.3.0-3
eval "$(conda shell.bash hook)"; conda activate /rds/projects/c/chenhp-graphsyn/envs/digress
python -m venv --system-site-packages .venv && . .venv/bin/activate
pip install pygsp sortedcontainers
python -c "import torch_sparse, torch_scatter, pygsp, sortedcontainers; print('ok')"
```

After that, `. .venv/bin/activate` (after the module/conda lines) before running.
If `torch_sparse` or `torch_scatter` is missing, install the wheels matching
`python -c "import torch; print(torch.__version__)"` from data.pyg.org.

## Running

```bash
python dataset_stats.py data/facebook_combined.txt                      # seconds
python coarsening_stats.py data/facebook_combined.txt --ks 1 2 3 4 5   # seconds
python kmis_eps.py data/facebook_combined.txt --ks 1 2 3 4 5           # ~1 min
python heavy_edge_vn.py data/facebook_combined.txt --rs 0.930 0.999    # ~5 min
```

Run the ε scripts as a batch job (`sbatch`), not on the login node, for anything
larger than Facebook-ego.

## Adding a dataset

Put the file in `data/` and pass its path. Formats: `.txt` (edge list; ids need
not be contiguous, they are renumbered), `.mtx` (1-indexed, e.g.
networkrepository.com), `.mat` (SuiteSparse).

**Every graph is treated as undirected.** `load_graph.py` symmetrises the adjacency
(A + Aᵀ), merges duplicate edges and drops self-loops; isolated nodes are kept. For a
citation network this means: u cites v gives the edge u–v, mutual citations become
one edge, self-citations are dropped, and papers with no in-dataset citation are
isolated nodes (single-node supernodes).

### Converting other formats

Both converters write an undirected `.mtx` and print what they kept and dropped.
They run in any Python with numpy (`-I` keeps a downloaded folder's files off the
import path):

```bash
# Netzschleuder graph-tool file, e.g. us_roads (VT): the Vermont road network,
# 9th DIMACS Challenge, TIGER/Line 2000 (97,975 nodes, 106,242 edges)
zstd -dc VT.gt.zst > VT.gt
python -I convert_gt.py VT.gt data/VT.mtx                                 # seconds

# AMiner DBLP-Citation-network V19 (7,289,177 papers, 81,929,433 undirected edges:
# 85.0M references, minus 2.9M to papers outside the file and 53,317 self-citations,
# duplicates merged)
python -I convert_aminer.py DBLP-Citation-network-V19.zip data/dblp_v19.mtx   # ~15 min
```

`convert_aminer.py` reads only `id` and `references` from each line and checks
that against a full `json.loads` on the first 20,000 lines (`--check`).

### Filling the spreadsheet

```bash
python coarsening_stats.py data/VT.mtx --ks 3 4 5 6 7 --out results/vt.tsv
python -I to_excel.py results/vt.tsv results/excel_vt.tsv
```

`excel_vt.tsv` has one row per k and the columns #components, #supernodes,
#supernodes with n_i ≥ 3, mean / median / max / 95th percentile n_i, #super edges,
#connected supernode pairs, median / mean m_i. Paste it next to the k column.
`to_excel.py` refuses a table whose check row (Σ m_i + #super edges = m) failed.

Graphs the size of DBLP V19 (7.3M nodes, 82M edges) need a batch job with plenty
of memory, e.g. `sbatch --mem=64G --time=4:00:00 --wrap "python coarsening_stats.py ..."`.

**Size limits:**
- `dataset_stats.py` and `coarsening_stats.py` scale to millions of edges.
  socfb-Penn94 (41,536 nodes, 1.36M edges) takes about 10 s for k = 1–5.
- **The ε scripts don't scale that far.** Both use a dense N × N eigendecomposition
  and a dense N × |E| gradient matrix. The library adds a dense N × N identity on top.
  - That's fine up to roughly 10k nodes and 100k edges.
  - socfb-Brown11 (8,600 nodes, 384,526 edges) needs a ~26 GB dense gradient
    matrix; socfb-Penn94 needs ~450 GB. Neither will run as is.
  - Larger graphs need a sparse version of ε: sparse S, and `eigsh` for only the
    first kmax eigenpairs.
