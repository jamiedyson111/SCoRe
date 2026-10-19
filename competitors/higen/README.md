# HiGen [5]: Hierarchical Graph Generative Networks (Karami, 2024)

Competitor for the assembled-graph table. Trained on the same 100 Luxembourg
regions as ICLR [2] (`../data/lux_regions.pkl`), then asked for five graphs of
6105 nodes. It sees only the node count.

Repo: <https://github.com/Karami-m/HiGen_main>, cloned on BlueBEAR to
`/rds/projects/c/chenhp-graphsyn/competitors/HiGen_main`; env
`/rds/projects/c/chenhp-graphsyn/envs/higen` (Python 3.8, torch 1.13.1 + CUDA 11.7).

| file | what |
|---|---|
| `setup_bluebear.sh` | clone, build the env (numpy < 1.24, pyyaml 5.4.1), apply the patch, convert data |
| `convert_data.py` | `lux_regions.pkl` → `$REPO/data/LuxRegions2/lux_regions_nx.pkl` (120 graphs, 100 train first) |
| `higen_lux.yaml` | the repo's DD2 config with dataset, split, coarsening (SC0), run length and size changed |
| `apply_patch.py` | adds the `LuxRegions2` dataset, size control, and a faster spectral search (below) |
| `preprocess.py` / `run_preprocess.sh` | builds the 120 two-level hierarchies in parallel on CPU |
| `run_train.sh` | one 12 h training job; resumes from the newest snapshot |
| `run_sample.sh` | 5 graphs of 6105 nodes, exported to `competitors/out/HiGen/` on BlueBEAR |
| `export.py` | HiGen's `.pickle.xz` output → `graph_s<i>.npz` |

## Run (BlueBEAR)

```bash
cd /rds/projects/c/chenhp-graphsyn
bash competitors/higen/setup_bluebear.sh                   # once, on a LOGIN node
P=$(sbatch --parsable competitors/higen/run_preprocess.sh)   # CPU, minutes
sbatch --dependency=afterok:$P competitors/higen/run_train.sh   # resubmit after each 12 h timeout
sbatch competitors/higen/run_sample.sh                     # after training
```

## Changes to the repo (`apply_patch.py`)

1. **`LuxRegions2` dataset**, built like the repo's FIRSTMM2 (its largest graphs).
2. **Size control.** HiGen draws each sample's total nodes/edges from the training
   graphs; with `test.target_num_nodes: 6105` they are rescaled to 6105 nodes,
   edges in proportion (the edges-per-node ratio still comes from Luxembourg).
3. **Coarsening.** The released code implements only spectral clustering (SC0;
   its Louvain option raises "not implemented"). SC0 tries every cluster count
   K in [0.7√n, 1.3√n] with 1000 k-means restarts on a dense matrix: 100-200 h for
   120 regions. The patch tries 5 evenly spaced K in the same range with 10
   restarts on a sparse matrix: about 20 s per region. Largest community on a
   5403-node region: 248 nodes (the model allows 500).

## Output

On BlueBEAR: `competitors/out/HiGen/graph_s0.npz … graph_s4.npz`, each `{n, edges (m, 2)}`. Copy to the
local `results/competitors/HiGen/` and run `syn_repeats_summary.py`.
