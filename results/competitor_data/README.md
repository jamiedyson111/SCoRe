# Oldenburg and Vermont with our train / test split (for the competitors)

The split used for every result in the paper: k-MIS coarsening at **k = 7**, then **80 / 20 by
supernode** (seed 0). A supernode, and every node in it, is either training or test. The same
split is used by D_intra, D_inter, B and C.

| | Oldenburg | Vermont |
|---|---|---|
| nodes / edges | 6,105 / 7,029 | 97,975 / 106,242 |
| supernodes | 317 | 6,075 |
| **train** supernodes / nodes / edges | 254 / 4,953 / 5,610 | 4,418 / 76,848 / 82,159 |
| **test** supernodes / nodes / edges | 63 / 1,152 / 1,183 | 1,104 / 20,024 / 20,341 |
| excluded (supernodes with n < 3) | none | 553 supernodes, 1,103 nodes |
| edges between train and test (or excluded) | 236 | 3,742 |

**No test node and no test edge appears in any training file.** Edges joining a training
supernode to a test supernode are in neither the train nor the test graph (last row). Vermont's 553
supernodes with fewer than 3 nodes were never part of our split ("excluded"); they belong to
neither side.

## Files (one folder per graph)

| file | contents |
|---|---|
| `full_graph.edges` | the whole graph: one edge per line, `u v` with `u < v`, 0-based node ids |
| `nodes.csv` | `node,supernode,split,is_centroid`, with `split` = `train` / `test` / `excluded` |
| `train_graph.edges` | the subgraph induced on all training nodes (both endpoints train), original node ids |
| `test_graph.edges` | the same for the test nodes |
| `train_supernodes.jsonl` | one training supernode per line, as its own small graph |
| `test_supernodes.jsonl` | the same for the test supernodes |
| `split.json` | our split file: supernode ids in `train` / `test`, and each supernode's size `n` |
| `summary.json` | the counts above |

Each line of `*_supernodes.jsonl` is

```json
{"supernode": 12, "n": 15, "centroid": 0, "nodes": [4021, 4022, ...], "edges": [[0, 3], [0, 7], ...]}
```

`nodes` are global node ids; a node's local index is its position in `nodes`; `edges` use local
indices (`u < v`); `centroid` is the local index of the supernode's k-MIS centre.

**Node ids.** Oldenburg: the ids of `OL.cedge` (Brinkhoff). Vermont: `VT.mtx` (DIMACS9) id minus 1.

## Which file for which setting (paper, Section VI-E)

- **Target-only (e.g. ILE-Target, HiGen, HiGGs)**: train on the training part only, either
  `train_graph.edges` (one large graph) or `train_supernodes.jsonl` (a set of small graphs),
  whichever the method expects. Never use `test_*` or `full_graph.edges` for training.
- **Local evaluation** on held-out instances: `test_supernodes.jsonl` (each test supernode's real
  subgraph) or `test_graph.edges`.
- **Full-graph evaluation**: compare a generated graph of the target's size (6,105 / 97,975 nodes)
  against `full_graph.edges`, as for our assembled graphs.
- **Source-only (ZS) and fine-tuned (FT)**: pretrain on Luxembourg (all of it is source data), then
  fine-tune on the training part above.

Made by `baselines/export_competitor_data.py`, which regenerates the supernodes with the same
k-MIS call our datasets were built with and checks every one against `split.json`.

---

# DBLP, Collab, LiveJournal and YouTube (November 2026)

Same file layout as above, one folder per graph (`dblp/`, `collab/`, `livejournal/`,
`youtube/`). Two transfer pairs: **DBLP → Collab** (k = 3) and **LiveJournal → YouTube**
(k = 4). The supernodes are **Huiping's coarsenings, never re-coarsened**:

| graph | partition file (`coarsen_compare_expr/output/...`) | method |
|---|---|---|
| DBLP | `dblp_collab/com_dblp_csr_kmis_sized_k3.npz` | K-MIS sized, k = 3 |
| Collab | `dblp_collab/ogbl_collab_full_csr_kmis_sized_k3.npz` | K-MIS sized, k = 3 |
| LiveJournal | `livejournal_youtube/com_livejournal_csr_kmis_cb_v4_k4_b500_l10.npz` | CB-KMIS, k = 4, b = 500, l = 10 |
| YouTube | `livejournal_youtube/com_youtube_csr_kmis_cb_v4_k4_b500_l10.npz` | CB-KMIS, k = 4, b = 500, l = 10 |

**Node ids are the .npz's** (row / column index of its CSR adjacency), not the SNAP /
OGB ids. `centroid` in the jsonl files is the .npz's `centroids` entry.

Split: 80 / 20 by supernode, seed 0, over the supernodes with n >= 3 (smaller ones are
"excluded" and belong to neither side), identical to the one every dataset of ours was
built with (`split.json` = `D_intra/data/splits/<graph>_k<k>_s80.json`). Checked: the
1,787 Collab test supernodes match our test records node for node and edge for edge,
and no node appears in both the train and the test files.

| | DBLP | Collab | LiveJournal | YouTube |
|---|---|---|---|---|
| nodes / edges | 317,080 / 1,049,866 | 235,868 / 967,632 | 3,997,962 / 34,681,189 | 1,134,890 / 2,987,624 |
| supernodes | 16,511 | 9,532 | 124,112 | 91,313 |
| **train** supernodes / nodes / edges | 12,064 / 251,674 / 772,223 | 7,147 / 188,281 / 722,622 | 98,888 / 3,199,023 / 23,515,871 | 44,065 / 871,143 / 2,023,394 |
| **test** supernodes / nodes / edges | 3,016 / 62,544 / 142,889 | 1,787 / 46,391 / 132,779 | 24,722 / 798,417 / 2,634,037 | 11,016 / 222,948 / 302,346 |
| excluded supernodes (n < 3) / nodes | 1,431 / 2,862 | 598 / 1,196 | 502 / 522 | 36,232 / 40,799 |
| edges between train and test (or excluded) | 134,754 | 112,231 | 8,531,281 | 661,884 |

In the transfer setting the source graph (DBLP, LiveJournal) is all source data; its
split matters only for our configuration A (train source, test source).

Made by `baselines/export_partition_competitor_data.py` (job `baselines/jobs/export_social.sh`).

## Runtime of our models

`runtime/runtime_memory_{oldenburg,vermont}.tsv`: training time, test-set sampling time
(per seed) and whole-graph inference time (D_intra, D_inter, assembly; per seed), with
peak GPU memory. The new datasets' rows will be added as their runs finish.

## Evaluation code

The latest scorers, on BlueBEAR in place:

| part | entry point | README |
|---|---|---|
| D_intra (per test supernode) | `D_intra/evaluation/repeat_summary.py` | `D_intra/evaluation/README.md` (sample format for other methods at the end) |
| D_inter (per test pair) | `D_inter/evaluation/transfer_summary.py` | `D_inter/evaluation/README.md` |
| whole graph | `D_inter/private_inference/syn_repeats_summary.py` (+ `graph_mmd.py`: clustering / orbit / spectrum MMD) | `D_inter/private_inference/README.md` |
