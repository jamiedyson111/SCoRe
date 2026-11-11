"""Oldenburg and Vermont with OUR train / test split, as plain files for the competitors.

    cd D_intra/extraction && python ../../baselines/export_competitor_data.py   (kmis env)

For each graph (k = 7, the 80 / 20 split by supernode of D_intra/data/splits/
<graph>_k7_s80.json, seed 0) writes results/competitor_data/<graph>/:

  full_graph.edges        every edge once, "u v" with u < v, 0-based node ids
  nodes.csv               node, supernode, split (train / test / excluded), is_centroid
  train_graph.edges       edges with BOTH endpoints in training supernodes
  test_graph.edges        edges with BOTH endpoints in test supernodes
  train_supernodes.jsonl  one training supernode per line: its induced subgraph
  test_supernodes.jsonl   the same for the test supernodes
                          {"supernode", "n", "centroid" (local index), "nodes" (global
                          ids, local index = position), "edges" (local ids, u < v)}
  split.json              the split file itself
  summary.json            the counts

The supernodes are this repository's k-MIS (kmis_pairs.extract_pairs, the call the
datasets were built with); every one is checked against the split file's sizes.
Supernodes with n < 3 were never in our split (Vermont: 553; Oldenburg: none) and are
"excluded". Edges between a training and a test supernode are in neither the train
nor the test graph: no test node or edge appears in any training file.
"""
import json
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.getcwd())
from kmis_pairs import extract_pairs, load_graph  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "competitor_data")
SPLITS = os.path.join(HERE, "..", "D_intra", "data", "splits")
K = 7


def write_edges(path, A):
    U = sp.triu(A, k=1).tocoo()
    order = np.lexsort((U.col, U.row))
    with open(path, "w") as f:
        f.write("".join(f"{u} {v}\n" for u, v in zip(U.row[order], U.col[order])))
    return len(order)


def export(graph):
    out = os.path.join(OUT, graph)
    os.makedirs(out, exist_ok=True)
    split = json.load(open(os.path.join(SPLITS, f"{graph}_k{K}_s80.json")))
    train, test = set(split["train"]), set(split["test"])
    _, A = load_graph(graph)
    A = sp.csr_matrix(A)
    n = A.shape[0]
    recs = extract_pairs(graph, K, cond=("n",), min_n=1, verbose=False)

    sn_of = np.full(n, -1)
    side = {}
    for r in recs:
        sn, size = int(r["sn"]), int(r["n"])
        sn_of[r["nodes"].numpy()] = sn
        if size >= 3:
            if split["n"].get(str(sn)) != size:
                raise SystemExit(f"{graph}: supernode {sn} has {size} nodes here, "
                                 f"{split['n'].get(str(sn))} in the split file")
            side[sn] = "train" if sn in train else "test" if sn in test else None
            if side[sn] is None:
                raise SystemExit(f"{graph}: supernode {sn} is in neither split")
        else:
            side[sn] = "excluded"
    if (sn_of < 0).any() or len([s for s in side.values() if s != "excluded"]) != len(split["n"]):
        raise SystemExit(f"{graph}: the coarsening does not cover the split")

    centre = {int(r["centre"]) for r in recs}
    with open(os.path.join(out, "nodes.csv"), "w") as f:
        f.write("node,supernode,split,is_centroid\n")
        f.write("".join(f"{v},{sn_of[v]},{side[int(sn_of[v])]},{int(v in centre)}\n" for v in range(n)))

    m = write_edges(os.path.join(out, "full_graph.edges"), A)
    counts = {"graph": graph, "k": K, "nodes": n, "edges": m, "supernodes": len(recs)}
    for part in ("train", "test"):
        keep = np.array([side[int(s)] == part for s in sn_of])
        D = sp.diags(keep.astype(float))
        counts[f"{part}_edges"] = write_edges(os.path.join(out, f"{part}_graph.edges"), D @ A @ D)
        counts[f"{part}_nodes"] = int(keep.sum())
        rows = []
        for r in recs:
            if side[int(r["sn"])] != part:
                continue
            nodes = r["nodes"].numpy()
            ei = r["edge_index"].numpy()
            e = sorted({(int(a), int(b)) for a, b in zip(*ei) if a < b})
            rows.append(json.dumps({"supernode": int(r["sn"]), "n": int(r["n"]),
                                    "centroid": int(np.flatnonzero(nodes == r["centre"])[0]),
                                    "nodes": nodes.tolist(), "edges": e}))
        with open(os.path.join(out, f"{part}_supernodes.jsonl"), "w") as f:
            f.write("\n".join(rows) + "\n")
        counts[f"{part}_supernodes"] = len(rows)
    ex = np.array([side[int(s)] == "excluded" for s in sn_of])
    counts["excluded_supernodes"] = sum(v == "excluded" for v in side.values())
    counts["excluded_nodes"] = int(ex.sum())
    counts["edges_between_train_and_test_or_excluded"] = m - counts["train_edges"] - counts["test_edges"]
    json.dump(split, open(os.path.join(out, "split.json"), "w"))
    json.dump(counts, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    for g in sys.argv[1:] or ["oldenburg", "vermont"]:
        export(g)
