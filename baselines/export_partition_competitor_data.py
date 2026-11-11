"""DBLP, Collab, LiveJournal and YouTube with OUR train / test split, as plain files
for the competitors -- the same layout as export_competitor_data.py's Oldenburg and
Vermont folders.

    python export_partition_competitor_data.py --graph collab \\
        --npz .../ogbl_collab_full_csr_kmis_sized_k3.npz \\
        --split ../D_intra/data/splits/collab_k3_s80.json --out .../competitor_data/collab

The supernodes are Huiping's coarsening (the .npz: fine CSR adjacency, `assignment`,
`centroids`), never re-coarsened; the split is the one every dataset of ours was
built with (80 / 20 by supernode, seed 0, supernodes with n >= 3). Node ids are the
.npz's (row / column index of its CSR adjacency).

Writes to --out:
  full_graph.edges        every edge once, "u v" with u < v
  nodes.csv               node, supernode, split (train / test / excluded), is_centroid
  train_graph.edges       edges with BOTH endpoints in training supernodes
  test_graph.edges        edges with BOTH endpoints in test supernodes
  train_supernodes.jsonl  one training supernode per line: its induced subgraph
  test_supernodes.jsonl   {"supernode", "n", "centroid" (local index), "nodes" (global
                          ids, local index = position), "edges" (local ids, u < v)}
  split.json, summary.json
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp


def write_edges(path, u, v):
    order = np.lexsort((v, u))
    u, v = u[order], v[order]
    with open(path, "w") as f:
        for s in range(0, len(u), 1_000_000):
            np.savetxt(f, np.column_stack([u[s:s + 1_000_000], v[s:s + 1_000_000]]), fmt="%d")
    return len(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with np.load(args.npz, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        n = int(meta["n"])
        A = sp.csr_matrix((np.ones(len(z["adj_indices"]), dtype=np.int8), z["adj_indices"],
                           z["adj_indptr"]), shape=(n, n))
        assign = z["assignment"].astype(np.int64)
        centroids = z["centroids"].astype(np.int64)
    if not np.array_equal(assign[centroids], np.arange(len(centroids))):
        raise SystemExit("assignment[centroids[i]] != i")
    split = json.load(open(args.split))
    k = int(split["k"])
    if k != int(meta["parameters"]["k"]):
        raise SystemExit(f"split k={k}, partition k={meta['parameters']['k']}")

    c = int(assign.max()) + 1
    size = np.bincount(assign, minlength=c)
    side = np.full(c, "excluded", dtype=object)            # n < 3: never in our split
    side[np.array(split["train"], dtype=np.int64)] = "train"
    side[np.array(split["test"], dtype=np.int64)] = "test"
    for s, m in split["n"].items():                          # the split's sizes must be ours
        if size[int(s)] != m:
            raise SystemExit(f"supernode {s}: {size[int(s)]} nodes here, {m} in the split")
    if (size[side != "excluded"] < 3).any() or (size[side == "excluded"] >= 3).any():
        raise SystemExit("the split does not cover exactly the supernodes with n >= 3")

    U = sp.triu(A, k=1).tocoo()
    u, v = U.row.astype(np.int64), U.col.astype(np.int64)
    m = write_edges(os.path.join(args.out, "full_graph.edges"), u, v)

    node_side = side[assign]
    is_c = np.zeros(n, dtype=np.int8)
    is_c[centroids] = 1
    with open(os.path.join(args.out, "nodes.csv"), "w") as f:
        f.write("node,supernode,split,is_centroid\n")
        for s in range(0, n, 1_000_000):
            idx = np.arange(s, min(n, s + 1_000_000))
            f.write("".join(f"{i},{a},{p},{q}\n" for i, a, p, q in
                            zip(idx, assign[idx], node_side[idx], is_c[idx])))

    # local index of each node = its position among its supernode's members (sorted ids)
    order = np.lexsort((np.arange(n), assign))
    start = np.searchsorted(assign[order], np.arange(c))
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n) - start[assign[order]]
    inside = assign[u] == assign[v]
    iu, iv, isn = u[inside], v[inside], assign[u[inside]]
    lu, lv = pos[iu], pos[iv]
    lo, hi = np.minimum(lu, lv), np.maximum(lu, lv)
    eord = np.lexsort((hi, lo, isn))
    lo, hi, isn = lo[eord], hi[eord], isn[eord]
    estart = np.searchsorted(isn, np.arange(c))
    eend = np.searchsorted(isn, np.arange(c), side="right")

    counts = {"graph": args.graph, "k": k, "partition": os.path.basename(args.npz),
              "nodes": n, "edges": m, "supernodes": c}
    for part in ("train", "test"):
        keep = (node_side[u] == part) & (node_side[v] == part)
        counts[f"{part}_edges"] = write_edges(os.path.join(args.out, f"{part}_graph.edges"),
                                              u[keep], v[keep])
        counts[f"{part}_nodes"] = int((node_side == part).sum())
        sns = np.flatnonzero(side == part)
        with open(os.path.join(args.out, f"{part}_supernodes.jsonl"), "w") as f:
            for s in sns:
                nodes = order[start[s]:start[s] + size[s]]
                e = np.column_stack([lo[estart[s]:eend[s]], hi[estart[s]:eend[s]]])
                f.write(json.dumps({"supernode": int(s), "n": int(size[s]),
                                    "centroid": int(pos[centroids[s]]),
                                    "nodes": nodes.tolist(), "edges": e.tolist()}) + "\n")
        counts[f"{part}_supernodes"] = int(len(sns))
    counts["excluded_supernodes"] = int((side == "excluded").sum())
    counts["excluded_nodes"] = int((node_side == "excluded").sum())
    counts["edges_between_train_and_test_or_excluded"] = m - counts["train_edges"] - counts["test_edges"]
    json.dump(split, open(os.path.join(args.out, "split.json"), "w"))
    json.dump(counts, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
