"""HiGen test output -> competitors/out/HiGen/graph_s<i>.npz (n, edges).

    python export.py SAVE_DIR/gen_graphs_epoch_N.pickle.xz OUT_DIR

Each sample is a list of networkx graphs, one per hierarchy level (test() uses
level='all'); the last non-empty one is the generated leaf graph. Self-loops
(HiGen models a weighted self-edge per community) are dropped.
"""
import lzma
import os
import pickle
import sys

import numpy as np


def main():
    path, out = sys.argv[1], sys.argv[2]
    with lzma.open(path, "rb") as f:
        samples = pickle.load(f)
    os.makedirs(out, exist_ok=True)
    for i, levels in enumerate(samples):
        G = [g for g in (levels if isinstance(levels, list) else [levels]) if g is not None][-1]
        idx = {u: k for k, u in enumerate(G.nodes())}
        e = np.array(sorted({(min(idx[u], idx[v]), max(idx[u], idx[v]))
                             for u, v in G.edges() if u != v}), dtype=np.int32).reshape(-1, 2)
        np.savez(os.path.join(out, f"graph_s{i}.npz"), n=G.number_of_nodes(), edges=e)
        print(f"graph_s{i}: {G.number_of_nodes()} nodes, {len(e)} edges")
    print(f"wrote {len(samples)} graphs to {out}")


if __name__ == "__main__":
    main()
