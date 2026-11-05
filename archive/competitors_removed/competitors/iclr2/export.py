"""ICLR [2] test-mode output -> competitors/out/ICLR2/graph_s<i>.npz (n, edges).

    python export.py RUN/test/step_N.pkl OUT_DIR BETA

The test pickle holds {'ema_<beta>': {'pred_graphs': [networkx graphs], ...}}.
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np


def main():
    path, out, beta = sys.argv[1], sys.argv[2], sys.argv[3]
    res = pickle.load(open(path, "rb"))
    key = f"ema_{float(beta)}" if f"ema_{float(beta)}" in res else f"ema_{beta}"
    graphs = res[key]["pred_graphs"]
    os.makedirs(out, exist_ok=True)
    for i, G in enumerate(graphs):
        G = nx.convert_node_labels_to_integers(G)
        e = np.array([(u, v) for u, v in G.edges() if u != v], dtype=np.int32).reshape(-1, 2)
        np.savez(os.path.join(out, f"graph_s{i}.npz"), n=G.number_of_nodes(), edges=e)
        print(f"graph_s{i}: {G.number_of_nodes()} nodes, {len(e)} edges")
    print(f"wrote {len(graphs)} graphs to {out}")


if __name__ == "__main__":
    main()
