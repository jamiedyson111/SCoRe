"""Basic statistics of one or more graphs: the dataset table.

graph-coarsening/graph_features.py's characterize(), for any graph file and
without networkx:

    python dataset_stats.py data/facebook_combined.txt path/to/other.mtx

|V|, |E|, density 2|E| / (|V|(|V|-1)), min / max / mean / median degree,
connected components and the size of the largest. Prints a TSV (pastes into Excel).
Needs numpy and scipy only; fine for millions of edges.
"""
import argparse
import os
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
from load_graph import load_adjacency, name_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    args = ap.parse_args()
    cols = ["graph", "|V|", "|E|", "density", "min deg", "max deg", "mean deg",
            "median deg", "components", "largest component"]
    print("\t".join(cols))
    for path in args.graphs:
        A = load_adjacency(path)
        n, m = A.shape[0], A.nnz // 2
        deg = np.diff(A.indptr)
        nc, lab = connected_components(A, directed=False)
        print("\t".join([name_of(path), str(n), str(m), f"{2 * m / (n * (n - 1)):.2e}",
                         str(deg.min()), str(deg.max()), f"{deg.mean():.2f}",
                         f"{np.median(deg):g}", str(nc), str(np.bincount(lab).max())]),
              flush=True)


if __name__ == "__main__":
    main()
