"""k-MIS coarsening statistics of one or more graphs, for several k.

    python coarsening_stats.py data/facebook_combined.txt path/to/socfb-Penn94.mtx --ks 1 2 3 4 5

Any graph load_graph.py reads (.txt edge list, .mtx, .mat). Needs torch, torch_sparse,
torch_scatter, torch_geometric, numpy, scipy; no eigendecomposition, so it scales
to millions of edges (Penn94, 1.36M edges, k = 1..5: about 10 s).

The coarsening is the D_intra datasets' own (kmis_pairs.extract_pairs), call for call (Greedy degree
ranking, maximal k-independent set, cluster_k_mis), so the numbers are
comparable with Oldenburg's and Luxembourg's. Every supernode counts, isolated
nodes included (each is a supernode of size 1).

Rows, one column per (graph, k):
    n, m                        the simple undirected graph (self-loops dropped)
    #components                 connected components of the graph (the same at every k:
                                a supernode never spans two components)
    #supernodes M
    #supernodes with n_i >= 3   those D_intra would model (its datasets drop n_i < 3)
    n_i mean                    n / M
    n_i median / 95th percentile / max      supernode sizes (nodes)
    #super edges                original edges BETWEEN two different supernodes,
                                sum over pairs of w_ij  (= m - sum_i m_i)
    #connected supernode pairs  pairs (i, j) joined by at least one edge, |E_c|
    m_i median / mean           edges INSIDE a supernode

Output: results/coarsening_stats.tsv in this folder (pastes into Excel as cells).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # -I leaves the script's own dir off the path

import numpy as np                          # noqa: E402
import scipy.sparse as sp                   # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
import torch                                # noqa: E402
from torch_sparse import SparseTensor       # noqa: E402

from kmis import orderings, reduce, sample  # noqa: E402
from load_graph import load_adjacency, name_of  # noqa: E402

OUT = os.path.join(HERE, "results", "coarsening_stats.tsv")


def load(path):
    """(SparseTensor adj, csr A) from load_graph.load_adjacency."""
    A = load_adjacency(path).tocoo()
    n = A.shape[0]
    adj = SparseTensor(row=torch.from_numpy(A.row.astype(np.int64)),
                       col=torch.from_numpy(A.col.astype(np.int64)),
                       value=torch.ones(A.nnz), sparse_sizes=(n, n)).coalesce()
    return adj, A.tocsr()


def coarsen(adj, k):
    """Supernode id of every node, as kmis_pairs.extract_pairs computes it."""
    deg = adj.sum(dim=1)
    rank = orderings.Greedy()(deg, adj)
    mis = sample.maximal_k_independent_set(adj, k, rank)
    return reduce.cluster_k_mis(mis, adj, k, rank).cpu().numpy()


def stats(A, cluster, k):
    n, m = A.shape[0], int(A.nnz // 2)
    M = int(cluster.max()) + 1
    sizes = np.bincount(cluster, minlength=M)
    P = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, M))
    C = (P.T @ A @ P).tocsr()
    m_in = (C.diagonal() / 2).astype(np.int64)           # edges inside each supernode
    off = sp.triu(C, k=1)
    ncomp = int(connected_components(A, directed=False)[0])
    return [("n", n), ("m", m), ("k", k),
            ("#components", ncomp),
            ("#supernodes M", M),
            ("#supernodes with n_i >= 3", int((sizes >= 3).sum())),
            ("n_i mean", n / M),
            ("n_i median", float(np.median(sizes))),
            ("n_i 95th percentile", float(np.percentile(sizes, 95))),
            ("n_i max", int(sizes.max())),
            ("#super edges (edges between supernodes)", int(off.sum())),
            ("#connected supernode pairs", int(off.nnz)),
            ("m_i median", float(np.median(m_in))),
            ("m_i mean", float(m_in.mean())),
            ("check: sum m_i + super edges = m", "yes" if int(m_in.sum() + off.sum()) == m else "NO")]


def fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)) or float(v).is_integer():
        return str(int(v))
    return f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+", help=".mtx edge lists")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    cols, rows = [], None
    for path in args.graphs:
        name = name_of(path)
        t0 = time.time()
        adj, A = load(path)
        deg = np.diff(A.indptr)
        print(f"{name}: {A.shape[0]} nodes, {A.nnz // 2} edges ({int((deg == 0).sum())} isolated "
              f"nodes), degree mean {deg.mean():.1f} max {deg.max()}, loaded in {time.time() - t0:.0f} s",
              flush=True)
        for k in args.ks:
            t0 = time.time()
            st = stats(A, coarsen(adj, k), k)
            v = dict(st)
            print(f"  k={k}: M={v['#supernodes M']}, max n_i={v['n_i max']}, "
                  f"pairs={v['#connected supernode pairs']} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            cols.append(f"{name} k={k}")
            rows = [[q] for q, _ in st] if rows is None else rows
            for r, (_, v) in zip(rows, st):
                r.append(fmt(v))
            lines = write(args.out, cols, rows)         # after every k, so a stopped run keeps its ks

    print("\n" + "\n".join(lines) + f"\n\nwrote {os.path.abspath(args.out)}")


def write(out, cols, rows):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    lines = ["\t".join(["Quantity"] + cols)] + ["\t".join(r) for r in rows]
    open(out, "w").write("\n".join(lines) + "\n")
    return lines


if __name__ == "__main__":
    main()
