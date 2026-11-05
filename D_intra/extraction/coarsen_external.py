"""Coarsening statistics of an external graph (MatrixMarket .mtx) at several k.

Run LOCALLY in the `kmis` env, from this directory, with -I (the edge lists are
downloaded files, so nothing is imported from beside them):

    python -I coarsen_external.py ../../datasets_external/socfb/Brown11/socfb-Brown11.mtx \\
                                  ../../datasets_external/socfb/Penn94/socfb-Penn94.mtx

Graphs: Rossi & Ahmed, The Network Data Repository, socfb-Brown11 and
socfb-Penn94 (https://networkrepository.com/socfb-Brown11.php, -Penn94.php).

The coarsening is kmis_pairs.extract_pairs's own, call for call (Greedy degree
ranking, maximal k-independent set, cluster_k_mis), so the numbers are
comparable with Oldenburg's and Luxembourg's. Every supernode counts, isolated
nodes included (each is a supernode of size 1).

Rows, one column per (graph, k):
    n, m                        the simple undirected graph (self-loops dropped)
    #supernodes M
    n_i median / 95th percentile / max      supernode sizes (nodes)
    #super edges                original edges BETWEEN two different supernodes,
                                sum over pairs of w_ij  (= m - sum_i m_i)
    #connected supernode pairs  pairs (i, j) joined by at least one edge, |E_c|
    m_i median / mean           edges INSIDE a supernode

Output: results/costs/coarsening_external.tsv (pastes into Excel as cells).
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                    # -I leaves the script's own dir off the path

import numpy as np                          # noqa: E402
import scipy.sparse as sp                   # noqa: E402
import torch                                # noqa: E402
from torch_sparse import SparseTensor       # noqa: E402

from kmis import orderings, reduce, sample  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "results", "costs", "coarsening_external.tsv")


def load_mtx(path):
    """(SparseTensor adj, csr A) of the simple undirected graph in a coordinate .mtx.

    networkrepository writes the header with a single '%' ("%MatrixMarket ..."),
    ids are 1-indexed, and "symmetric" lists each edge once.
    """
    with open(path) as f:
        header = f.readline()
        if "MatrixMarket" not in header or "coordinate" not in header:
            raise SystemExit(f"{path}: not a MatrixMarket coordinate file")
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        n, _, n_lines = (int(x) for x in line.split()[:3])
        uv = np.loadtxt(f, usecols=(0, 1), dtype=np.int64, ndmin=2) - 1
    if len(uv) != n_lines:
        raise SystemExit(f"{path}: header says {n_lines} edge lines, read {len(uv)}")
    loops = int((uv[:, 0] == uv[:, 1]).sum())
    uv = uv[uv[:, 0] != uv[:, 1]]
    A = sp.coo_matrix((np.ones(len(uv)), (uv[:, 0], uv[:, 1])), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.eliminate_zeros()
    A = A.tocoo()
    adj = SparseTensor(row=torch.from_numpy(A.row.astype(np.int64)),
                       col=torch.from_numpy(A.col.astype(np.int64)),
                       value=torch.ones(A.nnz), sparse_sizes=(n, n)).coalesce()
    return adj, A.tocsr(), n_lines, loops


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
    return [("n", n), ("m", m), ("k", k),
            ("#supernodes M", M),
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
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        adj, A, n_lines, loops = load_mtx(path)
        deg = np.diff(A.indptr)
        print(f"{name}: {A.shape[0]} nodes, {A.nnz // 2} edges ({n_lines} lines in the file, "
              f"{loops} self-loops dropped, {int((deg == 0).sum())} isolated nodes), "
              f"degree mean {deg.mean():.1f} max {deg.max()}, loaded in {time.time() - t0:.0f} s",
              flush=True)
        for k in args.ks:
            t0 = time.time()
            st = stats(A, coarsen(adj, k), k)
            print(f"  k={k}: M={st[3][1]}, max n_i={st[6][1]}, pairs={st[8][1]} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            cols.append(f"{name} k={k}")
            rows = [[q] for q, _ in st] if rows is None else rows
            for r, (_, v) in zip(rows, st):
                r.append(fmt(v))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lines = ["\t".join(["Quantity"] + cols)] + ["\t".join(r) for r in rows]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
