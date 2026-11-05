"""Read a graph file into a symmetric 0/1 scipy CSR adjacency (no self-loops).

Shared by every script in this folder, so all methods see exactly the same graph.

Supported, by extension:
    .txt / .edges / .el   edge list, one "u v" per line (SNAP style, e.g.
                          facebook_combined.txt, ca-GrQc.txt); lines starting with # or %
                          skipped. Ids need not be 0..n-1: the nodes that appear are
                          renumbered 0..n-1 in id order (SNAP ids have gaps -- ca-GrQc's
                          5,242 nodes run up to id ~26,000), so an edge list never
                          creates isolated nodes that are not in the file.
    .mtx                  MatrixMarket coordinate file, 1-indexed (networkrepository.com,
                          SuiteSparse); a header with one or two '%' is accepted
    .mat                  SuiteSparse MATLAB file (the sparse matrix in Problem.A)

Edges are made undirected, weights and duplicates collapse to 1, self-loops are
dropped. Isolated nodes are kept (they matter for the supernode counts).
"""
import os

import numpy as np
import scipy.sparse as sp


def load_adjacency(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".mat":
        import scipy.io as sio
        prob = sio.loadmat(path)["Problem"][0][0]
        A = next(f for f in prob if sp.issparse(f)).tocoo()
        n, u, v = A.shape[0], A.row, A.col
    elif ext == ".mtx":
        with open(path) as f:
            header = f.readline()
            if "MatrixMarket" not in header or "coordinate" not in header:
                raise SystemExit(f"{path}: not a MatrixMarket coordinate file")
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            n = int(line.split()[0])
            uv = np.loadtxt(f, usecols=(0, 1), dtype=np.int64, ndmin=2) - 1
        u, v = uv[:, 0], uv[:, 1]
    else:
        uv = np.loadtxt(path, usecols=(0, 1), dtype=np.int64, ndmin=2, comments=("#", "%"))
        ids, uv = np.unique(uv, return_inverse=True)          # compact 0..n-1
        uv = uv.reshape(-1, 2)
        n = len(ids)
        u, v = uv[:, 0], uv[:, 1]
    keep = u != v
    A = sp.coo_matrix((np.ones(int(keep.sum())), (u[keep], v[keep])), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.eliminate_zeros()
    return A.tocsr()


def name_of(path):
    return os.path.splitext(os.path.basename(path))[0]
