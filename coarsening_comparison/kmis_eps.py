"""k-MIS coarsening + the same Laplacian spectral error eps, for several k.

This is k-mis-pool/facebook_demo.py with the graph and settings as arguments.
The eps code is transcribed from the Loukas library (coarsening_utils.coarsening_quality
/ graph_utils.get_S) in numpy/scipy only, so the numbers are directly comparable
with heavy_edge_vn.py's:

    eps_k = | || S Pi U_k Lambda_k^{-1/2} ||_2 - 1 |

where L = S S^T is the combinatorial Laplacian, (U, Lambda) its first kmax
eigenpairs, and Pi = C^T C the projection of the coarsening matrix C, whose
row i is 1/sqrt(n_i) on the members of supernode i.

    python kmis_eps.py data/facebook_combined.txt                     # k = 1..8, both rankings
    python kmis_eps.py data/facebook_combined.txt --ks 1 2 3 --rankings degree

Rankings: "degree" (Greedy, highest degree first -- the one the datasets use) and
"divksum" (DivKSum, inverse k-path centrality, the k-MIS paper's). Time is the
median of runs 2..--runs of the coarsening alone.

Needs torch, torch_sparse, torch_scatter, torch_geometric, numpy, scipy.
MEMORY: the dense N x N eigendecomposition and the dense N x |E| gradient matrix
limit this, like heavy_edge_vn.py, to roughly <= 10k nodes and <= 100k edges.
"""
import argparse
import os
import sys
import time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import scipy.sparse as sp
import torch
from torch_sparse import SparseTensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
from kmis import orderings, sample, reduce
from load_graph import load_adjacency, name_of


def laplacian_and_gradient(W):
    W = sp.csr_matrix(W)
    deg = np.asarray(W.sum(1)).ravel()
    L = (sp.diags(deg) - W).tocsc()

    Wu = sp.triu(W, k=1).tocoo()          # one column per undirected edge
    m = Wu.row.size
    S = np.zeros((W.shape[0], m))
    sw = np.sqrt(Wu.data)
    S[Wu.row, np.arange(m)] = sw
    S[Wu.col, np.arange(m)] = -sw
    return L, S


def first_eigenpairs(L, kmax):
    w, V = np.linalg.eigh(L.toarray())
    idx = np.argsort(w)
    return V[:, idx[:kmax]], w[idx[:kmax]]


def coarsening_quality(L, S, U, l, C, kmax):
    kmax = min(kmax, C.shape[0])          # p cannot exceed the coarse graph size
    l = l[:kmax].copy()
    U = U[:, :kmax]
    mask = l < 1e-8                       # zero out the (possibly several) null modes
    l[mask] = 1.0
    linv = l ** (-0.5); linv[mask] = 0.0

    Pi = (C.T @ C)                        # N x N
    Lc = (C @ L @ C.T).toarray()          # c x c
    lc = np.sort(np.linalg.eigvalsh(Lc))[:kmax]

    error_eigenvalue = np.abs(l - lc) / l
    error_eigenvalue[mask] = 0.0

    M = S.T @ (Pi @ (U @ np.diag(linv)))  # |E| x kmax
    # ||M[:, :k]||_2 = sqrt(lambda_max(M[:, :k]^T M[:, :k])), via the small Gram matrix
    G = M.T @ M
    error_subspace = np.zeros(kmax)
    for k in range(1, kmax):
        lam = np.linalg.eigvalsh(G[: k + 1, : k + 1])[-1]
        error_subspace[k] = abs(np.sqrt(max(lam, 0.0)) - 1.0)

    return error_eigenvalue, error_subspace


def build_C(cluster, n):
    """Orthonormal coarsening matrix (c x n) from a k-MIS cluster assignment."""
    c = int(cluster.max() + 1)
    sizes = np.bincount(cluster, minlength=c).astype(float)
    return sp.coo_matrix(
        (1.0 / np.sqrt(sizes[cluster]), (cluster, np.arange(n))), shape=(c, n)
    ).tocsc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", help="edge list (.txt), MatrixMarket (.mtx) or SuiteSparse (.mat)")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--rankings", nargs="+", default=["divksum", "degree"],
                    choices=["divksum", "degree"])
    ap.add_argument("--kmax", type=int, default=100, help="eigenvectors eps is measured on")
    ap.add_argument("--runs", type=int, default=30, help="timing repeats")
    args = ap.parse_args()
    torch.manual_seed(0)

    W = load_adjacency(args.graph)
    n = W.shape[0]
    co = W.tocoo()
    adj = SparseTensor(row=torch.as_tensor(co.row, dtype=torch.long),
                       col=torch.as_tensor(co.col, dtype=torch.long),
                       value=torch.ones(co.nnz, dtype=torch.float), sparse_sizes=(n, n))
    L, S = laplacian_and_gradient(W)
    U, l = first_eigenpairs(L, args.kmax)
    deg = adj.sum(dim=1)

    def make_rank(kind, k):
        if kind == "divksum":              # paper ranking: inverse k-path centrality
            return orderings.DivKSum(k)(torch.ones((n,), dtype=torch.float), adj)
        return orderings.Greedy()(deg, adj)    # degree ranking: highest degree first

    fmt = "%-16s n=%5d  r=%.3f  time_med=%.4fs  eps_max=%.4f  eps_median=%.4f"
    print(f"{name_of(args.graph)}: N={n}, |E|={W.nnz // 2}, kmax={args.kmax}, "
          f"time = median of runs 2-{args.runs}\n")
    labels = {"divksum": "DivKSum", "degree": "Degree"}
    for k in args.ks:
        for kind in args.rankings:
            times = []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                rank = make_rank(kind, k)
                mis = sample.maximal_k_independent_set(adj, k, rank)
                cluster = reduce.cluster_k_mis(mis, adj, k, rank).cpu().numpy().astype(np.int64)
                times.append(time.perf_counter() - t0)
            t_med = float(np.median(times[1:]))

            C = build_C(cluster, n)
            _, es = coarsening_quality(L, S, U, l, C, args.kmax)
            nc = C.shape[0]
            # eps over eigenvalues 1..kmax (index 0 is the trivial constant vector)
            print(fmt % (f"{labels[kind]} (k={k})", nc, 1 - nc / n, t_med,
                         es[1:].max(), np.median(es[1:])), flush=True)
        print()


if __name__ == "__main__":
    main()
