"""Heavy Edge and Variation Neighbourhoods coarsening + Laplacian spectral error eps.

Loukas, "Graph reduction with spectral and cut guarantees", JMLR 2019. Both methods
and the error come from his graph_coarsening library (in ./graph_coarsening, with
NumPy >= 1.24 fixes: np.float/np.bool/np.Inf -> float/bool/inf; max_levels 20).
This is graph-coarsening/facebook_demo.py with the graph and settings as arguments.

    python heavy_edge_vn.py data/facebook_combined.txt                 # as before: r = 0.930, 0.999
    python heavy_edge_vn.py data/facebook_combined.txt --rs 0.9 --methods heavy_edge

For each reduction ratio r and method it prints the coarse size n, the achieved
r = 1 - n/N, the coarsening time and

    eps_k = | || S Pi U_k Lambda_k^{-1/2} ||_2 - 1 |      (restricted spectral approximation)

for k = 1..kmax (coarsening_quality(...)["error_subspace"]; k = 0, the constant
eigenvector, is skipped), summarised as eps_max and eps_median.

Needs pygsp, sortedcontainers, numpy, scipy, matplotlib. MEMORY: compute_fourier_basis
is a dense N x N eigendecomposition and coarsening_quality builds a dense N x |E|
gradient matrix and an N x N identity, so this fits graphs of roughly <= 10k nodes
and <= 100k edges (Facebook-ego: 4,039 nodes, 88,234 edges).
"""
import argparse
import os
import sys
import time
import warnings; warnings.filterwarnings("ignore")

import numpy as np
from pygsp import graphs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # also under python -I
from graph_coarsening.coarsening_utils import coarsen, coarsening_quality
from load_graph import load_adjacency, name_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph", help="edge list (.txt), MatrixMarket (.mtx) or SuiteSparse (.mat)")
    ap.add_argument("--rs", nargs="+", type=float, default=[0.930, 0.999],
                    help="target reduction ratios r = 1 - n/N")
    ap.add_argument("--methods", nargs="+", default=["heavy_edge", "variation_neighborhoods"])
    ap.add_argument("--K", type=int, default=100, help="eigenvectors the coarsening preserves")
    ap.add_argument("--kmax", type=int, default=100, help="eigenvectors eps is measured on")
    args = ap.parse_args()

    G = graphs.Graph(load_adjacency(args.graph))
    print(f"{name_of(args.graph)}: N={G.N}, |E|={G.Ne}, K={args.K}, kmax={args.kmax}\n")
    G.compute_fourier_basis()

    fmt = "%-24s n=%5d  r=%.3f  time=%.4fs  eps_max=%.4f  eps_median=%.4f"
    for r in args.rs:
        for method in args.methods:
            t0 = time.perf_counter()
            C, Gc, Call, Gall = coarsen(G, K=args.K, r=r, method=method)
            dt = time.perf_counter() - t0
            es = coarsening_quality(G, C, kmax=args.kmax)["error_subspace"]
            print(fmt % (method, Gc.N, 1 - Gc.N / G.N, dt,
                         float(np.max(es[1:])), float(np.median(es[1:]))), flush=True)
        print()


if __name__ == "__main__":
    main()
