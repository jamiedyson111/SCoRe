"""Coarsening statistics and state sizes (Sections IV-B, V-D, VI-E), at several K.

Run LOCALLY in the `kmis` env, from this directory. CPU only, no model needed:

    python coarsening_stats.py                       # Oldenburg + Luxembourg, K = 7 (seconds)
    python coarsening_stats.py --ks 3 5 7 9 11       # several K
    python coarsening_stats.py --graphs oldenburg --ks 7

The coarsening is kmis_pairs.extract_pairs's own, call for call (Greedy degree
ranking, maximal k-independent set, cluster_k_mis), so K = 7 here is exactly the
partition the datasets were built from. Every supernode counts, including the
ones under min_n that the datasets drop.

Rows, one column per (graph, K):

  coarsening   n, m, M, M/n, the bound n / (floor(K/2) + 1) and whether M meets it;
               supernode size median, mean, max (b = max n_i); m_c = |E_c|, the
               coarse edges (supernode pairs joined by at least one edge), and m_c / m
  D_intra      sum_i n_i^2, its dense adjacency entries, against the bound n*b
  D_inter      sum_{(i,j) in E_c} (n_i + n_j)^2, its dense state, against 4 m_c b^2
  reference    n^2, a whole-graph dense diffusion model's adjacency
  T            reverse diffusion steps (both models), cost is linear in it

Output: results/costs/coarsening_stats.tsv (pastes into Excel as cells).
"""
import argparse
import os
import time

import numpy as np
import scipy.sparse as sp

from kmis import orderings, reduce, sample
from kmis_pairs import load_graph

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "results", "costs")
T_STEPS = 500        # model.diffusion_steps in configs/experiment/{cluster_lux,pair_lux}.yaml


def coarsen(adj, K):
    """Supernode id of every node, as kmis_pairs.extract_pairs computes it."""
    deg = adj.sum(dim=1)
    rank = orderings.Greedy()(deg, adj)
    mis = sample.maximal_k_independent_set(adj, K, rank)
    return reduce.cluster_k_mis(mis, adj, K, rank).cpu().numpy()


def stats(A, cluster, K):
    n, m = A.shape[0], int(A.nnz // 2)
    M = int(cluster.max()) + 1
    sizes = np.bincount(cluster, minlength=M).astype(np.int64)
    b = int(sizes.max())
    # Coarse graph P^T A P: off-diagonal non-zeros are the coarse edges E_c.
    P = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, M))
    C = sp.triu(P.T @ A @ P, k=1).tocoo()
    m_c = int(C.nnz)
    pair_state = int(((sizes[C.row] + sizes[C.col]) ** 2).sum())
    bound_M = n / (K // 2 + 1)
    intra = int((sizes ** 2).sum())
    return [
        ("n", n), ("m", m), ("K", K),
        ("M (supernodes)", M), ("M / n", M / n),
        ("bound n / (floor(K/2)+1)", bound_M), ("M <= bound", "yes" if M <= bound_M else "NO"),
        ("supernode size median", float(np.median(sizes))),
        ("supernode size mean", float(sizes.mean())),
        ("b = max n_i", b),
        ("m_c = |E_c| (coarse edges)", m_c), ("m_c / m", m_c / m),
        ("D_intra: sum n_i^2", intra), ("bound n*b", n * b),
        ("sum n_i^2 / (n*b)", intra / (n * b)),
        ("D_inter: sum over E_c of (n_i+n_j)^2", pair_state),
        ("bound 4*m_c*b^2", 4 * m_c * b * b),
        ("sum (n_i+n_j)^2 / (4 m_c b^2)", pair_state / (4 * m_c * b * b)),
        ("full-graph dense: n^2", n * n),
        ("sum n_i^2 / n^2", intra / (n * n)),
        ("sum (n_i+n_j)^2 / n^2", pair_state / (n * n)),
        ("T (reverse diffusion steps)", T_STEPS),
    ]


def fmt(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.6g}" if abs(v) < 1e5 else f"{v:.4e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", nargs="+", default=["oldenburg", "luxembourg"])
    ap.add_argument("--ks", nargs="+", type=int, default=[7])
    ap.add_argument("--out", default=os.path.join(OUT, "coarsening_stats.tsv"))
    args = ap.parse_args()

    os.chdir(HERE)                    # load_graph resolves datasets/ from here
    cols, rows = [], None
    for g in args.graphs:
        adj, A = load_graph(g)
        for K in args.ks:
            t0 = time.time()
            st = stats(A, coarsen(adj, K), K)
            print(f"  {g} K={K}: M={st[3][1]}, b={st[9][1]}, m_c={st[10][1]} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            cols.append(f"{g} K={K}")
            rows = [[k] for k, _ in st] if rows is None else rows
            for r, (_, v) in zip(rows, st):
                r.append(fmt(v))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    lines = ["\t".join(["Quantity"] + cols)] + ["\t".join(r) for r in rows]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nwrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
