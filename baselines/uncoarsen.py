"""BS1 and BS2: uncoarsening baselines that see only the release.

Both place exactly the released edge count in every block -- round(rho_in *
n(n-1)/2) inside a supernode, w_ij between a connected pair -- so they match the
condition perfectly and differ from D_intra / D_inter only in WHERE the edges go.

    BS1  random        every free slot of the block is equally likely
    BS2  Chung-Lu      every node u carries an expected degree D_u drawn from the
                       PUBLIC graph's (Luxembourg's) degree distribution, and a
                       slot (u, v) is picked with probability proportional to
                       D_u * D_v -- the degree-corrected / configuration-style
                       expansion of a coarse graph

Edges are drawn without replacement over the block's slots, so the count is
exact and there are no multi-edges or self-loops. One D_u per node is used for
every block the node takes part in, internal and cross.
"""
import numpy as np


def _pick(weights, m, rng):
    """m distinct slot indices, with probability proportional to weights."""
    m = min(int(m), len(weights))
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    w = np.asarray(weights, float)
    if (w > 0).sum() < m:                      # too few weighted slots: fall back
        w = w + 1e-12                          # to uniform over the rest
    return rng.choice(len(w), size=m, replace=False, p=w / w.sum())


def internal_block(n, m, rng, D=None):
    """Undirected L_i as a (2, 2m) edge_index; D=None is BS1, else BS2 with degrees D."""
    iu, ju = np.triu_indices(n, k=1)
    w = np.ones(len(iu)) if D is None else D[iu] * D[ju]
    k = _pick(w, m, rng)
    i, j = iu[k], ju[k]
    return np.vstack([np.concatenate([i, j]), np.concatenate([j, i])])


def cross_block(n_i, n_j, w_ij, rng, D_i=None, D_j=None):
    """B_ij as a (2, w) edge_index of local (i-side, j-side) ids."""
    if D_i is None:
        wts = np.ones(n_i * n_j)
    else:
        wts = np.outer(D_i, D_j).ravel()
    k = _pick(wts, w_ij, rng)
    return np.vstack([k // n_j, k % n_j])


def internal_edges(rho_in, n):
    """The released internal edge count, as assemble_syn.random_blocks takes it."""
    return int(round(float(rho_in) * n * (n - 1) / 2.0))


def public_degrees(graph="luxembourg"):
    """Degree multiset of the public graph, the pool BS2 draws D_u from."""
    import os
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "D_intra", "extraction")
    cwd = os.getcwd()
    sys.path.insert(0, os.path.abspath(root))
    os.chdir(root)
    try:
        from kmis_pairs import load_graph
        _, A = load_graph(graph)
    finally:
        os.chdir(cwd)
    return np.asarray(A.sum(1)).ravel().astype(float)
