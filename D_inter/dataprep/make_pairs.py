"""Build (G_i, c_i) training pairs for the intra-cluster diffusion model D_intra.

Run LOCALLY in the `kmis` conda env. Writes one file of plain torch tensors that
the digress env can load without torch_sparse.

For each supernode c_i of a K-MIS coarsening of the public graph:
    target    G_i = G[V_i], the subgraph induced by that supernode's nodes
    condition c_i, a fixed-length vector describing the supernode

The condition features, per the brief:
    n         number of nodes in the supernode
    rho_in    internal density, 2 * m_in / (n (n-1))
    d_c       number of OTHER supernodes connected to this one
    dbar_out  average external edges per node, b_i / n
    h         hop distribution from the centroid, k values
    k         the coarsening scale

COND below selects which are used. The brief says to drop three for v1; the
highlighting did not survive the paste, so this defaults to the three that
describe the induced subgraph itself. Change COND to switch -- everything
downstream reads the width from the file.

NOTE ON 'k': it is constant across a single-k dataset, so it contributes nothing
to learning. It only becomes informative if you later train one model across
several coarsening scales. Harmless to include; just do not expect it to help yet.
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import breadth_first_order

from benchmark import datasets
from kmis import orderings, reduce, sample

# ===========================================================================
# EASILY-CHANGEABLE INPUTS
# ===========================================================================
NAME, GROUP = "luxembourg_osm", "DIMACS10"
K = 7                  # coarsening scale. Check the reported median size.
MIN_N = 3              # discard supernodes smaller than this
SEED = 0
SPLIT = (0.75, 0.15, 0.10)      # train / val / test, per the brief

# available: 'n', 'rho_in', 'd_c', 'dbar_out', 'h', 'k'    ('h' expands to K values)
COND = ['n', 'rho_in', 'k']

OUT = "pairs_%s_k%d.pt"
# ===========================================================================


def load_fine():
    adj = datasets.load_graph(NAME, GROUP, "datasets/")
    n = adj.size(0)
    row, col, _ = adj.coo()
    A = sp.coo_matrix((np.ones(row.numel()), (row.numpy(), col.numpy())), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return adj, A


def hop_histogram(A, centre, members, k):
    """Proportion of non-centroid members at each shortest-path distance 1..k.

    Distances are in the full graph G, as the brief specifies. Uses a BFS
    truncated at k hops rather than a full single-source shortest path, so this
    stays cheap even on a 100k-node graph.
    """
    h = np.zeros(k)
    if len(members) <= 1:
        return h
    member_set = set(members.tolist())
    frontier, seen = {int(centre)}, {int(centre)}
    for r in range(1, k + 1):
        nxt = set()
        for u in frontier:
            nxt.update(A.indices[A.indptr[u]:A.indptr[u + 1]].tolist())
        nxt -= seen
        seen |= nxt
        h[r - 1] = len(nxt & member_set)
        frontier = nxt
        if not frontier:
            break
    return h / (len(members) - 1)


def build_condition(feats, k):
    """Assemble the selected features into one flat float vector."""
    out = []
    for name in COND:
        out.extend(feats['h'].tolist() if name == 'h' else [float(feats[name])])
    return torch.tensor(out, dtype=torch.float)


def cond_names(k):
    names = []
    for name in COND:
        names.extend([f'h{r}' for r in range(1, k + 1)] if name == 'h' else [name])
    return names


def main():
    torch.manual_seed(SEED)
    adj, A = load_fine()
    n = adj.size(0)
    print(f"{NAME}: {n} nodes, {int(A.nnz / 2)} edges, k={K}")

    rank = orderings.Greedy()(adj.sum(dim=1), adj)
    mis = sample.maximal_k_independent_set(adj, K, rank)
    cluster = reduce.cluster_k_mis(mis, adj, K, rank).cpu().numpy()
    c = int(cluster.max()) + 1
    print(f"{c} supernodes\n")

    # cluster id -> its centroid node
    centre_of = np.full(c, -1, dtype=np.int64)
    for node in np.where(mis.cpu().numpy())[0]:
        centre_of[cluster[node]] = node

    order = np.argsort(cluster, kind='stable')
    starts = np.searchsorted(cluster[order], np.arange(c))
    ends = np.searchsorted(cluster[order], np.arange(c), side='right')

    recs, dropped, no_edges = [], 0, 0
    for ci in range(c):
        members = order[starts[ci]:ends[ci]]
        n_i = len(members)
        if n_i < MIN_N:
            dropped += 1
            continue

        sub = A[members][:, members].tocoo()          # induced subgraph, local ids
        m_in = int(sub.nnz / 2)
        b_i = int(A[members].sum() - sub.nnz)          # edges leaving the supernode

        if sub.nnz == 0:
            no_edges += 1
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            edge_index = torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long)

        nbr_clusters = set(cluster[A[members].tocoo().col].tolist()) - {ci}

        feats = {
            'n':        n_i,
            'rho_in':   2.0 * m_in / (n_i * (n_i - 1)) if n_i > 1 else 0.0,
            'd_c':      len(nbr_clusters),
            'dbar_out': b_i / n_i,
            'k':        K,
            'h':        hop_histogram(A, centre_of[ci], members, K) if 'h' in COND
                        else np.zeros(K),
        }

        recs.append({
            'sn': ci,
            'n': n_i,
            'edge_index': edge_index,                          # local ids, symmetric
            'y_raw': build_condition(feats, K),
            'centre': int(centre_of[ci]),
            'nodes': torch.tensor(members, dtype=torch.long),   # for evaluation only
        })

    if not recs:
        raise SystemExit("no supernodes survived -- lower MIN_N or change K")

    rng = np.random.default_rng(SEED)
    rng.shuffle(recs)
    ntr = int(SPLIT[0] * len(recs))
    nva = int(SPLIT[1] * len(recs))
    splits = [recs[:ntr], recs[ntr:ntr + nva], recs[ntr + nva:]]

    # Standardise the condition using TRAIN statistics only, so val/test carry no
    # information back into training. Raw values are kept for interpretability.
    Y = torch.stack([r['y_raw'] for r in splits[0]])
    mean, std = Y.mean(0), Y.std(0).clamp(min=1e-6)
    for split in splits:
        for r in split:
            r['y'] = (r['y_raw'] - mean) / std

    names = cond_names(K)
    sizes = np.array([r['n'] for r in recs])
    blob = {'k': K, 'source': NAME, 'cond_names': names,
            'cond_mean': mean, 'cond_std': std, 'splits': splits}

    path = OUT % (NAME, K)
    torch.save(blob, path)

    print(f"kept {len(recs)} supernodes (dropped {dropped} with n < {MIN_N})")
    print(f"  size n: {sizes.min()}-{sizes.max()}, median {int(np.median(sizes))}")
    print(f"  {no_edges} have no internal edges ({100 * no_edges / len(recs):.1f}%)")
    print(f"  condition ({len(names)} dims): {names}")
    print(f"  raw mean: {[round(v, 3) for v in mean.tolist()]}")
    print(f"  raw std : {[round(v, 3) for v in std.tolist()]}")
    print(f"  split: {len(splits[0])} train / {len(splits[1])} val / {len(splits[2])} test")
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    if np.median(sizes) < 5:
        print("\nWARNING: median supernode is tiny. Lower K.")
    if np.median(sizes) > 60:
        print("\nWARNING: median supernode is large -- dense E tensors. Raise K or cap size.")


if __name__ == "__main__":
    main()
