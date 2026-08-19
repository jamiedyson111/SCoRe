"""Extraction of (G_i, c_i) supernode pairs from a graph. Shared by the dataset
builders; contains no CLI so it can be imported safely.

Run in the `kmis` conda env (needs torch_sparse).
"""
import numpy as np
import scipy.sparse as sp
import torch

from benchmark import datasets
from kmis import orderings, reduce, sample

GRAPHS = {
    "luxembourg": ("luxembourg_osm", "DIMACS10"),
    "minnesota":  ("minnesota", "Gleich"),
    "facebook":   ("ego-Facebook", "SNAP"),
}


def load_graph(key, root="datasets/"):
    name, group = GRAPHS[key]
    adj = datasets.load_graph(name, group, root)
    n = adj.size(0)
    row, col, _ = adj.coo()
    A = sp.coo_matrix((np.ones(row.numel()), (row.numpy(), col.numpy())), shape=(n, n)).tocsr()
    A = A + A.T
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return adj, A


def hop_histogram(A, centre, members, k):
    """Proportion of non-centroid members at each shortest-path distance 1..k in G.

    `k` here is the WIDTH of the histogram, which is not necessarily the k the
    graph was coarsened at. A cluster built at k_cluster < k simply has zeros in
    rings k_cluster+1..k, and that is a fact about the cluster rather than
    padding: nearest-centre assignment puts every member within k_cluster hops of
    its centre, so those rings are provably empty. `sum(h) == 1` either way, which
    is what keeps a k=4 cluster describable on a k=7 model's input width.
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


def build_condition(feats, cond, k):
    out = []
    for name in cond:
        out.extend(feats["h"].tolist() if name == "h" else [float(feats[name])])
    return torch.tensor(out, dtype=torch.float)


def cond_names(cond, k):
    """Condition column names. `k` is the hop-histogram WIDTH -- see hop_histogram."""
    names = []
    for name in cond:
        names.extend([f"h{r}" for r in range(1, k + 1)] if name == "h" else [name])
    return names


def extract_pairs(graph_key, k, cond=("n", "rho_in", "k"), min_n=3, root="datasets/",
                  verbose=True, k_hist=None):
    """Coarsen `graph_key` at locality k and return one record per supernode.

    Records carry y_raw (unstandardised). Standardisation is deliberately left to
    the caller: in a transfer setting the TRAINING graph's statistics must be
    applied to the test graph, so it cannot be done per-graph here.

    `k_hist` is the hop-histogram width, defaulting to `k`. It is separate from
    `k` so that graphs coarsened at DIFFERENT localities can share one condition
    width, which is what lets a model trained on a k=7 graph be evaluated on a
    k=4 one. The `k` column itself still reports this graph's true locality.
    """
    k_hist = k if k_hist is None else k_hist
    if k_hist < k:
        raise ValueError(
            f"k_hist={k_hist} < k={k}: the histogram would truncate rings that can "
            f"legitimately be occupied at locality {k}, so sum(h) would stop being 1.")
    adj, A = load_graph(graph_key, root)
    deg = adj.sum(dim=1)

    rank = orderings.Greedy()(deg, adj)                    # descending degree
    mis = sample.maximal_k_independent_set(adj, k, rank)
    cluster = reduce.cluster_k_mis(mis, adj, k, rank).cpu().numpy()
    c = int(cluster.max()) + 1

    centre_of = np.full(c, -1, dtype=np.int64)
    for node in np.where(mis.cpu().numpy())[0]:
        centre_of[cluster[node]] = node

    order = np.argsort(cluster, kind="stable")
    starts = np.searchsorted(cluster[order], np.arange(c))
    ends = np.searchsorted(cluster[order], np.arange(c), side="right")

    recs, dropped = [], 0
    for ci in range(c):
        members = order[starts[ci]:ends[ci]]
        n_i = len(members)
        if n_i < min_n:
            dropped += 1
            continue

        sub = A[members][:, members].tocoo()
        m_in = int(sub.nnz / 2)
        b_i = int(A[members].sum() - sub.nnz)
        edge_index = (torch.zeros(2, 0, dtype=torch.long) if sub.nnz == 0
                      else torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long))
        nbrs = set(cluster[A[members].tocoo().col].tolist()) - {ci}

        feats = {
            "n": n_i,
            "rho_in": 2.0 * m_in / (n_i * (n_i - 1)) if n_i > 1 else 0.0,
            "d_c": len(nbrs),
            "dbar_out": b_i / n_i,
            # The graph's TRUE locality, not the histogram width: a k=4 supernode
            # is reported as k=4 even when described on a 7-wide condition.
            "k": k,
            "h": (hop_histogram(A, centre_of[ci], members, k_hist) if "h" in cond
                  else np.zeros(k_hist)),
        }

        recs.append({
            "sn": ci,
            "graph": graph_key,
            "n": n_i,
            "edge_index": edge_index,
            "y_raw": build_condition(feats, cond, k),
            "centre": int(centre_of[ci]),
            "nodes": torch.tensor(members, dtype=torch.long),
        })

    if verbose:
        sizes = np.array([r["n"] for r in recs]) if recs else np.array([0])
        width = f"k={k}" if k_hist == k else f"k={k} (h width {k_hist})"
        print(f"  {graph_key:<12} {width}  {c} supernodes -> {len(recs)} kept "
              f"(dropped {dropped} with n<{min_n}), size {sizes.min()}-{sizes.max()} "
              f"median {int(np.median(sizes))}")
    return recs


def standardise(train_recs, all_splits):
    """Standardise every split using TRAIN statistics only.

    Features with (near-)zero training variance are left on their raw scale
    rather than divided by a tiny epsilon. Dividing by ~1e-6 would map any unseen
    value -- e.g. k=4 for a model trained only on k=7 -- to a magnitude of ~1e6
    and destroy the forward pass. This matters specifically for transfer, where
    the test graph may be coarsened at a different k.
    """
    Y = torch.stack([r["y_raw"] for r in train_recs])
    mean = Y.mean(0)
    std = Y.std(0)
    constant = std < 1e-6
    std = torch.where(constant, torch.ones_like(std), std)
    for split in all_splits:
        for r in split:
            r["y"] = (r["y_raw"] - mean) / std
    return mean, std, constant


def split_records(recs, fracs=(0.75, 0.15, 0.10), seed=0, shuffle=True):
    recs = list(recs)
    if shuffle:
        np.random.default_rng(seed).shuffle(recs)
    ntr = int(fracs[0] * len(recs))
    nva = int(fracs[1] * len(recs))
    return [recs[:ntr], recs[ntr:ntr + nva], recs[ntr + nva:]]
