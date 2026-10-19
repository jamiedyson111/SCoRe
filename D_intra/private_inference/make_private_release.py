"""STAGE 0 of private inference: coarsen the private graph and emit the release.

Run LOCALLY in the `kmis` conda env (needs torch_sparse):

    python make_private_release.py --graph oldenburg --k 7

Everything downstream sees ONLY this file. It stands in for what a data holder
would publish: the coarse graph and per-supernode summary statistics, never the
fine structure inside a supernode or between two supernodes.

    supernodes   {sn, n, rho_in, centre, h, y_raw_intra, nodes}
    pairs        {sn_i, sn_j, w}          -- one per CONNECTED supernode pair

`h` is the k-bin hop histogram D_intra is conditioned on: the fraction of a
supernode's non-centre members at shortest-path distance 1..k from its centre.
It is part of the release because the model requires it, which makes the release
7 numbers per supernode larger than the brief's [n, rho_in] and is the natural
place to look first if a DP budget is ever attached to this.

`nodes` is the list of fine node ids in the supernode. It is not part of the
release in the privacy sense -- it is the alignment map that lets stage 3 place
generated blocks back into a graph with the same node set as G^priv, so that
L^syn can be compared to it node-for-node. Nothing generative reads it.

NO NOISE IS ADDED. n, rho_in and w_ij are exact. The hats in the brief denote
"released", and privacy accounting is deliberately a separate piece of work; if
a DP mechanism is added later it belongs here, between the exact statistic and
the file, and nothing downstream needs to change.

EVERY SUPERNODE IS KEPT, including those below the MIN_N=3 that the training
sets dropped. A supernode that was too small to be worth training on still has
nodes in G^priv, and dropping it here would silently shrink L^syn. Stage 1 fills
those blocks without the model (see `modelled`).

THE COARSENING MUST MATCH the one the models were trained under: same Greedy
degree ordering, same maximal k-independent set, same cluster assignment. This
file calls the same three functions in the same order as
kmis_pairs.extract_pairs, rather than reimplementing them.
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import torch

HERE = pathlib.Path(__file__).resolve().parent
EXTRACTION = HERE.parent / "extraction"

# kmis_pairs.load_graph resolves the .mat files relative to the cwd, and the
# kmis / benchmark packages are only importable from inside extraction/.
os.chdir(EXTRACTION)
sys.path.insert(0, str(EXTRACTION))

from kmis_pairs import (GRAPHS, build_condition, cond_names,   # noqa: E402
                        hop_histogram, load_graph)
from kmis import orderings, reduce, sample         # noqa: E402

# The condition D_intra was trained on. MUST match COND in
# extraction/make_transfer_datasets.py -- "h" expands to h1..hk, so this is a
# 10-vector at k=7, not a 3-vector. The release stores the assembled y_raw per
# supernode rather than the ingredients, so stage 1 never re-derives it and a
# future change to COND cannot silently desynchronise the two.
INTRA_COND = ["n", "rho_in", "k", "h"]
# The condition D_inter was trained on, in order. Must match COND_NAMES in
# D_inter/dataprep/make_pair_blocks.py.
INTER_COND_NAMES = ['n_i', 'rho_i', 'n_j', 'rho_j', 'w_ij', 'k']

MIN_N_TRAINED = 3        # what the training sets dropped; recorded, not applied
MAX_COMBINED_TRAINED = 128


def density(n, m):
    """Internal density 2m / (n(n-1)); 0 for a single node."""
    return 2.0 * m / (n * (n - 1)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default='oldenburg', choices=sorted(GRAPHS),
                    help='the graph to treat as private')
    ap.add_argument('--k-hist', type=int, default=7,
                    help='hop-histogram WIDTH, separate from --k. The models were '
                         'trained with 7 columns, so this stays 7 even when the '
                         'private graph is recoarsened at k=4: rings 5-7 of a k=4 '
                         'cluster are provably empty, sum(h) is still 1, and the '
                         'condition stays 10-dimensional so the checkpoints load '
                         'unchanged. This mirrors extract_pairs(..., k_hist=...), '
                         'which is what built the k7k4 training datasets.')
    ap.add_argument('--k', type=int, default=7,
                    help='coarsening scale; must match the trained models')
    ap.add_argument('--seed', type=int, default=0,
                    help='orientation randomisation only; the coarsening is deterministic')
    ap.add_argument('--out-dir', default=str(HERE.parent.parent / 'results' / 'private_inference'))
    args = ap.parse_args()
    # A histogram narrower than the coarsening scale would truncate rings that
    # can legitimately be occupied; the reverse is safe and is the whole point.
    if args.k_hist < args.k:
        raise SystemExit(f"--k-hist {args.k_hist} < --k {args.k}: the histogram "
                         f"would drop rings {args.k_hist + 1}..{args.k}, which a "
                         f"k={args.k} cluster can occupy.")

    rng = np.random.default_rng(args.seed)

    adj, A = load_graph(args.graph)
    n_fine = adj.size(0)
    m_fine = int(A.nnz / 2)
    print(f"{args.graph}: {n_fine} nodes, {m_fine} edges, k={args.k}")

    # Identical to kmis_pairs.extract_pairs -- same functions, same order.
    rank = orderings.Greedy()(adj.sum(dim=1), adj)
    mis = sample.maximal_k_independent_set(adj, args.k, rank)
    cluster = reduce.cluster_k_mis(mis, adj, args.k, rank).cpu().numpy()
    c = int(cluster.max()) + 1
    print(f"{c} supernodes")

    # The k-MIS node each supernode was grown from. hop_histogram measures
    # distances from it, so it has to be recovered the same way extract_pairs
    # does rather than picked by any other rule.
    centre_of = np.full(c, -1, dtype=np.int64)
    for node in np.where(mis.cpu().numpy())[0]:
        centre_of[cluster[node]] = node

    # Members of each supernode, as one contiguous sort rather than c scans.
    order = np.argsort(cluster, kind='stable')
    starts = np.searchsorted(cluster[order], np.arange(c))
    ends = np.searchsorted(cluster[order], np.arange(c), side='right')

    supernodes, rows, true_clusters = {}, {}, {}
    for ci in range(c):
        mem = order[starts[ci]:ends[ci]]
        rows[ci] = A[mem]                                   # (n_i, n_fine) csr
        n_i = len(mem)
        sub = sp.coo_matrix(rows[ci][:, mem])
        m_in = int(sub.nnz / 2)

        # Built with the same helpers extract_pairs uses, so the released
        # condition is bit-for-bit the one the training records carry. Note
        # hop_histogram walks the FULL graph A, not the induced subgraph: a
        # member two hops from the centre may be reached through a node in
        # another supernode. h is therefore a statistic about the supernode's
        # embedding in G, not purely about its interior.
        feats = {
            'n': n_i,
            'rho_in': density(n_i, m_in),
            'k': args.k,
            'h': hop_histogram(A, centre_of[ci], mem, args.k_hist),
        }
        supernodes[ci] = {
            'sn': ci,
            'n': n_i,
            'rho_in': feats['rho_in'],
            'centre': int(centre_of[ci]),
            'h': torch.tensor(feats['h'], dtype=torch.float),
            'y_raw_intra': build_condition(feats, INTRA_COND, args.k_hist),
            'nodes': torch.tensor(mem, dtype=torch.long),
            'modelled': n_i >= MIN_N_TRAINED,
            'm_in_true': m_in,          # ground truth, for stage 3 scoring only
        }
        # The REAL internal block, in exactly the format stage 1 emits. Written
        # to a separate file so that stage 2 can be validated against true
        # context without importing anything from the extraction stack -- that
        # needs h5py and torch_sparse, which the GPU env does not have.
        true_clusters[ci] = {
            'n': n_i, 'modelled': True,
            'edge_index': (torch.zeros(2, 0, dtype=torch.long) if sub.nnz == 0 else
                           torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long)),
        }

    # Coarse adjacency C = P^T A P; entry (i,j) is the number of fine edges
    # between supernodes i and j. One sparse product instead of scanning all
    # O(c^2) candidate pairs.
    P = sp.csr_matrix((np.ones(n_fine), (np.arange(n_fine), cluster)), shape=(n_fine, c))
    C = (P.T @ A @ P).tocoo()

    pairs = []
    for ci, cj, w in zip(C.row, C.col, C.data):
        ci, cj, w = int(ci), int(cj), int(w)
        if ci >= cj or w == 0:               # each unordered pair once, no self-loops
            continue

        # Randomise which supernode takes the i slot, exactly as
        # make_pair_blocks.py does when building the training set. k-MIS numbers
        # supernodes by degree rank, so a deterministic i<j would put the larger
        # cluster in slot i systematically -- the very skew training randomised
        # away. Feeding that skew in at inference would be a train/test mismatch.
        if rng.random() < 0.5:
            ci, cj = cj, ci

        pairs.append({
            'sn_i': ci, 'sn_j': cj,
            'n_i': supernodes[ci]['n'], 'n_j': supernodes[cj]['n'],
            'w': w,
        })

    blob = {
        'graph': args.graph, 'k': args.k, 'seed': args.seed,
        'n_fine': n_fine, 'm_fine': m_fine,
        'n_supernodes': c,
        'supernodes': [supernodes[i] for i in range(c)],
        'pairs': pairs,
        'intra_cond_names': cond_names(INTRA_COND, args.k_hist),
        'inter_cond_names': INTER_COND_NAMES,
        'min_n_trained': MIN_N_TRAINED,
        'max_combined_trained': MAX_COMBINED_TRAINED,
        'cluster': torch.tensor(cluster, dtype=torch.long),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"release_{args.graph}_k{args.k}.pt")
    torch.save(blob, path)

    true_path = os.path.join(args.out_dir, f"clusters_{args.graph}_k{args.k}_true.pt")
    torch.save({'graph': args.graph, 'k': args.k,
                'checkpoint': None, 'dataset': 'TRUE INTERNAL BLOCKS -- validation only',
                'clusters': true_clusters}, true_path)

    # ---- invariants --------------------------------------------------------
    # Cheap, and they catch a wrong coarsening immediately rather than three
    # stages later when the only symptom is a bad L^syn.
    sizes = np.array([s['n'] for s in blob['supernodes']])
    assert int(sizes.sum()) == n_fine, \
        f"supernode sizes sum to {int(sizes.sum())}, expected {n_fine} fine nodes"
    m_internal = sum(s['m_in_true'] for s in blob['supernodes'])
    m_cross = sum(p['w'] for p in pairs)
    assert m_internal + m_cross == m_fine, \
        f"{m_internal} internal + {m_cross} cross != {m_fine} fine edges"

    n_small = int((sizes < MIN_N_TRAINED).sum())
    comb = np.array([p['n_i'] + p['n_j'] for p in pairs]) if pairs else np.array([0])
    n_big = int((comb > MAX_COMBINED_TRAINED).sum())

    print(f"\n  supernode sizes: {sizes.min()}-{sizes.max()}, "
          f"median {int(np.median(sizes))}, mean {sizes.mean():.2f}")
    print(f"  {n_small} supernodes below the MIN_N={MIN_N_TRAINED} used in training "
          f"-> stage 1 fills these without the model")
    print(f"  {len(pairs)} connected pairs, w_ij {min(p['w'] for p in pairs)}-"
          f"{max(p['w'] for p in pairs)}, median {int(np.median([p['w'] for p in pairs]))}")
    print(f"  combined size: {comb.min()}-{comb.max()}, median {int(np.median(comb))}")
    if n_big:
        print(f"  {n_big} pairs exceed the MAX_COMBINED={MAX_COMBINED_TRAINED} seen in "
              f"training -- stage 2 will warn; it is a memory limit, not a model one")
    print(f"  edges: {m_internal} internal + {m_cross} cross = {m_fine}  [checks out]")
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    print(f"wrote {true_path} ({os.path.getsize(true_path) / 1e6:.1f} MB)"
          f"  -- real internal blocks, for validating stage 2 without stage 1")


if __name__ == "__main__":
    main()
