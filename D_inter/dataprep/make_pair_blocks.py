"""Build (H_ij, c_ij) training samples for the inter-cluster diffusion model D_inter.

Run LOCALLY in the `kmis` conda env. Writes one file of plain torch tensors that
the digress env can load without torch_sparse.

For each PAIR of supernodes (c_i, c_j) of a K-MIS coarsening that has at least
one edge between them:

    target    B_ij, the cross-subgraph adjacency block  |V_i| x |V_j|
    context   X_i, X_j, each node's normalised internal degree d_in(v) / (n_i - 1),
              computed from L_i, L_j by the dataset class (src/datasets/pair_dataset.py)
    condition c_ij = [n_i, n_j, w_ij, r_{i->j}, r_{j->i}]               (13 dims)

CONDITION (November 2026, the paper's Section B). r_{i->j} is described below;
r_{j->i} is the same profile from the other side, the fraction of V_j nodes with
d edges into V_i. n_i, n_j and w_ij are standardised on the training pairs, the r
blocks are proportions and stay raw. rho and k are no longer part of c_ij.

CONDITION V2. r_{i->j} = [r_0, r_1, r_2, r_3, r_{>=4}] is the out-degree profile
of the block: r_d is the fraction of nodes of V_i with exactly d edges into V_j,
and r_{>=4} the fraction with four or more. Fixed width 5 whatever n_i and n_j
are, which is the point -- it describes the SHAPE of the block without scaling
with its size.

MEASURED on luxembourg_osm k=7, all 10902 pairs: the maximum cross out-degree
anywhere is 2, so r_3 and r_{>=4} are identically zero and r_2 is nonzero in
0.88% of pairs. When every out-degree is 0 or 1 the count of degree-1 nodes IS
w_ij, so r = [1 - w_ij/n_i, w_ij/n_i, 0, 0, 0] exactly -- r is a deterministic
function of (n_i, w_ij), both already in V1, for 10806/10902 pairs (99.12%).
Expect V2 to move very little on this dataset. It is worth carrying anyway for
denser domains, where the bins should be re-chosen to match the degree range.

The three are stored as ONE combined graph, exactly the H of the brief:

        H_ij = [ L_i    B_ij  ]      local ids 0 .. n_i-1        are V_i
               [ B_ij^T L_j   ]      local ids n_i .. n_i+n_j-1  are V_j

so an entry of H is a cross entry precisely when its two endpoints lie on
different sides. The dataset class turns that ordering into a 2-class node
feature, and the model derives the cross-block mask from it. Nothing downstream
needs to know how the nodes were numbered here -- only that the side is carried
in X. The records keep the whole H, internal edges included: the dataset class
reduces L_i to the degree feature X_i and gives the model only the cross edges,
while the evaluation scripts and the baselines read the real blocks from here.

ORIENTATION. Which supernode of a pair takes the i slot is randomised. It must
be: k-MIS numbers supernodes by centroid rank, which is degree-ordered, so a
naive i<j convention makes the lower id systematically the LARGER cluster
(measured: mean n_i 16.85 vs n_j 13.91). The model would learn that skew, and it
is an artefact of the numbering, not of the graph.

SPLITTING (October 2026). Train / test only, decided by SUPERNODE and shared with
D_intra: D_intra's make_transfer_datasets.py writes the supernode split to
D_intra/data/splits/<graph>_k<k>_s80.json and this reads it. A pair is a
TRAINING pair only if both its supernodes are training supernodes; every pair
that touches a test supernode is a TEST pair, so no test cross block is ever
trained on and D_inter trains on exactly D_intra's training supernodes. There is
no validation split; the "val" slot holds a sample of training pairs (logging
only).
"""
import os
import pathlib
import sys

import numpy as np
import scipy.sparse as sp
import torch

# ===========================================================================
# EASILY-CHANGEABLE INPUTS
# ===========================================================================
NAME, GROUP = "luxembourg_osm", "DIMACS10"
K = 7                  # coarsening scale. Must match the D_intra run.
MIN_N = 3              # discard supernodes smaller than this
MAX_COMBINED = 128     # discard pairs with n_i + n_j above this -- E is dense,
                       # so cost is quadratic in the combined size
SEED = 0
SPLIT_TAG = "s80"               # the shared 80/20 supernode split (D_intra/data/splits/)
VAL_FROM_TRAIN = 256            # training pairs in the "val" slot, logging only


# where the K-MIS code and the raw graph live
EXTRACTION = "/home/jamie/Projects/Placement/k-mis-pool"

# The name MUST change when the condition or the node features change:
# PairGraphDataset derives its processed_file_names from this, so reusing a name
# loads the stale cache in data/pair_lux/processed/ and trains on the old
# condition with no error anywhere. "_deg": X_i = internal degree, c_ij of the
# paper's Section B. The 6-wide V1 files (no suffix) are the previous runs'.
VERSION = "_deg"
OUT = "pairs_inter_%s_k%d_" + SPLIT_TAG + VERSION + ".pt"
# ===========================================================================

# Resolve output BEFORE the chdir below. __file__ may be relative, in which case
# resolving it afterwards silently lands the file in the wrong project.
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "pair_lux" / "raw"
SPLIT_DIR = HERE.parents[1] / "D_intra" / "data" / "splits"

os.chdir(EXTRACTION)
sys.path.insert(0, EXTRACTION)   # sys.path[0] is THIS script's dir, not the cwd

from benchmark import datasets                    # noqa: E402
from kmis import orderings, reduce, sample        # noqa: E402

R_BINS = 5             # r_0, r_1, r_2, r_3, r_{>=4}
R_NAMES = ['0', '1', '2', '3', '4+']
COND_NAMES = (['n_i', 'n_j', 'w_ij'] + [f'rij{b}' for b in R_NAMES]
              + [f'rji{b}' for b in R_NAMES])

# Which condition columns get the train-mean/std treatment. The r blocks do
# NOT: it is already a set of proportions in [0,1], and on Luxembourg k=7 the
# train std of r_3 and r_{>=4} is exactly 0. Standardising would divide by the
# clamped 1e-6, so a Minnesota pair with a single degree-3 node would arrive as
# a z-score of ~1e5 -- the same trap sample_private_blocks.py already guards
# against for k, and just as silent.
STANDARDISE = np.array([True] * 3 + [False] * 2 * R_BINS)


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


def density(n, m):
    """Internal density 2m / (n(n-1)); 0 for a single node."""
    return 2.0 * m / (n * (n - 1)) if n > 1 else 0.0


def out_degree_hist(rows, n_i, n_bins=R_BINS):
    """r_{i->j}: the fraction of V_i nodes with cross out-degree 0, 1, .. >=n_bins-1.

    `rows` is the row index of every nonzero of B_ij, i.e. B.row of the COO --
    each cross edge contributes its V_i endpoint once. Nodes with no cross edge
    are counted into r_0; they are part of the distribution, not absent from it.
    Sums to 1 by construction, so one of the bins is redundant.
    """
    d = np.bincount(rows, minlength=n_i)[:n_i]
    return np.bincount(np.clip(d, 0, n_bins - 1), minlength=n_bins) / n_i


def split_pairs(recs, graph, k, split_tag=SPLIT_TAG):
    """(train, test) pairs from D_intra's supernode split for this graph.

    train: both supernodes are training supernodes. test: at least one is a test
    supernode. Supernode sizes are checked against the split file, which catches
    a coarsening that numbered supernodes differently from D_intra's.
    """
    import json
    path = SPLIT_DIR / f"{graph}_k{k}_{split_tag}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run D_intra/extraction/"
                         f"make_transfer_datasets.py first; it writes the split")
    sp_ = json.load(open(path))
    test, n_of = set(sp_["test"]), {int(a): b for a, b in sp_["n"].items()}
    bad = [r for r in recs if n_of.get(int(r['sn_i'])) != int(r['n_i'])
           or n_of.get(int(r['sn_j'])) != int(r['n_j'])]
    if bad:
        raise SystemExit(f"{len(bad)} pairs disagree with {path.name} on supernode "
                         f"sizes -- the two coarsenings do not match")
    train = [r for r in recs if int(r['sn_i']) not in test and int(r['sn_j']) not in test]
    held = [r for r in recs if int(r['sn_i']) in test or int(r['sn_j']) in test]
    return train, held


def standardise(train, splits):
    """Train-only mean / std for the STANDARDISE columns; writes r['y'] on `splits`.

    The r columns get mean 0, std 1 (left raw, see STANDARDISE). So does a column
    with (near-)zero training variance: it keeps its raw scale rather than being
    divided by an epsilon, the rule of D_intra/extraction/kmis_pairs.py:standardise().
    Returns (mean, std, constant), stored in the blob so that the samplers
    standardise a released condition exactly the same way.
    """
    Y = torch.stack([r['y_raw'] for r in train])
    mean, std = Y.mean(0), Y.std(0)
    raw = torch.from_numpy(~STANDARDISE)
    constant = std < 1e-6
    mean = torch.where(raw, torch.zeros_like(mean), mean)
    std = torch.where(raw | constant, torch.ones_like(std), std)
    for split in splits:
        for r in split:
            r['y'] = (r['y_raw'] - mean) / std
    return mean, std, constant


def val_from_train(train, n=VAL_FROM_TRAIN, seed=SEED):
    """A few TRAINING pairs for the val slot (logging only; nothing is held out)."""
    idx = np.random.default_rng(seed + 1).permutation(len(train))[:n]
    return [train[i] for i in idx]


def extract_pair_records(adj, A, k, rng, min_n=MIN_N, max_combined=MAX_COMBINED,
                         verbose=True):
    """Coarsen at locality k and return (records, n dropped for size), one per pair.

    Every connected pair of supernodes with n >= min_n, oriented at random with
    `rng` (one draw per pair, in coarse-COO order) and dropped when n_i + n_j >
    max_combined. Shared with make_transfer_pairs.py, which runs it on Oldenburg;
    main() passes the same rng it later shuffles the split with, so the Luxembourg
    output is unchanged by the refactor.
    """
    n = adj.size(0)
    rank = orderings.Greedy()(adj.sum(dim=1), adj)
    mis = sample.maximal_k_independent_set(adj, k, rank)
    cluster = reduce.cluster_k_mis(mis, adj, k, rank).cpu().numpy()
    c = int(cluster.max()) + 1
    if verbose:
        print(f"{c} supernodes")

    # members of each supernode, as one contiguous sort
    order = np.argsort(cluster, kind='stable')
    starts = np.searchsorted(cluster[order], np.arange(c))
    ends = np.searchsorted(cluster[order], np.arange(c), side='right')

    # Keep only supernodes big enough to be worth modelling, and cache each one's
    # rows of A once. Every pair below is a column-slice of these, which is far
    # cheaper than re-slicing the 114k-square matrix per pair.
    members, rows, n_of, rho_of = {}, {}, {}, {}
    for ci in range(c):
        mem = order[starts[ci]:ends[ci]]
        if len(mem) < min_n:
            continue
        members[ci] = mem
        rows[ci] = A[mem]                                  # (n_i, n) csr
        n_of[ci] = len(mem)
        rho_of[ci] = density(len(mem), int(rows[ci][:, mem].nnz / 2))
    if verbose:
        print(f"{len(members)} supernodes with n >= {min_n}")

    # Coarse adjacency: P^T A P, entry (i,j) = number of edges between i and j.
    # One sparse product replaces a scan over all O(c^2) candidate pairs.
    P = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, c))
    C = (P.T @ A @ P).tocoo()

    recs, too_big = [], 0
    for ci, cj, w in zip(C.row, C.col, C.data):
        ci, cj, w = int(ci), int(cj), int(w)
        if ci >= cj or w == 0:                 # each unordered pair once, self-loops out
            continue
        if ci not in members or cj not in members:
            continue

        # Randomise which supernode takes the i slot. See ORIENTATION above.
        if rng.random() < 0.5:
            ci, cj = cj, ci

        mem_i, mem_j = members[ci], members[cj]
        n_i, n_j = n_of[ci], n_of[cj]
        if n_i + n_j > max_combined:
            too_big += 1
            continue

        L_i = rows[ci][:, mem_i].tocoo()
        L_j = rows[cj][:, mem_j].tocoo()
        B = rows[ci][:, mem_j].tocoo()         # (n_i, n_j), the target block

        # Combined graph H, local ids, both directions of every edge.
        src = np.concatenate([L_i.row, L_j.row + n_i, B.row,        B.col + n_i])
        dst = np.concatenate([L_i.col, L_j.col + n_i, B.col + n_i,  B.row])
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        r = out_degree_hist(B.row, n_i)            # r_{i->j}
        r_ji = out_degree_hist(B.col, n_j)         # r_{j->i}: B^T's rows are B's columns
        y_raw = torch.tensor([n_i, n_j, B.nnz, *r, *r_ji], dtype=torch.float)

        recs.append({
            'sn_i': ci, 'sn_j': cj,
            'n_i': n_i, 'n_j': n_j,
            'edge_index': edge_index,
            'w': int(B.nnz),
            'r': torch.tensor(r, dtype=torch.float),          # r_{i->j}; SC-RG reads 'r'
            'r_ji': torch.tensor(r_ji, dtype=torch.float),
            'rho_i': rho_of[ci], 'rho_j': rho_of[cj],
            'y_raw': y_raw,
        })

    return recs, too_big


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    adj, A = load_fine()
    n = adj.size(0)
    print(f"{NAME}: {n} nodes, {int(A.nnz / 2)} edges, k={K}")

    recs, too_big = extract_pair_records(adj, A, K, rng)

    if not recs:
        raise SystemExit("no connected pairs -- lower MIN_N or change K")

    # ---- r diagnostics (the r_{i->j} columns) -----------------------------------
    R = np.stack([r['y_raw'][3:3 + R_BINS].numpy() for r in recs])

    # ---- the shared supernode split ------------------------------------------
    train, test = split_pairs(recs, "luxembourg", K)
    splits = [train, val_from_train(train), test]
    discarded = len(recs) - len(train) - len(test)
    mean, std, constant = standardise(train, splits)

    blob = {'k': K, 'source': NAME, 'graph': 'luxembourg', 'cond_names': COND_NAMES,
            'cond_mean': mean, 'cond_std': std, 'cond_constant': constant,
            'split_by': f'supernode ({SPLIT_TAG}, shared with D_intra)',
            'splits': splits}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / (OUT % (NAME, K))
    torch.save(blob, path)

    # ---- report -----------------------------------------------------------
    ni = np.array([r['n_i'] for r in recs])
    nj = np.array([r['n_j'] for r in recs])
    comb = ni + nj
    w = np.array([r['w'] for r in recs])
    dens = w / (ni * nj)

    print(f"\nkept {len(recs)} connected pairs"
          + (f" (dropped {too_big} with n_i+n_j > {MAX_COMBINED})" if too_big else ""))
    print(f"  n_i  mean {ni.mean():.2f}   n_j mean {nj.mean():.2f}   "
          f"(these must match -- if not, orientation is biased)")
    print(f"  combined size: {comb.min()}-{comb.max()}, median {int(np.median(comb))}")
    print(f"  w_ij: {w.min()}-{w.max()}, median {int(np.median(w))}")
    print(f"  cross-block density w/(n_i n_j): mean {dens.mean():.5f}, "
          f"median {np.median(dens):.5f}")
    print(f"  condition ({len(COND_NAMES)} dims): {COND_NAMES}")
    print(f"  train mean: {[round(v, 3) for v in mean.tolist()]}")
    print(f"  train std : {[round(v, 3) for v in std.tolist()]}")

    # r diagnostics. If the top bins are all-zero the condition is carrying less
    # than its 5 dimensions suggest, and if r is recoverable from (n_i, w_ij) it
    # is carrying nothing NEW at all -- both are true on Luxembourg k=7. Print it
    # rather than leave it to be rediscovered.
    print(f"\n  r_i->j mean {R.mean(0).round(4).tolist()}")
    print(f"  r_i->j std  {R.std(0).round(4).tolist()}")
    dead = [COND_NAMES[3 + b] for b in range(R_BINS) if R[:, b].std() == 0]
    if dead:
        print(f"  NOTE: {dead} are constant across every pair -- those bins are "
              f"dead weight at this k. They are exempt from standardisation, so "
              f"this is safe, but re-bin r for a denser graph.")
    implied = np.stack([[1 - r['w'] / r['n_i'], r['w'] / r['n_i']]
                        + [0.0] * (R_BINS - 2) for r in recs])
    # 1e-6, not an exact compare: R comes back through y_raw as float32 while
    # `implied` is float64, so an exact test measures rounding, not agreement.
    same = int((np.abs(R - implied).max(axis=1) < 1e-6).sum())
    print(f"  r is exactly implied by (n_i, w_ij) for {same}/{len(recs)} pairs "
          f"({same / len(recs):.2%}) -- V2 only adds information on the rest.")
    print(f"  split by supernode ({SPLIT_TAG}): {len(train)} train pairs / {len(test)} test "
          f"pairs (val slot: {len(splits[1])} training pairs)" + (f", {discarded} unassigned" if discarded else ""))
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if dens.mean() < 0.005:
        print("\nNOTE: the cross block is very sparse. Predicting 'no edge' everywhere "
              "will already score a low CE -- read W_out-degree, not CE alone.")
    if comb.max() > 96:
        print(f"\nWARNING: largest combined graph is {comb.max()} nodes. E is dense "
              f"({comb.max()**2} entries per sample); lower MAX_COMBINED if memory bites.")


if __name__ == "__main__":
    main()
