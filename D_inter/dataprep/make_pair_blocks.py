"""Build (H_ij, c_ij) training samples for the inter-cluster diffusion model D_inter.

Run LOCALLY in the `kmis` conda env. Writes one file of plain torch tensors that
the digress env can load without torch_sparse.

For each PAIR of supernodes (c_i, c_j) of a K-MIS coarsening that has at least
one edge between them:

    target    B_ij, the cross-subgraph adjacency block  |V_i| x |V_j|
    context   L_i, L_j, the two internal adjacency matrices (given, never noised)
    condition c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k, r_{i->j}]        (V2)

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
in X.

ORIENTATION. Which supernode of a pair takes the i slot is randomised. It must
be: k-MIS numbers supernodes by centroid rank, which is degree-ordered, so a
naive i<j convention makes the lower id systematically the LARGER cluster
(measured: mean n_i 16.85 vs n_j 13.91). The model would learn that skew, and it
is an artefact of the numbering, not of the graph.

SPLITTING. By pair, not by supernode -- see SPLIT_BY below.
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
SPLIT = (0.75, 0.15, 0.10)      # train / val / test

# 'pair'      -- split the list of PAIRS. Recommended.
# 'supernode' -- disjoint supernodes between splits; strictly cleaner, but a
#                split holding fraction p of supernodes keeps only ~p^2 of
#                pairs, so most of the data is thrown away (measured: 4470 of
#                10902 discarded). The fractions below are square-rooted to
#                compensate. The model never sees supernode identity and is
#                permutation-equivariant, so there is no index along which it
#                could memorise across a pair-level split.
SPLIT_BY = "pair"

# where the K-MIS code and the raw graph live
EXTRACTION = "/home/jamie/Projects/Placement/k-mis-pool"

# _v2 for the V2 condition. The name MUST change when the condition changes:
# PairGraphDataset derives its processed_file_names from this, so reusing the
# name loads the stale 6-wide cache in data/pair_lux/processed/ and trains a V2
# run on a V1 condition with no error anywhere.
OUT = "pairs_inter_%s_k%d_v2.pt"
# ===========================================================================

# Resolve output BEFORE the chdir below. __file__ may be relative, in which case
# resolving it afterwards silently lands the file in the wrong project.
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "pair_lux" / "raw"

os.chdir(EXTRACTION)
sys.path.insert(0, EXTRACTION)   # sys.path[0] is THIS script's dir, not the cwd

from benchmark import datasets                    # noqa: E402
from kmis import orderings, reduce, sample        # noqa: E402

R_BINS = 5             # r_0, r_1, r_2, r_3, r_{>=4}
COND_NAMES = ['n_i', 'rho_i', 'n_j', 'rho_j', 'w_ij', 'k',
              'r0', 'r1', 'r2', 'r3', 'r4+']

# Which condition columns get the train-mean/std treatment. The r block does
# NOT: it is already a set of proportions in [0,1], and on Luxembourg k=7 the
# train std of r_3 and r_{>=4} is exactly 0. Standardising would divide by the
# clamped 1e-6, so a Minnesota pair with a single degree-3 node would arrive as
# a z-score of ~1e5 -- the same trap sample_private_blocks.py already guards
# against for k, and just as silent.
STANDARDISE = np.array([True] * 6 + [False] * R_BINS)


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


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    adj, A = load_fine()
    n = adj.size(0)
    print(f"{NAME}: {n} nodes, {int(A.nnz / 2)} edges, k={K}")

    rank = orderings.Greedy()(adj.sum(dim=1), adj)
    mis = sample.maximal_k_independent_set(adj, K, rank)
    cluster = reduce.cluster_k_mis(mis, adj, K, rank).cpu().numpy()
    c = int(cluster.max()) + 1
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
        if len(mem) < MIN_N:
            continue
        members[ci] = mem
        rows[ci] = A[mem]                                  # (n_i, n) csr
        n_of[ci] = len(mem)
        rho_of[ci] = density(len(mem), int(rows[ci][:, mem].nnz / 2))
    print(f"{len(members)} supernodes with n >= {MIN_N}")

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
        if n_i + n_j > MAX_COMBINED:
            too_big += 1
            continue

        L_i = rows[ci][:, mem_i].tocoo()
        L_j = rows[cj][:, mem_j].tocoo()
        B = rows[ci][:, mem_j].tocoo()         # (n_i, n_j), the target block

        # Combined graph H, local ids, both directions of every edge.
        src = np.concatenate([L_i.row, L_j.row + n_i, B.row,        B.col + n_i])
        dst = np.concatenate([L_i.col, L_j.col + n_i, B.col + n_i,  B.row])
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        r = out_degree_hist(B.row, n_i)
        y_raw = torch.tensor([n_i, rho_of[ci], n_j, rho_of[cj], B.nnz, K, *r],
                             dtype=torch.float)

        recs.append({
            'sn_i': ci, 'sn_j': cj,
            'n_i': n_i, 'n_j': n_j,
            'edge_index': edge_index,
            'w': int(B.nnz),
            'r': torch.tensor(r, dtype=torch.float),
            'y_raw': y_raw,
        })

    if not recs:
        raise SystemExit("no connected pairs -- lower MIN_N or change K")

    # ---- splits -----------------------------------------------------------
    if SPLIT_BY == "pair":
        rng.shuffle(recs)
        ntr = int(SPLIT[0] * len(recs))
        nva = int(SPLIT[1] * len(recs))
        splits = [recs[:ntr], recs[ntr:ntr + nva], recs[ntr + nva:]]
        discarded = 0
    elif SPLIT_BY == "supernode":
        # A pair survives only if BOTH its supernodes fall in the same split, so
        # a split of size p keeps ~p^2 of the pairs. Square-root the fractions so
        # the resulting PAIR split is roughly the one asked for.
        ids = np.array(sorted(members))
        rng.shuffle(ids)
        f = np.sqrt(np.array(SPLIT))
        f = f / f.sum()
        a, b = int(f[0] * len(ids)), int((f[0] + f[1]) * len(ids))
        side = {}
        for s, group in enumerate((ids[:a], ids[a:b], ids[b:])):
            side.update({int(g): s for g in group})
        splits = [[], [], []]
        discarded = 0
        for r in recs:
            if side[r['sn_i']] == side[r['sn_j']]:
                splits[side[r['sn_i']]].append(r)
            else:
                discarded += 1
    else:
        raise SystemExit(f"SPLIT_BY must be 'pair' or 'supernode', got {SPLIT_BY!r}")

    if min(len(s) for s in splits) == 0:
        raise SystemExit("a split came out empty -- too few pairs for this SPLIT_BY")

    # Standardise on TRAIN statistics only, so val/test leak nothing back.
    # The r columns are exempted by neutralising their mean/std rather than by
    # slicing them out, so cond_mean/cond_std keep the full condition width and
    # every downstream consumer -- sample_private_blocks.py above all -- can go
    # on applying (y_raw - mean) / std to the whole vector unchanged.
    #
    # A feature with (near-)zero training variance is left on its RAW scale
    # rather than divided by a tiny epsilon -- the same rule as
    # D_intra/extraction/kmis_pairs.py:standardise(), and for the same reason.
    # `k` is constant here (one graph, one locality), so clamping its std to
    # 1e-6 mapped any unseen k to a magnitude of ~1e6 and destroyed the forward
    # pass instead of degrading. Setting the divisor to 1 leaves training
    # unchanged -- (k - k)/1 == (k - k)/1e-6 == 0 -- and turns an unseen k into
    # a shift of a few units. Existing checkpoints stay valid for exactly that
    # reason: the values they trained on do not move.
    Y = torch.stack([r['y_raw'] for r in splits[0]])
    mean, std = Y.mean(0), Y.std(0)
    constant = std < 1e-6
    std = torch.where(constant, torch.ones_like(std), std)
    keep = torch.tensor(STANDARDISE)
    mean = torch.where(keep, mean, torch.zeros_like(mean))
    std = torch.where(keep, std, torch.ones_like(std))
    for split in splits:
        for r in split:
            r['y'] = (r['y_raw'] - mean) / std

    blob = {'k': K, 'source': NAME, 'cond_names': COND_NAMES,
            'cond_mean': mean, 'cond_std': std, 'cond_constant': constant,
            'split_by': SPLIT_BY,
            'standardise': torch.tensor(STANDARDISE), 'r_bins': R_BINS,
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
    print(f"  raw mean: {[round(v, 3) for v in mean.tolist()]}   "
          f"(r columns zeroed -- not standardised)")
    print(f"  raw std : {[round(v, 3) for v in std.tolist()]}   "
          f"(r columns set to 1)")

    # r diagnostics. If the top bins are all-zero the condition is carrying less
    # than its 5 dimensions suggest, and if r is recoverable from (n_i, w_ij) it
    # is carrying nothing NEW at all -- both are true on Luxembourg k=7. Print it
    # rather than leave it to be rediscovered.
    R = np.stack([r['y_raw'][6:].numpy() for r in recs])
    print(f"\n  r_i->j mean {R.mean(0).round(4).tolist()}")
    print(f"  r_i->j std  {R.std(0).round(4).tolist()}")
    dead = [COND_NAMES[6 + b] for b in range(R_BINS) if R[:, b].std() == 0]
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
    print(f"  split by {SPLIT_BY}: {len(splits[0])} train / {len(splits[1])} val / "
          f"{len(splits[2])} test" + (f", {discarded} discarded" if discarded else ""))
    print(f"\nwrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    if dens.mean() < 0.005:
        print("\nNOTE: the cross block is very sparse. Predicting 'no edge' everywhere "
              "will already score a low CE -- read W_out-degree, not CE alone.")
    if comb.max() > 96:
        print(f"\nWARNING: largest combined graph is {comb.max()} nodes. E is dense "
              f"({comb.max()**2} entries per sample); lower MAX_COMBINED if memory bites.")


if __name__ == "__main__":
    main()
