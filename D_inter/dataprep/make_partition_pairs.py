"""D_inter's A / B / C pair datasets from two coarsenings made elsewhere (.npz partitions).

Run LOCALLY in the `kmis` conda env, AFTER D_intra/extraction/make_partition_datasets.py
(it writes the supernode split this reads):

    python make_partition_pairs.py --public dblp --public-npz DBLP.npz \\
        --private collab --private-npz COLLAB.npz            # every pair; --max-combined 128 caps

The road-network datasets (make_pair_blocks.py for A, make_transfer_pairs.py for B
and C, make_nor_pairs.py for no-r) with the supernodes read from a coarsening .npz
(coarsen_compare_expr/coarsen_io.py's format) instead of coarsened here.
partition_pair_records is make_pair_blocks.extract_pair_records' pair loop,
unchanged: the same record format, condition [n_i, n_j, w_ij, r_{i->j}, r_{j->i}],
orientation draw and standardisation.

  A  train public -> test public                 pairs_inter_A_<tag>_deg[_nor].pt
  B  train public -> test private (A's model)    pairs_inter_B_<tag>_deg[_nor].pt
  C  A fine-tuned on private train -> private    pairs_inter_C_<tag>_deg[_nor].pt
     (A's condition statistics and cross-block edge marginal, as make_transfer_pairs.py)

_nor: the condition cut to [n_i, n_j, w_ij], as make_nor_pairs.py.

R BINS. r_{i->j}[b] is the fraction of V_i nodes whose cross out-degree d falls in
bin b, edges e_0 = 0 < e_1 < ... : bin b is e_b <= d < e_{b+1}, the last bin open.
The default, 0 1 2 3 4, is the road networks' r_0, r_1, r_2, r_3, r_{>=4}
(make_pair_blocks.out_degree_hist). Denser graphs need wider bins: on DBLP and
ogbl-collab the cross out-degree reaches the hundreds, so --r-edges 0 1 2 4 8
gives 0, 1, 2-3, 4-7, >=8. Always 5 bins, so the condition stays 13 wide. The
edges are stored in the files (r_edges) and name the columns; anything scoring or
releasing r for these graphs must bin the same way.

Train pairs have both supernodes in D_intra's training split; test pairs touch a
test supernode. Every pair is kept unless --max-combined is given, which drops
pairs with n_i + n_j above it and puts _c<max-combined> in the tag, matching
D_intra's --max-n (make_pair_blocks.py's road runs used 128).
"""
import argparse
import copy
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_pair_blocks as mpb       # noqa: E402  (chdirs into k-mis-pool)
import scipy.sparse as sp            # noqa: E402
import torch                         # noqa: E402
from make_nor_pairs import KEEP      # noqa: E402
from make_transfer_pairs import cross_marginal, describe   # noqa: E402


def load_partition(path):
    """(A csr, assignment, k) from a coarsening .npz; allow_pickle=False."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        n = int(meta["n"])
        A = sp.csr_matrix((z["adj_data"].astype(np.float64), z["adj_indices"], z["adj_indptr"]),
                          shape=(n, n))
        cluster = z["assignment"].astype(np.int64)
    return A, cluster, int(meta["parameters"]["k"])


def out_degree_hist(rows, n_i, edges):
    """r_{i->j} with bins [edges[b], edges[b+1]), the last open; make_pair_blocks'
    out_degree_hist when edges = 0 1 2 3 4."""
    d = np.bincount(rows, minlength=n_i)[:n_i]
    return np.bincount(np.digitize(d, edges[1:]), minlength=len(edges)) / n_i


def bin_names(edges):
    """'0', '1', '2-3', ..., '8+' for the r columns."""
    names = [str(a) if b - a == 1 else f"{a}-{b - 1}" for a, b in zip(edges, edges[1:])]
    return names + [f"{edges[-1]}+"]


def partition_pair_records(A, cluster, rng, min_n=mpb.MIN_N, max_combined=mpb.MAX_COMBINED,
                           r_edges=(0, 1, 2, 3, 4)):
    """(records, n dropped for size): make_pair_blocks.extract_pair_records' loop.
    max_combined=None keeps every pair."""
    n = A.shape[0]
    c = int(cluster.max()) + 1
    order = np.argsort(cluster, kind='stable')
    starts = np.searchsorted(cluster[order], np.arange(c))
    ends = np.searchsorted(cluster[order], np.arange(c), side='right')

    members, rows, n_of, rho_of = {}, {}, {}, {}
    for ci in range(c):
        mem = order[starts[ci]:ends[ci]]
        if len(mem) < min_n:
            continue
        members[ci] = mem
        rows[ci] = A[mem]
        n_of[ci] = len(mem)
        rho_of[ci] = mpb.density(len(mem), int(rows[ci][:, mem].nnz / 2))

    P = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, c))
    C = (P.T @ A @ P).tocoo()

    recs, too_big = [], 0
    for ci, cj, w in zip(C.row, C.col, C.data):
        ci, cj, w = int(ci), int(cj), int(w)
        if ci >= cj or w == 0:
            continue
        if ci not in members or cj not in members:
            continue
        if rng.random() < 0.5:                 # make_pair_blocks.py's ORIENTATION
            ci, cj = cj, ci

        mem_i, mem_j = members[ci], members[cj]
        n_i, n_j = n_of[ci], n_of[cj]
        if max_combined and n_i + n_j > max_combined:
            too_big += 1
            continue

        L_i = rows[ci][:, mem_i].tocoo()
        L_j = rows[cj][:, mem_j].tocoo()
        B = rows[ci][:, mem_j].tocoo()

        src = np.concatenate([L_i.row, L_j.row + n_i, B.row,        B.col + n_i])
        dst = np.concatenate([L_i.col, L_j.col + n_i, B.col + n_i,  B.row])
        edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

        r = out_degree_hist(B.row, n_i, r_edges)
        r_ji = out_degree_hist(B.col, n_j, r_edges)
        y_raw = torch.tensor([n_i, n_j, B.nnz, *r, *r_ji], dtype=torch.float)

        recs.append({
            'sn_i': ci, 'sn_j': cj,
            'n_i': n_i, 'n_j': n_j,
            'edge_index': edge_index,
            'w': int(B.nnz),
            'r': torch.tensor(r, dtype=torch.float),
            'r_ji': torch.tensor(r_ji, dtype=torch.float),
            'rho_i': rho_of[ci], 'rho_j': rho_of[cj],
            'y_raw': y_raw,
        })
    return recs, too_big


def drop_r(blob):
    """make_nor_pairs.py's cut: the condition to KEEP = [n_i, n_j, w_ij]."""
    d = len(KEEP)
    assert list(blob['cond_names'])[:d] == KEEP
    return dict(blob, cond_names=KEEP, cond_mean=blob['cond_mean'][:d],
                cond_std=blob['cond_std'][:d], cond_constant=blob['cond_constant'][:d],
                splits=[[dict(r, y_raw=r['y_raw'][:d], y=r['y'][:d]) for r in s]
                        for s in blob['splits']])


def pairs(name, path, max_combined, r_edges):
    A, cluster, k = load_partition(path)
    print(f"{name}: {A.shape[0]} nodes, {int(A.nnz / 2)} edges, k={k}")
    recs, too_big = partition_pair_records(A, cluster, np.random.default_rng(mpb.SEED),
                                           max_combined=max_combined, r_edges=r_edges)
    for r in recs:
        r['graph'] = name
    print(f"  kept {len(recs)} connected pairs"
          + (f" (dropped {too_big} with n_i+n_j > {max_combined})" if max_combined else ""))
    tr, te = mpb.split_pairs(recs, name, k)
    print(f"  split: {len(tr)} train / {len(te)} test pairs")
    return k, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--public-npz", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--private-npz", required=True)
    ap.add_argument("--max-combined", type=int, default=None,
                    help="drop pairs with n_i + n_j above this (default: keep every pair)")
    ap.add_argument("--r-edges", type=int, nargs=5, default=[0, 1, 2, 3, 4],
                    help="lower edges of the 5 r bins (default: the road bins 0 1 2 3 4+; "
                         "collaboration graphs: 0 1 2 4 8)")
    args = ap.parse_args()
    if args.r_edges[0] != 0 or sorted(set(args.r_edges)) != args.r_edges:
        ap.error("--r-edges must start at 0 and increase")
    torch.manual_seed(mpb.SEED)

    k, pub_tr, pub_te = pairs(args.public, args.public_npz, args.max_combined, args.r_edges)
    k_priv, priv_tr, priv_te = pairs(args.private, args.private_npz, args.max_combined,
                                     args.r_edges)
    if k != k_priv:
        raise SystemExit(f"k differs ({k} vs {k_priv})")
    tag = (f"{args.public[:4]}2{args.private[:4]}_k{k}_{mpb.SPLIT_TAG}"
           + (f"_c{args.max_combined}" if args.max_combined else "") + mpb.VERSION)

    print("\nwhere the two graphs differ (the transfer gap):")
    print(describe(args.public, pub_tr))
    print(describe(args.private, priv_tr))

    names = bin_names(args.r_edges)
    cond_names = (['n_i', 'n_j', 'w_ij'] + [f'rij{b}' for b in names]
                  + [f'rji{b}' for b in names])
    common = {'k': k, 'cond_names': cond_names, 'r_edges': list(args.r_edges),
              'public': args.public,
              'private': args.private, 'max_combined': args.max_combined, 'source': tag,
              'split_by': f"supernode ({mpb.SPLIT_TAG}, shared with D_intra)"}

    # A: public train / test, standardised on public train
    A_splits = [pub_tr, mpb.val_from_train(pub_tr), pub_te]
    mean, std, constant = mpb.standardise(pub_tr, A_splits)
    stats = {'cond_mean': mean, 'cond_std': std, 'cond_constant': constant}
    A = dict(common, config='A', graph=args.public, **stats, splits=A_splits)

    def a_standardise(recs):
        recs = copy.deepcopy(recs)
        for r in recs:
            r['y'] = (r['y_raw'] - mean) / std
        return recs

    # B: A's training data, private test pairs on A's statistics
    B = dict(common, config='B', **stats, splits=[pub_tr, A_splits[1], a_standardise(priv_te)])

    # C: fine-tune A on private train; A's statistics and edge marginal
    c_tr = a_standardise(priv_tr)
    C = dict(common, config='C', **stats, edge_marginal=cross_marginal(pub_tr), finetune_of='A',
             splits=[c_tr, mpb.val_from_train(c_tr), a_standardise(priv_te)])

    print()
    for blob in (A, B, C):
        for nor in ("", "_nor"):
            path = mpb.OUT_DIR / f"pairs_inter_{blob['config']}_{tag}{nor}.pt"
            torch.save(drop_r(blob) if nor else blob, path)
            tr, va, te = (len(s) for s in blob['splits'])
            print(f"  {path.name:<46} {tr:>6} train / {va:>4} val / {te:>5} test"
                  f"   ({path.stat().st_size / 1e6:.0f} MB)")
    R = np.stack([r['y_raw'][3:8].numpy() for r in pub_tr])
    print(f"\n  r bins {names}; r_i->j mean on {args.public} train: {R.mean(0).round(4).tolist()}")
    print(f"  C: A's cross-block edge marginal {[round(v, 5) for v in C['edge_marginal'].tolist()]}, "
          f"{args.private}'s own {[round(v, 5) for v in cross_marginal(priv_tr).tolist()]}")


if __name__ == "__main__":
    main()
