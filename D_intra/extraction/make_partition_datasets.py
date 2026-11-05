"""D_intra's transfer datasets from two coarsenings made elsewhere (.npz partitions).

    python make_partition_datasets.py --public dblp --public-npz DBLP.npz \\
        --private collab --private-npz COLLAB.npz            # every supernode; --max-n 128 caps
    python make_finetune_datasets.py --tag dblp2coll_k3_s80          # then Cft and O

The datasets make_transfer_datasets.py builds for the road networks, but with the
supernodes read from a coarsening .npz in coarsen_compare_expr/coarsen_io.py's
format -- the fine graph as CSR (adj_*), `assignment` (node -> supernode) and
`centroids` (supernode -> its centre, assignment[centroids[i]] == i) -- instead of
coarsened here. partition_records is kmis_pairs.extract_pairs' loop over the
supernodes, unchanged; the condition, split, standardisation and C's oversampling
are make_transfer_datasets.py's.

  A  train public -> test public       pairs_A_<tag>[_nohop|_uncond].pt
  B  train public -> test private      pairs_B_<tag>[_nohop|_uncond].pt  (A's model)
  C  public + private train -> private pairs_C_<tag>[_nohop|_uncond].pt  (make_finetune_datasets.py
                                                                          reads Cft's records from it)

Condition [n, rho_in, k, h1..hk]; _nohop [n, rho_in, k]; _uncond none.

SIZE LIMIT. --max-n drops supernodes larger than it from every split. The split
(../data/splits/<graph>_k<k>_s80.json, read by D_inter) is drawn over ALL
supernodes with n >= 3 first, so it does not depend on the limit. The limit is in
the tag (_c<max-n>).
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
import torch

from kmis_pairs import build_condition, cond_names, hop_histogram, standardise
from make_transfer_datasets import (COND, COND_NO_HOPS, COND_NONE, MIN_N, OUT_DIR,
                                    OVERSAMPLE_TARGET, SEED, SPLIT_DIR, TEST_FRAC,
                                    VAL_FROM_TRAIN, describe_sizes, save,
                                    split_by_supernode, val_from_train, write_split)

VARIANTS = {"": COND, "_nohop": COND_NO_HOPS, "_uncond": COND_NONE}


def load_partition(path):
    """(A csr, assignment, centroid of each supernode, k); allow_pickle=False."""
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        n = int(meta["n"])
        A = sp.csr_matrix((z["adj_data"].astype(np.float64), z["adj_indices"], z["adj_indptr"]),
                          shape=(n, n))
        cluster = z["assignment"].astype(np.int64)
        if "centroids" not in z.files:
            raise SystemExit(f"{path} has no `centroids` array")
        centre_of = z["centroids"].astype(np.int64)
    if not np.array_equal(cluster[centre_of], np.arange(len(centre_of))):
        raise SystemExit(f"{path}: assignment[centroids[i]] != i")
    return A, cluster, centre_of, int(meta["parameters"]["k"])


def partition_records(graph_key, A, cluster, centre_of, k, cond, min_n=MIN_N, k_hist=None):
    """One record per supernode with n >= min_n: kmis_pairs.extract_pairs' loop."""
    k_hist = k if k_hist is None else k_hist
    c = int(cluster.max()) + 1
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

    sizes = np.array([r["n"] for r in recs])
    print(f"  {graph_key:<12} k={k}  {c} supernodes -> {len(recs)} kept "
          f"(dropped {dropped} with n<{min_n}), size {sizes.min()}-{sizes.max()} "
          f"median {int(np.median(sizes))}")
    return recs


def records(name, path, max_n):
    """All records of one graph with the full condition, split, then size-limited."""
    A, cluster, centre_of, k = load_partition(path)
    recs = partition_records(name, A, cluster, centre_of, k, COND)
    h = torch.stack([r["y_raw"][3:] for r in recs])
    if not torch.allclose(h.sum(1), torch.ones(len(recs))):
        raise SystemExit(f"{name}: a member lies more than k={k} hops from its centroid")
    tr, te = split_by_supernode(recs, TEST_FRAC, SEED)
    write_split(name, k, recs, te, TEST_FRAC, SEED, SPLIT_DIR)
    if max_n:
        tr, te = [r for r in tr if r["n"] <= max_n], [r for r in te if r["n"] <= max_n]
    print(f"  {name}: {len(tr)} train / {len(te)} test supernodes"
          + (f" ({len(recs) - len(tr) - len(te)} with n > {max_n} dropped)" if max_n else ""))
    return k, tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True, help="training graph's name, e.g. dblp")
    ap.add_argument("--public-npz", required=True)
    ap.add_argument("--private", required=True, help="target graph's name, e.g. collab")
    ap.add_argument("--private-npz", required=True)
    ap.add_argument("--max-n", type=int, default=None, help="drop supernodes larger than this")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    torch.manual_seed(SEED)

    k, pub_tr0, pub_te0 = records(args.public, args.public_npz, args.max_n)
    k_priv, priv_tr0, priv_te0 = records(args.private, args.private_npz, args.max_n)
    if k != k_priv:
        raise SystemExit(f"k differs ({k} vs {k_priv})")
    pct = int(round(100 * (1 - TEST_FRAC)))
    base = (f"{args.public[:4]}2{args.private[:4]}_k{k}_s{pct}"
            + (f"_c{args.max_n}" if args.max_n else ""))
    print(f"\n  {args.public} clusters: {describe_sizes(pub_tr0 + pub_te0)}")
    print(f"  {args.private} clusters: {describe_sizes(priv_tr0 + priv_te0)}\n")

    for suffix, cond in VARIANTS.items():
        tag = base + suffix
        names = cond_names(cond, k)
        d = len(names)
        cut = lambda recs: [dict(r, y_raw=r["y_raw"][:d]) for r in recs]   # noqa: E731
        pub_tr, pub_te, priv_tr, priv_te = cut(pub_tr0), cut(pub_te0), cut(priv_tr0), cut(priv_te0)
        pub_val = val_from_train(pub_tr, VAL_FROM_TRAIN, SEED)
        common = {"cond_names": names, "k": k, "k_public": k, "k_private": k,
                  "public": args.public, "private": args.private, "max_n": args.max_n}
        print(f"condition{suffix} ({d} dims): {names}")

        A = dict(common, config="A", desc=f"train {args.public}, test {args.public}",
                 splits=[list(pub_tr), list(pub_val), list(pub_te)])
        mean, std, const = standardise(A["splits"][0], A["splits"])
        A.update(cond_mean=mean, cond_std=std, cond_constant=const)
        save(os.path.join(args.out_dir, f"pairs_A_{tag}.pt"), A)

        B = dict(common, config="B", desc=f"train {args.public}, test {args.private}",
                 splits=[list(pub_tr), list(pub_val), list(priv_te)])
        mean, std, const = standardise(B["splits"][0], B["splits"])
        B.update(cond_mean=mean, cond_std=std, cond_constant=const)
        save(os.path.join(args.out_dir, f"pairs_B_{tag}.pt"), B)

        reps = 1
        if OVERSAMPLE_TARGET > 0 and priv_tr:
            want = OVERSAMPLE_TARGET / (1 - OVERSAMPLE_TARGET) * len(pub_tr)
            reps = max(1, int(round(want / len(priv_tr))))
        combined = list(pub_tr) + list(priv_tr) * reps
        np.random.default_rng(SEED).shuffle(combined)
        C = dict(common, config="C",
                 desc=f"train {args.public}+{args.private}, test {args.private}",
                 private_oversample_factor=reps,
                 splits=[combined, val_from_train(combined, VAL_FROM_TRAIN, SEED), list(priv_te)])
        mean, std, const = standardise(C["splits"][0], C["splits"])
        C.update(cond_mean=mean, cond_std=std, cond_constant=const)
        save(os.path.join(args.out_dir, f"pairs_C_{tag}.pt"), C)
        print()

    print(f"Next: python make_finetune_datasets.py --tag {base}   (Cft and O)")


if __name__ == "__main__":
    main()
