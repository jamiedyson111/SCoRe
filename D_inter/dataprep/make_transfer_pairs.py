"""Build D_inter's transfer configurations B and C: Luxembourg -> Oldenburg.

Run LOCALLY in the `kmis` conda env:

    cd D_inter/dataprep && python make_transfer_pairs.py                     # Oldenburg
    cd D_inter/dataprep && python make_transfer_pairs.py --private vermont   # Vermont

Mirrors D_intra/extraction/make_transfer_datasets.py:

  A  train Luxembourg                          -> test Luxembourg   make_pair_blocks.py
  B  train Luxembourg                          -> test Oldenburg    no training: A's model
  C  A's model fine-tuned on Oldenburg train   -> test Oldenburg    A's checkpoint, fine-tuned

A is pair_lux_s80_deg, built by make_pair_blocks.py (run that first).

B is A's training data with an Oldenburg test split. The Luxembourg records and
A's cond_mean / cond_std are copied, not recomputed, so B's training data is A's
exactly and A's checkpoint IS the B model. Only the test split is new,
standardised with the Luxembourg statistics the model was trained under.

C FINE-TUNES A's checkpoint on the Oldenburg training pairs alone (the paper's
"limited target adaptation subset"); scripts/run_pair.sh does it with
general.finetune_from. Its training split is therefore Oldenburg only, with no
oversampling. Everything the pre-trained network's inputs depend on is kept at
A's values, so fine-tuning starts from a model that already reads them:
  - cond_mean / cond_std are A's (Luxembourg's), not re-estimated on Oldenburg;
  - edge_marginal is A's cross-block edge marginal, which pair_dataset.py uses in
    place of C's own, so the noise model and its limit distribution stay A's.
The fine-tuning pairs carry the TRUE X_i, X_j (from the real L_i, L_j), as
training always does; generated context is an inference-time matter.

THE CONDITION is make_pair_blocks.COND_NAMES, [n_i, n_j, w_ij, r_{i->j}, r_{j->i}].

SPLIT (October 2026): 80 / 20 by supernode, the split D_intra wrote to
D_intra/data/splits/oldenburg_k7_s80.json. Oldenburg training pairs are those
whose two supernodes are both training supernodes; every pair touching a test
supernode is a test pair. B and C share the test pairs. There is no validation
split; the "val" slot holds training pairs (logging only).
"""
import argparse
import copy
import pathlib
import sys

import numpy as np

# Resolve paths before importing make_pair_blocks: importing it chdirs into
# k-mis-pool, after which a relative __file__ would resolve to the wrong place.
HERE = pathlib.Path(__file__).resolve().parent
PLACEMENT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import make_pair_blocks as mpb      # noqa: E402  (chdirs; puts kmis on the path)
import scipy.sparse as sp           # noqa: E402
import torch                        # noqa: E402

# ===========================================================================
# The private graphs, read exactly as D_intra/extraction/kmis_pairs.py reads them
# (EDGE_LISTS there), so the coarsening -- and so the supernode ids the split file
# names -- is the same. split_pairs checks every supernode size against it.
DATASETS = PLACEMENT / "D_intra" / "extraction" / "datasets"
PRIVATE_GRAPHS = {"oldenburg": DATASETS / "Brinkhoff" / "oldenburg" / "raw" / "OL.cedge",
                  "vermont":   DATASETS / "DIMACS9" / "vermont" / "VT.mtx"}
PUBLIC_BLOB = mpb.OUT_DIR / (mpb.OUT % (mpb.NAME, mpb.K))      # = A
# ===========================================================================


def load_edge_list(path):
    """(adj SparseTensor, A csr) for an "edge_id u v length" file (.cedge, 0-indexed)
    or a MatrixMarket file (.mtx, 1-indexed, n from its header), as load_fine returns.

    Same reading as D_intra/extraction/kmis_pairs.py:_load_edge_list: self-loops
    and duplicate rows dropped, both directions stored, unit weights.
    """
    from torch_sparse import SparseTensor
    path = str(path)
    if path.endswith(".mtx"):
        with open(path) as f:
            line = f.readline()
            while line.startswith("%"):
                line = f.readline()
            n = int(line.split()[0])
            uv = np.loadtxt(f, usecols=(0, 1), dtype=np.int64, ndmin=2) - 1
    else:
        uv = np.loadtxt(path, usecols=(1, 2), dtype=np.int64)
        n = int(uv.max()) + 1
    uv = uv[uv[:, 0] != uv[:, 1]]
    row = torch.from_numpy(np.concatenate([uv[:, 0], uv[:, 1]]))
    col = torch.from_numpy(np.concatenate([uv[:, 1], uv[:, 0]]))
    adj = SparseTensor(row=row, col=col, value=torch.ones(row.numel()),
                       sparse_sizes=(n, n)).coalesce(reduce="max")
    r, c, _ = adj.coo()
    A = sp.coo_matrix((np.ones(r.numel()), (r.numpy(), c.numpy())), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return adj, A


def cross_marginal(recs):
    """[no-edge, edge] marginal over the cross blocks of `recs`.

    PairGraphDataModule.edge_counts' quantity, computed from the records: both
    triangles of E, so 2 n_i n_j slots and 2 w_ij edges per pair.
    """
    slots = 2.0 * sum(int(r['n_i']) * int(r['n_j']) for r in recs)
    edges = 2.0 * sum(int(r['w']) for r in recs)
    return torch.tensor([slots - edges, edges]) / slots


def describe(name, recs):
    ni = np.array([r['n_i'] for r in recs])
    w = np.array([r['w'] for r in recs])
    return (f"  {name:<11} {len(recs):>6} pairs   n_i mean {ni.mean():6.2f}   "
            f"w mean {w.mean():.3f}   w=1 {100 * np.mean(w == 1):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", default="oldenburg", choices=sorted(PRIVATE_GRAPHS))
    PRIVATE = ap.parse_args().private
    TAG = f"lux2{PRIVATE[:4]}_k{mpb.K}_{mpb.SPLIT_TAG}{mpb.VERSION}"
    pub = torch.load(PUBLIC_BLOB, map_location="cpu")
    assert list(pub['cond_names']) == mpb.COND_NAMES, pub['cond_names']
    k = int(pub['k'])
    lux_tr, lux_va, lux_te = pub['splits']
    stats = {'cond_mean': pub['cond_mean'], 'cond_std': pub['cond_std'],
             'cond_constant': pub['cond_constant']}

    def lux_standardise(recs):
        for r in recs:
            r['y'] = (r['y_raw'] - pub['cond_mean']) / pub['cond_std']
        return recs

    # ---- the private graph's pairs, same extraction as Luxembourg ---------
    rng = np.random.default_rng(mpb.SEED)
    adj, A = load_edge_list(PRIVATE_GRAPHS[PRIVATE])
    print(f"{PRIVATE}: {adj.size(0)} nodes, {int(A.nnz / 2)} edges, k={k}")
    recs, too_big = mpb.extract_pair_records(adj, A, k, rng)
    for r in recs:
        r['graph'] = PRIVATE
    print(f"kept {len(recs)} connected pairs"
          + (f" (dropped {too_big} with n_i+n_j > {mpb.MAX_COMBINED})" if too_big else ""))

    # The supernode split D_intra wrote; the same rule as Luxembourg's.
    olde_tr, olde_te = mpb.split_pairs(recs, PRIVATE, k)

    print("\nwhere the two graphs differ (the transfer gap):")
    print(describe("luxembourg", lux_tr))
    print(describe(PRIVATE, recs))

    common = {'k': k, 'cond_names': list(pub['cond_names']),
              'split_by': f"supernode ({mpb.SPLIT_TAG}, shared with D_intra)",
              'public': 'luxembourg', 'private': PRIVATE, 'source': TAG}

    # ---- B: Luxembourg train/val, Oldenburg test, Luxembourg statistics ---
    # The test records are copied before y is written, so B and C each carry
    # their own standardisation of the same Oldenburg pairs.
    B = dict(common, config='B', **stats,
             splits=[lux_tr, lux_va, lux_standardise(copy.deepcopy(olde_te))])

    # ---- C: fine-tune on Oldenburg train, Oldenburg test, A's statistics ----
    c_tr = lux_standardise(copy.deepcopy(olde_tr))
    c_va, c_te = mpb.val_from_train(c_tr), lux_standardise(copy.deepcopy(olde_te))
    C = dict(common, config='C', **stats, edge_marginal=cross_marginal(lux_tr),
             finetune_of='A', splits=[c_tr, c_va, c_te])

    print(f"\n{PRIVATE} split: {len(olde_tr)} train / {len(olde_te)} test pairs "
          f"(B and C share the test pairs; no validation split)")
    for blob in (B, C):
        path = mpb.OUT_DIR / f"pairs_inter_{blob['config']}_{TAG}.pt"
        torch.save(blob, path)
        tr, va, te = (len(s) for s in blob['splits'])
        print(f"  {path.name:<34} {tr:>6} train / {va:>5} val / {te:>4} test")
    print(f"\n  config C: fine-tunes A's model on the {len(c_tr)} {PRIVATE} training pairs "
          f"(A's condition statistics; A's edge marginal "
          f"{[round(v, 5) for v in C['edge_marginal'].tolist()]}, "
          f"{PRIVATE}'s own {[round(v, 5) for v in cross_marginal(olde_tr).tolist()]})")
    print("  A is the Luxembourg file itself (make_pair_blocks.py); B uses A's model.")
    print(f"\nDataset configs: configs/dataset/pair_{{B,C}}_{TAG}.yaml")


if __name__ == "__main__":
    main()
