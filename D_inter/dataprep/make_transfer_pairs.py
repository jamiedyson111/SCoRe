"""Build D_inter's transfer configurations B and C: Luxembourg -> Oldenburg.

Run LOCALLY in the `kmis` conda env:

    cd D_inter/dataprep && python make_transfer_pairs.py

Mirrors D_intra/extraction/make_transfer_datasets.py:

  A  train Luxembourg              -> test Luxembourg   the existing pair_lux, unchanged
  B  train Luxembourg              -> test Oldenburg    no training: same model as A
  C  train Luxembourg + Oldenburg  -> test Oldenburg    a new model

A IS pair_lux. Its pairs file (the V1 blob, pairs_inter_luxembourg_osm_k7.pt) and
its checkpoint already exist, so this script does not rebuild it.

B is A's training and validation data with an Oldenburg test split. The
Luxembourg records and the V1 blob's cond_mean / cond_std are copied, not
recomputed, so B's training data is A's exactly and the existing pair_lux
checkpoint IS the B model. Only the test split is new, standardised with the
Luxembourg statistics the model was trained under.

C trains on Luxembourg plus Oldenburg. Oldenburg yields far fewer pairs (219
train against 8176), so the Oldenburg training pairs are repeated to
OVERSAMPLE_TARGET of C's training set -- repetition, not new data, as in D_intra.
C validates on Oldenburg and is re-standardised on its own training set.

THE CONDITION IS V1, [n_i, rho_i, n_j, rho_j, w_ij, k]: what the existing
checkpoint takes and what the private release publishes. extract_pair_records
builds the 11-wide V2 vector; the r columns are dropped here.

Oldenburg is split (0.40, 0.10, 0.50) like D_intra's private graph, so B and C
are scored on the same, reasonably large, set of held-out pairs.
"""
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
PRIVATE = "oldenburg"
EDGE_LIST = (PLACEMENT / "D_intra" / "extraction" / "datasets" / "Brinkhoff"
             / "oldenburg" / "raw" / "OL.cedge")
PUBLIC_BLOB = mpb.OUT_DIR / "pairs_inter_luxembourg_osm_k7.pt"     # the V1 blob = A
SPLIT_PRIVATE = (0.40, 0.10, 0.50)
OVERSAMPLE_TARGET = 0.25
V1 = 6                              # [n_i, rho_i, n_j, rho_j, w_ij, k]
TAG = "lux2olde_k7"
# ===========================================================================


def load_edge_list(path):
    """(adj SparseTensor, A csr) for an "edge_id u v length" file, as load_fine returns.

    Same reading as D_intra/extraction/kmis_pairs.py:_load_edge_list: self-loops
    and duplicate rows dropped, both directions stored, unit weights.
    """
    from torch_sparse import SparseTensor
    uv = np.loadtxt(path, usecols=(1, 2), dtype=np.int64)
    uv = uv[uv[:, 0] != uv[:, 1]]
    n = int(uv.max()) + 1
    row = torch.from_numpy(np.concatenate([uv[:, 0], uv[:, 1]]))
    col = torch.from_numpy(np.concatenate([uv[:, 1], uv[:, 0]]))
    adj = SparseTensor(row=row, col=col, value=torch.ones(row.numel()),
                       sparse_sizes=(n, n)).coalesce(reduce="max")
    r, c, _ = adj.coo()
    A = sp.coo_matrix((np.ones(r.numel()), (r.numpy(), c.numpy())), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return adj, A


def standardise(train, splits):
    """Train-only statistics; a constant column keeps its raw scale (std := 1).

    The rule make_pair_blocks.py uses. Writes r['y'] on every record of `splits`.
    """
    Y = torch.stack([r['y_raw'] for r in train])
    mean, std = Y.mean(0), Y.std(0)
    constant = std < 1e-6
    std = torch.where(constant, torch.ones_like(std), std)
    for split in splits:
        for r in split:
            r['y'] = (r['y_raw'] - mean) / std
    return mean, std, constant


def describe(name, recs):
    ni = np.array([r['n_i'] for r in recs])
    w = np.array([r['w'] for r in recs])
    return (f"  {name:<11} {len(recs):>6} pairs   n_i mean {ni.mean():6.2f}   "
            f"w mean {w.mean():.3f}   w=1 {100 * np.mean(w == 1):5.1f}%")


def main():
    pub = torch.load(PUBLIC_BLOB, map_location="cpu")
    assert list(pub['cond_names']) == mpb.COND_NAMES[:V1], pub['cond_names']
    k = int(pub['k'])
    lux_tr, lux_va, lux_te = pub['splits']

    # ---- Oldenburg pairs, same extraction as Luxembourg ------------------
    rng = np.random.default_rng(mpb.SEED)
    adj, A = load_edge_list(EDGE_LIST)
    print(f"{PRIVATE}: {adj.size(0)} nodes, {int(A.nnz / 2)} edges, k={k}")
    recs, too_big = mpb.extract_pair_records(adj, A, k, rng)
    for r in recs:
        r['y_raw'] = r['y_raw'][:V1]                # V1: drop the r block
        r['graph'] = PRIVATE
    print(f"kept {len(recs)} connected pairs"
          + (f" (dropped {too_big} with n_i+n_j > {mpb.MAX_COMBINED})" if too_big else ""))

    # The split shuffle continues the orientation rng, as make_pair_blocks does.
    rng.shuffle(recs)
    ntr = int(SPLIT_PRIVATE[0] * len(recs))
    nva = int(SPLIT_PRIVATE[1] * len(recs))
    olde_tr, olde_va, olde_te = recs[:ntr], recs[ntr:ntr + nva], recs[ntr + nva:]

    print("\nwhere the two graphs differ (the transfer gap):")
    print(describe("luxembourg", lux_tr))
    print(describe(PRIVATE, recs))

    common = {'k': k, 'cond_names': list(pub['cond_names']), 'split_by': 'pair',
              'public': 'luxembourg', 'private': PRIVATE, 'source': TAG}

    # ---- B: Luxembourg train/val, Oldenburg test, Luxembourg statistics ---
    # The test records are copied before y is written, so B and C each carry
    # their own standardisation of the same Oldenburg pairs.
    b_te = copy.deepcopy(olde_te)
    for r in b_te:
        r['y'] = (r['y_raw'] - pub['cond_mean']) / pub['cond_std']
    B = dict(common, config='B', cond_mean=pub['cond_mean'], cond_std=pub['cond_std'],
             splits=[lux_tr, lux_va, b_te])

    # ---- C: Luxembourg + oversampled Oldenburg train, Oldenburg val/test ---
    want = OVERSAMPLE_TARGET / (1 - OVERSAMPLE_TARGET) * len(lux_tr)
    reps = max(1, int(round(want / len(olde_tr))))
    c_tr = copy.deepcopy(lux_tr) + [copy.deepcopy(r) for r in olde_tr for _ in range(reps)]
    np.random.default_rng(mpb.SEED).shuffle(c_tr)
    c_va, c_te = copy.deepcopy(olde_va), copy.deepcopy(olde_te)
    mean, std, constant = standardise(c_tr, [c_tr, c_va, c_te])
    C = dict(common, config='C', cond_mean=mean, cond_std=std, cond_constant=constant,
             private_oversample_factor=reps, splits=[c_tr, c_va, c_te])

    print(f"\n{PRIVATE} split: {len(olde_tr)} train / {len(olde_va)} val / "
          f"{len(olde_te)} test  (B and C share the test pairs)")
    for blob in (B, C):
        path = mpb.OUT_DIR / f"pairs_inter_{blob['config']}_{TAG}.pt"
        torch.save(blob, path)
        tr, va, te = (len(s) for s in blob['splits'])
        print(f"  {path.name:<34} {tr:>6} train / {va:>5} val / {te:>4} test")
    n_priv = len(olde_tr) * reps
    print(f"\n  config C: {len(olde_tr)} {PRIVATE} train pairs repeated x{reps} -> "
          f"{n_priv} of {len(c_tr)} ({100 * n_priv / len(c_tr):.0f}% of the training set)")
    print("  A is pair_lux itself; its V1 blob and checkpoint are unchanged.")
    print(f"\nDataset configs: configs/dataset/pair_{{B,C}}_{TAG}.yaml")


if __name__ == "__main__":
    main()
