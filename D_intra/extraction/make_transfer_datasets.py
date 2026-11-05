"""Build the three train/test configurations for the D_intra transfer study.

The scenario is public -> private transfer: D_intra is pretrained on a PUBLIC
graph and used to reconstruct supernodes of a PRIVATE graph, so that the private
graph's fine structure is never needed at training time.

    python make_transfer_datasets.py                          # Luxembourg -> Oldenburg
    python make_transfer_datasets.py --no-cond                # plain DiGress baseline (no condition)
    python make_transfer_datasets.py --private minnesota --k-private 4   # the old private graph
    python make_transfer_datasets.py --public luxembourg --private facebook --k 7
    python make_transfer_datasets.py --k-public 7 --k-private 4    # differing localities

  A  train public  -> test public     (in-distribution reference)
  B  train public  -> test private    (transfer, zero private training data)
  C  train both    -> test private    (does a little private data help?)

Writes three .pt files in the format cluster_dataset.py expects.

k PER GRAPH. k is a parameter of the coarsening pipeline, not a tuning knob: if
the private graph's coarse graph is built at k, then D_intra must reconstruct
k-supernodes. The default is one k throughout, and at equal k a denser private
graph yields larger supernodes (Minnesota's mean degree is 2.50 against
Luxembourg's 2.09, giving median cluster sizes of 28 vs 11 at k=7).

Oldenburg, the default private graph, needs no override. Its mean degree is 2.30
and at k=7 it gives 317 supernodes of median size 15, with a mean hop histogram
of [0.20, 0.20, 0.23, 0.20, 0.11, 0.04, 0.01] against Luxembourg's
[0.23, 0.22, 0.23, 0.17, 0.09, 0.04, 0.01]: the same locality and close to the
same shape, so the k column stays in-distribution for B.

--k-private lets the private graph be coarsened at its own locality, which is
what the k=4 Minnesota configuration did: 226 supernodes of median size 11
against 94 of median size 28, so the private clusters land on the SAME size
distribution the model trained on, and the shared B/C test set grows from 47
supernodes to 113. Two consequences, both real and both to be reported:

  * The hop histogram keeps the public graph's width (see hop_histogram). Rings
    beyond the private k are provably empty, so sum(h) is still 1 and the model's
    input width is unchanged -- but h's DISTRIBUTION shifts hard. At k=4
    Minnesota's mean h is [0.32, 0.40, 0.22, 0.06, 0, 0, 0] against Luxembourg's
    [0.23, 0.22, 0.23, 0.17, 0.09, 0.04, 0.01]; standardised on Luxembourg
    statistics, h2 sits about 3.2 sigma above the training mean.
  * The `k` column becomes genuinely informative in C (it has variance there, so
    it is standardised normally) and genuinely unseen in B, whose training set is
    public-only and therefore has k constant. standardise() leaves a constant
    feature on its raw scale rather than dividing by an epsilon, so B receives
    (4 - 7)/1 = -3 on a column that was always 0 in training. That is the honest
    encoding of "the granularity changed" and it is a real part of what B is
    being asked to do; it is not a bug, but B's result should not be read as if
    the condition were in-distribution.

THE SPLIT (October 2026). Train / test only, 80 / 20 by SUPERNODE, the same
supernodes for D_intra and D_inter: the supernode assignment is written to
../data/splits/<graph>_k<k>_s80.json and D_inter's builders read that file, so
D_inter trains only on pairs whose two supernodes are both training supernodes,
and tests on every pair that touches a test supernode. There is no held-out
validation split: models train for a fixed number of epochs and the final
checkpoint is used. The "val" slot the training code still expects holds a small
sample of TRAINING records (VAL_FROM_TRAIN), used for logging only -- it is not
held-out data and selects nothing.

TRAINING SET IMBALANCE. The public graph typically yields far more supernodes
than the private one (43x here), so an unweighted union in configuration C would
be ~99% public and C would be indistinguishable from B. OVERSAMPLE_TARGET repeats
the private training pairs to a stated share. This is repetition, not new data,
and must be reported as such.
"""
import argparse
import json
import os

import numpy as np
import torch

from kmis_pairs import GRAPHS, cond_names, extract_pairs, standardise

# ===========================================================================
DEFAULT_PUBLIC = "luxembourg"
DEFAULT_PRIVATE = "oldenburg"
DEFAULT_K = 7

# "h" expands to k columns: the hop histogram, the proportion of non-centroid
# members at each shortest-path distance 1..k from the centroid. It is the only
# entry here that carries information independent of cluster size -- n is imposed
# at sampling, k is constant, and rho_in is degenerate with n (R2 0.997 on
# Luxembourg) because every cluster is a near-tree once connectivity is enforced.
COND = ["n", "rho_in", "k", "h"]
# The ablation: same pipeline, same splits, same seed, condition truncated to
# [n, rho_in, k]. Whatever the two models differ by is attributable to h alone,
# and h is the part of the release that goes beyond the brief's [n, rho_in].
COND_NO_HOPS = ["n", "rho_in", "k"]
# Plain DiGress, one supernode at a time: NO condition (y has width 0). Same
# supernodes, split and seed; the sampler still fixes each graph's size to the
# target supernode's n_i, so this measures what the whole condition c_i adds.
COND_NONE = []
MIN_N = 3
SEED = 0

# Both graphs are split the same way; B and C share the private test supernodes.
TEST_FRAC = 0.20          # 80 / 20 train / test, by supernode, no validation split
VAL_FROM_TRAIN = 256      # training records placed in the "val" slot, logging only
OVERSAMPLE_TARGET = 0.25
OUT_DIR = "../data/cluster_lux/raw"
SPLIT_DIR = "../data/splits"
# ===========================================================================


def save(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(blob, path)
    tr, va, te = (len(s) for s in blob["splits"])
    print(f"  {os.path.basename(path):<30} {tr:>6} train /{va:>5} val /{te:>5} test"
          f"   ({os.path.getsize(path)/1e6:.1f} MB)")


def split_by_supernode(recs, test_frac, seed):
    """Train / test record lists from a seeded shuffle of the supernode ids."""
    ids = np.array(sorted(int(r["sn"]) for r in recs))
    np.random.default_rng(seed).shuffle(ids)
    n_test = int(round(test_frac * len(ids)))
    test = set(ids[len(ids) - n_test:].tolist())
    train_recs = [r for r in recs if int(r["sn"]) not in test]
    test_recs = [r for r in recs if int(r["sn"]) in test]
    return train_recs, test_recs


def write_split(graph, k, recs, test_recs, test_frac, seed, out_dir):
    """The shared supernode split, read by D_inter's dataset builders."""
    os.makedirs(out_dir, exist_ok=True)
    test = sorted(int(r["sn"]) for r in test_recs)
    train = sorted(int(r["sn"]) for r in recs if int(r["sn"]) not in set(test))
    path = os.path.join(out_dir, f"{graph}_k{k}_s{int(round(100 * (1 - test_frac)))}.json")
    json.dump({"graph": graph, "k": k, "seed": seed, "test_frac": test_frac,
               "train": train, "test": test,
               "n": {str(int(r["sn"])): int(r["n"]) for r in recs}},
              open(path, "w"))
    print(f"  split -> {path}: {len(train)} train / {len(test)} test supernodes")


def val_from_train(train, n, seed):
    """A few TRAINING records for the val slot (logging only; nothing is held out)."""
    idx = np.random.default_rng(seed + 1).permutation(len(train))[:n]
    return [train[i] for i in idx]


def describe_sizes(recs):
    a = np.array([r["n"] for r in recs])
    return f"median {int(np.median(a))}, range {a.min()}-{a.max()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", default=DEFAULT_PUBLIC, choices=sorted(GRAPHS))
    ap.add_argument("--private", default=DEFAULT_PRIVATE, choices=sorted(GRAPHS))
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="coarsening locality for both graphs, unless overridden below")
    ap.add_argument("--k-public", type=int, default=None,
                    help="locality for the public (training) graph; defaults to --k")
    ap.add_argument("--k-private", type=int, default=None,
                    help="locality for the private (test) graph; defaults to --k. "
                         "Differing values get a k<pub>k<priv> tag so they cannot "
                         "overwrite the single-k datasets or share their cache.")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--tag", default=None, help="filename tag; defaults to pub2priv")
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC,
                    help="share of supernodes held out for test (default 0.2: an 80/20 split)")
    ap.add_argument("--no-hops", action="store_true",
                    help="ablation: drop h1..hk, leaving the condition [n, rho_in, k]. "
                         "Appends _nohop to the tag so it cannot overwrite the main "
                         "datasets and gets its own processed cache.")
    ap.add_argument("--no-cond", action="store_true",
                    help="plain DiGress baseline: no condition at all (y of width 0); "
                         "only the size n_i is imposed, at sampling. Appends _uncond.")
    args = ap.parse_args()
    if args.no_hops and args.no_cond:
        ap.error("--no-hops and --no-cond are alternatives")

    cond = COND_NONE if args.no_cond else COND_NO_HOPS if args.no_hops else COND
    k_pub = args.k_public if args.k_public is not None else args.k
    k_priv = args.k_private if args.k_private is not None else args.k
    # One condition width for both graphs, or the two could not be fed to the same
    # model. max(), not k_pub: a private graph coarsened MORE loosely than the
    # public one would occupy rings the public width does not have, and
    # extract_pairs refuses to silently truncate them.
    k_hist = max(k_pub, k_priv)

    ktag = f"k{k_pub}" if k_pub == k_priv else f"k{k_pub}k{k_priv}"
    pct = int(round(100 * (1 - args.test_frac)))
    tag = args.tag or f"{args.public[:3]}2{args.private[:4]}_{ktag}_s{pct}"
    if args.no_hops and not args.tag:
        tag += "_nohop"
    if args.no_cond and not args.tag:
        tag += "_uncond"
    torch.manual_seed(SEED)

    print(f"public  = {args.public} (k={k_pub})   private = {args.private} (k={k_priv})")
    if k_pub != k_priv:
        print(f"  differing localities -> hop histogram is {k_hist} wide for both; "
              f"rings beyond each graph's own k are provably empty")
    print("extracting supernode pairs")
    pub = extract_pairs(args.public, k_pub, cond=cond, min_n=MIN_N, k_hist=k_hist)
    priv = extract_pairs(args.private, k_priv, cond=cond, min_n=MIN_N, k_hist=k_hist)

    names = cond_names(cond, k_hist)
    print(f"\ncondition ({len(names)} dims): {names}")
    print(f"  {args.public:<12} clusters: {describe_sizes(pub)}")
    print(f"  {args.private:<12} clusters: {describe_sizes(priv)}")
    pm = np.median([r["n"] for r in pub])
    qm = np.median([r["n"] for r in priv])
    if max(pm, qm) / max(min(pm, qm), 1) > 1.5:
        print(f"  NOTE: private clusters are {qm/pm:.1f}x the public median at the same k.")
        print("  This is a real property of the transfer problem (density differs), not")
        print("  an artefact -- report it alongside the results.")

    pub_tr, pub_te = split_by_supernode(pub, args.test_frac, SEED)
    priv_tr, priv_te = split_by_supernode(priv, args.test_frac, SEED)
    # The split depends on the supernodes only, so the hop and no-hop datasets
    # (and D_inter) get the same one; writing it again is harmless.
    write_split(args.public, k_pub, pub, pub_te, args.test_frac, SEED, SPLIT_DIR)
    write_split(args.private, k_priv, priv, priv_te, args.test_frac, SEED, SPLIT_DIR)
    print(f"  private split: {len(priv_tr)} train / {len(priv_te)} test supernodes "
          f"(B and C share this test set); no validation split\n")
    pub_val = val_from_train(pub_tr, VAL_FROM_TRAIN, SEED)

    common = {"cond_names": names, "k": k_hist, "k_public": k_pub, "k_private": k_priv,
              "public": args.public, "private": args.private}

    # ---- A: train public, test public ------------------------------------
    A = dict(common, config="A", desc=f"train {args.public}, test {args.public}",
             splits=[list(pub_tr), list(pub_val), list(pub_te)])
    mean, std, const = standardise(A["splits"][0], A["splits"])
    A.update(cond_mean=mean, cond_std=std, cond_constant=const)
    save(os.path.join(args.out_dir, f"pairs_A_{tag}.pt"), A)

    # ---- B: train public, test private -----------------------------------
    # Zero private data: B's training records are A's exactly, so A's model is
    # B's model. Conditions are standardised with PUBLIC statistics -- the only
    # scale the model has seen.
    B = dict(common, config="B", desc=f"train {args.public}, test {args.private}",
             splits=[list(pub_tr), list(pub_val), list(priv_te)])
    mean, std, const = standardise(B["splits"][0], B["splits"])
    B.update(cond_mean=mean, cond_std=std, cond_constant=const)
    save(os.path.join(args.out_dir, f"pairs_B_{tag}.pt"), B)

    # ---- C: train public + private, test private -------------------------
    n_pub = len(pub_tr)
    reps = 1
    if OVERSAMPLE_TARGET > 0 and priv_tr:
        want = OVERSAMPLE_TARGET / (1 - OVERSAMPLE_TARGET) * n_pub
        reps = max(1, int(round(want / len(priv_tr))))
    priv_train = list(priv_tr) * reps
    combined = list(pub_tr) + priv_train
    np.random.default_rng(SEED).shuffle(combined)

    C = dict(common, config="C",
             desc=f"train {args.public}+{args.private}, test {args.private}",
             private_oversample_factor=reps,
             splits=[combined, val_from_train(combined, VAL_FROM_TRAIN, SEED), list(priv_te)])
    mean, std, const = standardise(C["splits"][0], C["splits"])
    C.update(cond_mean=mean, cond_std=std, cond_constant=const)
    save(os.path.join(args.out_dir, f"pairs_C_{tag}.pt"), C)

    print(f"\n  config C: {len(priv_tr)} private train supernodes repeated x{reps} "
          f"-> {len(priv_train)} of {len(combined)} "
          f"({100*len(priv_train)/len(combined):.0f}% of the training set)")

    if const.any():
        flagged = [names[i] for i in range(len(names)) if bool(const[i])]
        print(f"\n  NOTE: {flagged} had ~zero variance in C's training and were left")
        print("  unstandardised rather than divided by an epsilon. A constant feature")
        print("  cannot inform the model, but this keeps it numerically harmless.")

    if k_pub != k_priv:
        # B trains public-only, so k is constant there and left on its raw scale;
        # the private test records then arrive at (k_priv - k_pub). Print the
        # number rather than describing it, so it is checkable against the log.
        print("\n  NOTE: k differs between train and test. In B, k has zero training")
        print(f"  variance, so the private test records enter the model at "
              f"{k_priv - k_pub:+d} on")
        print("  the k column where training was always 0. C's training set contains")
        print("  both localities, so there k is standardised in the ordinary way.")

    print(f"\nDataset configs to create: configs/dataset/cluster_{{A,B,C}}_{tag}.yaml")


if __name__ == "__main__":
    main()
