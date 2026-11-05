"""Oldenburg-only datasets: C as a FINE-TUNE of A, and plain DiGress on Oldenburg 80/20.

    python make_finetune_datasets.py              # after make_transfer_datasets.py (and --no-hops, --no-cond)

Built from the existing A / B / C files, so the supernodes, the split and the
records are exactly theirs; nothing is re-extracted.

  Cft  (pairs_Cft_<tag>[_nohop].pt)   C, fine-tuned: A's model, trained further on the
       254 Oldenburg TRAINING supernodes only (no oversampling, no Luxembourg).
       Conditions are standardised with A's statistics and the noise process keeps
       A's edge marginal (`edge_marginal`), so the fine-tune starts from exactly the
       model A trained. Test: the 63 Oldenburg test supernodes, as B and C.
  O    (pairs_O_<tag>_uncond.pt)      plain DiGress, no condition, trained from scratch
       on the same 254 Oldenburg training supernodes and tested on the same 63.

The Oldenburg training records are C's (C holds each one 9 times; one copy is
kept). The "val" slot holds training records again (at most VAL_FROM_TRAIN, a seeded
sample), for logging only.
"""
import argparse
import copy
import os

import numpy as np
import torch

TAG = "lux2olde_k7_s80"
VAL_FROM_TRAIN = 256      # as make_transfer_datasets.py: the val slot is for logging only
RAW = "../data/cluster_lux/raw"


def load(name):
    return torch.load(os.path.join(RAW, name))


def save(name, blob):
    path = os.path.join(RAW, name)
    torch.save(blob, path)
    tr, va, te = (len(s) for s in blob["splits"])
    print(f"  {name:<36} {tr:>5} train /{va:>4} val /{te:>4} test   ({os.path.getsize(path)/1e6:.1f} MB)")


def private_train(C):
    """One copy of each Oldenburg training record in C, by supernode id."""
    seen = {}
    for r in C["splits"][0]:
        if r["graph"] == C["private"] and int(r["sn"]) not in seen:
            seen[int(r["sn"])] = r
    return [copy.deepcopy(seen[s]) for s in sorted(seen)]


def val_slot(tr):
    """The training records themselves when there are few (Oldenburg: all 254), else a
    seeded sample of VAL_FROM_TRAIN of them, so logging stays cheap on a large graph."""
    if len(tr) <= VAL_FROM_TRAIN:
        return list(tr)
    idx = np.random.default_rng(1).permutation(len(tr))[:VAL_FROM_TRAIN]
    return [tr[i] for i in sorted(idx)]


def edge_marginal(recs):
    """[no-edge, edge] over ordered node pairs, as AbstractDataModule.edge_counts."""
    pairs = sum(int(r["n"]) * (int(r["n"]) - 1) for r in recs)
    edges = sum(int(r["edge_index"].size(1)) for r in recs)
    return torch.tensor([pairs - edges, edges], dtype=torch.float) / pairs


def restandardise(recs, mean, std):
    for r in recs:
        r["y"] = (r["y_raw"] - mean) / std
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG)
    args = ap.parse_args()

    for suffix in ("", "_nohop"):
        A, B, C = (load(f"pairs_{c}_{args.tag}{suffix}.pt") for c in "ABC")
        tr = restandardise(private_train(C), A["cond_mean"], A["cond_std"])
        assert len(tr) == len(set(int(r["sn"]) for r in tr))
        assert not ({int(r["sn"]) for r in tr} & {int(r["sn"]) for r in B["splits"][2]})
        marg = edge_marginal(A["splits"][0])
        Cft = {k: v for k, v in C.items() if k not in ("splits", "private_oversample_factor")}
        Cft.update(config="Cft", finetune_of="A",
                   desc=f"A's model fine-tuned on {C['private']} train, test {C['private']}",
                   cond_mean=A["cond_mean"], cond_std=A["cond_std"],
                   cond_constant=A["cond_constant"], edge_marginal=marg,
                   splits=[tr, val_slot(tr), list(B["splits"][2])])
        save(f"pairs_Cft_{args.tag}{suffix}.pt", Cft)
        print(f"    A's edge marginal kept: {[round(float(x), 4) for x in marg]} "
              f"({C['private']} train alone: {[round(float(x), 4) for x in edge_marginal(tr)]})")

    A, B, C = (load(f"pairs_{c}_{args.tag}_uncond.pt") for c in "ABC")
    tr = private_train(C)
    O = {k: v for k, v in C.items() if k not in ("splits", "private_oversample_factor")}
    O.update(config="O", desc=f"plain DiGress, train {C['private']}, test {C['private']}",
             splits=[tr, val_slot(tr), list(B["splits"][2])])
    save(f"pairs_O_{args.tag}_uncond.pt", O)
    print("\nDataset configs: configs/dataset/cluster_Cft_<tag>[_nohop].yaml, cluster_O_<tag>_uncond.yaml")


if __name__ == "__main__":
    main()
