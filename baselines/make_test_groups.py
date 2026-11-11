"""The fixed test groups every method is evaluated on (Collab, YouTube), LOCAL:

    python make_test_groups.py

Five NON-OVERLAPPING groups per graph and model, drawn once with seed 0 and saved
next to the competitor data (results/competitor_data/<graph>/):

  test_groups_d_intra.json  {"groups": [[supernode id, ...] x 5], ...}
      Collab: its 1,787 test supernodes allow 5 non-overlapping groups of 357 only
      (5 x 1,000 would need 5,000); YouTube: 5 x 1,000 of its 11,016.
  test_groups_d_inter.json  {"groups": [[{"idx", "sn_i", "sn_j"}, ...] x 5], ...}
      Collab: 5 x 1,000 of the 34,768 test pairs (pairs with at least one test
      supernode, both supernodes n >= 3); idx is the pair's position in the test
      split of D_inter/data/pair_lux/raw/pairs_inter_{B,C}_dblp2coll_k3_s80_deg.pt
      (the same pairs in the same order in both).
Supernode ids are those of the partition .npz (nodes.csv, *_supernodes.jsonl).
"""
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "competitor_data")
SPLITS = os.path.join(ROOT, "D_intra", "data", "splits")
GROUPS, SEED = 5, 0


def groups_of(items, size):
    perm = np.random.default_rng(SEED).permutation(len(items))
    return [[items[i] for i in sorted(perm[g * size:(g + 1) * size])] for g in range(GROUPS)]


def save(graph, name, blob):
    os.makedirs(os.path.join(OUT, graph), exist_ok=True)
    path = os.path.join(OUT, graph, name)
    json.dump(blob, open(path, "w"))
    sizes = [len(g) for g in blob["groups"]]
    print(f"  {path}: {GROUPS} groups of {sizes}")


def d_intra(graph, split_file, size):
    test = sorted(json.load(open(os.path.join(SPLITS, split_file)))["test"])
    size = min(size, len(test) // GROUPS)
    save(graph, "test_groups_d_intra.json",
         {"graph": graph, "what": "D_intra test supernodes", "seed": SEED,
          "non_overlapping": True, "of": len(test), "groups": groups_of(test, size)})


def d_inter(graph, pairs_file, size):
    test = torch.load(pairs_file, map_location="cpu", weights_only=False)["splits"][2]
    items = [{"idx": i, "sn_i": int(r["sn_i"]), "sn_j": int(r["sn_j"])} for i, r in enumerate(test)]
    save(graph, "test_groups_d_inter.json",
         {"graph": graph, "what": "D_inter test pairs (at least one test supernode)",
          "seed": SEED, "non_overlapping": True, "of": len(items),
          "pairs_file": os.path.basename(pairs_file), "groups": groups_of(items, size)})


if __name__ == "__main__":
    d_intra("collab", "collab_k3_s80.json", 1000)
    d_intra("youtube", "youtube_k4_s80.json", 1000)
    d_inter("collab", os.path.join(ROOT, "D_inter", "data", "pair_lux", "raw",
                                   "pairs_inter_C_dblp2coll_k3_s80_deg.pt"), 1000)
