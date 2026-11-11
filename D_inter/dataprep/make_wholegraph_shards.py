"""Shards of the Collab D_inter pairs still to generate for whole-graph reconstruction, LOCAL.

    python make_wholegraph_shards.py [--shards 6]
    python make_wholegraph_shards.py --group 1 --shards 2

--group K (Huiping, 9 Nov, one GPU for Collab): whole graph with only the 357 test
supernodes of D_intra group K (test_groups_d_intra.json) generated; all other supernodes,
test ones included, keep their original structure. EVERY pair touching group K is generated
(none reused from coll_pairs_*, whose context has all test supernodes generated):
    data/wholegraph/collab_group<K>_shard<k>.json, collab_group<K>_supernodes.json

Whole-graph protocol (Huiping, 9 Nov 2026): D_inter is run on EVERY pair with at least
one D_intra test supernode; all other supernodes and pairs keep their original structure.
Those pairs are exactly the D_inter test split (34,768 pairs, both supernodes n >= 3).
The 5 x 1,000 evaluation groups (test_groups_d_inter.json) are already sampled by
coll_pairs_{B,C}; this writes the REMAINING pairs as shards, balanced by estimated cost
(~ (n_i + n_j)^2, from bench_pair_sampling.py), in the sampler's +subset_file format:

    data/wholegraph/collab_rest_shard<k>.json   {"groups": [[{"idx", "sn_i", "sn_j"}, ...]]}
"""
import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--pairs", default=os.path.join(HERE, "..", "data", "pair_lux", "raw",
                                                     "pairs_inter_C_dblp2coll_k3_s80_deg.pt"))
    ap.add_argument("--groups", default=os.path.join(ROOT, "results", "competitor_data", "collab",
                                                      "test_groups_d_inter.json"))
    ap.add_argument("--group", type=int, default=None, help="1-based D_intra group")
    ap.add_argument("--intra-groups", default=os.path.join(ROOT, "results", "competitor_data", "collab",
                                                            "test_groups_d_intra.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data", "wholegraph"))
    args = ap.parse_args()

    test = torch.load(args.pairs, map_location="cpu", weights_only=False)["splits"][2]
    if args.group:
        sel = set(json.load(open(args.intra_groups))["groups"][args.group - 1])
        done = set()
        rest = [i for i, r in enumerate(test) if int(r["sn_i"]) in sel or int(r["sn_j"]) in sel]
        prefix = f"collab_group{args.group}"
    else:
        done = {p["idx"] for g in json.load(open(args.groups))["groups"] for p in g}
        rest = [i for i in range(len(test)) if i not in done]
        prefix = "collab_rest"
    cost = lambda i: max(1.7e-4 * (int(test[i]["n_i"]) + int(test[i]["n_j"])) ** 2, 0.3)
    shards, load = [[] for _ in range(args.shards)], [0.0] * args.shards
    for i in sorted(rest, key=cost, reverse=True):          # greedy: largest first
        k = load.index(min(load))
        shards[k].append(i)
        load[k] += cost(i)

    os.makedirs(args.out, exist_ok=True)
    if args.group:
        json.dump({"graph": "collab", "group": args.group, "supernodes": sorted(sel)},
                  open(os.path.join(args.out, f"{prefix}_supernodes.json"), "w"))
    print(f"{len(test)} test pairs, {len(done)} already sampled, {len(rest)} to generate")
    for k, idx in enumerate(shards):
        items = [{"idx": i, "sn_i": int(test[i]["sn_i"]), "sn_j": int(test[i]["sn_j"])} for i in sorted(idx)]
        path = os.path.join(args.out, f"{prefix}_shard{k}.json")
        json.dump({"graph": "collab", "what": f"whole-graph D_inter pairs, shard {k} of {args.shards}",
                   "pairs_file": os.path.basename(args.pairs), "groups": [items]}, open(path, "w"))
        print(f"  {path}: {len(items)} pairs, ~{load[k] * 31.29 / 28.21 / 3600:.1f} GPU-h (likely)")


if __name__ == "__main__":
    main()
