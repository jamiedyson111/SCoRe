"""ER (BS1) and SC-RG (BS2) on the fixed test groups of Collab and YouTube, LOCAL (`kmis` env):

    python group_baselines.py intra --pairs ../D_intra/data/cluster_lux/raw/pairs_B_dblp2coll_k3_s80.pt \\
        --groups ../results/competitor_data/collab/test_groups_d_intra.json \\
        --out-dir ../results/samples/D_intra/collab_groups
    python group_baselines.py inter --pairs ../D_inter/data/pair_lux/raw/pairs_inter_C_dblp2coll_k3_s80_deg.pt \\
        --groups ../results/competitor_data/collab/test_groups_d_inter.json \\
        --out-dir ../results/samples/D_inter/collab_groups

Huiping's generators (competitors/Baseline/), called block by block on the instances of
the groups only, one generation (seed 42, her run 0), so they are scored exactly like our
models (repeat_summary.py / transfer_summary.py --groups):

  intra  BS1 ER_intra.sample_gnm_adjacency (m_i edges uniformly);
         BS2 SC_RG_intra.build_hop_aware_graph (hop layers from h_i, m_i edges);
         written in the conditional_samples format with the true centroid marked, as
         BS1_s0.txt / BS2_s0.txt.
  inter  BS1 ER_inter.sample_bipartite_edges (w_ij edges uniformly);
         BS2 SC_RG_inter.make_i_side_degree_sequence + sample_edges_from_i_degrees from
         the i-side cross-degree profile in HER bins 0, 1, 2, 3, 4+ (what her generator
         takes), measured on the real block -- Collab's stored r uses 0, 1, 2-3, 4-7, 8+
         and cannot be passed as is; written as BS1_s0.pt / BS2_s0.pt (pair_samples format).
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "competitors", "Baseline"))
import ER_inter, ER_intra, SC_RG_inter, SC_RG_intra  # noqa: E402,E401

SEED = 42


def load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def intra(args, blob, groups):
    want = {int(s) for g in groups for s in g}
    items = [r for r in blob["splits"][2] if int(r["sn"]) in want]
    if len(items) != len(want):
        raise SystemExit(f"{len(want) - len(items)} group supernodes are not in the test split")
    items.sort(key=lambda r: int(r["sn"]))
    for name in ("BS1", "BS2"):
        rng = random.Random(SEED)
        path = os.path.join(args.out_dir, f"{name}_s0.txt")
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                n, m = int(item["n"]), ER_intra.infer_m(item)
                if name == "BS1":
                    adj = ER_intra.sample_gnm_adjacency(n, m, rng)
                else:
                    _, _, k, h = SC_RG_intra.get_raw_condition(item)
                    adj, _ = SC_RG_intra.build_hop_aware_graph(
                        n=n, m=m, centroid=SC_RG_intra.get_centroid_local_index(item),
                        layer_counts=SC_RG_intra.h_to_layer_counts(h=h, n=n), rng=rng)
                ER_intra.write_block(f, item, adj)
        print(f"  {path}: {len(items)} supernodes")


def i_side_profile(rec):
    """r_{i->j} in her bins 0, 1, 2, 3, 4+, from the record's real cross block."""
    n_i = int(rec["n_i"])
    src, dst = rec["edge_index"][0].numpy(), rec["edge_index"][1].numpy()
    cross = (src < n_i) & (dst >= n_i)
    d = np.bincount(src[cross], minlength=n_i)[:n_i]
    if int(d.sum()) != int(rec["w"]):
        raise SystemExit(f"pair ({rec['sn_i']}, {rec['sn_j']}): cross block has {d.sum()} edges, w = {rec['w']}")
    return (np.bincount(np.clip(d, 0, 4), minlength=5) / n_i).tolist()


def inter(args, blob, groups):
    test = blob["splits"][2]
    pairs = sorted({p["idx"] for g in groups for p in g})
    for name in ("BS1", "BS2"):
        rng = random.Random(SEED)
        blocks = []
        for idx in pairs:
            r = test[idx]
            n_i, n_j, w = int(r["n_i"]), int(r["n_j"]), int(r["w"])
            if name == "BS1":
                ei = ER_inter.sample_bipartite_edges(n_i, n_j, w, rng)
            else:
                deg, _ = SC_RG_inter.make_i_side_degree_sequence(n_i=n_i, n_j=n_j, w=w, r=i_side_profile(r), rng=rng)
                ei = SC_RG_inter.sample_edges_from_i_degrees(deg, n_j, rng)
            blocks.append({"idx": idx, "sn_i": int(r["sn_i"]), "sn_j": int(r["sn_j"]),
                           "n_i": n_i, "n_j": n_j, "w": w, "edge_index": torch.as_tensor(ei, dtype=torch.long)})
        path = os.path.join(args.out_dir, f"{name}_s0.pt")
        torch.save({"pairs_file": os.path.basename(args.pairs), "seed": SEED,
                    "checkpoint": f"{name} ({'ER' if name == 'BS1' else 'SC-RG'}) baseline", "blocks": blocks}, path)
        print(f"  {path}: {len(blocks)} pairs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=("intra", "inter"))
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    blob, groups = load(args.pairs), json.load(open(args.groups))["groups"]
    (intra if args.model == "intra" else inter)(args, blob, groups)


if __name__ == "__main__":
    main()
