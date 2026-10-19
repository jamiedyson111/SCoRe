"""lux_regions.pkl -> $REPO/data/LuxRegions2/lux_regions_nx.pkl, HiGen's input.

    python convert_data.py --repo $REPO

A list of 120 networkx graphs, the 100 TRAIN regions first and the 20 val after,
so HiGen's own split (train_ratio 0.834 of 120 -> the first 100) trains on
exactly the regions ICLR [2] trains on. Nothing from Oldenburg is included.
"""
import argparse
import os
import pickle

import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))


def to_nx(rec):
    G = nx.Graph()
    G.add_nodes_from(range(rec["n"]))
    G.add_edges_from(map(tuple, rec["edges"].tolist()))
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--regions", default=os.path.join(HERE, "..", "data", "lux_regions.pkl"))
    args = ap.parse_args()
    d = pickle.load(open(args.regions, "rb"))
    graphs = [to_nx(r) for r in d["train"]] + [to_nx(r) for r in d["val"]]
    out = os.path.join(args.repo, "data", "LuxRegions2")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "lux_regions_nx.pkl"), "wb") as f:
        pickle.dump(graphs, f, protocol=4)
    print(f"{len(graphs)} graphs (100 train first) -> {out}/lux_regions_nx.pkl")


if __name__ == "__main__":
    main()
