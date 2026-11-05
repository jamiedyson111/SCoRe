"""lux_regions.pkl -> the networkx pickles ICLR [2]'s `file` dataset loads.

Run on BlueBEAR in the graph-generation env, from this directory:

    python convert_data.py --repo $REPO

Writes two files into $REPO/data/ (main.py reads data/<dataset.name>.pkl):

    lux_regions.pkl          TRAINING. train = the 100 Luxembourg regions,
                             val = 4 of the 20 val regions, test = 2 more.
                             Everything is Luxembourg -- configuration B's
                             footing, nothing private reaches training or
                             model selection. Val is cut to 4 because every
                             validation samples it once per EMA beta.
    lux_regions_olde5.pkl    SAMPLING. test = 5 copies of Oldenburg, so test mode
                             generates five graphs of exactly 6105 nodes. Only
                             their size is taken from it; the generated graphs
                             are scored by syn_repeats_summary.py, not by ICLR
                             [2]'s own metrics.
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
    ap.add_argument("--repo", required=True, help="the cloned graph-generation repo")
    ap.add_argument("--regions", default=os.path.join(HERE, "..", "data", "lux_regions.pkl"))
    args = ap.parse_args()

    d = pickle.load(open(args.regions, "rb"))
    train = [to_nx(r) for r in d["train"]]
    val = [to_nx(r) for r in d["val"]]
    olde = to_nx(d["test"][0])

    out = os.path.join(args.repo, "data")
    with open(os.path.join(out, "lux_regions.pkl"), "wb") as f:
        pickle.dump({"train": train, "val": val[:4], "test": val[4:6]}, f)
    with open(os.path.join(out, "lux_regions_olde5.pkl"), "wb") as f:
        pickle.dump({"train": train, "val": val[:4], "test": [olde.copy() for _ in range(5)]}, f)
    print(f"train {len(train)}, val 4, test 2 (Luxembourg)  -> {out}/lux_regions.pkl")
    print(f"test 5 x Oldenburg ({olde.number_of_nodes()} nodes)  -> {out}/lux_regions_olde5.pkl")


if __name__ == "__main__":
    main()
