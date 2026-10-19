"""Training data for the full-graph competitors (ICLR [2], HiGen): Luxembourg regions.

Run LOCALLY in the `kmis` env:

    python make_lux_regions.py           # -> data/lux_regions.pkl

The competitors are unconditional generators trained on a SET of graphs, and are
compared with configuration B: trained on the public graph only, evaluated on the
private one. So the training set is cut from Luxembourg alone: 120 connected
regions, each the first T nodes reached by a breadth-first search from a random
node -- a compact patch of road network -- with T uniform in [2000, 6000], around
Oldenburg's 6105. Regions may overlap. 100 train, 20 val. Oldenburg is stored as
the single TEST graph; it is never trained on, only used to set the generated
size (6105) and to score against.

Stored as plain numpy, not networkx, so the file loads in every competitor's
environment whatever its networkx version:

    {'train' | 'val' | 'test': [ {'n': int, 'edges': int32 (m, 2), u < v} ],
     'source', 'seed', 'size_range'}

Each competitor's own converter (competitors/<method>/) turns this into its
format.
"""
import os
import pickle
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXTRACTION = os.path.join(HERE, "..", "D_intra", "extraction")
N_TRAIN, N_VAL = 100, 20
SIZE_RANGE = (2000, 6000)
SEED = 0


def load(graph):
    cwd = os.getcwd()
    sys.path.insert(0, os.path.abspath(EXTRACTION))
    os.chdir(EXTRACTION)
    try:
        from kmis_pairs import load_graph
        _, A = load_graph(graph)
    finally:
        os.chdir(cwd)
    return A.tocsr()


def bfs_region(A, start, size):
    seen, order, q = {start}, [start], deque([start])
    while q and len(order) < size:
        u = q.popleft()
        for v in A.indices[A.indptr[u]:A.indptr[u + 1]]:
            if v not in seen:
                seen.add(v)
                order.append(v)
                q.append(v)
                if len(order) == size:
                    break
    return np.array(order)


def as_record(A, nodes):
    sub = A[nodes][:, nodes].tocoo()
    keep = sub.row < sub.col
    return {"n": len(nodes),
            "edges": np.stack([sub.row[keep], sub.col[keep]], 1).astype(np.int32)}


def main():
    rng = np.random.default_rng(SEED)
    lux = load("luxembourg")
    regions = []
    while len(regions) < N_TRAIN + N_VAL:
        size = int(rng.integers(SIZE_RANGE[0], SIZE_RANGE[1] + 1))
        nodes = bfs_region(lux, int(rng.integers(lux.shape[0])), size)
        if len(nodes) == size:                  # a start in a small component falls short
            regions.append(as_record(lux, nodes))
    olde = load("oldenburg")
    out = {"train": regions[:N_TRAIN], "val": regions[N_TRAIN:],
           "test": [as_record(olde, np.arange(olde.shape[0]))],
           "source": "luxembourg BFS regions; test = oldenburg",
           "seed": SEED, "size_range": SIZE_RANGE}
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "lux_regions.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)

    for split in ("train", "val", "test"):
        ns = np.array([r["n"] for r in out[split]])
        ms = np.array([len(r["edges"]) for r in out[split]])
        print(f"{split:>5}: {len(ns):3d} graphs, n {ns.min()}-{ns.max()} (mean {ns.mean():.0f}), "
              f"mean degree {np.mean(2 * ms / ns):.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
