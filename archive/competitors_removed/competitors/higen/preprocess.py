"""Build HiGen's 2-level hierarchies of the 120 Luxembourg regions, in parallel.

    python preprocess.py --repo $REPO [--workers 32] [--only 0]

HiGen builds these itself, one graph at a time, at the start of training; with
spectral coarsening on 2000-6000-node graphs that is hours inside a GPU job. This
does the same per-graph work with the repo's own get_HG_of_orders, with the same
arguments its loader passes, and writes the same cache files
(data/LuxRegions2/LuxRegions2_<i>_BFSDC_multilevel_SC0.p). HiGen's loader then
finds them and skips the work. --only i builds one graph, for a timing test.
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

KW = None


def build(i):
    fn = f"data/LuxRegions2/LuxRegions2_{i}_BFSDC_multilevel_SC0.p"
    if os.path.exists(fn):
        return i, "cached", 0.0, None
    from utils.dataset.dataset_preprocessing import get_HG_of_orders
    G = pickle.load(open("data/LuxRegions2/lux_regions_nx.pkl", "rb"))[i]
    t = time.time()
    hg = get_HG_of_orders(G, num_levels=2, node_orders=["BFSDC"], g_id=f"lx{i}", **KW)[0]
    with open(fn + ".tmp", "wb") as f:
        pickle.dump(hg, f)
    os.replace(fn + ".tmp", fn)
    # largest community: HiGen's model is sized for at most max_num_nodes (500)
    parts = [p.num_nodes for p in hg[-1].part_ls] if hasattr(hg[-1], "part_ls") else []
    return i, "built", time.time() - t, max(parts) if parts else None


def main():
    global KW
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", type=int)
    args = ap.parse_args()
    os.chdir(args.repo)
    sys.path.insert(0, args.repo)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    # exactly what get_multilevel_graph_dataset hands get_HG_of_orders for LuxRegions2
    KW = dict(graph_name="LuxRegions2", coarsening_alg="SC0", level_startFromRoot=False,
              load_afew=np.inf,
              splice_out_inds={2: [], 3: [1], 4: [0, 2], 5: [0, 2, 3], 6: [0, 2, 3, 4],
                               7: [0, 2, 3, 4, 5]})
    n = len(pickle.load(open("data/LuxRegions2/lux_regions_nx.pkl", "rb")))
    idx = [args.only] if args.only is not None else list(range(n))
    with Pool(min(args.workers, len(idx))) as pool:
        for i, status, dt, big in pool.imap_unordered(build, idx):
            print(f"graph {i:3d}: {status} {dt:7.1f}s  largest community {big}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
