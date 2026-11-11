"""E_h for samples that mark no centroid (plain DiGress), with a centre chosen after
generation, plus the checks that make the number interpretable.

    python eh_posthoc.py                                   # Oldenburg
    python eh_posthoc.py --tag lux2verm_k7_s80 --samples-dir ../../results/samples/D_intra/s80_verm

The rule: the GRAPH CENTRE -- within the largest connected component, the node of
smallest eccentricity (greatest shortest-path distance to the others); ties go to the
highest degree, then the lowest index. E_h is then repeat_summary's JS form from that
node, against each test supernode's true h_i.

Columns, mean ± std over seeds:
  E_h (marked centroid)   the table's E_h; blank for DiGress, which marks none
  E_h (graph centre)      the rule applied to every configuration alike
  E_h (best-matching node)  the node whose profile is closest to h_i: optimistic,
                          the same for every configuration, 0 on the real supernodes
and a Real row: the graph-centre rule applied to the REAL test supernodes, i.e. how
far that rule's centre is from the true k-MIS centroid (the floor it puts under E_h).
"""
import argparse
import glob
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from evaluate_clusters import centroid_index, dense_adj, hop_hist
from make_intra_table import load_column
from repeat_summary import CONFIGS, hop_targets, js_hop, seed_of


def graph_centre(A):
    n = A.shape[0]
    if n <= 1:
        return 0
    _, lab = connected_components(csr_matrix(A), directed=False)
    keep = np.flatnonzero(lab == np.bincount(lab).argmax())
    D = shortest_path(csr_matrix(A[np.ix_(keep, keep)]), unweighted=True, directed=False)
    ecc = D.max(axis=1)
    deg = A[np.ix_(keep, keep)].sum(axis=1)
    best = np.lexsort((np.arange(len(keep)), -deg, ecc))[0]
    return int(keep[best])


def eh(graphs, h_target, k, centre_of):
    return float(np.mean([js_hop(hop_hist(A, centre_of(A, x), k), h_target[sn])
                          for A, x, sn in graphs]))


def eh_best(graphs, h_target, k):
    """E_h from the best-matching node: the smallest JS over every node as the centre.
    An optimistic bound, the same for every model; 0 on the real supernodes."""
    out = []
    for A, _, sn in graphs:
        D = shortest_path(csr_matrix(A), unweighted=True, directed=False)
        n = A.shape[0]
        best = 1.0
        for c in range(n):
            h = np.array([np.count_nonzero(D[c] == r) for r in range(1, k + 1)]) / max(n - 1, 1)
            best = min(best, js_hop(h, h_target[sn]))
        out.append(best)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="../../results/samples/D_intra/s80")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--tag", default="lux2olde_k7_s80")
    ap.add_argument("--configs", nargs="+", default=["O_uncond", "T_v3", "Cft_v3", "B_v3"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    h_target = hop_targets(os.path.join(args.pairs_dir, f"pairs_B_{args.tag}.pt"))
    k = len(next(iter(h_target.values())))
    rows, real_done = [], False
    for cfg in args.configs:
        label, letter, psuffix = CONFIGS[cfg]
        pairs = os.path.join(args.pairs_dir, f"pairs_{letter}_{args.tag}{psuffix}.pt")
        marked, centre, best = [], [], []
        for f in sorted(glob.glob(os.path.join(args.samples_dir, f"{cfg}_s*.txt")), key=seed_of):
            col = load_column(f, pairs)
            gen = [(g["A"], g["x"], g["supernode"]) for g, _ in col["pairs"]]
            if not real_done:
                real = [(dense_adj(r["edge_index"], int(r["n"])), None, int(r["sn"]))
                        for _, r in col["pairs"]]
                floor = eh(real, h_target, k, lambda A, x: graph_centre(A))
                rows.append(("Real", "0.0000", f"{floor:.4f}", f"{eh_best(real, h_target, k):.4f}"))
                real_done = True
            if all(centroid_index(x) is not None for _, x, _ in gen):
                marked.append(eh(gen, h_target, k, lambda A, x: centroid_index(x)))
            centre.append(eh(gen, h_target, k, lambda A, x: graph_centre(A)))
            best.append(eh_best(gen, h_target, k))
        fmt = lambda v: f"{np.mean(v):.4f} ± {np.std(v, ddof=1):.4f}" if v else ""
        rows.append((label, fmt(marked), fmt(centre), fmt(best)))
        print(f"  {cfg}: {len(centre)} runs")

    out = ["Configuration\tE_h (marked centroid)\tE_h (graph centre)\tE_h (best-matching node)"] + ["\t".join(r) for r in rows]
    print("\n".join(out))
    if args.out:
        open(args.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
