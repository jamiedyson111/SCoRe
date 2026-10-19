"""Paired bootstrap between two sampled runs scored on the SAME supernodes.

    python _bootstrap_pair.py A A_nohop lux2olde_k7 lux2olde_k7_nohop
    python _bootstrap_pair.py oldenburg/B oldenburg/B_v3 lux2olde_k7 lux2olde_k7

Resamples the supernode index and recomputes both sides from the same draw, so
the between-cluster variance cancels and what is left is the difference between
the two models. A CI that straddles zero means the difference is not
distinguishable from sampling noise, whichever way it points.
"""
import sys

import numpy as np
import torch

from evaluate_clusters import (agg, dense_adj, describe, dist_to_real,
                               parse_generated)

RNG = np.random.default_rng(0)
N_BOOT = 2000

L, R, TAG_L, TAG_R = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]


def load(name, tag):
    gen = parse_generated(f"../../results/samples/D_intra/{name}.txt")
    cfg = name.split("/")[-1].split("_")[0]
    blob = torch.load(f"../data/cluster_lux/raw/pairs_{cfg}_{tag}.pt")
    test = {int(r["sn"]): r for r in blob["splits"][2]}
    pairs = [(g, test[g["supernode"]]) for g in gen if g.get("supernode") in test]
    pairs.sort(key=lambda t: t[0]["supernode"])
    return pairs


pl, pr = load(L, TAG_L), load(R, TAG_R)
assert [g["supernode"] for g, _ in pl] == [g["supernode"] for g, _ in pr], \
    "the two runs are not on the same supernodes -- not a paired comparison"
n = len(pl)

gl = [describe(g["A"]) for g, _ in pl]
gr = [describe(g["A"]) for g, _ in pr]
rs = [describe(dense_adj(r["edge_index"], int(r["n"]))) for _, r in pl]
nreq = np.array([float(r["n"]) for _, r in pl])
m_true = np.array([s["m"] for s in rs], float)

print(f"\n{L}  vs  {R}      {n} paired supernodes")
print(f"{'metric':<20}{'real':>9}{L:>11}{R:>11}{'diff':>9}{'95% CI on diff':>22}")
print("-" * 82)

for label, key in [("connected %", "connected"), ("acyclic %", "forest"),
                   ("components", "components"), ("edges (mean)", "mean_m"),
                   ("density", "mean_density"), ("node degree", "mean_deg")]:
    a, b, rv = agg(gl)[key], agg(gr)[key], agg(rs)[key]
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        draws[i] = agg([gr[j] for j in idx])[key] - agg([gl[j] for j in idx])[key]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{label:<20}{rv:>9.4f}{a:>11.4f}{b:>11.4f}{b - a:>9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

ml = np.abs(np.array([s["m"] for s in gl], float) - m_true)
mr = np.abs(np.array([s["m"] for s in gr], float) - m_true)
naive = np.abs((nreq - 1) - m_true)
draws = np.array([(mr[i] - ml[i]).mean()
                  for i in (RNG.integers(0, n, n) for _ in range(N_BOOT))])
lo, hi = np.percentile(draws, [2.5, 97.5])
tag = "" if lo <= 0 <= hi else "  <-- real"
print(f"{'edge-count MAE':<20}{naive.mean():>9.4f}{ml.mean():>11.4f}{mr.mean():>11.4f}"
      f"{mr.mean() - ml.mean():>9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")
print(f"{'':<20}{'(m=n-1)':>9}")

for label, which in [("degree dist.", 0), ("spectrum", 1)]:
    a, b = dist_to_real(gl, rs)[which], dist_to_real(gr, rs)[which]
    draws = np.empty(N_BOOT // 8)
    for i in range(len(draws)):
        idx = RNG.integers(0, n, n)
        gli, gri, rr = ([gl[j] for j in idx], [gr[j] for j in idx],
                        [rs[j] for j in idx])
        draws[i] = dist_to_real(gri, rr)[which] - dist_to_real(gli, rr)[which]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{label:<20}{0.0:>9.4f}{a:>11.4f}{b:>11.4f}{b - a:>9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")
