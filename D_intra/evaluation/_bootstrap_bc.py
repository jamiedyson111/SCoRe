"""Paired bootstrap of the B vs C differences on the 48 shared Minnesota clusters.

B and C are scored on the SAME supernodes in the same order, so resample the
supernode INDEX and recompute both sides from the same draw. That keeps the
pairing and removes the between-cluster variance that would otherwise swamp
the comparison.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "../evaluation")
from evaluate_clusters import (agg, dense_adj, describe, dist_to_real,  # noqa: E402
                               parse_generated)

RNG = np.random.default_rng(0)
B_BOOT = 4000


def load(cfg):
    gen = parse_generated(f"../results/samples/{cfg}.txt")
    blob = torch.load(f"../data/cluster_lux/raw/pairs_{cfg}_lux2minn_k7.pt")
    test = {int(r["sn"]): r for r in blob["splits"][2]}
    pairs = [(g, test[g["supernode"]]) for g in gen if g.get("supernode") in test]
    pairs.sort(key=lambda t: t[0]["supernode"])          # identical order for B and C
    return pairs


pb, pc = load("B"), load("C")
sn_b = [g["supernode"] for g, _ in pb]
sn_c = [g["supernode"] for g, _ in pc]
assert sn_b == sn_c, "B and C are not on the same supernodes -- comparison is not paired"
n = len(pb)
print(f"{n} paired supernodes\n")

gs_b = [describe(g["A"]) for g, _ in pb]
gs_c = [describe(g["A"]) for g, _ in pc]
rs = [describe(dense_adj(r["edge_index"], int(r["n"]))) for _, r in pb]

nreq = np.array([float(r["n"]) for _, r in pb])
m_true = np.array([s["m"] for s in rs], float)
m_b = np.array([s["m"] for s in gs_b], float)
m_c = np.array([s["m"] for s in gs_c], float)

SCALAR = [("connected %", "connected"), ("acyclic %", "forest"),
          ("components", "components"), ("edges (mean)", "mean_m"),
          ("density", "mean_density"), ("node degree", "mean_deg")]

print(f"{'metric':<22}{'B':>9}{'C':>9}{'C-B':>9}{'95% CI on C-B':>22}{'':>4}")
print("-" * 75)

for label, key in SCALAR:
    obs = agg(gs_c)[key] - agg(gs_b)[key]
    draws = np.empty(B_BOOT)
    for i in range(B_BOOT):
        idx = RNG.integers(0, n, n)
        draws[i] = (agg([gs_c[j] for j in idx])[key]
                    - agg([gs_b[j] for j in idx])[key])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{label:<22}{agg(gs_b)[key]:>9.4f}{agg(gs_c)[key]:>9.4f}"
          f"{obs:>9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

# edge-count MAE
mae_b, mae_c = np.abs(m_b - m_true), np.abs(m_c - m_true)
obs = mae_c.mean() - mae_b.mean()
draws = np.array([(mae_c[i] - mae_b[i]).mean()
                  for i in (RNG.integers(0, n, n) for _ in range(B_BOOT))])
lo, hi = np.percentile(draws, [2.5, 97.5])
tag = "" if lo <= 0 <= hi else "  <-- real"
print(f"{'edge-count MAE':<22}{mae_b.mean():>9.4f}{mae_c.mean():>9.4f}"
      f"{obs:>9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

# set-level distances: resample the index, recompute both against the same real draw
for label, which in [("degree dist.", 0), ("spectrum", 1)]:
    ob = dist_to_real(gs_b, rs)[which]
    oc = dist_to_real(gs_c, rs)[which]
    draws = np.empty(B_BOOT // 8)                        # these are the slow ones
    for i in range(len(draws)):
        idx = RNG.integers(0, n, n)
        gb, gc, rr = ([gs_b[j] for j in idx], [gs_c[j] for j in idx],
                      [rs[j] for j in idx])
        draws[i] = dist_to_real(gc, rr)[which] - dist_to_real(gb, rr)[which]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{label:<22}{ob:>9.4f}{oc:>9.4f}{oc - ob:>9.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

print("\nCI straddling 0 => the difference is not distinguishable from")
print("sampling noise on 48 clusters, whichever direction it points.")

# how concentrated is C's private training slice?
blob = torch.load("../data/cluster_lux/raw/pairs_C_lux2minn_k7.pt")
tr = blob["splits"][0]
sns = [int(r["sn"]) for r in tr]
uniq, counts = np.unique(sns, return_counts=True)
dup = counts[counts > 1]
print(f"\nC train: {len(tr)} records, {len(uniq)} distinct supernodes")
if len(dup):
    print(f"  {len(dup)} supernodes are repeated, up to {counts.max()}x")
    print(f"  repeated records account for {dup.sum()}/{len(tr)} "
          f"= {100 * dup.sum() / len(tr):.1f}% of the training set")
