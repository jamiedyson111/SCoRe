"""Full condition vs the no-hop ablation, on the SAME held-out supernodes.

Run LOCALLY in the `kmis` env once both configurations have been sampled:

    python compare_ablation.py --cfg A \
        --gen-full ../../results/samples/D_intra/A.txt \
        --gen-abl  ../../results/samples/D_intra/A_nohop.txt --latex

The two datasets are built from the same seed and carry identical splits, so
every held-out supernode is scored by both models. That makes this a PAIRED
comparison: resample the supernode index and recompute both sides from the same
draw, which removes the between-cluster variance that would otherwise swamp a
small effect.

WHY CONFIGURATION A. The ablation asks whether h1..h7 carries information beyond
what n already implies. Answering it needs statistical power, and A has 882
held-out supernodes against B and C's 48. The B/C comparison is worth running for
completeness but cannot resolve anything: the paired bootstrap on those 48 puts a
95% interval of roughly +/-0.04 on the degree-distribution distance, several
times any plausible effect.

WHAT A NULL RESULT WOULD MEAN. h is 7 of the 10 numbers the private release has
to publish per supernode. If the ablation matches the full condition, the release
can be cut to [n, rho_in, k] with no loss -- which is a stronger privacy position
than the current design, and a result worth reporting in its own right.
"""
import argparse
import os

import numpy as np
import torch

from scipy.stats import wasserstein_distance

from evaluate_clusters import (agg, dense_adj, describe, dist_to_real,
                               parse_generated)

SEED = 0
N_BOOT = 4000
N_BOOT_SLOW = 2000       # degree distance only; see the note where it is used

SCALAR = [("connected %", "connected"), ("acyclic %", "forest"),
          ("components", "components"), ("edges (mean)", "mean_m"),
          ("density", "mean_density"), ("node degree", "mean_deg")]


def _load_blob(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_matched(gen_full, gen_abl, pairs_full, pairs_abl):
    """Generated pairs from both models, restricted to supernodes common to both.

    Both blobs' test splits are asserted identical rather than assumed: the whole
    comparison rests on the two models having been scored on the same graphs, and
    a regenerated dataset with a different seed would silently break that.
    """
    bf, ba = _load_blob(pairs_full), _load_blob(pairs_abl)
    sn_f = [int(r["sn"]) for r in bf["splits"][2]]
    sn_a = [int(r["sn"]) for r in ba["splits"][2]]
    assert sn_f == sn_a, (
        "the two datasets have different test splits, so the comparison is not "
        "paired. Rebuild both with the same --k and SEED.")
    assert len(ba["cond_names"]) < len(bf["cond_names"]), (
        f"--gen-abl blob has {len(ba['cond_names'])} condition dims and --gen-full "
        f"has {len(bf['cond_names'])}; the ablation should be the SMALLER one")

    test = {int(r["sn"]): r for r in bf["splits"][2]}
    gf = {g["supernode"]: g for g in parse_generated(gen_full) if g.get("supernode") in test}
    ga = {g["supernode"]: g for g in parse_generated(gen_abl) if g.get("supernode") in test}
    common = sorted(set(gf) & set(ga))
    if not common:
        raise SystemExit("no supernodes in common -- are these the same configuration?")
    return common, [gf[s] for s in common], [ga[s] for s in common], \
        [test[s] for s in common], bf["cond_names"], ba["cond_names"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="A", choices=("A", "B", "C"))
    ap.add_argument("--gen-full", required=True, help="samples from the 10-dim model")
    ap.add_argument("--gen-abl", required=True, help="samples from the 3-dim model")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--tag", default="lux2olde_k7")
    ap.add_argument("--abl-suffix", default="_nohop")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    pf = os.path.join(args.pairs_dir, f"pairs_{args.cfg}_{args.tag}.pt")
    pa = os.path.join(args.pairs_dir, f"pairs_{args.cfg}_{args.tag}{args.abl_suffix}.pt")
    sns, gen_f, gen_a, real, names_f, names_a = load_matched(
        args.gen_full, args.gen_abl, pf, pa)

    rng = np.random.default_rng(SEED)
    n = len(sns)
    print(f"configuration {args.cfg}: {n} paired supernodes")
    print(f"  full     ({len(names_f)} dims): {list(names_f)}")
    print(f"  ablation ({len(names_a)} dims): {list(names_a)}\n")

    sf = [describe(g["A"]) for g in gen_f]
    sa = [describe(g["A"]) for g in gen_a]
    rs = [describe(dense_adj(r["edge_index"], int(r["n"]))) for r in real]

    hdr = f"{'metric':<22}{'full':>9}{'no-hop':>9}{'real':>9}{'abl-full':>10}{'95% CI':>22}"
    print(hdr)
    print("-" * len(hdr))

    af, aa, ar = agg(sf), agg(sa), agg(rs)

    # Per-graph values, so a bootstrap draw is an array index rather than 882
    # dict rebuilds. `mean_deg` is a POOLED mean over all nodes, not a mean of
    # per-graph means, so it is bootstrapped as a ratio of sums.
    def per_graph(stats):
        return {"connected": np.array([100.0 * s["connected"] for s in stats]),
                "forest": np.array([100.0 * s["is_forest"] for s in stats]),
                "components": np.array([float(s["n_components"]) for s in stats]),
                "mean_m": np.array([float(s["m"]) for s in stats]),
                "mean_density": np.array([s["density"] for s in stats]),
                "_deg_sum": np.array([s["degrees"].sum() for s in stats]),
                "_deg_cnt": np.array([float(len(s["degrees"])) for s in stats])}

    bg_f, bg_a = per_graph(sf), per_graph(sa)
    IDX = rng.integers(0, n, size=(N_BOOT, n))

    def boot(p, key):
        if key == "mean_deg":
            return p["_deg_sum"][IDX].sum(1) / p["_deg_cnt"][IDX].sum(1)
        return p[key][IDX].mean(1)

    for label, key in SCALAR:
        obs = aa[key] - af[key]
        draws = boot(bg_a, key) - boot(bg_f, key)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        tag = "" if lo <= 0 <= hi else "  <-- real"
        print(f"{label:<22}{af[key]:>9.4f}{aa[key]:>9.4f}{ar[key]:>9.4f}"
              f"{obs:>10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

    # edge-count MAE against the requested n's true edge count
    m_true = np.array([s["m"] for s in rs], float)
    mae_f = np.abs(np.array([s["m"] for s in sf], float) - m_true)
    mae_a = np.abs(np.array([s["m"] for s in sa], float) - m_true)
    obs = mae_a.mean() - mae_f.mean()
    draws = np.array([(mae_a[i] - mae_f[i]).mean()
                      for i in (rng.integers(0, n, n) for _ in range(N_BOOT))])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{'edge-count MAE':<22}{mae_f.mean():>9.4f}{mae_a.mean():>9.4f}{0.0:>9.4f}"
          f"{obs:>10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

    # Set-level distances. Real scores 0 by construction -- it is being compared
    # with itself -- so that column is the reference, not an attainable target.
    dist = {}
    df, da = dist_to_real(sf, rs), dist_to_real(sa, rs)
    dist["degree dist. vs real"] = (df[0], da[0])
    dist["spectrum vs real"] = (df[1], da[1])

    # Degree distance bootstraps cheaply: one concatenation and one Wasserstein
    # per draw. Precompute the per-graph degree arrays so a draw is an index.
    deg_f = [s["degrees"] for s in sf]
    deg_a = [s["degrees"] for s in sa]
    deg_r = [s["degrees"] for s in rs]
    draws = np.empty(N_BOOT_SLOW)
    for i in range(len(draws)):
        idx = rng.integers(0, n, n)
        rr = np.concatenate([deg_r[j] for j in idx])
        draws[i] = (wasserstein_distance(np.concatenate([deg_a[j] for j in idx]), rr)
                    - wasserstein_distance(np.concatenate([deg_f[j] for j in idx]), rr))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    tag = "" if lo <= 0 <= hi else "  <-- real"
    print(f"{'degree dist. vs real':<22}{df[0]:>9.4f}{da[0]:>9.4f}{0.0:>9.4f}"
          f"{da[0] - df[0]:>10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{tag}")

    # The spectral distance is a median over every (generated, real) PAIR of
    # graphs, so one evaluation is O(n^2) Wasserstein calls -- 778k at n=882.
    # Bootstrapping it is not affordable here (it is, at the n=48 of B and C,
    # which is why _bootstrap_bc.py does it). Point estimates only, stated as
    # such rather than quietly omitted.
    print(f"{'spectrum vs real':<22}{df[1]:>9.4f}{da[1]:>9.4f}{0.0:>9.4f}"
          f"{da[1] - df[1]:>10.4f}{'(no CI, see below)':>22}")

    print("\nCI straddling 0 => dropping h1..h7 makes no measurable difference to")
    print("that metric, whichever direction the point estimate points.")
    print(f"Spectrum has no interval: it is O(n^2) per draw and n = {n}.")

    if args.latex:
        print("\n% ---- LaTeX: hop-histogram ablation ----")
        print("\\begin{tabular}{lrrr}")
        print("\\toprule")
        print(" & Full $c_i$ & No hops & Real \\\\")
        print(f" & \\textsc{{{len(names_f)}-dim}} & \\textsc{{{len(names_a)}-dim}} & \\\\")
        print("\\midrule")
        print(f"Connected (\\%) & {af['connected']:.1f} & {aa['connected']:.1f} & {ar['connected']:.1f} \\\\")
        print(f"Acyclic (\\%) & {af['forest']:.1f} & {aa['forest']:.1f} & {ar['forest']:.1f} \\\\")
        print(f"Edges (mean) & {af['mean_m']:.2f} & {aa['mean_m']:.2f} & {ar['mean_m']:.2f} \\\\")
        print(f"Density (mean) & {af['mean_density']:.4f} & {aa['mean_density']:.4f} & {ar['mean_density']:.4f} \\\\")
        print("\\midrule")
        for label in ("degree dist. vs real", "spectrum vs real"):
            of, oa = dist[label]
            nice = label.replace("dist.", "dist.\\").replace(" vs real", "")
            print(f"{nice.strip().capitalize()} vs.\\ real $\\downarrow$ & "
                  f"{of:.4f} & {oa:.4f} & 0.0000 \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()
