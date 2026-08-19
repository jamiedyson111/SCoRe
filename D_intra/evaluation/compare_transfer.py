"""Produce the three comparison tables for the D_intra transfer study.

Run LOCALLY in the `kmis` env once all three configurations have been sampled:

    python compare_transfer.py \
        --a conditional_samples_A.txt \
        --b conditional_samples_B.txt \
        --c conditional_samples_C.txt

Emits three tables:
  1. Realism   -- generated vs real, per configuration, with a random floor
  2. Transfer  -- B and C side by side on the SAME held-out Minnesota supernodes
  3. Condition -- whether rho_in is used, per configuration

Also writes LaTeX for tables 1 and 2 with --latex.
"""
import argparse
import os

import numpy as np
import torch

from evaluate_clusters import (agg, dense_adj, describe, dist_to_real, er_like,
                               parse_generated, partial_corr)

SEED = 0
DEFAULT_TAG = "lux2minn_k7"
DESCRIPTIONS = {
    "A": "train Lux -> test Lux",
    "B": "train Lux -> test Minn",
    "C": "train Lux+Minn -> test Minn",
}


def configs(tag):
    """(key, pairs filename, description) for a given dataset tag."""
    return [(k, f"pairs_{k}_{tag}.pt", DESCRIPTIONS[k]) for k in ("A", "B", "C")]


CONFIGS = configs(DEFAULT_TAG)


def load(gen_path, pairs_path):
    gen = parse_generated(gen_path)
    try:
        blob = torch.load(pairs_path, weights_only=False)
    except TypeError:
        blob = torch.load(pairs_path)
    test = {int(r["sn"]): r for r in blob["splits"][2]}
    pairs = [(g, test[g["supernode"]]) for g in gen
             if g.get("supernode") in test]
    return pairs, blob


def analyse(pairs, blob, rng):
    g_stats = [describe(g["A"]) for g, _ in pairs]
    r_stats = [describe(dense_adj(r["edge_index"], int(r["n"]))) for _, r in pairs]
    e_stats = [describe(er_like(g["A"], rng)) for g, _ in pairs]
    gdw, gsp = dist_to_real(g_stats, r_stats)
    edw, esp = dist_to_real(e_stats, r_stats)

    names = blob["cond_names"]
    n = np.array([float(r["n"]) for _, r in pairs])
    rho_gen = np.array([s["density"] for s in g_stats])
    rho_req = np.array([float(g["cond"][names.index("rho_in")]) for g, _ in pairs])
    m_gen = np.array([s["m"] for s in g_stats], float)
    m_true = np.array([s["m"] for s in r_stats], float)

    return {
        "n_graphs": len(pairs),
        "gen": agg(g_stats), "real": agg(r_stats), "er": agg(e_stats),
        "deg_w": gdw, "spec": gsp, "deg_w_er": edw, "spec_er": esp,
        "partial_2n": partial_corr(rho_gen, rho_req, 2.0 / n),
        "excess": float(np.corrcoef(rho_gen - 2.0 / n, rho_req - 2.0 / n)[0, 1]),
        "m_err": float(np.abs(m_gen - m_true).mean()),
        "m_err_naive": float(np.abs((n - 1) - m_true).mean()),
        "m_within1": 100.0 * float(np.mean(np.abs(m_gen - m_true) <= 1)),
        "m_within1_naive": 100.0 * float(np.mean(np.abs((n - 1) - m_true) <= 1)),
    }


def table_realism(res):
    print("\nTABLE 1 -- Realism of generated clusters")
    print("=" * 78)
    hdr = f"{'':<26}" + "".join(f"{c:>17}" for c, _, _ in CONFIGS if c in res)
    print(hdr)
    print(f"{'':<26}" + "".join(f"{'gen / real':>17}" for c, _, _ in CONFIGS if c in res))
    print("-" * 78)
    rows = [("connected %", "connected", "{:.1f}"), ("acyclic %", "forest", "{:.1f}"),
            ("components", "components", "{:.2f}"), ("edges", "mean_m", "{:.2f}"),
            ("density", "mean_density", "{:.4f}"), ("node degree", "mean_deg", "{:.2f}")]
    for label, key, fmt in rows:
        line = f"{label:<26}"
        for c, _, _ in CONFIGS:
            if c not in res:
                continue
            r = res[c]
            line += f"{fmt.format(r['gen'][key]) + ' / ' + fmt.format(r['real'][key]):>17}"
        print(line)
    print("-" * 78)
    for label, key, ekey in [("degree dist. vs real", "deg_w", "deg_w_er"),
                             ("spectrum vs real", "spec", "spec_er")]:
        line = f"{label:<26}"
        for c, _, _ in CONFIGS:
            if c not in res:
                continue
            line += f"{res[c][key]:.4f} ({res[c][ekey]:.3f}){'':>3}"
        print(line)
    print("  (value, random-ER floor in brackets; lower is better)")


def table_transfer(res):
    if not ({"B", "C"} <= set(res)):
        return
    print("\nTABLE 2 -- Transfer to Minnesota (identical held-out supernodes)")
    print("=" * 78)
    print(f"{'':<30}{'B: Lux only':>16}{'C: Lux + Minn':>16}{'delta':>12}")
    print("-" * 78)
    # "close" == the real graph is the target and BOTH models can overshoot it,
    # so nearer to real wins; "low" == a distance that is already 0-is-best.
    for label, path, fmt, better in [
        ("connected %", ("gen", "connected"), "{:.1f}", "close"),
        ("acyclic %", ("gen", "forest"), "{:.1f}", "close"),
        ("edges (mean)", ("gen", "mean_m"), "{:.2f}", "close"),
        ("degree dist. vs real", ("deg_w",), "{:.4f}", "low"),
        ("spectrum vs real", ("spec",), "{:.4f}", "low"),
        ("edge-count MAE", ("m_err",), "{:.2f}", "low"),
    ]:
        def get(c):
            v = res[c]
            for p in path:
                v = v[p]
            return v
        b, cc = get("B"), get("C")
        d = cc - b
        arrow = ""
        if better == "low":
            arrow = "  better" if d < 0 else ("  worse" if d > 0 else "")
        elif better == "close":
            real = res["B"]["real"][path[-1]]
            gap = abs(cc - real) - abs(b - real)
            arrow = "  better" if gap < 0 else ("  worse" if gap > 0 else "")
        print(f"{label:<30}{fmt.format(b):>16}{fmt.format(cc):>16}{d:>+12.4f}{arrow}")
    print("-" * 78)
    print(f"  real Minnesota reference: connected {res['B']['real']['connected']:.1f}%, "
          f"acyclic {res['B']['real']['forest']:.1f}%, edges {res['B']['real']['mean_m']:.2f}")
    print(f"  test set: {res['B']['n_graphs']} supernodes (shared by B and C)")


def table_condition(res):
    print("\nTABLE 3 -- Is rho_in used, beyond what n implies?")
    print("=" * 78)
    print(f"{'':<30}{'partial|2/n':>14}{'excess corr':>14}"
          f"{'MAE model':>12}{'MAE m=n-1':>12}")
    print("-" * 78)
    for c, _, desc in CONFIGS:
        if c not in res:
            continue
        r = res[c]
        print(f"{c + ': ' + desc:<30}{r['partial_2n']:>14.3f}{r['excess']:>14.3f}"
              f"{r['m_err']:>12.2f}{r['m_err_naive']:>12.2f}")
    print("-" * 78)
    print("  partial|2/n near 0 => the model conditions on SIZE alone.")
    print("  'MAE m=n-1' is a rule using only the node count; if it beats the")
    print("  model, the model adds nothing on edge count.")


def latex_transfer(res):
    if not ({"B", "C"} <= set(res)):
        return
    print("\n% ---- LaTeX: transfer table ----")
    print("\\begin{tabular}{lrrr}")
    print("\\toprule")
    print(" & B: Lux only & C: Lux + Minn & Real Minnesota \\\\")
    print("\\midrule")
    for label, path, fmt in [("Connected (\\%)", ("gen", "connected"), "{:.1f}"),
                             ("Acyclic (\\%)", ("gen", "forest"), "{:.1f}"),
                             ("Edges (mean)", ("gen", "mean_m"), "{:.2f}"),
                             ("Density (mean)", ("gen", "mean_density"), "{:.4f}")]:
        def get(c):
            v = res[c]
            for p in path:
                v = v[p]
            return v
        real = res["B"]["real"][path[-1]]
        print(f"{label} & {fmt.format(get('B'))} & {fmt.format(get('C'))} "
              f"& {fmt.format(real)} \\\\")
    print("\\midrule")
    print(f"Degree dist. vs.\\ real $\\downarrow$ & {res['B']['deg_w']:.4f} "
          f"& {res['C']['deg_w']:.4f} & 0.0000 \\\\")
    print(f"Spectrum vs.\\ real $\\downarrow$ & {res['B']['spec']:.4f} "
          f"& {res['C']['spec']:.4f} & --- \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--c")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help="dataset tag, i.e. the part after pairs_X_ in the filenames")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    global CONFIGS
    CONFIGS = configs(args.tag)

    res = {}
    for key, pairs_file, desc in CONFIGS:
        gen_path = getattr(args, key.lower())
        if not gen_path:
            continue
        if not os.path.exists(gen_path):
            print(f"  skipping {key}: {gen_path} not found")
            continue
        pairs, blob = load(gen_path, os.path.join(args.pairs_dir, pairs_file))
        if not pairs:
            print(f"  skipping {key}: no generated graph matched a test supernode")
            continue
        res[key] = analyse(pairs, blob, rng)
        print(f"  {key}: {len(pairs)} matched graphs  ({desc})")

    if not res:
        raise SystemExit("nothing to compare -- pass --a/--b/--c")

    table_realism(res)
    table_transfer(res)
    table_condition(res)
    if args.latex:
        latex_transfer(res)


if __name__ == "__main__":
    main()
