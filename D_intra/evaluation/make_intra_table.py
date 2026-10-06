"""Structural statistics of generated supernode blocks against real held-out ones.

Emits one table with a column per configuration and a Real column, in the layout
used for the D_inter cross-block table. Run LOCALLY in the `kmis` env:

    python make_intra_table.py --tag lux2olde_k7 \
        --b ../results/samples/oldenburg/B.txt        --c ../results/samples/oldenburg/C.txt \
        --b-nohop ../results/samples/oldenburg/B_nohop.txt \
        --c-nohop ../results/samples/oldenburg/C_nohop.txt --latex

Every configuration passed must be scored on the SAME held-out supernodes, which
is asserted rather than assumed: B and C share a test split by construction, and
the hop and no-hop datasets are built from the same seed, so a mismatch means a
dataset was rebuilt with different settings and the columns are not comparable.

THE REAL COLUMN IS EXACT HERE. Unlike the D_inter table, generation is 1:1 --
one generated block per test supernode, conditioned on that supernode's own c_i
-- so the real column is the same set of blocks the generated column is scored
against and its degree distance to itself is 0 by construction, not by rounding.
What is NOT free is the resolution of W_1 at this sample size, so --latex also
reports a bootstrap floor: resample the real blocks with replacement to the same
count and measure their distance to the full real set. A generated value at or
below that floor is indistinguishable from having drawn the real blocks again.
"""
import argparse
import os

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from evaluate_clusters import dense_adj, describe, parse_generated

SEED = 0
N_BOOT = 2000

# (flag, column label, filename letter). Order fixes the column order.
COLUMNS = [("b", "B (hops)", "B"), ("b_nohop", "B (no hops)", "B"),
           ("c", "C (hops)", "C"), ("c_nohop", "C (no hops)", "C")]


def _load_blob(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_column(gen_path, pairs_path):
    """Generated/real describe() stats for one configuration, ordered by supernode."""
    gen = parse_generated(gen_path)
    blob = _load_blob(pairs_path)
    test = {int(r["sn"]): r for r in blob["splits"][2]}
    pairs = [(g, test[g["supernode"]]) for g in gen if g.get("supernode") in test]
    pairs.sort(key=lambda t: t[0]["supernode"])          # same order in every column
    if not pairs:
        raise SystemExit(f"no generated graph in {gen_path} matched a test supernode "
                         f"of {os.path.basename(pairs_path)} -- wrong pairs file?")

    # Supernode ids are 0..c-1 and are reused across coarsenings, so a samples file
    # from one k matches a pairs file from another on id alone -- silently, on the
    # subset of ids the two happen to share, producing a table of nonsense. Sizes
    # are the disambiguator: the sampler forces n_i from the record it generated
    # for, so every matched pair must agree on n. This is the check that catches a
    # k=7 samples file pointed at the k=4 dataset.
    bad = [(g["supernode"], g["n"], int(r["n"])) for g, r in pairs
           if g["n"] != int(r["n"])]
    if bad:
        sn, gn, rn = bad[0]
        raise SystemExit(
            f"{os.path.basename(gen_path)} does not belong to "
            f"{os.path.basename(pairs_path)}: {len(bad)} of {len(pairs)} matched "
            f"supernodes disagree on size (e.g. supernode {sn}: generated {gn} nodes, "
            f"dataset says {rn}). Check --tag.")
    if len(pairs) < len(test):
        print(f"    note: {len(test) - len(pairs)} of {len(test)} test supernodes "
              f"had no generated graph")
    return {
        "sn": [g["supernode"] for g, _ in pairs],
        "gen": [describe(g["A"]) for g, _ in pairs],
        "real": [describe(dense_adj(r["edge_index"], int(r["n"]))) for _, r in pairs],
        "blob": blob,
    }


def describe_split(blob):
    """'Oldenburg $k{=}7$', trained on 'Luxembourg $k{=}7$' -- read off the dataset.

    Taken from the blob rather than the filename so the caption cannot drift from
    what was actually built. k_public/k_private are absent from datasets written
    before per-graph k existed; those are single-k by definition.
    """
    k = blob.get("k", 7)
    k_pub, k_priv = blob.get("k_public", k), blob.get("k_private", k)
    test_graph = blob["private"] if blob["config"] in ("B", "C") else blob["public"]
    k_test = k_priv if blob["config"] in ("B", "C") else k_pub
    return (f"{test_graph.capitalize()} $k{{=}}{k_test}$",
            f"{blob['public'].capitalize()} $k{{=}}{k_pub}$")


def stats(block):
    """The row values for one set of blocks."""
    m = np.array([s["m"] for s in block], float)
    deg = np.concatenate([s["degrees"] for s in block])
    return {
        "m_median": np.median(m),
        "m_std": m.std(ddof=1),
        "m_mean": m.mean(),
        "density": np.mean([s["density"] for s in block]),
        "deg_median": np.median(deg),
        "deg_mean": deg.mean(),
        "connected": 100.0 * np.mean([s["connected"] for s in block]),
        "forest": 100.0 * np.mean([s["is_forest"] for s in block]),
    }


def degree_w1(block, real):
    return wasserstein_distance(np.concatenate([s["degrees"] for s in block]),
                                np.concatenate([s["degrees"] for s in real]))


def resolution_floor(real, rng, n_boot=N_BOOT):
    """W_1 of a resample of the real blocks against the full real set.

    The metric's noise floor at this sample size. Reported as mean +/- sd.
    """
    n = len(real)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[i] = degree_w1([real[j] for j in idx], real)
    return draws.mean(), draws.std(ddof=1)


ROWS = [("Edges (median)", "m_median", "{:.0f}"),
        ("Edges (std)", "m_std", "{:.3f}"),
        ("Edges (mean)", "m_mean", "{:.3f}"),
        ("Density (mean)", "density", "{:.4f}"),
        ("Degree (median)", "deg_median", "{:.0f}"),
        ("Degree (mean)", "deg_mean", "{:.4f}"),
        ("Connected (\\%)", "connected", "{:.1f}"),
        ("Acyclic (\\%)", "forest", "{:.1f}")]


def emit_plain(labels, cols, real, w1, floor, n_blocks):
    width = max(len(l) for l in labels + ["Degree distribution vs. real"]) + 2
    print(f"\n{n_blocks} held-out supernodes, scored by every column\n")
    print(" " * width + "".join(f"{l:>14}" for l in labels) + f"{'Real':>14}")
    print("-" * (width + 14 * (len(labels) + 1)))
    for label, key, fmt in ROWS:
        line = f"{label.replace(chr(92) + '%', '%'):<{width}}"
        line += "".join(f"{fmt.format(c[key]):>14}" for c in cols)
        print(line + f"{fmt.format(real[key]):>14}")
    print("-" * (width + 14 * (len(labels) + 1)))
    print(f"{'Degree dist. vs real':<{width}}"
          + "".join(f"{v:>14.4f}" for v in w1) + f"{0.0:>14.4f}")
    print(f"\n  resolution floor of the degree distance at n={n_blocks}: "
          f"{floor[0]:.4f} +/- {floor[1]:.4f}")
    print("  (real blocks resampled to the same count, measured against the real set;")
    print("   a generated value at or below this is not distinguishable from real)")


def emit_latex(labels, cols, real, w1, floor, n_blocks, split, label_key):
    test_desc, train_desc = split
    spec = "l" + "r" * (len(cols) + 1)
    print("\n% ---- LaTeX ----")
    print("\\begin{table}[H]")
    print("\\centering")
    print("\\caption{Structural statistics of generated supernode blocks against real")
    print(f"held-out blocks, on the {test_desc} test split, for a model trained on")
    print(f"{train_desc}. For the upper block, closer to the real column is")
    print("better. Generation is one block per held-out supernode, so all columns are")
    print(f"scored on the same {n_blocks} blocks and the real column is compared against")
    print("itself, giving exactly $0$ on the lower block. Resampling the real blocks to")
    print(f"the same count gives a $W_1$ of ${floor[0]:.4f} \\pm {floor[1]:.4f}$ rather")
    print("than $0$, which is the resolution limit of the metric at this sample size.}")
    print(f"\\label{{tab:dintra-{label_key}-stats}}")
    print(f"\\begin{{tabular}}{{{spec}}}")
    print("\\toprule")
    print(" & " + " & ".join(labels) + " & Real \\\\")
    print("\\midrule")
    for label, key, fmt in ROWS:
        vals = " & ".join(fmt.format(c[key]) for c in cols)
        print(f"{label:<34} & {vals} & {fmt.format(real[key])} \\\\")
    print("\\midrule")
    w1s = " & ".join(f"{v:.4f}" for v in w1)
    # Assembled outside the f-string: the kmis env is on Python 3.8, where an
    # f-string expression may not contain a backslash.
    w1_label = "Degree distribution vs.\\ real $\\downarrow$"
    print(f"{w1_label:<34} & {w1s} & 0.0000 \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def main():
    ap = argparse.ArgumentParser()
    for flag, _, _ in COLUMNS:
        ap.add_argument(f"--{flag.replace('_', '-')}")
    ap.add_argument("--tag", default="lux2olde_k7",
                    help="dataset tag, i.e. the part after pairs_X_ in the filenames. "
                         "The _nohop columns get _nohop appended automatically.")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    labels, cols, w1, real_ref, sn_ref, split = [], [], [], None, None, None
    for flag, label, letter in COLUMNS:
        gen_path = getattr(args, flag)
        if not gen_path:
            continue
        if not os.path.exists(gen_path):
            print(f"  skipping {label}: {gen_path} not found")
            continue
        tag = args.tag + ("_nohop" if flag.endswith("_nohop") else "")
        col = load_column(gen_path, os.path.join(args.pairs_dir,
                                                 f"pairs_{letter}_{tag}.pt"))

        # The whole table rests on every column describing the same blocks.
        # A rebuilt dataset with a different seed would break this silently.
        if sn_ref is None:
            sn_ref, real_ref = col["sn"], col["real"]
            split = describe_split(col["blob"])
        elif col["sn"] != sn_ref:
            raise SystemExit(
                f"{label} is scored on different supernodes than the first column "
                f"({len(col['sn'])} vs {len(sn_ref)}); the columns are not comparable. "
                f"Rebuild the datasets from one seed.")

        labels.append(label)
        cols.append(stats(col["gen"]))
        w1.append(degree_w1(col["gen"], col["real"]))
        print(f"  {label}: {len(col['sn'])} matched blocks")

    if not cols:
        raise SystemExit("nothing to tabulate -- pass --b/--c/--b-nohop/--c-nohop")

    real = stats(real_ref)
    floor = resolution_floor(real_ref, rng)
    emit_plain(labels, cols, real, w1, floor, len(sn_ref))
    if args.latex:
        emit_latex(labels, cols, real, w1, floor, len(sn_ref), split,
                   args.tag.replace("_", "-"))


if __name__ == "__main__":
    main()
