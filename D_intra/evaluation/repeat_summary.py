"""Mean (or median) and standard deviation of every metric across repeated sampling runs.

Run LOCALLY in the `kmis` env, after scripts/run_olde_repeats.sh has run on
BlueBEAR and its files have been copied to ../results/samples/D_intra/oldenburg/repeats/
as <CFG>_s<seed>.txt:

    python repeat_summary.py                                   # B and C, on Oldenburg
    python repeat_summary.py --configs A_nohop_centroid A_v3   # A, on Luxembourg
    python repeat_summary.py --stat median                     # median instead of mean
    python repeat_summary.py --configs B C --samples-dir ../../results/samples/D_intra/oldenburg/repeats

Each run is scored exactly as the single-run tables score it -- the metric
definitions are imported from make_intra_table.py and evaluate_clusters.py, not
reimplemented -- and then every metric is summarised across the runs as its
mean (default; --stat median for the median) and its sample standard deviation
(ddof=1).

E_h, THE HOP ERROR. For each test block, the L1 distance between the hop
histogram of the generated block and the target condition h_i:

    E_h = (1/N_test) sum_i sum_{r=1..K} | h^_i(r) - h_i(r) |

h^_i(r) is the fraction of the block's non-centroid nodes at shortest-path
distance r from its CENTROID, so it is only defined when the sample marks one:
v3 and the *_centroid models fix node 0 as the centroid at sampling. Models
with one node type (plain *_nohop, the raw-hop A/B/C) never say which node it
is, and get a blank cell. h_i is read from the HOPS pairs file even for the
no-hop configurations -- they are scored on the same supernodes, they just
were not given h. The Real column measures the real blocks from their true
k-MIS centre and must come out at 0, which checks hop_hist against the stored h.

The output is TAB-separated so it pastes into Excel as cells:

    ../results/D_intra/tables/oldenburg/repeats_summary.tsv     one row per metric, one
                                                     "mean ± std" cell per config
    ../results/D_intra/tables/oldenburg/repeats_per_seed.tsv    every run's raw value

A is written to repeats_summary_A.tsv / repeats_per_seed_A.tsv instead. It is
tested on Luxembourg's 882 supernodes, not Oldenburg's 160, so it cannot share
a sheet -- or a Real column -- with B and C, and asking for both at once stops
with an error rather than mixing them.

The spread is SAMPLING noise only: every run of a configuration uses the same
trained checkpoint, so it says nothing about what retraining would change.
"""
import argparse
import glob
import os
import re

import numpy as np

from evaluate_clusters import dense_adj, dist_to_real, has_centroid, hop_hist
from make_intra_table import _load_blob, degree_w1, load_column, stats

# (key, column label, letter of the pairs file, nohop pairs file?)
# v3 and *_centroid read the same pairs file as their base model: the centroid
# node type and the hop embedding are applied at load / model time, not stored
# in the dataset.
CONFIGS = {}
for _L in "ABC":
    CONFIGS.update({
        f"{_L}":                (f"{_L} (hops)",                _L, False),
        f"{_L}_nohop":          (f"{_L} (no hops)",             _L, True),
        f"{_L}_nohop_centroid": (f"{_L} (no hops + centroid)",  _L, True),
        f"{_L}_v3":             (f"{_L} v3 (hops + centroid)",  _L, False),
    })
# Baselines from baselines/make_baseline_samples.py, scored on the same test
# supernodes: BS1/BS2 on Oldenburg's (the B/C split), A_BS1/A_BS2 on Luxembourg's.
CONFIGS.update({
    "BS1":   ("BS1 (random)",   "B", False),
    "BS2":   ("BS2 (Chung-Lu)", "B", False),
    "A_BS1": ("BS1 (random)",   "A", False),
    "A_BS2": ("BS2 (Chung-Lu)", "A", False),
    # Huiping's baselines (competitors/Baseline on BlueBEAR), Oldenburg test set
    "ER":    ("ER",             "B", False),
    "SC_RG": ("SC_RG",          "B", False),
})

# (key, label, decimals). Order is the row order of the spreadsheet.
# The (std) rows are spreads WITHIN one run -- across the test blocks for edges
# and density, across all nodes pooled for degree -- not the spread across runs,
# which is the "± std" beside every cell.
METRICS = [
    ("m_median",    "Edges (median)",            1),
    ("m_mean",      "Edges (mean)",              3),
    ("m_std",       "Edges (std)",               3),
    ("density",     "Density (mean)",            4),
    ("density_std", "Density (std)",             4),
    ("deg_median",  "Degree (median)",           1),
    ("deg_mean",    "Degree (mean)",             4),
    ("deg_std",     "Degree (std)",              4),
    ("connected",  "Connected (%)",             2),
    ("forest",     "Acyclic (%)",               2),
    ("components", "Components (mean)",         4),
    ("m_mae",      "Edge-count MAE",            4),
    ("deg_w1",     "Degree dist. vs real (W1)", 4),
    ("spec",       "Spectrum vs real",          4),
    ("e_h",        "Hop error E_h",             4),
]


def hop_targets(pairs_path):
    """{supernode: h_i} from the test split of a pairs file that carries h."""
    blob = _load_blob(pairs_path)
    names = blob["cond_names"]
    idx = [i for i, nm in enumerate(names) if re.fullmatch(r"h\d+", nm)]
    if not idx:
        raise SystemExit(f"{os.path.basename(pairs_path)} has no h columns")
    return {int(r["sn"]): r["y_raw"][idx].numpy().astype(float)
            for r in blob["splits"][2]}


def hop_error(pairs, h_target):
    """E_h for one run, or None when the samples do not mark a centroid."""
    marked = [has_centroid(g["x"]) for g, _ in pairs]
    if not any(marked):
        return None
    if not all(marked):
        raise SystemExit(f"{sum(marked)} of {len(marked)} samples mark a centroid -- "
                         "a run should be all or nothing")
    k = len(next(iter(h_target.values())))
    return float(np.mean([np.abs(hop_hist(g["A"], 0, k) - h_target[g["supernode"]]).sum()
                          for g, _ in pairs]))


def spreads(block):
    """The within-run std rows that make_intra_table.stats does not report."""
    return {"density_std": float(np.std([s["density"] for s in block], ddof=1)),
            "deg_std": float(np.concatenate([s["degrees"] for s in block]).std(ddof=1))}


def score(col, h_target):
    """Every metric for one run, from load_column's output."""
    gen, real = col["gen"], col["real"]
    out = stats(gen)
    out.update(spreads(gen))
    out["components"] = float(np.mean([s["n_components"] for s in gen]))
    out["m_mae"] = float(np.mean([abs(g["m"] - r["m"]) for g, r in zip(gen, real)]))
    out["deg_w1"] = degree_w1(gen, real)
    out["spec"] = dist_to_real(gen, real)[1]
    out["e_h"] = hop_error(col["pairs"], h_target)
    return out


def real_row(col, h_target):
    """The real blocks scored the same way. Distances to themselves are 0.

    E_h is measured, not assumed: each real block from its true centre against
    the stored target. Anything above float rounding means hop_hist and the
    condition disagree on the definition, and E_h for the models is suspect.
    """
    real = col["real"]
    out = stats(real)
    out.update(spreads(real))
    out["components"] = float(np.mean([s["n_components"] for s in real]))
    out["m_mae"] = 0.0
    out["deg_w1"] = 0.0
    out["spec"] = 0.0
    k = len(next(iter(h_target.values())))
    errs = []
    for _, r in col["pairs"]:
        centre = r["nodes"].tolist().index(int(r["centre"]))
        A = dense_adj(r["edge_index"], int(r["n"]))
        errs.append(np.abs(hop_hist(A, centre, k) - h_target[int(r["sn"])]).sum())
    if max(errs) > 1e-4:
        raise SystemExit(f"real blocks do not reproduce their stored h (max E_h "
                         f"{max(errs):.2e}) -- hop_hist disagrees with the condition")
    out["e_h"] = float(np.mean(errs))
    return out


def fmt(v, dp):
    """A number as text for the sheet; None or nan as a blank cell."""
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dp}f}"


def seed_of(path):
    return int(re.search(r"_s(\d+)\.txt$", path).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="../../results/samples/D_intra/oldenburg/repeats")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--tag", default="lux2olde_k7")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    # No hops vs v3, both with the centroid node type, so the two
                    # differ in h_i alone. The others can be named explicitly.
                    default=["B_nohop_centroid", "B_v3", "C_nohop_centroid", "C_v3"])
    ap.add_argument("--stat", choices=("mean", "median"), default="mean",
                    help="summary across runs, reported beside the std")
    ap.add_argument("--out-dir", default="../../results/D_intra/tables/oldenburg")
    args = ap.parse_args()
    test_on_public = {CONFIGS[c][1] == "A" for c in args.configs}
    if len(test_on_public) > 1:
        raise SystemExit("A is tested on the public graph and B/C on the private one; "
                         "summarise them in separate runs")
    suffix = "_A" if test_on_public.pop() else ""

    per_run, sn_ref, real, n_runs, test_graph = {}, None, None, {}, None
    targets = {}                                   # hops pairs file -> {sn: h_i}
    for cfg in args.configs:
        label, letter, nohop = CONFIGS[cfg]
        files = sorted(glob.glob(os.path.join(args.samples_dir, f"{cfg}_s*.txt")), key=seed_of)
        if not files:
            print(f"  {cfg}: no {cfg}_s*.txt in {args.samples_dir}, skipped")
            continue
        pairs = os.path.join(args.pairs_dir,
                             f"pairs_{letter}_{args.tag}{'_nohop' if nohop else ''}.pt")
        hops_pairs = os.path.join(args.pairs_dir, f"pairs_{letter}_{args.tag}.pt")
        if hops_pairs not in targets:
            targets[hops_pairs] = hop_targets(hops_pairs)
        h_target = targets[hops_pairs]
        runs = []
        for f in files:
            col = load_column(f, pairs)
            # Every run of every configuration must be scored on the same blocks,
            # or the medians are of different things.
            if sn_ref is None:
                sn_ref, real = col["sn"], real_row(col, h_target)
                blob = col["blob"]
                test_graph = blob["public" if blob["config"] == "A" else "private"]
            elif col["sn"] != sn_ref:
                raise SystemExit(f"{f} is scored on different supernodes from the "
                                 f"first file read -- not comparable")
            runs.append((seed_of(f), score(col, h_target)))
        per_run[cfg] = runs
        n_runs[cfg] = len(runs)
        print(f"  {cfg}: {len(runs)} runs (seeds {[s for s, _ in runs]}), "
              f"{len(sn_ref)} supernodes")

    if not per_run:
        raise SystemExit("nothing to summarise")
    if len(set(n_runs.values())) > 1:
        print(f"  NOTE: configurations have different run counts: {n_runs}")

    os.makedirs(args.out_dir, exist_ok=True)
    cfgs = list(per_run)

    # ---- mean (or median) / std, one row per metric ---------------------------
    centre_fn = np.mean if args.stat == "mean" else np.median
    # One "mean ± std" cell per configuration, ready to paste into the sheet.
    header = ["Metric", "Real"] + [CONFIGS[c][0] for c in cfgs]
    rows = [header]
    for key, label, dp in METRICS:
        row = [label, fmt(real[key], dp)]
        for c in cfgs:
            vals = [r[key] for _, r in per_run[c]]
            if any(v is None for v in vals):       # E_h without a centroid
                row.append("")
                continue
            v = np.array(vals, float)
            # One run has no spread: the cell is the value alone.
            sd = fmt(v.std(ddof=1), dp) if len(v) > 1 else ""
            row.append(fmt(float(centre_fn(v)), dp) + (f" \u00b1 {sd}" if sd else ""))
        rows.append(row)
    rows.append([])
    rows.append([f"{len(sn_ref)} held-out {test_graph.capitalize()} supernodes; "
                 f"runs per configuration: {', '.join(f'{c}={n_runs[c]}' for c in cfgs)}; "
                 f"{args.stat} and sample std (ddof=1) across runs, i.e. sampling noise only; "
                 "E_h is blank where the samples mark no centroid"])
    summary = os.path.join(args.out_dir, f"repeats_summary{suffix}.tsv")
    with open(summary, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    # ---- every run's raw value, for checking ----------------------------------
    header = ["Metric"] + [f"{CONFIGS[c][0]} s{s}" for c in cfgs for s, _ in per_run[c]]
    rows = [header]
    for key, label, dp in METRICS:
        rows.append([label] + [fmt(r[key], dp) for c in cfgs for _, r in per_run[c]])
    per_seed = os.path.join(args.out_dir, f"repeats_per_seed{suffix}.tsv")
    with open(per_seed, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    print("\n" + open(summary).read())
    print(f"wrote {os.path.abspath(summary)}")
    print(f"wrote {os.path.abspath(per_seed)}")


if __name__ == "__main__":
    main()
