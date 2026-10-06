"""Median and standard deviation of every metric across repeated sampling runs.

Run LOCALLY in the `kmis` env, after scripts/run_olde_repeats.sh has run on
BlueBEAR and its files have been copied to results/samples/oldenburg/repeats/
as <CFG>_s<seed>.txt:

    python repeat_summary.py                          # B and C, on Oldenburg
    python repeat_summary.py --configs A_nohop A_v3   # A, on Luxembourg
    python repeat_summary.py --configs B C --samples-dir ../results/samples/oldenburg/repeats

Each run is scored exactly as the single-run tables score it -- the metric
definitions are imported from make_intra_table.py and evaluate_clusters.py, not
reimplemented -- and then every metric is summarised across the runs as its
median and its sample standard deviation (ddof=1).

The output is TAB-separated so it pastes into Excel as cells:

    results/tables/oldenburg/repeats_summary.tsv     one row per metric,
                                                     median and std columns
    results/tables/oldenburg/repeats_per_seed.tsv    every run's raw value

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

from evaluate_clusters import dist_to_real
from make_intra_table import degree_w1, load_column, stats

# (key, column label, letter of the pairs file, nohop pairs file?)
# v3 reads the same pairs file as the base model: mark_centroid and the hop
# embedding are applied at load / model time, not stored in the dataset.
CONFIGS = {
    "A":       ("A (hops)",    "A", False),
    "A_nohop": ("A (no hops)", "A", True),
    "A_v3":    ("A v3",        "A", False),
    "B":       ("B (hops)",    "B", False),
    "B_nohop": ("B (no hops)", "B", True),
    "B_v3":    ("B v3",        "B", False),
    "C":       ("C (hops)",    "C", False),
    "C_nohop": ("C (no hops)", "C", True),
    "C_v3":    ("C v3",        "C", False),
}

# (key, label, decimals). Order is the row order of the spreadsheet.
METRICS = [
    ("m_median",   "Edges (median)",            1),
    ("m_std",      "Edges (std across blocks)", 3),
    ("m_mean",     "Edges (mean)",              3),
    ("density",    "Density (mean)",            4),
    ("deg_median", "Degree (median)",           1),
    ("deg_mean",   "Degree (mean)",             4),
    ("connected",  "Connected (%)",             2),
    ("forest",     "Acyclic (%)",               2),
    ("components", "Components (mean)",         4),
    ("m_mae",      "Edge-count MAE",            4),
    ("deg_w1",     "Degree dist. vs real (W1)", 4),
    ("spec",       "Spectrum vs real",          4),
]


def score(col):
    """Every metric for one run, from load_column's output."""
    gen, real = col["gen"], col["real"]
    out = stats(gen)
    out["components"] = float(np.mean([s["n_components"] for s in gen]))
    out["m_mae"] = float(np.mean([abs(g["m"] - r["m"]) for g, r in zip(gen, real)]))
    out["deg_w1"] = degree_w1(gen, real)
    out["spec"] = dist_to_real(gen, real)[1]
    return out


def real_row(col):
    """The real blocks scored the same way. Distances to themselves are 0."""
    real = col["real"]
    out = stats(real)
    out["components"] = float(np.mean([s["n_components"] for s in real]))
    out["m_mae"] = 0.0
    out["deg_w1"] = 0.0
    out["spec"] = 0.0
    return out


def seed_of(path):
    return int(re.search(r"_s(\d+)\.txt$", path).group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="../results/samples/oldenburg/repeats")
    ap.add_argument("--pairs-dir", default="../data/cluster_lux/raw")
    ap.add_argument("--tag", default="lux2olde_k7")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS),
                    # No hops vs v3 is the comparison; the raw-histogram
                    # models (A, B, C) can still be named explicitly.
                    default=["B_nohop", "B_v3", "C_nohop", "C_v3"])
    ap.add_argument("--out-dir", default="../results/tables/oldenburg")
    args = ap.parse_args()
    test_on_public = {CONFIGS[c][1] == "A" for c in args.configs}
    if len(test_on_public) > 1:
        raise SystemExit("A is tested on the public graph and B/C on the private one; "
                         "summarise them in separate runs")
    suffix = "_A" if test_on_public.pop() else ""

    per_run, sn_ref, real, n_runs, test_graph = {}, None, None, {}, None
    for cfg in args.configs:
        label, letter, nohop = CONFIGS[cfg]
        files = sorted(glob.glob(os.path.join(args.samples_dir, f"{cfg}_s*.txt")), key=seed_of)
        if not files:
            print(f"  {cfg}: no {cfg}_s*.txt in {args.samples_dir}, skipped")
            continue
        pairs = os.path.join(args.pairs_dir,
                             f"pairs_{letter}_{args.tag}{'_nohop' if nohop else ''}.pt")
        runs = []
        for f in files:
            col = load_column(f, pairs)
            # Every run of every configuration must be scored on the same blocks,
            # or the medians are of different things.
            if sn_ref is None:
                sn_ref, real = col["sn"], real_row(col)
                blob = col["blob"]
                test_graph = blob["public" if blob["config"] == "A" else "private"]
            elif col["sn"] != sn_ref:
                raise SystemExit(f"{f} is scored on different supernodes from the "
                                 f"first file read -- not comparable")
            runs.append((seed_of(f), score(col)))
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

    # ---- median / std, one row per metric ------------------------------------
    header = ["Metric", "Real"]
    for c in cfgs:
        header += [f"{CONFIGS[c][0]} median", f"{CONFIGS[c][0]} std"]
    rows = [header]
    for key, label, dp in METRICS:
        row = [label, f"{real[key]:.{dp}f}"]
        for c in cfgs:
            v = np.array([r[key] for _, r in per_run[c]], float)
            # One run has no spread; a blank cell reads better in Excel than nan.
            sd = f"{v.std(ddof=1):.{dp}f}" if len(v) > 1 else ""
            row += [f"{np.median(v):.{dp}f}", sd]
        rows.append(row)
    rows.append([])
    rows.append([f"{len(sn_ref)} held-out {test_graph.capitalize()} supernodes; "
                 f"runs per configuration: {', '.join(f'{c}={n_runs[c]}' for c in cfgs)}; "
                 "std is the sample std (ddof=1) across runs, i.e. sampling noise only"])
    summary = os.path.join(args.out_dir, f"repeats_summary{suffix}.tsv")
    with open(summary, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    # ---- every run's raw value, for checking ----------------------------------
    header = ["Metric"] + [f"{CONFIGS[c][0]} s{s}" for c in cfgs for s, _ in per_run[c]]
    rows = [header]
    for key, label, dp in METRICS:
        rows.append([label] + [f"{r[key]:.{dp}f}" for c in cfgs for _, r in per_run[c]])
    per_seed = os.path.join(args.out_dir, f"repeats_per_seed{suffix}.tsv")
    with open(per_seed, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    print("\n" + open(summary).read())
    print(f"wrote {os.path.abspath(summary)}")
    print(f"wrote {os.path.abspath(per_seed)}")


if __name__ == "__main__":
    main()
