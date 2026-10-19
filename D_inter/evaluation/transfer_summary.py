"""Mean and standard deviation of D_inter's metrics across repeated sampling runs.

Run LOCALLY in the `kmis` env once scripts/run_pairs_repeats.sh has run on
BlueBEAR and its files are copied to ../results/samples/D_inter/transfer/<CFG>_s<seed>.pt:

    python transfer_summary.py                      # B and C, on Oldenburg
    python transfer_summary.py --configs A          # A, on Luxembourg
    python transfer_summary.py --configs B C BS1 BS2              # with the baselines
    python transfer_summary.py --configs A A_BS1 A_BS2

Every generated cross block is scored against the real block of the SAME test
pair (matched by position in the test split, checked on sn_i, sn_j, n_i, n_j).
Eight rows:

    Cross edges (mean/std)     |B_ij| over the test pairs
    Cross density (mean/std)   |B_ij| / (n_i n_j) over the test pairs
    D_out (mean/std)           cross out-degree of a V_i node, pooled over pairs
    Degree dist. vs real (W1)  cross-degree over both sides, pooled
    Out-degree profile E_r     the r analogue of D_intra's hop error E_h:

        E_r = 1/N_test sum_i sum_{d=0}^{4} | r^_i(d) - r_i(d) |

    where r_i = r_{i->j} is make_pair_blocks.py's out-degree profile of test
    pair i -- the fraction of V_i nodes with exactly d cross edges into V_j,
    d >= 4 pooled into the last bin -- and r^_i the same for its generated
    block. 0 is a perfect match, 2 the maximum. On road networks almost every
    out-degree is 0 or 1, so r is nearly fixed by (n_i, w_ij) and E_r mostly
    tracks edge-count error, scaled by 1/n_i.

The std rows are spreads ACROSS PAIRS within one run (ddof=1). Each row is then
summarised across runs as the mean and sample std (ddof=1). Degree-W1 uses
pair_stats.py's own definitions (real_pair_stats / summarise are imported, not
copied), so it lines up with the existing Luxembourg table.

Output, tab-separated so it pastes into Excel as cells ("mean ± std" per config):
    ../results/D_inter/tables/transfer/pairs_summary.tsv      (pairs_summary_A.tsv for A)
    ../results/D_inter/tables/transfer/pairs_per_seed.tsv     (pairs_per_seed_A.tsv)

A is tested on Luxembourg's 1091 pairs and B/C on Oldenburg's 276, so they are
written to separate sheets and asking for both at once is refused. The std is
sampling noise only: every run of a configuration uses one trained checkpoint.
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pair_stats import real_pair_stats, summarise  # noqa: E402

PAIRS = {"A": "pairs_inter_luxembourg_osm_k7.pt",
         "B": "pairs_inter_B_lux2olde_k7.pt",
         "C": "pairs_inter_C_lux2olde_k7.pt",
         # baselines/make_baseline_samples.py, on the same test pairs
         "BS1": "pairs_inter_B_lux2olde_k7.pt",
         "BS2": "pairs_inter_B_lux2olde_k7.pt",
         "A_BS1": "pairs_inter_luxembourg_osm_k7.pt",
         "A_BS2": "pairs_inter_luxembourg_osm_k7.pt"}
LABELS = {"A": "A (Lux -> Lux)", "B": "B (Lux -> Olde)", "C": "C (Lux+Olde -> Olde)",
          "BS1": "BS1 (random)", "BS2": "BS2 (Chung-Lu)",
          "A_BS1": "BS1 (random)", "A_BS2": "BS2 (Chung-Lu)"}

R_BINS = 5      # r_0 .. r_3, r_{>=4}: make_pair_blocks.py's R_BINS

# (key, label, decimals). Order is the row order of the sheet.
METRICS = [
    ("edges_mean", "Cross edges (mean)",         4),
    ("edges_std",  "Cross edges (std)",          4),
    ("rho_mean",   "Cross density (mean)",       5),
    ("rho_std",    "Cross density (std)",        5),
    ("dout_mean",  "D_out (mean)",               4),
    ("dout_std",   "D_out (std)",                4),
    ("deg_w1",     "Degree dist. vs real (W1)",  5),
    ("e_r",        "Out-degree profile E_r",     4),
]


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def gen_entry(b):
    """real_pair_stats' tuple for a generated block (edge_index: i-side x j-side)."""
    n_i, n_j = int(b["n_i"]), int(b["n_j"])
    src, dst = b["edge_index"][0].numpy(), b["edge_index"][1].numpy()
    d_ij = np.bincount(src, minlength=n_i)[:n_i]
    d_ji = np.bincount(dst, minlength=n_j)[:n_j]
    return int(len(src)), n_i, n_j, d_ij, np.concatenate([d_ij, d_ji])


def r_profile(d_ij):
    """r_{i->j}: fraction of V_i nodes with cross out-degree 0, 1, .. >=R_BINS-1."""
    return np.bincount(np.clip(d_ij, 0, R_BINS - 1), minlength=R_BINS) / len(d_ij)


def column(entries):
    """The descriptive rows of one column (real, or one generated run)."""
    s = summarise(entries)
    rho = np.array([e[0] / (e[1] * e[2]) for e in entries])
    dout = np.concatenate([e[3] for e in entries])
    return dict(edges_mean=s["edges_mean"], edges_std=s["edges_std"],
                rho_mean=s["rho_mean"], rho_std=float(rho.std(ddof=1)),
                dout_mean=s["dout_mean"], dout_std=float(dout.std(ddof=1)),
                deg=s["deg"])


def score(gen, real):
    """Every metric for one run; gen and real are aligned entry lists."""
    g, r = column(gen), column(real)
    out = {k: g[k] for k, _, _ in METRICS if k in g}
    out["deg_w1"] = float(wasserstein_distance(g["deg"], r["deg"]))
    out["e_r"] = float(np.mean([np.abs(r_profile(eg[3]) - r_profile(er[3])).sum()
                                for eg, er in zip(gen, real)]))
    return out


def real_row(real):
    r = column(real)
    out = {k: r[k] for k, _, _ in METRICS if k in r}
    out.update(deg_w1=0.0, e_r=0.0)
    return out


def seed_of(path):
    return int(re.search(r"_s(\d+)\.pt$", path).group(1))


def fmt(v, dp):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dp}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default="../../results/samples/D_inter/transfer")
    ap.add_argument("--pairs-dir", default="../data/pair_lux/raw")
    ap.add_argument("--configs", nargs="+", choices=list(PAIRS), default=["B", "C"])
    ap.add_argument("--out-dir", default="../../results/D_inter/tables/transfer")
    args = ap.parse_args()
    on_lux = {c.startswith("A") for c in args.configs}
    if len(on_lux) > 1:
        raise SystemExit("A is tested on Luxembourg and B/C on Oldenburg; "
                         "summarise them in separate runs")
    suffix = "_A" if on_lux.pop() else ""

    per_run, real, key_ref = {}, None, None
    for cfg in args.configs:
        files = sorted(glob.glob(os.path.join(args.samples_dir, f"{cfg}_s*.pt")), key=seed_of)
        if not files:
            print(f"  {cfg}: no {cfg}_s*.pt in {args.samples_dir}, skipped")
            continue
        test = _load(os.path.join(args.pairs_dir, PAIRS[cfg]))["splits"][2]
        real_entries = [real_pair_stats(r) for r in test]
        # B and C must be scored on the same pairs, in the same order.
        keys = [(int(r["sn_i"]), int(r["sn_j"])) for r in test]
        if key_ref is None:
            key_ref, real = keys, real_row(real_entries)
        elif keys != key_ref:
            raise SystemExit(f"{cfg}'s test split is not the same pairs as the first "
                             f"configuration's -- not comparable")
        runs = []
        for f in files:
            blob = _load(f)
            if os.path.basename(blob["pairs_file"]) != PAIRS[cfg]:
                raise SystemExit(f"{f} was sampled from {blob['pairs_file']}, "
                                 f"not {PAIRS[cfg]}")
            blocks = blob["blocks"]
            if len(blocks) != len(test):
                raise SystemExit(f"{f}: {len(blocks)} blocks for {len(test)} test pairs")
            for b, r in zip(blocks, test):
                if (b["sn_i"], b["sn_j"], b["n_i"], b["n_j"]) != (
                        int(r["sn_i"]), int(r["sn_j"]), int(r["n_i"]), int(r["n_j"])):
                    raise SystemExit(f"{f}: block {b['idx']} does not match its test pair")
            runs.append((seed_of(f), score([gen_entry(b) for b in blocks], real_entries)))
        per_run[cfg] = runs
        print(f"  {cfg}: {len(runs)} runs (seeds {[s for s, _ in runs]}), {len(test)} pairs")

    if not per_run:
        raise SystemExit("nothing to summarise")
    os.makedirs(args.out_dir, exist_ok=True)
    cfgs = list(per_run)

    # One "mean ± std" cell per configuration, as in the D_intra sheet.
    header = ["Metric", "Real"] + [LABELS[c] for c in cfgs]
    rows = [header]
    for key, label, dp in METRICS:
        row = [label, fmt(real[key], dp)]
        for c in cfgs:
            v = np.array([r[key] for _, r in per_run[c]], float)
            sd = fmt(v.std(ddof=1), dp) if len(v) > 1 else ""
            row.append(fmt(float(v.mean()), dp) + (f" \u00b1 {sd}" if sd else ""))
        rows.append(row)
    graph = "Luxembourg" if suffix else "Oldenburg"
    rows += [[], [f"{len(key_ref)} held-out {graph} pairs; runs per configuration: "
                  f"{', '.join(f'{c}={len(per_run[c])}' for c in cfgs)}; mean and sample "
                  "std (ddof=1) across runs, i.e. sampling noise only"]]
    summary = os.path.join(args.out_dir, f"pairs_summary{suffix}.tsv")
    with open(summary, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    header = ["Metric"] + [f"{LABELS[c]} s{s}" for c in cfgs for s, _ in per_run[c]]
    rows = [header] + [[label] + [fmt(r[key], dp) for c in cfgs for _, r in per_run[c]]
                       for key, label, dp in METRICS]
    per_seed = os.path.join(args.out_dir, f"pairs_per_seed{suffix}.tsv")
    with open(per_seed, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")

    print("\n" + open(summary).read())
    print(f"wrote {os.path.abspath(summary)}")
    print(f"wrote {os.path.abspath(per_seed)}")


if __name__ == "__main__":
    main()
