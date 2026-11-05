"""Relative errors of the whole assembled graph, mean ± std over seeds.

    python syn_re_summary.py ../../results/D_inter/tables/syn_s80_final/syn_per_seed.tsv

Reads syn_repeats_summary.py's per-seed table (whole graph) and writes, beside it,
syn_re_summary.tsv: one row per error, one "mean ± std" cell per column (B, C, BS1,
BS2, ...), every error computed PER SEED and then averaged (sample std, ddof=1):

    RE mean degree   |m^ - m| / m          mean degree 2m/n and density 2m/(n(n-1)) are
    RE density       |m^ - m| / m          fixed multiples of the edge count m, so their
                                           relative errors are identical; both are taken
                                           from the exact edge counts, not rounded values
    RE components    |c^ - c| / c
    RE_GCC, RE_APL   as syn_repeats_summary.py computes them
    degW             the degree-distribution W1 distance (a distance, not a relative error)
    Clustering, Orbit, Spectrum
                     graph_mmd.py's squared MMDs against the real graph (DiGress's
                     statistics); the spectrum is blank above graph_mmd.SPECTRUM_MAX_N nodes

All REs in %; degW and the MMDs as is.
"""
import os
import sys

import numpy as np

ROWS = [("RE mean degree (%)", "Edges",                  100, 2),
        ("RE density (%)",     "Edges",                  100, 2),
        ("RE components (%)",  "Components",             100, 2),
        ("RE_GCC (%)",         "RE_GCC (lower better)",  100, 2),
        ("degW",               "degW (lower better)",    1,   4),
        ("RE_APL (%)",         "RE_APL (lower better)",  100, 2),
        ("Clustering",         "Clustering MMD (lower better)", 1, 4),
        ("Orbit",              "Orbit MMD (lower better)",      1, 4),
        ("Spectrum",           "Spectrum MMD (lower better)",   1, 4)]
ALREADY_RE = {"RE_GCC (lower better)", "RE_APL (lower better)", "degW (lower better)",
              "Clustering MMD (lower better)", "Orbit MMD (lower better)",
              "Spectrum MMD (lower better)"}


def main():
    path = sys.argv[1]
    rows = {r[0]: r[1:] for r in (l.rstrip("\n").split("\t") for l in open(path)) if r[0]}
    hdr = rows["Metric"]
    cols = []
    for h in hdr[:-1]:                       # the last column is Real
        c = h.rsplit(" s", 1)[0]
        if c not in cols:
            cols.append(c)
    out = [["Error"] + cols]
    for label, src, scale, dp in ROWS:
        real = float(rows[src][-1])
        cells = []
        for c in cols:
            idx = [i for i, h in enumerate(hdr[:-1]) if h.rsplit(" s", 1)[0] == c]
            v = np.array([float(rows[src][i]) for i in idx])
            e = scale * (v if src in ALREADY_RE else np.abs(v - real) / real)
            if np.isnan(e).all():                  # e.g. no spectrum for a large graph
                cells.append("")
                continue
            sd = f" ± {e.std(ddof=1):.{dp}f}" if len(e) > 1 else ""
            cells.append(f"{e.mean():.{dp}f}{sd}")
        out.append([label] + cells)
    seeds = {c: sum(h.rsplit(" s", 1)[0] == c for h in hdr[:-1]) for c in cols}
    out += [[], [f"whole graph; seeds per column: {', '.join(f'{c}={n}' for c, n in seeds.items())}; "
                 "each error computed per seed, then mean and sample std (ddof=1)"]]
    dst = os.path.join(os.path.dirname(path), "syn_re_summary.tsv")
    open(dst, "w").write("\n".join("\t".join(r) for r in out) + "\n")
    print("\n".join("\t".join(r) for r in out) + f"\n\nwrote {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
