"""Summary statistics for generated D_inter cross blocks vs real held-out ones.

Run LOCALLY in the `kmis` env (numpy / scipy / torch, plus pandas+matplotlib for
the optional CSV and plot output):

    python pair_stats.py --generated generated_samples1.txt \
                         --pairs data/pair_lux/raw/pairs_inter_luxembourg_osm_k7.pt \
                         --split test --out-dir results/pair_stats --plots

Reports four things.

  1 EDGES PER PAIR          median and std of |B_ij|.

  2 BIPARTITE DENSITY       rho_cross = |B_ij| / (n_i * n_j).  The bipartite
                            denominator, i.e. the same normalisation the training
                            CE uses.  Should sit near the 0.0045 cross-block
                            marginal that seeded the transition matrix; a visible
                            drift means the noise model and the sampler disagree.

  3 D_out                   median of d_{i->j}(u) = sum_v B_ij(u,v), pooled over
                            every u of every V_i, zeros included.  Definition is
                            imported from evaluate_pairs.py so the two scripts
                            cannot drift apart.

  4 DEGREE DISTRIBUTION     W_1 between the pooled cross-degree multi-set of all
                            generated pairs and that of all real ones.  Zero is
                            ideal, lower is better.

                            This is the D_inter analogue of the D_intra degree
                            distribution metric.  It cannot be the whole-graph
                            degree multi-set: L_i and L_j are handed to the model
                            as observed context, so whole-graph degrees are
                            dominated by the input and would flatter any model.
                            The model-attributable part is the cross-degree, so
                            that is what is pooled -- over BOTH sides, unlike
                            D_out which looks only at V_i.  Pooling both sides
                            also makes the metric invariant to which supernode
                            was labelled i, which the arbitrary side assignment
                            demands.

LIMITATION.  generated_samples.txt records X and E but not the condition each
sample was drawn with, so every comparison here is distributional -- no statistic
is matched sample-to-record.  Checking per-sample that w_ij was honoured, or that
L_i / L_j survived sampling untouched, needs the condition logged at generation
time.
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_pairs import parse_generated, out_degrees_from_dense  # noqa: E402

SEED = 0
CROSS_MARGINAL = 0.0045   # what MarginalUniformTransition was seeded with


# --------------------------------------------------------------------------
# real side
# --------------------------------------------------------------------------
def real_pair_stats(rec):
    """(cross edge count, n_i, n_j, out-degrees over V_i, cross-degrees over V_i+V_j).

    edge_index stores both directions, so a cross edge appears once as i->j and
    once as j->i.  |B_ij| is the count of one orientation.
    """
    n_i, n_j = int(rec['n_i']), int(rec['n_j'])
    src, dst = rec['edge_index'][0].numpy(), rec['edge_index'][1].numpy()

    fwd = (src < n_i) & (dst >= n_i)
    rev = (src >= n_i) & (dst < n_i)

    d_ij = np.bincount(src[fwd], minlength=n_i)[:n_i]
    d_ji = np.bincount(src[rev] - n_i, minlength=n_j)[:n_j]

    return int(fwd.sum()), n_i, n_j, d_ij, np.concatenate([d_ij, d_ji])


def gen_pair_stats(side, E):
    """Same tuple as real_pair_stats, from a dense generated sample."""
    vi = np.flatnonzero(side == 0)
    vj = np.flatnonzero(side == 1)
    if len(vi) == 0 or len(vj) == 0:
        return None

    d_ij = out_degrees_from_dense(side, E)
    d_ji = (E[np.ix_(vj, vi)] == 1).sum(axis=1)

    return int(d_ij.sum()), len(vi), len(vj), d_ij, np.concatenate([d_ij, d_ji])


# --------------------------------------------------------------------------
# baselines, so the W_1 numbers are readable
# --------------------------------------------------------------------------
def _uniform_degrees(rec, rng):
    """Cross-degrees of a block with the right edge count placed at random."""
    n_i, n_j, w = int(rec['n_i']), int(rec['n_j']), int(rec['w'])
    slots = rng.integers(0, n_i * n_j, size=w)
    return np.concatenate([np.bincount(slots // n_j, minlength=n_i)[:n_i],
                           np.bincount(slots % n_j, minlength=n_j)[:n_j]])


def baselines(recs, real_deg, n_gen, rng, repeats=200):
    """Reference points for W_1, all evaluated at the SAME sample size as the model.

    W_1 is inflated by finite-sample noise, so a baseline computed over all
    len(recs) pairs is not comparable to a model scored on n_gen < len(recs)
    samples -- that comparison flatters the baseline and was, in the first
    version of this script, large enough to invent a gap that is not there.
    Each baseline is therefore drawn on n_gen pairs and averaged over `repeats`.

    The NOISE FLOOR is real pairs scored against the full real set. It is what a
    perfect model scores at this sample size, i.e. the point below which the
    metric can no longer resolve anything.
    """
    zero = np.zeros_like(real_deg)
    w1_zero = wasserstein_distance(zero, real_deg)

    unif, floor = [], []
    for _ in range(repeats):
        idx = rng.choice(len(recs), min(n_gen, len(recs)), replace=False)
        unif.append(wasserstein_distance(
            np.concatenate([_uniform_degrees(recs[i], rng) for i in idx]), real_deg))
        floor.append(wasserstein_distance(
            np.concatenate([real_pair_stats(recs[i])[4] for i in idx]), real_deg))
    return w1_zero, np.array(unif), np.array(floor)


# --------------------------------------------------------------------------
def zero_entry(rec):
    n_i, n_j = int(rec['n_i']), int(rec['n_j'])
    return 0, n_i, n_j, np.zeros(n_i, int), np.zeros(n_i + n_j, int)


def uniform_entry(rec, rng):
    """The right number of edges, given by w_ij, at uniformly random positions."""
    n_i, n_j, w = int(rec['n_i']), int(rec['n_j']), int(rec['w'])
    slots = rng.integers(0, n_i * n_j, size=w)
    d_ij = np.bincount(slots // n_j, minlength=n_i)[:n_i]
    d_ji = np.bincount(slots % n_j, minlength=n_j)[:n_j]
    return int(d_ij.sum()), n_i, n_j, d_ij, np.concatenate([d_ij, d_ji])


def summarise(entries):
    """Descriptive statistics for one column of the results table."""
    cnt = np.array([e[0] for e in entries])
    rho = np.array([e[0] / (e[1] * e[2]) for e in entries])
    dout = np.concatenate([e[3] for e in entries])
    deg = np.concatenate([e[4] for e in entries])
    return dict(
        n_pairs=len(entries),
        # Pairs that came back with at least one cross edge. Every pair in the
        # dataset is a CONNECTED pair -- w_ij >= 1 by construction -- so the real
        # column is 100% definitionally and the generated column measures how
        # often the model leaves a pair that should be joined completely
        # disconnected. The count is not comparable across columns when the
        # sample sizes differ; the percentage is.
        n_connected=int((cnt >= 1).sum()),
        pct_connected=100.0 * (cnt >= 1).mean(),
        edges_median=np.median(cnt), edges_std=cnt.std(ddof=1), edges_mean=cnt.mean(),
        rho_mean=rho.mean(),
        rho_pooled=cnt.sum() / sum(e[1] * e[2] for e in entries),
        dout_median=np.median(dout), dout_mean=dout.mean(), dout_max=int(dout.max()),
        deg=deg,
    )


def collect(pairs, is_real):
    entries, rows = [], []
    for item in pairs:
        got = real_pair_stats(item) if is_real else gen_pair_stats(*item)
        if got is None:
            continue
        w, n_i, n_j, d_ij, d_all = got
        entries.append(got)
        rows.append(dict(n_i=n_i, n_j=n_j, cross_edges=w, rho_cross=w / (n_i * n_j),
                         slots=n_i * n_j, mean_d_out=d_ij.mean(),
                         max_cross_degree=int(d_all.max())))
    return entries, rows


ROWS = [
    ("Connected pairs",       'n_connected',  "{:.0f}"),
    ("Connected pairs (%)",   'pct_connected', "{:.1f}"),
    ("Edges (median)",        'edges_median', "{:.0f}"),
    ("Edges (std)",           'edges_std',    "{:.3f}"),
    ("Edges (mean)",          'edges_mean',   "{:.3f}"),
    ("Density (mean)",        'rho_mean',     "{:.4f}"),
    ("Density (pooled)",      'rho_pooled',   "{:.5f}"),
    ("D_out (median)",        'dout_median',  "{:.0f}"),
    ("D_out (mean)",          'dout_mean',    "{:.4f}"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--out-dir', default='results/pair_stats')
    ap.add_argument('--plots', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        blob = torch.load(args.pairs, weights_only=False)
    except TypeError:
        blob = torch.load(args.pairs)
    recs = blob['splits'][{'val': 1, 'test': 2}[args.split]]

    r_entries, r_rows = collect(recs, True)
    g_entries, g_rows = collect(parse_generated(args.generated), False)
    if not g_entries:
        raise SystemExit("no usable generated pairs -- every sample was one-sided")

    rng = np.random.default_rng(SEED)

    # Two reference models, evaluated on the same real conditions. Uniform is
    # handed the true w_ij, so its edge counts and density match real exactly by
    # construction -- only WHERE the edges go differs.
    cols = {
        'gen':  summarise(g_entries),
        'unif': summarise([uniform_entry(r, rng) for r in recs]),
        'zero': summarise([zero_entry(r) for r in recs]),
        'real': summarise(r_entries),
    }
    r_deg = cols['real']['deg']
    for key in cols:
        cols[key]['w1'] = wasserstein_distance(cols[key]['deg'], r_deg)

    n_gen = len(g_entries)
    _, unif_m, floor_m = baselines(recs, r_deg, n_gen, rng)

    # ---- report -----------------------------------------------------------
    head = ['Generated', 'Uniform', 'Zero', 'Real']
    L = [f"{os.path.basename(args.generated)}: {n_gen} generated pairs "
         f"vs {len(r_entries)} real {args.split} pairs\n\n",
         f"{'':<20}" + "".join(f"{h:>12}" for h in head) + "\n",
         "-" * 68 + "\n"]
    for label, key, fmt in ROWS:
        L.append(f"{label:<20}" + "".join(
            f"{fmt.format(cols[c][key]):>12}" for c in ('gen', 'unif', 'zero', 'real')) + "\n")
    L.append("-" * 68 + "\n")
    L.append(f"{'Degree dist. vs real':<20}" + "".join(
        f"{cols[c]['w1']:>12.4f}" for c in ('gen', 'unif', 'zero', 'real')) + "\n")

    L.append(f"\ntransition-matrix prior for density: {CROSS_MARGINAL}\n")
    L.append(f"noise floor at n={n_gen} (real vs real): "
             f"{floor_m.mean():.5f} +/- {floor_m.std():.5f}\n")
    L.append(f"uniform at n={n_gen}:                   "
             f"{unif_m.mean():.5f} +/- {unif_m.std():.5f}\n")
    resolved = cols['gen']['w1'] > floor_m.mean() + 2 * floor_m.std()
    L.append(f"-> generated W_1 {cols['gen']['w1']:.5f} is "
             f"{'OUTSIDE' if resolved else 'INSIDE'} the noise floor: "
             f"{'a real difference' if resolved else 'indistinguishable from perfect at this n'}\n")

    report = "".join(L)
    print(report)
    with open(os.path.join(args.out_dir, 'stats_summary.txt'), 'w') as f:
        f.write(report)

    # ---- LaTeX ------------------------------------------------------------
    with open(os.path.join(args.out_dir, 'stats_table.tex'), 'w') as f:
        f.write(latex_table(cols, n_gen, len(r_entries), floor_m, args.split))
    with open(os.path.join(args.out_dir, 'stats_table_simple.tex'), 'w') as f:
        f.write(latex_table(cols, n_gen, len(r_entries), floor_m, args.split,
                            order=('gen', 'real'), label='dinter-stats-simple'))

    g_deg, r_deg = cols['gen']['deg'], cols['real']['deg']
    g_cnt = np.array([e[0] for e in g_entries])
    r_cnt = np.array([e[0] for e in r_entries])
    g_rho = np.array([e[0] / (e[1] * e[2]) for e in g_entries])
    r_rho = np.array([e[0] / (e[1] * e[2]) for e in r_entries])

    # ---- CSVs -------------------------------------------------------------
    try:
        import pandas as pd
        pd.DataFrame(g_rows).assign(source='generated').to_csv(
            os.path.join(args.out_dir, 'per_pair_generated.csv'), index=False)
        pd.DataFrame(r_rows).assign(source='real').to_csv(
            os.path.join(args.out_dir, 'per_pair_real.csv'), index=False)

        hi = max(g_deg.max(), r_deg.max())
        pd.DataFrame({
            'cross_degree': np.arange(hi + 1),
            'generated': np.bincount(g_deg, minlength=hi + 1) / len(g_deg),
            'real': np.bincount(r_deg, minlength=hi + 1) / len(r_deg),
        }).to_csv(os.path.join(args.out_dir, 'degree_distribution.csv'), index=False)
    except ImportError:
        print("(pandas missing -- CSVs skipped)")

    if args.plots:
        make_plots(args.out_dir, g_cnt, r_cnt, g_rho, r_rho, g_deg, r_deg)

    print(f"written to {args.out_dir}/")


TEX_ROWS = [
    (r"Connected pairs",                         'n_connected',  "{:.0f}"),
    (r"Connected pairs (\%)",                    'pct_connected', "{:.1f}"),
    (r"Edges (median)",                          'edges_median', "{:.0f}"),
    (r"Edges (std)",                             'edges_std',    "{:.3f}"),
    (r"Edges (mean)",                            'edges_mean',   "{:.3f}"),
    (r"Density (mean)",                          'rho_mean',     "{:.4f}"),
    (r"$D_{\mathrm{out}}$ (median)",             'dout_median',  "{:.0f}"),
    (r"$D_{\mathrm{out}}$ (mean)",               'dout_mean',    "{:.4f}"),
]

ORDER = ('gen', 'unif', 'zero', 'real')


HEADERS = {'gen': 'Generated', 'unif': 'Uniform', 'zero': 'Zero', 'real': 'Real'}

BASELINE_NOTE = r"""Uniform places the correct number of edges, given by
$w_{ij}$ in the condition, at uniformly random positions, so it matches the real
column on every count-based row by construction and differs only in \emph{where}
the edges go. Zero emits an empty cross block. """


def latex_table(cols, n_gen, n_real, floor, split, order=ORDER, label='dinter-stats'):
    body = "\n".join(
        f"{lab:<42}" + "".join(f"& {fmt.format(cols[c][key])} " for c in order) + r"\\"
        for lab, key, fmt in TEX_ROWS)
    w1 = "".join(f"& {cols[c]['w1']:.4f} " for c in order)
    note = BASELINE_NOTE if len(order) > 2 else ""

    return rf"""% D_inter cross-block statistics, {split} split
\begin{{table}}[H]
\centering
\caption{{Structural statistics of generated cross blocks against real held-out
blocks, on the Luxembourg $k{{=}}7$ {split} split. For the upper block, closer to
the real column is better. {note}The real column is compared against
itself and is therefore exactly $0$ on the lower block. Note that the generated
column is drawn from {n_gen} pairs against {n_real} real ones; resampling real
pairs down to {n_gen} gives a $W_1$ of ${floor.mean():.4f} \pm {floor.std():.4f}$
rather than $0$, which is the resolution limit of the metric at this sample
size.}}\label{{tab:{label}}}
\begin{{tabular}}{{l{'r' * len(order)}}}
\toprule
& {' & '.join(HEADERS[c] for c in order)} \\
\midrule
{body}
\midrule
Degree distribution vs.\ real $\downarrow$ {w1}\\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_plots(out_dir, g_cnt, r_cnt, g_rho, r_rho, g_deg, r_deg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    hi = int(max(g_cnt.max(), r_cnt.max()))
    bins = np.arange(-0.5, hi + 1.5)
    ax[0].hist(r_cnt, bins=bins, density=True, alpha=.55, label='real')
    ax[0].hist(g_cnt, bins=bins, density=True, alpha=.55, label='generated')
    ax[0].set(title='cross edges per pair', xlabel='$|B_{ij}|$', ylabel='fraction')
    ax[0].legend()

    ax[1].hist(r_rho, bins=40, density=True, alpha=.55, label='real')
    ax[1].hist(g_rho, bins=40, density=True, alpha=.55, label='generated')
    ax[1].set(title=r'bipartite density', xlabel=r'$\rho_{cross}$')
    ax[1].legend()

    hi = int(max(g_deg.max(), r_deg.max()))
    x = np.arange(hi + 1)
    ax[2].bar(x - .2, np.bincount(r_deg, minlength=hi + 1) / len(r_deg),
              width=.4, label='real')
    ax[2].bar(x + .2, np.bincount(g_deg, minlength=hi + 1) / len(g_deg),
              width=.4, label='generated')
    ax[2].set(title='pooled cross-degree', xlabel='degree', yscale='log')
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'pair_stats.png'), dpi=150)


if __name__ == "__main__":
    main()
