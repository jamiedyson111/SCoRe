"""Mean and std of the assembled-graph table across repeated private-inference runs.

Runs LOCALLY in the `kmis` env, once stages 1 and 2 have run N times on BlueBEAR
(run_private_clusters_repeats.sh, run_private_blocks_repeats.sh) and their files
are copied into results/samples/assembled/ (the release is in results/private_inference/):

    python syn_repeats_summary.py                         # B_v3-B and C_v3-C
    python syn_repeats_summary.py --configs B_v3-B        # one configuration

A configuration is <D_intra cfg>-<D_inter cfg>. Seed s of it is the graph
assembled from clusters_<graph>_k<k>_<D_intra cfg>_s<s>.pt (stage 1) and
blocks_<graph>_k<k>_<cfg>_s<s>.pt (stage 2, generated around those clusters).

Every graph is assembled and scored with assemble_syn.py's own functions --
assemble, stats, topology -- so the rows mean exactly what they mean in the
single-run reconstruction table:

    Nodes, Edges (internal = D_intra's, cross = D_inter's), Mean degree,
    Density, Components, GCC, APL,
    RE_GCC = |GCC(G) - GCC(G^)| / GCC(G)
    degW   = W1 between the degree distributions of G^ and G
    RE_APL = |APL(G) - APL(G^)| / APL(G)

BS1 is assemble_syn's Random baseline (every released count honoured exactly,
edges placed uniformly). BS2 places the same counts Chung-Lu style: each fine
node gets one expected degree D_u drawn from public Luxembourg's degree
distribution, and every block, internal and cross, picks its slots with
probability proportional to D_u * D_v (baselines/uncoarsen.py). Both are drawn
once per seed so they carry a spread too. Real is the private graph, one number. Each cell is the mean and sample std (ddof=1) across
seeds: sampling noise only, every seed uses the same trained checkpoints.

Output, in --out-dir:
    syn_summary.tsv     "mean ± std" cells, pastes into Excel
    syn_per_seed.tsv    every graph's raw value
    syn_table.tex       the LaTeX table
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from assemble_syn import (SEED, _load, assemble, load_true_graph, random_blocks,
                          stats, supernode_of, topology)

HERE = os.path.dirname(os.path.abspath(__file__))
COMPETITORS = os.path.join(HERE, '..', '..', 'competitors')
sys.path.insert(0, os.path.join(HERE, '..', '..', 'baselines'))
from uncoarsen import cross_block, internal_block, internal_edges, public_degrees  # noqa: E402
RESULTS = os.path.join(HERE, '..', '..', 'results')
OUT = os.path.join(RESULTS, 'private_inference')
SAMPLES = os.path.join(RESULTS, 'samples', 'assembled')

LABELS = {'B': 'B (Lux -> Olde)', 'C': 'C (Lux+Olde -> Olde)'}
BASELINE_LABELS = {'BS1': 'BS1 (random)', 'BS2': 'BS2 (Chung-Lu)'}
TEX_LABELS = {'B': r'B', 'C': r'C'}

# (key, label, tex label, decimals). Order is the row order of both tables.
ROWS = [('nodes',          'Nodes',               r'Nodes',                       0),
        ('edges',          'Edges',               r'Edges',                       1),
        ('edges_internal', '  internal (D_intra)', r'\quad internal (D\_intra)',  1),
        ('edges_cross',    '  cross (D_inter)',   r'\quad cross (D\_inter)',      1),
        ('mean_degree',    'Mean degree',         r'Mean degree',                 3),
        ('density',        'Density',             r'Density',                     6),
        ('components',     'Components',          r'Components',                  1),
        ('gcc',            'GCC',                 r'GCC',                         4),
        ('apl',            'APL',                 r'APL',                         2)]
ERROR_ROWS = [('regcc', 'RE_GCC (lower better)', r'$RE_{GCC}$ $\downarrow$',          4),
              ('degw',  'degW (lower better)',   r'$\mathrm{deg}W$ $\downarrow$',     4),
              ('reapl', 'RE_APL (lower better)', r'$RE_{APL}$ $\downarrow$',          4)]


def bs2_blocks(release, pool, rng):
    """BS2 for the whole release: one expected degree per fine node, used by every block."""
    D = {int(s['sn']): rng.choice(pool, size=int(s['n']), replace=True)
         for s in release['supernodes']}
    internal = {}
    for s in release['supernodes']:
        sn, n = int(s['sn']), int(s['n'])
        internal[sn] = torch.as_tensor(
            internal_block(n, internal_edges(s['rho_in'], n), rng, D[sn]), dtype=torch.long)
    blocks = {}
    for p in release['pairs']:
        si, sj = int(p['sn_i']), int(p['sn_j'])
        n_i, n_j, w = int(p['n_i']), int(p['n_j']), int(p['w'])
        blocks[(si, sj)] = {'n_i': n_i, 'n_j': n_j, 'w_requested': w,
                            'edge_index': torch.as_tensor(
                                cross_block(n_i, n_j, w, rng, D[si], D[sj]), dtype=torch.long)}
    return internal, blocks


def seed_of(path):
    return int(re.search(r'_s(\d+)\.pt$', path).group(1))


def score(A, A_real, sn_of, real, apl_sources):
    """Every row for one assembled graph. real is the Real column's dict, or None."""
    m = stats(A, sn_of)
    m.update(topology(A, np.random.default_rng(SEED), apl_sources))
    if real is not None:
        m['degw'] = wasserstein_distance(m['deg'], real['deg'])
        m['regcc'] = abs(real['gcc'] - m['gcc']) / real['gcc']
        m['reapl'] = abs(real['apl'] - m['apl']) / real['apl']
    return m


def load_run(release, samples_dir, cfg, seed):
    """(internal, blocks) for one seed of one configuration, checked for completeness."""
    icfg = cfg.split('-')[0]
    gk = f"{release['graph']}_k{release['k']}"
    cpath = os.path.join(samples_dir, f"clusters_{gk}_{icfg}_s{seed}.pt")
    bpath = os.path.join(samples_dir, f"blocks_{gk}_{cfg}_s{seed}.pt")
    for p in (cpath, bpath):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")
    bc, bb = _load(cpath), _load(bpath)
    for name, b in (('clusters', bc), ('blocks', bb)):
        if (b['graph'], b['k']) != (release['graph'], release['k']):
            raise SystemExit(f"{name} seed {seed} of {cfg} is for {b['graph']} "
                             f"k={b['k']}, the release is {gk}")
        if b.get('limit'):
            raise SystemExit(f"{name} seed {seed} of {cfg} was a +limit smoke test")
    # The cross blocks must have been generated around THESE internal blocks.
    if os.path.basename(str(bb.get('clusters', ''))) != os.path.basename(cpath):
        raise SystemExit(f"{bpath} was generated around {bb.get('clusters')}, "
                         f"not {os.path.basename(cpath)}")
    internal = {int(sn): v['edge_index'] for sn, v in bc['clusters'].items()}
    blocks = bb['blocks']
    if len(internal) != len(release['supernodes']):
        raise SystemExit(f"{cpath}: {len(internal)} of {len(release['supernodes'])} supernodes")
    if len(blocks) != len(release['pairs']):
        raise SystemExit(f"{bpath}: {len(blocks)} of {len(release['pairs'])} pairs")
    return internal, blocks


def load_competitor(path):
    """A competitor's generated graph (competitors/out/<METHOD>/graph_s<seed>.npz) as csr."""
    import scipy.sparse as sp
    z = np.load(path)
    n, e = int(z['n']), z['edges'].astype(np.int64)
    A = sp.coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def digress_oom(path):
    """True when competitors/digress_scale.tsv says one-shot DiGress cannot fit Oldenburg."""
    if not os.path.isfile(path):
        return False
    rows = [l.split('\t') for l in open(path) if l[0].isdigit()]
    return any(r[0] == '6105' and r[1] == 'OOM' for r in rows)


def cell(v, dp):
    if isinstance(v, str):
        return v
    v = np.asarray(v, float)
    if np.isnan(v).all():
        return ""                     # a row the column has no value for
    if len(v) == 1:
        return f"{v[0]:.{dp}f}"
    return f"{v.mean():.{dp}f} ± {v.std(ddof=1):.{dp}f}"


def tex_cell(v, dp):
    if isinstance(v, str):
        return v
    v = np.asarray(v, float)
    if np.isnan(v).all():
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{dp}f}"
    return f"{v.mean():.{dp}f} $\\pm$ {v.std(ddof=1):.{dp}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default=os.path.join(OUT, 'release_oldenburg_k7.pt'))
    ap.add_argument('--samples-dir', default=SAMPLES,
                    help='where the clusters_*/blocks_* files of every seed are')
    ap.add_argument('--configs', nargs='+', default=['B_v3-B', 'C_v3-C'])
    ap.add_argument('--graph-root',
                    default=os.path.join(HERE, '..', '..', 'D_intra', 'extraction'))
    ap.add_argument('--apl-sources', type=int, default=100,
                    help='BFS sources for APL above 6500 nodes; Oldenburg is exact')
    ap.add_argument('--competitors-dir', default=os.path.join(RESULTS, 'competitors'),
                    help='one subdirectory per full-graph competitor, holding '
                         'graph_s<seed>.npz (n, edges) files')
    ap.add_argument('--baseline-seeds', type=int, default=5,
                    help='BS1/BS2 seeds when no configuration has been sampled yet')
    ap.add_argument('--out-dir', default=os.path.join(RESULTS, 'D_inter', 'tables', 'syn'))
    args = ap.parse_args()

    release = _load(args.release)
    A_real = load_true_graph(release, os.path.abspath(args.graph_root))
    sn_of = supernode_of(release, A_real.shape[0])
    real = score(A_real, A_real, sn_of, None, args.apl_sources)
    real.update(degw=0.0, regcc=0.0, reapl=0.0)
    print(f"{release['graph']} k={release['k']}: {real['nodes']} nodes, "
          f"{real['edges']} edges, {len(release['pairs'])} connected pairs")

    columns, seeds_used = {}, {}
    for cfg in args.configs:
        files = sorted(glob.glob(os.path.join(
            args.samples_dir, f"blocks_{release['graph']}_k{release['k']}_{cfg}_s*.pt")),
            key=seed_of)
        if not files:
            print(f"  {cfg}: no blocks_*_{cfg}_s*.pt in {args.samples_dir}, skipped")
            continue
        runs = []
        for f in files:
            s = seed_of(f)
            A = assemble(release, *load_run(release, args.samples_dir, cfg, s))
            runs.append((s, score(A, A_real, sn_of, real, args.apl_sources)))
            print(f"  {cfg} seed {s}: {runs[-1][1]['edges']} edges, "
                  f"degW {runs[-1][1]['degw']:.4f}", flush=True)
        columns[cfg], seeds_used[cfg] = runs, [s for s, _ in runs]
    # Full-graph competitors (ICLR [2], HiGen): unconditional, so they never saw
    # the release, and their nodes are not Oldenburg's -- the internal/cross rows
    # have no meaning for them and stay blank. Everything else is scored as above.
    for d in sorted(glob.glob(os.path.join(args.competitors_dir, '*', ''))):
        name = os.path.basename(os.path.dirname(d))
        files = sorted(glob.glob(os.path.join(d, 'graph_s*.npz')),
                       key=lambda f: int(re.search(r'_s(\d+)\.npz$', f).group(1)))
        if not files:
            continue
        runs = []
        for f in files:
            m = score(load_competitor(f), A_real, None, real, args.apl_sources)
            m['edges_internal'] = m['edges_cross'] = float('nan')
            runs.append((int(re.search(r'_s(\d+)\.npz$', f).group(1)), m))
        columns[name], seeds_used[name] = runs, [s for s, _ in runs]
        print(f"  competitor {name}: seeds {seeds_used[name]}")
    if digress_oom(os.path.join(RESULTS, 'competitors', 'digress_scale.tsv')):
        columns['DiGress'] = 'OOM'
        print("  competitor DiGress: OOM at 6105 nodes (digress_scale.tsv)")

    # The baselines, once per seed of the longest configuration.
    rseeds = (max(seeds_used.values(), key=len) if seeds_used
              else list(range(args.baseline_seeds)))
    pool = public_degrees('luxembourg')
    columns['BS1'], columns['BS2'] = [], []
    for s in rseeds:
        A = assemble(release, *random_blocks(release, np.random.default_rng(s)))
        columns['BS1'].append((s, score(A, A_real, sn_of, real, args.apl_sources)))
        A = assemble(release, *bs2_blocks(release, pool, np.random.default_rng([s, 2])))
        columns['BS2'].append((s, score(A, A_real, sn_of, real, args.apl_sources)))
    print(f"  BS1, BS2: seeds {rseeds}")
    columns['Real'] = [(None, real)]

    names = list(columns)

    def label(c, tex=False):
        if c in BASELINE_LABELS:
            return c if tex else BASELINE_LABELS[c]
        if c == 'Real' or '-' not in c:          # Real, or a full-graph competitor
            return c
        icfg, inter = c.split('-')
        known = icfg == f"{inter}_v3" and inter in LABELS
        return ((TEX_LABELS if tex else LABELS)[inter] if known else c)

    os.makedirs(args.out_dir, exist_ok=True)
    def vals(c, key):
        """The per-seed values of one row, or the column's text (DiGress: 'OOM')."""
        return columns[c] if isinstance(columns[c], str) else [r[key] for _, r in columns[c]]
    seeded = [c for c in names if not isinstance(columns[c], str)]
    n_seeds = ', '.join(f"{c}={len(columns[c])}" for c in seeded if c != 'Real')

    # ---- TSV: "mean ± std" cells ---------------------------------------------
    rows = [['Metric'] + [label(c) for c in names]]
    for key, lab, _, dp in ROWS + ERROR_ROWS:
        rows.append([lab.strip()] + [cell(vals(c, key), dp) for c in names])
    rows += [[], [f"{release['graph']} k={release['k']}; seeds per column: {n_seeds}; "
                  "mean and sample std (ddof=1) across seeds, i.e. sampling noise only"]]
    summary = os.path.join(args.out_dir, 'syn_summary.tsv')
    with open(summary, 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in rows) + '\n')

    # ---- TSV: every graph -----------------------------------------------------
    head = ['Metric'] + [f"{label(c)} s{s}" if s is not None else label(c)
                         for c in seeded for s, _ in columns[c]]
    rows = [head] + [[lab.strip()] + [f"{r[key]:.{dp}f}" for c in seeded for _, r in columns[c]]
                     for key, lab, _, dp in ROWS + ERROR_ROWS]
    with open(os.path.join(args.out_dir, 'syn_per_seed.tsv'), 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in rows) + '\n')

    # ---- LaTeX ------------------------------------------------------------------
    def body(spec):
        return '\n'.join(f"{tlab:<30}" + ''.join(
            f"& {tex_cell(vals(c, key), dp)} " for c in names) + r'\\'
            for key, _, tlab, dp in spec)
    n_rep = len(rseeds)
    tex = rf"""\begin{{table}}[H]
\centering
\caption{{$\textbf{{Reconstruction quality of the assembled synthetic graph}}$
$\widehat{{G}}$ $\textbf{{ against the private graph }}$ $G$ ({release['graph'].capitalize()}, $k{{=}}{release['k']}$).
B uses D\_intra and D\_inter trained on Luxembourg only; C adds the Oldenburg
training split to both. Each synthetic column is the mean $\pm$ std over {n_rep}
independently sampled graphs (sampling noise; one trained model per column).
BS1 and BS2 reproduce every released count exactly---$\rho_{{\mathrm{{in}}}}$
per supernode and $w_{{ij}}$ per pair---so they differ from the synthetic columns
only in \emph{{where}} the edges go: BS1 places them uniformly, BS2 with
probability $\propto D_uD_v$ for expected degrees $D_u$ drawn from the public
graph's degree distribution (Chung--Lu). Internal
edges are D\_intra's, cross edges D\_inter's. GCC is the global clustering
coefficient $3\tau_{{\triangle}}/\tau_{{3}}$; APL is the mean geodesic distance,
computed exactly over all reachable pairs. Lower is better in the bottom block,
where the Real column is $0$ by construction.}}
\label{{tab:lsyn-{release['graph']}-k{release['k']}}}
\begin{{tabular}}{{l{'r' * len(names)}}}
\toprule
 & {' & '.join(label(c, tex=True) for c in names)} \\
\midrule
{body(ROWS)}
\midrule
{body(ERROR_ROWS)}
\bottomrule
\end{{tabular}}
\end{{table}}
"""
    with open(os.path.join(args.out_dir, 'syn_table.tex'), 'w') as f:
        f.write(tex)

    print('\n' + open(summary).read())
    print(f"wrote {os.path.abspath(args.out_dir)}/syn_summary.tsv, syn_per_seed.tsv, syn_table.tex")


if __name__ == '__main__':
    main()
