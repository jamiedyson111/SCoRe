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
    Clustering, Orbit, Spectrum: graph_mmd.py's squared MMDs of G^ against G (the
             DiGress statistics: local clustering coefficients, 4-node graphlet
             orbits, normalised-Laplacian spectrum), whole graph only

BS1 is Huiping's ER baseline and BS2 her SC-RG (competitors/Baseline/), run on
every supernode and pair of the release by baselines/full_graph.py. Both honour
every released count exactly (m_i from rho_in, w_ij). BS1 places the edges
uniformly; BS2 builds each supernode's hop layers from h_i and each cross block's
i-side out-degrees from r_{i->j}, measured on the real graph. Both are drawn once
per seed so they carry a spread too. Real is the private graph, one number. Each cell is the mean and sample std (ddof=1) across
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

from assemble_syn import (SEED, _load, assemble, load_true_graph,
                          stats, supernode_of, topology)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'baselines'))
import full_graph  # noqa: E402
import graph_mmd  # noqa: E402
RESULTS = os.path.join(HERE, '..', '..', 'results')
OUT = os.path.join(RESULTS, 'private_inference')
SAMPLES = os.path.join(RESULTS, 'samples', 'assembled')

LABELS = {'B': 'B (Lux -> Olde)', 'C': 'C (joint Lux+Olde -> Olde)',
          # the internal-degree D_inter (X_i = internal degree; C fine-tuned), run as
          #   python syn_repeats_summary.py --configs B_v3_s80-B_deg C_v3_s80-C_deg \
          #       --out-dir ../../results/D_inter/tables/syn_s80_deg
          'B_deg': 'B (Lux -> Olde)', 'C_deg': 'C (Lux, fine-tuned on Olde -> Olde)'}
BASELINE_LABELS = {'BS1': 'BS1 (ER)', 'BS2': 'BS2 (SC-RG)'}
TEX_LABELS = {'B': r'B', 'C': r'C', 'B_deg': r'B', 'C_deg': r'C'}

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
              ('reapl', 'RE_APL (lower better)', r'$RE_{APL}$ $\downarrow$',          4),
              ('clus_mmd',  'Clustering MMD (lower better)', r'Clustering $\downarrow$', 4),
              ('orbit_mmd', 'Orbit MMD (lower better)',      r'Orbit $\downarrow$',      4),
              ('spec_mmd',  'Spectrum MMD (lower better)',   r'Spectrum $\downarrow$',   4)]
MMD_KEYS = ('clus_mmd', 'orbit_mmd', 'spec_mmd')


def seed_of(path):
    return int(re.search(r'_s(\d+)\.pt$', path).group(1))


def score(A, A_real, sn_of, real, apl_sources, mmd=False):
    """Every row for one assembled graph. real is the Real column's dict, or None.
    mmd: also the clustering / orbit / spectrum discrepancies (real must carry
    'profiles'); NaN otherwise, which the tables leave blank."""
    m = stats(A, sn_of)
    m.update(topology(A, np.random.default_rng(SEED), apl_sources))
    m.update(dict.fromkeys(MMD_KEYS, float('nan')))
    if real is not None:
        m['degw'] = wasserstein_distance(m['deg'], real['deg'])
        m['regcc'] = abs(real['gcc'] - m['gcc']) / real['gcc']
        m['reapl'] = abs(real['apl'] - m['apl']) / real['apl']
        if mmd:
            m.update(graph_mmd.discrepancies(graph_mmd.profiles(A), real['profiles']))
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
    ap.add_argument('--configs', nargs='+', default=['B_v3_s80-B', 'C_v3_s80-C'])
    ap.add_argument('--split-file',
                    default=os.path.join(HERE, '..', '..', 'D_intra', 'data', 'splits',
                                         'oldenburg_k7_s80.json'),
                    help='the supernode split; its test supernodes define the test part')
    ap.add_argument('--graph-root',
                    default=os.path.join(HERE, '..', '..', 'D_intra', 'extraction'))
    ap.add_argument('--apl-sources', type=int, default=100,
                    help='BFS sources for APL above 6500 nodes; Oldenburg is exact')
    ap.add_argument('--baseline-seeds', type=int, default=5,
                    help='BS1/BS2 seeds when no configuration has been sampled yet')
    ap.add_argument('--out-dir', default=os.path.join(RESULTS, 'D_inter', 'tables', 'syn_s80'))
    args = ap.parse_args()

    release = _load(args.release)
    A_real = load_true_graph(release, os.path.abspath(args.graph_root))
    sn_of = supernode_of(release, A_real.shape[0])

    # Two parts are scored for every graph. 'whole': all of the graph. 'test': the
    # subgraph induced on the nodes of the held-out TEST supernodes -- their
    # generated insides and the generated cross edges between two test supernodes,
    # compared with the same induced subgraph of the real graph. No training
    # supernode is in it.
    import json
    split = json.load(open(args.split_file))
    test_sn = set(split['test'])
    test_idx = np.flatnonzero(np.isin(sn_of, list(test_sn)))
    parts = {'whole': None, 'test': test_idx}

    def cut(A, idx):
        return A if idx is None else A[idx][:, idx]

    real = {}
    for part, idx in parts.items():
        Ar = cut(A_real, idx)
        real[part] = score(Ar, Ar, sn_of if idx is None else sn_of[idx], None, args.apl_sources)
        real[part].update(degw=0.0, regcc=0.0, reapl=0.0)
        if part == 'whole':
            real[part]['profiles'] = graph_mmd.profiles(Ar)
            real[part].update(graph_mmd.discrepancies(real[part]['profiles'],
                                                      real[part]['profiles']))
    print(f"{release['graph']} k={release['k']}: {real['whole']['nodes']} nodes, "
          f"{real['whole']['edges']} edges; test part: {len(test_sn)} test supernodes, "
          f"{real['test']['nodes']} nodes, {real['test']['edges']} edges")

    def score_parts(A):
        out = {}
        for part, idx in parts.items():
            out[part] = score(cut(A, idx), cut(A_real, idx),
                              sn_of if idx is None else sn_of[idx], real[part], args.apl_sources,
                              mmd=part == 'whole')
        return out

    columns = {part: {} for part in parts}
    seeds_used = {}
    for cfg in args.configs:
        files = sorted(glob.glob(os.path.join(
            args.samples_dir, f"blocks_{release['graph']}_k{release['k']}_{cfg}_s*.pt")),
            key=seed_of)
        if not files:
            print(f"  {cfg}: no blocks_*_{cfg}_s*.pt in {args.samples_dir}, skipped")
            continue
        for part in parts:
            columns[part][cfg] = []
        for f in files:
            s_ = seed_of(f)
            m = score_parts(assemble(release, *load_run(release, args.samples_dir, cfg, s_)))
            for part in parts:
                columns[part][cfg].append((s_, m[part]))
            print(f"  {cfg} seed {s_}: {m['whole']['edges']} edges, degW whole "
                  f"{m['whole']['degw']:.4f}, test part {m['test']['degw']:.4f}", flush=True)
        seeds_used[cfg] = [s_ for s_, _ in columns['whole'][cfg]]

    # The baselines, once per seed of the longest configuration.
    rseeds = (max(seeds_used.values(), key=len) if seeds_used
              else list(range(args.baseline_seeds)))
    r = full_graph.out_degree_profile(A_real, release)
    for part in parts:
        columns[part]['BS1'], columns[part]['BS2'] = [], []
    for s_ in rseeds:
        # Huiping's seeds: run s uses 42 + s, as her scripts.
        m1 = score_parts(assemble(release, *full_graph.bs1_blocks(release, 42 + s_)))
        m2 = score_parts(assemble(release, *full_graph.bs2_blocks(release, r, 42 + s_)))
        for part in parts:
            columns[part]['BS1'].append((s_, m1[part]))
            columns[part]['BS2'].append((s_, m2[part]))
    print(f"  BS1, BS2: seeds {rseeds}")
    for part in parts:
        columns[part]['Real'] = [(None, real[part])]

    os.makedirs(args.out_dir, exist_ok=True)
    for part in parts:
        write_tables(columns[part], part, release, rseeds, len(test_sn), args.out_dir)


def label(c, tex=False):
    if c in BASELINE_LABELS:
        return c if tex else BASELINE_LABELS[c]
    if c == 'Real' or '-' not in c:
        return c
    icfg, inter = c.split('-')
    if icfg.startswith('Cft_v3') and inter == 'C_deg':     # both halves of C fine-tuned
        return 'C' if tex else 'C (fine-tuned)'
    known = icfg.startswith(f"{inter.split('_')[0]}_v3") and inter in LABELS
    return ((TEX_LABELS if tex else LABELS)[inter] if known else c)


def write_tables(columns, part, release, rseeds, n_test, out_dir):
    """syn_summary[_test].tsv, syn_per_seed[_test].tsv and syn_table[_test].tex."""
    sfx = '' if part == 'whole' else '_test'
    names = list(columns)

    def vals(c, key):
        """The per-seed values of one row, or the column's text if it has no runs."""
        return columns[c] if isinstance(columns[c], str) else [r[key] for _, r in columns[c]]
    seeded = [c for c in names if not isinstance(columns[c], str)]
    n_seeds = ', '.join(f"{c}={len(columns[c])}" for c in seeded if c != 'Real')
    what = (f"{release['graph']} k={release['k']}, whole graph" if part == 'whole' else
            f"{release['graph']} k={release['k']}, test part: the subgraph on the "
            f"{n_test} held-out test supernodes")

    # ---- TSV: "mean ± std" cells ---------------------------------------------
    rows = [['Metric'] + [label(c) for c in names]]
    for key, lab, _, dp in ROWS + ERROR_ROWS:
        rows.append([lab.strip()] + [cell(vals(c, key), dp) for c in names])
    rows += [[], [f"{what}; seeds per column: {n_seeds}; "
                  "mean and sample std (ddof=1) across seeds, i.e. sampling noise only"]]
    summary = os.path.join(out_dir, f'syn_summary{sfx}.tsv')
    with open(summary, 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in rows) + '\n')

    # ---- TSV: every graph -----------------------------------------------------
    head = ['Metric'] + [f"{label(c)} s{s_}" if s_ is not None else label(c)
                         for c in seeded for s_, _ in columns[c]]
    rows = [head] + [[lab.strip()] + [f"{r[key]:.{dp}f}" for c in seeded for _, r in columns[c]]
                     for key, lab, _, dp in ROWS + ERROR_ROWS]
    with open(os.path.join(out_dir, f'syn_per_seed{sfx}.tsv'), 'w') as f:
        f.write('\n'.join('\t'.join(r) for r in rows) + '\n')

    # ---- LaTeX ------------------------------------------------------------------
    def body(spec):
        return '\n'.join(f"{tlab:<30}" + ''.join(
            f"& {tex_cell(vals(c, key), dp)} " for c in names) + r'\\'
            for key, _, tlab, dp in spec)
    scope = ("" if part == 'whole' else
             rf" Restricted to the subgraph induced on the {n_test} held-out test supernodes"
             r" (their generated insides and the generated cross edges between them), against"
             r" the same subgraph of $G$; no training supernode appears in it.")
    if any('_deg' in c for c in names):
        c_desc = (r"B uses D\_intra and D\_inter trained on Luxembourg only. In C, D\_intra is "
                  r"trained jointly on Luxembourg and the Oldenburg training supernodes (80/20 "
                  r"split by supernode), and D\_inter is the Luxembourg model fine-tuned on the "
                  r"Oldenburg training pairs. D\_inter sees each node's normalised internal "
                  r"degree, computed from D\_intra's generated blocks.")
    else:
        c_desc = (r"B uses D\_intra and D\_inter trained on Luxembourg only; C trains both "
                  r"jointly, from scratch, on Luxembourg together with the Oldenburg training "
                  r"supernodes (80/20 split by supernode; no pre-training or fine-tuning stage).")
    tex = rf"""\begin{{table}}[H]
\centering
\caption{{$\textbf{{Reconstruction quality of the assembled synthetic graph}}$
$\widehat{{G}}$ $\textbf{{ against the private graph }}$ $G$ ({release['graph'].capitalize()}, $k{{=}}{release['k']}$).{scope}
{c_desc} Each synthetic column is the
mean $\pm$ std over {len(rseeds)} independently sampled graphs (sampling noise; one trained
model per column). BS1 and BS2 reproduce every released count exactly---$\rho_{{\mathrm{{in}}}}$
per supernode and $w_{{ij}}$ per pair---so they differ from the synthetic columns
only in \emph{{where}} the edges go: BS1 (Erd\H{{o}}s--R\'enyi $G(n,m)$) places them
uniformly; BS2 (SC-RG) places them subject to each supernode's hop layers $h_i$ and
each cross block's out-degree profile $r_{{i\to j}}$. Internal edges are D\_intra's, cross edges
D\_inter's. GCC is the global clustering coefficient $3\tau_{{\triangle}}/\tau_{{3}}$;
APL is the mean geodesic distance over reachable pairs. Lower is better in the
bottom block, where the Real column is $0$ by construction.}}
\label{{tab:lsyn-{release['graph']}-k{release['k']}{sfx.replace('_', '-')}}}
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
    with open(os.path.join(out_dir, f'syn_table{sfx}.tex'), 'w') as f:
        f.write(tex)

    print(f"\n== {what}\n" + open(summary).read())
    print(f"wrote {os.path.abspath(out_dir)}/syn_summary{sfx}.tsv, syn_per_seed{sfx}.tsv, "
          f"syn_table{sfx}.tex")


if __name__ == '__main__':
    main()
