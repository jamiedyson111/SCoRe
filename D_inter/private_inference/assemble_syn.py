"""STAGE 3 of private inference: assemble L^syn and score it against the private graph.

Runs LOCALLY in the `kmis` env (numpy / scipy / torch; pandas and matplotlib
optional). No GPU, no model.

    python assemble_syn.py --release  out/release_minnesota_k7.pt \
                           --clusters out/clusters_minnesota_k7.pt \
                           --blocks   out/blocks_minnesota_k7.pt \
                           --out-dir  out/syn_minnesota_k7

Assembly is the block matrix of the brief: each generated L_i on the diagonal,
each generated B_ij and its transpose off-diagonal, zero for any pair not
connected in the released coarse graph.

    L^syn = [ L_1   B_12  ...  B_1M ]
            [ B_12' L_2   ...  B_2M ]
            [ ...                   ]

NODE IDENTITY IS PRESERVED. Supernode i's local node u is written back to the
fine node id release['supernodes'][i]['nodes'][u], so L^syn has exactly the node
set of G^priv. That is what makes a degree-distribution comparison meaningful --
both multi-sets have the same length, so unlike the public-run pair metrics there
is no sample-size mismatch to correct for and a perfect model scores exactly 0.

    --real     ignore --clusters/--blocks and assemble from the private graph's
               TRUE blocks. This is the assembly self-test: it must reproduce
               G^priv edge for edge. If it does not, the bug is in this file, and
               no amount of staring at model output will reveal that.
"""
import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import wasserstein_distance

SEED = 0

# (label, key, format). One list, so the text table and the LaTeX cannot drift.
# These are the reconstruction-table rows: counts and shape first, then the three
# comparison metrics, which are errors against the real column and so are
# computed rather than measured per graph.
ROWS = [("Nodes", 'nodes', "{:d}"),
        ("Edges", 'edges', "{:d}"),
        ("  internal (D_intra)", 'edges_internal', "{:d}"),
        ("  cross (D_inter)", 'edges_cross', "{:d}"),
        ("Mean degree", 'mean_degree', "{:.4f}"),
        ("Density", 'density', "{:.6f}"),
        ("Components", 'components', "{:d}"),
        ("GCC", 'gcc', "{:.4f}"),
        ("APL", 'apl', "{:.3f}")]

ERROR_ROWS = [("REGCC", 'regcc', "{:.4f}"),
              ("degW", 'w1', "{:.4f}"),
              ("REAPL", 'reapl', "{:.4f}")]

# Measured too, and kept in the text report, but not part of the table above.
EXTRA_ROWS = [("Isolated nodes", 'isolated', "{:d}"),
              ("Largest comp. frac", 'largest_component_frac', "{:.4f}"),
              ("Reachable pairs", 'reachable_frac', "{:.4f}"),
              ("Triangles", 'triangles', "{:d}"),
              ("Avg. clustering", 'avg_clustering', "{:.4f}"),
              ("Degree assortativity", 'assortativity', "{:+.4f}")]

TEX_ROWS = [(r"Nodes", 'nodes', "{:d}"),
            (r"Edges", 'edges', "{:d}"),
            (r"\quad internal (D\_intra)", 'edges_internal', "{:d}"),
            (r"\quad cross (D\_inter)", 'edges_cross', "{:d}"),
            (r"Mean degree", 'mean_degree', "{:.3f}"),
            (r"Density", 'density', "{:.6f}"),
            (r"Components", 'components', "{:d}"),
            (r"GCC", 'gcc', "{:.4f}"),
            (r"APL", 'apl', "{:.2f}")]

TEX_ERROR_ROWS = [(r"REGCC $\downarrow$", 'regcc', "{:.4f}"),
                  (r"$\mathrm{deg}W$ $\downarrow$", 'w1', "{:.4f}"),
                  (r"REAPL $\downarrow$", 'reapl', "{:.4f}")]


def _load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_true_graph(release, graph_root):
    cwd = os.getcwd()
    sys.path.insert(0, graph_root)
    os.chdir(graph_root)
    try:
        from kmis_pairs import load_graph
        _, A = load_graph(release['graph'])
    finally:
        os.chdir(cwd)
    return A.tocsr()


def true_blocks(release, A):
    """The private graph's own L_i and B_ij, in the same shape the stages emit."""
    internal, blocks = {}, {}
    for s in release['supernodes']:
        mem = s['nodes'].numpy()
        sub = sp.csr_matrix(A[mem][:, mem]).tocoo()
        internal[int(s['sn'])] = (torch.zeros(2, 0, dtype=torch.long) if sub.nnz == 0
                                  else torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long))
    mem_of = {int(s['sn']): s['nodes'].numpy() for s in release['supernodes']}
    for p in release['pairs']:
        si, sj = int(p['sn_i']), int(p['sn_j'])
        sub = sp.csr_matrix(A[mem_of[si]][:, mem_of[sj]]).tocoo()
        blocks[(si, sj)] = {
            'n_i': int(p['n_i']), 'n_j': int(p['n_j']), 'w_requested': int(p['w']),
            'edge_index': torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long),
        }
    return internal, blocks


def _triu_pairs(n, m, rng):
    """m distinct unordered pairs from n nodes, without materialising the triangle."""
    total = n * (n - 1) // 2
    m = min(m, total)
    if m <= 0:
        return np.empty(0, int), np.empty(0, int)
    picks = rng.choice(total, size=m, replace=False)
    i = (n - 2 - np.floor(np.sqrt(-8 * picks + 4 * n * (n - 1) - 7) / 2.0 - 0.5)).astype(int)
    j = (picks + i + 1 - n * i + (i * (i + 1)) // 2).astype(int)
    return i, j


def random_blocks(release, rng):
    """Blocks that honour the release and nothing else -- the baseline column.

    Every count the release publishes is matched exactly: each internal block
    gets round(rho_in * n(n-1)/2) edges, each cross block gets w_ij, and both are
    placed uniformly at random. So this is not a straw man. It is what a
    generator scores by using the condition perfectly and learning nothing about
    structure, which is the only honest scale for the rows below -- a synthetic
    graph that merely matches edge counts lands here, not near the real column.
    """
    internal, blocks = {}, {}
    for s in release['supernodes']:
        n_i = int(s['n'])
        m = int(round(float(s['rho_in']) * n_i * (n_i - 1) / 2.0))
        i, j = _triu_pairs(n_i, m, rng)
        internal[int(s['sn'])] = torch.tensor(
            np.vstack([np.concatenate([i, j]), np.concatenate([j, i])]), dtype=torch.long)
    for p in release['pairs']:
        n_i, n_j, w = int(p['n_i']), int(p['n_j']), int(p['w'])
        slots = rng.choice(n_i * n_j, size=min(w, n_i * n_j), replace=False)
        blocks[(int(p['sn_i']), int(p['sn_j']))] = {
            'n_i': n_i, 'n_j': n_j, 'w_requested': w,
            'edge_index': torch.tensor(np.vstack([slots // n_j, slots % n_j]),
                                       dtype=torch.long),
        }
    return internal, blocks


def assemble(release, internal, blocks):
    """Global adjacency over the private graph's own node set."""
    n = int(release['n_fine'])
    mem_of = {int(s['sn']): s['nodes'].numpy() for s in release['supernodes']}

    src, dst = [], []
    for sn, ei in internal.items():
        if ei.numel() == 0:
            continue
        mem = mem_of[sn]
        src.append(mem[ei[0].numpy()])
        dst.append(mem[ei[1].numpy()])

    for (si, sj), blk in blocks.items():
        ei = blk['edge_index']
        if ei.numel() == 0:
            continue
        mi, mj = mem_of[si], mem_of[sj]
        u, v = mi[ei[0].numpy()], mj[ei[1].numpy()]
        src.append(u); dst.append(v)          # and the symmetric position
        src.append(v); dst.append(u)

    if not src:
        return sp.csr_matrix((n, n))
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    A = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def supernode_of(release, n):
    """node id -> supernode id, the partition every internal/cross split needs."""
    sn_of = np.full(n, -1, dtype=np.int64)
    for s in release['supernodes']:
        sn_of[s['nodes'].numpy()] = int(s['sn'])
    return sn_of


def stats(A, sn_of=None):
    n = A.shape[0]
    deg = np.asarray(A.sum(1)).ravel().astype(int)
    m = int(A.nnz // 2)
    ncomp, labels = connected_components(A, directed=False)
    _, sizes = np.unique(labels, return_counts=True)
    out = {
        'nodes': n, 'edges': m,
        'mean_degree': deg.mean(),
        'density': 2.0 * m / (n * (n - 1)) if n > 1 else 0.0,
        'isolated': int((deg == 0).sum()),
        'components': int(ncomp),
        'largest_component_frac': sizes.max() / n,
        'deg': deg,
    }
    if sn_of is not None:
        # Which block an edge lands in is which model produced it: an edge whose
        # endpoints share a supernode is D_intra's, one that crosses is
        # D_inter's. Splitting the count here makes the headline table an
        # attribution rather than a single pooled figure.
        U = sp.triu(A, k=1).tocoo()
        internal = int((sn_of[U.row] == sn_of[U.col]).sum())
        out['edges_internal'] = internal
        out['edges_cross'] = m - internal
    return out


# --------------------------------------------------------------------------
# structure beyond the degree sequence
# --------------------------------------------------------------------------
def triangles(A):
    """(triangles per node, total). A@A stays cheap on a near-planar graph."""
    B = A.astype(np.int64)
    per_node = np.asarray((B @ B).multiply(B).sum(1)).ravel() / 2.0
    return per_node, per_node.sum() / 3.0


def apl(A, rng, n_sources, exact_max_nodes=5000):
    """Average shortest-path length over reachable pairs, and that fraction.

        APL(G) = (1/|R|) sum_{(u,v) in R} d_G(u,v),   R = pairs with a finite path

    Exact -- every ordered pair -- whenever the graph is small enough that
    all-pairs BFS fits; Minnesota at 2642 nodes is. Luxembourg at 114599 is not:
    the distance matrix alone would be 1.3e10 entries, so there APL is estimated
    from n_sources BFS sweeps. Which one was used is reported, never assumed.

    Unreachable pairs are excluded, as the definition says, and the reachable
    fraction is returned beside it. The two have to be read together: a
    fragmented graph measures only within its fragments and so reports a
    SHORTER APL, not a longer one.
    """
    n = A.shape[0]
    exact = n <= exact_max_nodes
    idx = np.arange(n) if exact else rng.choice(n, size=min(n_sources, n), replace=False)
    d = shortest_path(A, method='D', unweighted=True, directed=False, indices=idx)
    off = ~np.eye(n, dtype=bool)[idx]
    finite = np.isfinite(d) & off
    return {
        'apl': d[finite].mean() if finite.any() else float('nan'),
        'apl_exact': exact,
        'reachable_frac': finite.sum() / max(off.sum(), 1),
    }


def topology(A, rng, n_sources=100):
    """Structure the degree sequence cannot see.

    GCC is the global clustering coefficient of the brief, 3 tau_triangle /
    tau_triple, where tau_triple counts connected triples -- i.e. transitivity,
    not the average of the local coefficients. Both are reported; only GCC feeds
    REGCC.

    Road networks are what makes these worth having: they are sparse,
    near-planar and almost triangle-free, so a generator can match the degree
    histogram exactly while producing something that is not a road network.
    """
    deg = np.asarray(A.sum(1)).ravel().astype(np.int64)
    tri_node, tri_total = triangles(A)

    tau3 = (deg * (deg - 1) // 2).sum()          # connected triples
    denom = np.where(deg >= 2, deg * (deg - 1) / 2.0, 1.0)
    local = np.where(deg >= 2, tri_node / denom, 0.0)

    # Degree assortativity: Pearson r over the degrees at the endpoints of every
    # edge, each edge counted in both orientations -- the standard undirected
    # form.
    src, dst = A.nonzero()
    r = np.corrcoef(deg[src], deg[dst])[0, 1] if len(src) > 1 else float('nan')

    out = {
        'triangles': int(round(tri_total)),
        'gcc': 3.0 * tri_total / tau3 if tau3 else 0.0,
        'avg_clustering': local.mean(),
        'assortativity': r,
    }
    out.update(apl(A, rng, n_sources))
    return out


# --------------------------------------------------------------------------
# edge-level agreement
# --------------------------------------------------------------------------
def edge_keys(A):
    """Upper-triangle edges as sorted int64 keys, for set arithmetic."""
    U = sp.triu(A, k=1).tocoo()
    return np.sort(U.row.astype(np.int64) * A.shape[0] + U.col.astype(np.int64))


def edge_agreement(A_syn, A_real, release):
    """Precision / recall against the real edge set, split by who produced it.

    This comparison is only available because assembly preserves node identity:
    L^syn and G^priv are graphs on the SAME vertices, so "did the model place
    this particular edge" is a well-posed question, not just "does the degree
    histogram match". Nothing else in the evaluation is this strict.

    The split matters more than the total. An internal edge is D_intra's to get
    right and a cross edge is D_inter's, so two numbers that would otherwise be
    pooled into one uninformative score become an attribution: which half of the
    pipeline is losing the edges.
    """
    n = A_real.shape[0]
    sn_of = supernode_of(release, n)

    def split(keys):
        u, v = keys // n, keys % n
        internal = sn_of[u] == sn_of[v]
        return {'all': keys, 'internal': keys[internal], 'cross': keys[~internal]}

    syn, real = split(edge_keys(A_syn)), split(edge_keys(A_real))

    out = {}
    for part in ('all', 'internal', 'cross'):
        g, r = syn[part], real[part]
        hit = np.intersect1d(g, r, assume_unique=True).size
        union = g.size + r.size - hit
        out[part] = {
            'n_syn': g.size, 'n_real': r.size, 'hit': hit,
            'precision': hit / g.size if g.size else float('nan'),
            'recall': hit / r.size if r.size else float('nan'),
            'jaccard': hit / union if union else 1.0,
        }
        p, rc = out[part]['precision'], out[part]['recall']
        out[part]['f1'] = 2 * p * rc / (p + rc) if p and rc and (p + rc) else 0.0
    return out


def er_baseline_degrees(n, m, rng):
    """Degrees of an Erdos-Renyi graph with the same n and m -- a scale for W_1.

    Shares _triu_pairs with the Random column rather than repeating the
    triangle-index inversion, which is exactly the kind of formula that is wrong
    in silence: the previous copy dropped an n and returned column indices past
    the end of the graph, so bincount returned an array LONGER than n and this
    baseline was measured against a degree multi-set that did not exist.
    """
    i, j = _triu_pairs(n, m, rng)
    return np.bincount(np.concatenate([i, j]), minlength=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', required=True)
    ap.add_argument('--clusters')
    ap.add_argument('--blocks')
    ap.add_argument('--real', action='store_true',
                    help='assemble from the private graph itself (assembly self-test)')
    ap.add_argument('--graph-root',
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         '..', '..', 'D_intra', 'extraction'),
                    help='where kmis_pairs.py and the raw graphs live; needed to '
                         'score against the real graph')
    ap.add_argument('--apl-sources', type=int, default=100,
                    help='BFS sources for APL when the graph is too big for '
                         'all-pairs (>5000 nodes); ignored when it is not')
    ap.add_argument('--no-baseline', action='store_true',
                    help='skip the random-from-release column (halves the runtime)')
    ap.add_argument('--out-dir', default='out/syn')
    args = ap.parse_args()

    release = _load(args.release)
    os.makedirs(args.out_dir, exist_ok=True)
    A_real = load_true_graph(release, os.path.abspath(args.graph_root))

    if args.real:
        internal, blocks = true_blocks(release, A_real)
        source = 'REAL blocks (assembly self-test)'
    else:
        assert args.clusters and args.blocks, "--clusters and --blocks, or --real"
        bc, bb = _load(args.clusters), _load(args.blocks)
        internal = {int(sn): v['edge_index'] for sn, v in bc['clusters'].items()}
        blocks = bb['blocks']
        source = f"generated ({os.path.basename(args.clusters)}, {os.path.basename(args.blocks)})"

    A_syn = assemble(release, internal, blocks)

    # Three columns: what the models produced, what the release alone buys you,
    # and the truth. The middle one is what stops every row being unreadable --
    # see random_blocks.
    columns = [('Synthetic', A_syn)]
    if not args.no_baseline:
        columns.append(('Random', assemble(release,
                                           *random_blocks(release,
                                                          np.random.default_rng(SEED)))))
    columns.append(('Real', A_real))

    real_deg = np.asarray(A_real.sum(1)).ravel().astype(int)
    sn_of = supernode_of(release, A_real.shape[0])
    M = {}
    for name, A in columns:
        m = stats(A, sn_of)
        # Same seed for every column, so where APL is sampled it is sampled from
        # the SAME sources and the columns differ only where the graphs do.
        m.update(topology(A, np.random.default_rng(SEED), args.apl_sources))
        m['w1'] = wasserstein_distance(m['deg'], real_deg)
        m['agree'] = edge_agreement(A, A_real, release)
        M[name] = m
    names = [n for n, _ in columns]

    # REGCC and REAPL are relative errors against the real graph, so the real
    # column is 0 by construction, exactly as degW is.
    def rel_err(got, ref):
        if ref == 0 or not np.isfinite(ref) or not np.isfinite(got):
            return float('nan')
        return abs(ref - got) / ref
    for n in names:
        M[n]['regcc'] = rel_err(M[n]['gcc'], M['Real']['gcc'])
        M[n]['reapl'] = rel_err(M[n]['apl'], M['Real']['apl'])

    rng = np.random.default_rng(SEED)
    w1_er = wasserstein_distance(er_baseline_degrees(M['Real']['nodes'],
                                                     M['Real']['edges'], rng), real_deg)
    w1_empty = wasserstein_distance(np.zeros_like(real_deg), real_deg)

    # ---- per-block fidelity -----------------------------------------------
    rho_of = {int(s['sn']): float(s['rho_in']) for s in release['supernodes']}
    n_of = {int(s['sn']): int(s['n']) for s in release['supernodes']}
    int_err, cross_err = [], []
    for sn, ei in internal.items():
        n_i = n_of[sn]
        want = rho_of[sn] * n_i * (n_i - 1) / 2.0
        int_err.append(ei.shape[1] // 2 - want)
    for (si, sj), blk in blocks.items():
        cross_err.append(blk['edge_index'].shape[1] - int(blk['w_requested']))
    int_err = np.array(int_err)
    cross_err = np.array(cross_err) if cross_err else np.zeros(1)

    exact_match = (A_syn != A_real).nnz == 0

    # ---- report -----------------------------------------------------------
    W = 56 + 14 * (len(names) - 2)
    L = [f"L^syn for {release['graph']} k={release['k']}\n",
         f"source: {source}\n\n",
         f"{'':<28}" + "".join(f"{n:>14}" for n in names) + "\n", "-" * W + "\n"]
    for label, key, fmt in ROWS:
        L.append(f"{label:<28}" + "".join(f"{fmt.format(M[n][key]):>14}"
                                          for n in names) + "\n")
    L.append("-" * W + "\n")
    for label, key, fmt in ERROR_ROWS:
        L.append(f"{label + ' (lower better)':<28}"
                 + "".join(f"{fmt.format(M[n][key]):>14}" for n in names) + "\n")

    apl_how = ("exact, all pairs" if M['Real']['apl_exact']
               else f"estimated from {args.apl_sources} BFS sources per graph")
    L.append(f"\n  APL: {apl_how}\n")
    L.append(f"  degW baselines: Erdos-Renyi (same n, m) {w1_er:.4f}, "
             f"empty graph {w1_empty:.4f}\n")

    L.append("\nALSO MEASURED (not part of the table above)\n")
    for label, key, fmt in EXTRA_ROWS:
        L.append(f"{label:<28}" + "".join(f"{fmt.format(M[n][key]):>14}"
                                          for n in names) + "\n")
    L.append(f"{'Edge F1 vs real':<28}"
             + "".join(f"{M[n]['agree']['all']['f1']:>14.4f}" for n in names) + "\n")

    # Only meaningful because assembly preserves node identity -- see
    # edge_agreement's docstring.
    L.append("\nEDGE-LEVEL AGREEMENT (shared node set, so edges compare one by one)\n")
    L.append(f"{'':<11}{'':<10}{'edges':>9}{'real':>9}{'matched':>9}"
             f"{'prec':>9}{'recall':>9}{'F1':>9}{'Jaccard':>9}\n")
    for n in names[:-1]:
        for part, who in (('all', ''), ('internal', '  <- D_intra'),
                          ('cross', '  <- D_inter')):
            a = M[n]['agree'][part]
            head = n if part == 'all' else ''
            L.append(f"{head:<11}{part:<10}{a['n_syn']:>9}{a['n_real']:>9}{a['hit']:>9}"
                     f"{a['precision']:>9.4f}{a['recall']:>9.4f}{a['f1']:>9.4f}"
                     f"{a['jaccard']:>9.4f}{who}\n")

    L.append(f"\nBLOCK FIDELITY vs the release (synthetic column)\n")
    L.append(f"  internal edges - rho_in target : mean {int_err.mean():+.3f}, "
             f"mean |err| {np.abs(int_err).mean():.3f}\n")
    L.append(f"  |B_ij| - w_ij                  : exact "
             f"{int((cross_err == 0).sum())}/{len(cross_err)} "
             f"({100.0 * (cross_err == 0).mean():.1f}%), "
             f"mean {cross_err.mean():+.3f}\n")

    if args.real:
        L.append(f"\nASSEMBLY SELF-TEST: L^syn {'EQUALS' if exact_match else 'DIFFERS FROM'} "
                 f"G^priv edge for edge -> {'PASS' if exact_match else 'FAIL'}\n")

    report = "".join(L)
    print(report)
    with open(os.path.join(args.out_dir, 'syn_summary.txt'), 'w') as f:
        f.write(report)

    torch.save({'graph': release['graph'], 'k': release['k'], 'source': source,
                'n': int(A_syn.shape[0]),
                'edge_index': torch.tensor(np.vstack(A_syn.tocoo().nonzero()), dtype=torch.long)},
               os.path.join(args.out_dir, f"L_syn_{release['graph']}_k{release['k']}.pt"))

    with open(os.path.join(args.out_dir, 'syn_table.tex'), 'w') as f:
        f.write(latex_table(M, names, release, source))

    try:
        import pandas as pd
        hi = max(M[n]['deg'].max() for n in names)
        pd.DataFrame(dict({'degree': np.arange(hi + 1)}, **{
            n.lower(): np.bincount(M[n]['deg'], minlength=hi + 1) / M[n]['nodes']
            for n in names})).to_csv(
                os.path.join(args.out_dir, 'degree_distribution.csv'), index=False)
    except ImportError:
        pass

    print(f"written to {args.out_dir}/")
    return 0 if (not args.real or exact_match) else 1


def latex_table(M, names, release, source):
    """The reconstruction table, ready to \\input."""
    body = "\n".join(
        f"{lab:<34}" + "".join(f"& {fmt.format(M[n][key])} " for n in names) + r"\\"
        for lab, key, fmt in TEX_ROWS)
    errs = "\n".join(
        f"{lab:<34}" + "".join(f"& {fmt.format(M[n][key])} " for n in names) + r"\\"
        for lab, key, fmt in TEX_ERROR_ROWS)
    apl_how = ("computed exactly over all reachable pairs"
               if M['Real']['apl_exact'] else
               "estimated from the same sampled BFS sources in every column")
    baseline_note = (
        r"The Random column places the released edge counts -- $\rho_{\mathrm{in}}$ "
        r"per supernode and $w_{ij}$ per pair -- uniformly at random, so it matches "
        r"the release exactly on the count rows and differs only in \emph{where} the "
        r"edges go. " if 'Random' in names else "")

    return rf"""% L^syn vs the private graph, {release['graph']} k={release['k']}
% source: {source}
\begin{{table}}[H]
\centering
\caption{{Reconstruction quality of the assembled synthetic graph
$\widehat{{G}}$ against the private graph $G$ ({release['graph']},
$k{{=}}{release['k']}$). D\_intra generates each diagonal block, D\_inter each
off-diagonal one, and the two are assembled into a single graph on $G$'s own node
set. {baseline_note}GCC is the global clustering coefficient
$3\tau_{{\triangle}}/\tau_{{3}}$; APL is the mean geodesic distance over
reachable pairs, {apl_how}. The three lower rows are discrepancies against $G$ --
REGCC and REAPL the relative errors of GCC and APL, $\mathrm{{deg}}W$ the first
Wasserstein distance between degree distributions -- so lower is better and the
Real column is $0$ by construction. Assembly preserves node identity, so the two
degree multi-sets have equal length and $\mathrm{{deg}}W$ needs no finite-sample
correction.}}
\label{{tab:lsyn-{release['graph']}}}
\begin{{tabular}}{{l{'r' * len(names)}}}
\toprule
& {' & '.join(names)} \\
\midrule
{body}
\midrule
{errs}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


if __name__ == "__main__":
    raise SystemExit(main())
