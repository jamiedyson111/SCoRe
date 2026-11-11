"""Whole-graph evaluation by PARTIAL reconstruction (Huiping, 9 Nov 2026), LOCALLY, `kmis` env:

    python partial_whole_graph.py --selftest                     # original parts in -> original out
    python partial_whole_graph.py --dinter coll_g1_C_0.pt coll_g1_C_1.pt

For a selected set of test supernodes S (Collab: D_intra group 1, 357 supernodes,
collab_group1_supernodes.json):
  * every supernode in S gets its GENERATED internal structure (D_intra C, seed 0);
  * every pair touching S whose two supernodes have n >= 3 -- the D_inter test pairs
    touching S, 7,917 on Collab -- gets its GENERATED cross block (D_inter C, whose
    context was S's generated structure and every other supernode's real one);
  * everything else keeps its ORIGINAL structure: the other supernodes (test ones
    included), their pairs, and the few pairs touching S whose partner has n < 3,
    which D_inter does not model.
Node k of a generated supernode is its k-th member in ascending node id, the order
D_inter's records (and so its generated blocks) use.

ER (BS1) and SC-RG (BS2) replace the SAME blocks with Huiping's generators, called block
by block as baselines/full_graph.py does: ER G(n, m_i) and w_ij uniform cross edges;
SC-RG hop layers from h_i and the i-side out-degree profile r_{i->j} (bins 0, 1, 2, 3,
4+, the form her generator takes) measured on the real block. 5 seeds each (42 + s).

Scored with syn_repeats_summary.score, so every row means what it means in the road
tables, on two parts: the WHOLE graph, and the S PART (the subgraph induced on S's
nodes: generated insides and the generated cross blocks between two S supernodes).
Spectrum is computed on the S part only: on the whole Collab graph one eigenvalue count
takes ~400 s (fill-in), ~22 h per histogram.

Writes, in --out-dir: partial_whole_<part>.tsv ("mean ± std" cells; Ours is one
generation) and partial_whole_<part>_per_run.tsv.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'D_intra', 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'competitors', 'Baseline'))
import graph_mmd                 # noqa: E402
import syn_repeats_summary as S  # noqa: E402
from evaluate_clusters import parse_generated  # noqa: E402
import ER_inter, ER_intra, SC_RG_inter, SC_RG_intra  # noqa: E402,E401


def load_graph(npz):
    with np.load(npz, allow_pickle=False) as z:
        n = int(json.loads(str(z['metadata']))['n'])
        A = sp.csr_matrix((np.ones(len(z['adj_indices'])), z['adj_indices'], z['adj_indptr']), shape=(n, n))
        a = z['assignment'].astype(np.int64)
    A.data[:] = 1.0
    return A, a


def members_of(a):
    order = np.argsort(a, kind='stable')
    starts = np.searchsorted(a[order], np.arange(a.max() + 2))
    return lambda s: order[starts[s]:starts[s + 1]]


def build(A, a, sel, pairs, internal, cross):
    """A with the internal edges of every supernode in sel and the cross edges of every
    pair in pairs removed, then internal / cross (global node ids) added."""
    U = sp.triu(A, k=1).tocoo()
    su, sv = a[U.row], a[U.col]
    c = a.max() + 1
    insel = np.zeros(c, bool)
    insel[list(sel)] = True
    pkey = np.array(sorted(min(p) * c + max(p) for p in pairs), dtype=np.int64)
    ekey = np.minimum(su, sv) * c + np.maximum(su, sv)
    drop = ((su == sv) & insel[su]) | np.isin(ekey, pkey)
    rows = [U.row[~drop]] + [e[0] for e in internal] + [e[0] for e in cross]
    cols = [U.col[~drop]] + [e[1] for e in internal] + [e[1] for e in cross]
    r, q = np.concatenate(rows), np.concatenate(cols)
    keep = r != q
    r, q = r[keep], q[keep]
    n = A.shape[0]
    B = sp.coo_matrix((np.ones(len(r)), (r, q)), shape=(n, n)).tocsr()
    B = ((B + B.T) > 0).astype(np.float64)
    return B.tocsr(), int(drop.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=os.path.join(ROOT, 'datasets_external', 'dblp_collab_k3',
                                                  'ogbl_collab_full_csr_kmis_sized_k3.npz'))
    ap.add_argument('--sel', default=os.path.join(ROOT, 'D_inter', 'data', 'wholegraph',
                                                  'collab_group1_supernodes.json'))
    ap.add_argument('--inter-pairs', default=os.path.join(ROOT, 'D_inter', 'data', 'pair_lux', 'raw',
                                                          'pairs_inter_C_dblp2coll_k3_s80_deg.pt'))
    ap.add_argument('--intra-pairs', default=os.path.join(ROOT, 'D_intra', 'data', 'cluster_lux', 'raw',
                                                          'pairs_B_dblp2coll_k3_s80.pt'))
    ap.add_argument('--dintra', default=os.path.join(ROOT, 'results', 'samples', 'D_intra',
                                                     'collab_groups', 'Cft_v3_s0.txt'))
    ap.add_argument('--dinter', nargs='*', default=[])
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--apl-sources', type=int, default=100)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'results', 'whole_graph', 'collab_group1'))
    args = ap.parse_args()
    t0 = time.time()

    A, a = load_graph(args.npz)
    mem = members_of(a)
    sel = set(json.load(open(args.sel))['supernodes'])
    test = S._load(args.inter_pairs)['splits'][2]
    recs = {(int(r['sn_i']), int(r['sn_j'])): r for r in test
            if int(r['sn_i']) in sel or int(r['sn_j']) in sel}
    # pairs touching S that D_inter does not model (partner n < 3) keep their edges
    U = sp.triu(A, k=1).tocoo()
    su, sv = a[U.row], a[U.col]
    touch = (np.isin(su, list(sel)) | np.isin(sv, list(sel))) & (su != sv)
    all_pairs = set(zip(np.minimum(su[touch], sv[touch]).tolist(), np.maximum(su[touch], sv[touch]).tolist()))
    modelled = {(min(p), max(p)) for p in recs}
    print(f"{A.shape[0]} nodes, {A.nnz // 2} edges; S = {len(sel)} supernodes, "
          f"{sum(len(mem(s)) for s in sel)} nodes; pairs touching S: {len(all_pairs)}, of which "
          f"{len(modelled)} generated by D_inter and {len(all_pairs - modelled)} kept (partner n < 3)")

    intra = {int(r['sn']): r for r in S._load(args.intra_pairs)['splits'][2] if int(r['sn']) in sel}
    if set(intra) != sel:
        raise SystemExit("some selected supernodes are not in the D_intra test split")
    for s, r in intra.items():
        if not np.array_equal(r['nodes'].numpy(), mem(s)):
            raise SystemExit(f"supernode {s}: record member order is not ascending node id")

    def put_internal(s, ei):           # local (2, k) -> global
        m = mem(s)
        return m[np.asarray(ei[0])], m[np.asarray(ei[1])]

    def put_cross(key, ei):
        mi, mj = mem(key[0]), mem(key[1])
        return mi[np.asarray(ei[0])], mj[np.asarray(ei[1])]

    sn_of = a
    parts = {'whole': None, 'S': np.flatnonzero(np.isin(a, list(sel)))}

    def cut(M, idx):
        return M if idx is None else M[idx][:, idx]

    def real_internal(s):
        m = mem(s)
        sub = A[m][:, m].tocoo()
        return np.vstack([sub.row, sub.col])

    def real_cross(key):
        sub = A[mem(key[0])][:, mem(key[1])].tocoo()
        return np.vstack([sub.row, sub.col])

    if args.selftest:
        G, dropped = build(A, a, sel, recs, [put_internal(s, real_internal(s)) for s in sel],
                           [put_cross(k, real_cross(k)) for k in recs])
        diff = abs(G - A).nnz
        print(f"selftest: removed {dropped} edges and put the original ones back; "
              f"{diff} entries differ from the original graph ({time.time() - t0:.0f}s)")
        raise SystemExit(0 if diff == 0 else 1)

    # --- real column
    real = {}
    for part, idx in parts.items():
        Ar = cut(A, idx)
        real[part] = S.score(Ar, Ar, sn_of if idx is None else sn_of[idx], None, args.apl_sources)
        real[part].update(degw=0.0, regcc=0.0, reapl=0.0)
        real[part]['profiles'] = profiles(Ar, spectrum=part == 'S')
        real[part].update(graph_mmd.discrepancies(real[part]['profiles'], real[part]['profiles']))
    print(f"real scored ({time.time() - t0:.0f}s)", flush=True)

    def score_parts(G):
        out = {}
        for part, idx in parts.items():
            Gp = cut(G, idx)
            m = S.score(Gp, cut(A, idx), sn_of if idx is None else sn_of[idx], real[part], args.apl_sources)
            m.update(graph_mmd.discrepancies(profiles(Gp, spectrum=part == 'S'), real[part]['profiles']))
            out[part] = m
        return out

    runs = {}
    # --- ours
    if args.dinter:
        gen = {g['supernode']: g for g in parse_generated(args.dintra) if g.get('supernode') in sel}
        if set(gen) != sel:
            raise SystemExit(f"{args.dintra}: {len(sel - set(gen))} selected supernodes have no sample")
        internal = []
        for s in sel:
            if gen[s]['A'].shape[0] != len(mem(s)):
                raise SystemExit(f"supernode {s}: generated size differs")
            i, j = np.nonzero(np.triu(gen[s]['A'], 1))
            internal.append(put_internal(s, (i, j)))
        blocks = {}
        for f in args.dinter:
            for b in S._load(f)['blocks']:
                if b is not None:
                    blocks[(b['sn_i'], b['sn_j'])] = b
        miss = [k for k in recs if k not in blocks]
        if miss:
            raise SystemExit(f"{len(miss)} of {len(recs)} pairs touching S have no generated block")
        cross = []
        for k in recs:
            b = blocks[k]
            if (b['n_i'], b['n_j']) != (len(mem(k[0])), len(mem(k[1]))):
                raise SystemExit(f"pair {k}: block size differs")
            cross.append(put_cross(k, b['edge_index'].numpy()))
        G, _ = build(A, a, sel, recs, internal, cross)
        runs['Ours (C)'] = [(0, score_parts(G))]
        print(f"ours scored ({time.time() - t0:.0f}s)", flush=True)

    # --- baselines
    hidx = None
    for name in ('ER', 'SC-RG'):
        runs[name] = []
        for s_ in range(args.seeds):
            rng = random.Random(42 + s_)
            internal, cross = [], []
            for s in sorted(sel):
                r, n = intra[s], len(mem(s))
                m_i = int(r['edge_index'].shape[1] // 2)
                if name == 'ER':
                    adj = ER_intra.sample_gnm_adjacency(n, m_i, rng)
                else:
                    names = list(S._load(args.intra_pairs)['cond_names']) if hidx is None else None
                    if hidx is None:
                        hidx = [i for i, nm in enumerate(names) if nm.startswith('h')]
                    h = [float(v) for v in r['y_raw'][hidx]]
                    centre = r['nodes'].tolist().index(int(r['centre']))
                    adj, _ = SC_RG_intra.build_hop_aware_graph(
                        n=n, m=m_i, centroid=centre,
                        layer_counts=SC_RG_intra.h_to_layer_counts(h=h, n=n), rng=rng)
                i, j = np.nonzero(np.triu(np.asarray(adj, dtype=np.int8), 1))
                internal.append(put_internal(s, (i, j)))
            for k in sorted(recs):
                r = recs[k]
                n_i, n_j, w = int(r['n_i']), int(r['n_j']), int(r['w'])
                if name == 'ER':
                    ei = ER_inter.sample_bipartite_edges(n_i, n_j, w, rng)
                else:
                    d = np.bincount(real_cross(k)[0], minlength=n_i)[:n_i]
                    rp = (np.bincount(np.clip(d, 0, 4), minlength=5) / n_i).tolist()
                    deg, _ = SC_RG_inter.make_i_side_degree_sequence(n_i=n_i, n_j=n_j, w=w, r=rp, rng=rng)
                    ei = SC_RG_inter.sample_edges_from_i_degrees(deg, n_j, rng)
                cross.append(put_cross(k, np.asarray(ei)))
            G, _ = build(A, a, sel, recs, internal, cross)
            runs[name].append((s_, score_parts(G)))
            print(f"{name} seed {s_} scored ({time.time() - t0:.0f}s)", flush=True)

    write(runs, real, args, len(sel))


def profiles(M, spectrum):
    """graph_mmd.profiles, with the spectrum only where asked (None -> blank)."""
    return {'clus': graph_mmd.clustering_hist(M), 'orbit': graph_mmd.orbit_profile(M),
            'spec': graph_mmd.spectrum_hist(M) if spectrum else None}


ROWS = [('nodes', 'Nodes', 0), ('edges', 'Edges', 0), ('mean_degree', 'Mean degree', 4),
        ('components', 'Components', 1), ('gcc', 'GCC', 4), ('apl', 'APL', 3)]
ERRORS = [('re_mean_degree', 'RE mean degree (%)', 2), ('re_density', 'RE density (%)', 2),
          ('re_components', 'RE components (%)', 2), ('regcc_pct', 'RE_GCC (%)', 2),
          ('degw', 'degW', 4), ('reapl_pct', 'RE_APL (%)', 2), ('clus_mmd', 'Clustering', 4),
          ('orbit_mmd', 'Orbit', 4), ('spec_mmd', 'Spectrum', 4)]


def write(runs, real, args, n_sel):
    os.makedirs(args.out_dir, exist_ok=True)
    for part in ('whole', 'S'):
        r0 = real[part]
        for name in runs:
            for _, m in runs[name]:
                p = m[part]
                p['re_mean_degree'] = 100 * abs(p['mean_degree'] - r0['mean_degree']) / r0['mean_degree']
                p['re_density'] = 100 * abs(p['density'] - r0['density']) / r0['density']
                p['re_components'] = 100 * abs(p['components'] - r0['components']) / r0['components']
                p['regcc_pct'] = 100 * p['regcc']
                p['reapl_pct'] = 100 * p['reapl']
        names = list(runs)

        def cell(name, key, dp):
            v = np.array([m[part][key] for _, m in runs[name]], float)
            if np.all(np.isnan(v)):
                return ''
            return f"{v.mean():.{dp}f}" + (f" ± {v.std(ddof=1):.{dp}f}" if len(v) > 1 else '')
        lines = ['\t'.join(['Metric', 'Real'] + names)]
        lines += ['\t'.join([lab, f"{r0[k]:.{dp}f}"] + [cell(nm, k, dp) for nm in names]) for k, lab, dp in ROWS]
        lines += ['', '\t'.join(['Error'] + names)]
        lines += ['\t'.join([lab] + [cell(nm, k, dp) for nm in names]) for k, lab, dp in ERRORS]
        what = ('the whole graph' if part == 'whole' else
                f"the S part: the subgraph induced on the {n_sel} selected supernodes' nodes")
        lines += ['', f"Partial reconstruction of Collab, {what}. Ours: one generation (D_intra C and "
                      f"D_inter C, seed 0); ER and SC-RG: {args.seeds} seeds, mean ± sample std. "
                      "APL from 100 BFS sources (the same in every column). Spectrum only on the S part."]
        path = os.path.join(args.out_dir, f"partial_whole_{part}.tsv")
        open(path, 'w').write('\n'.join(lines) + '\n')
        head = ['Metric'] + [f"{nm} s{s}" for nm in names for s, _ in runs[nm]]
        rows = ['\t'.join(head)] + ['\t'.join([lab] + [f"{m[part][k]:.{dp}f}" for nm in names for _, m in runs[nm]])
                                    for k, lab, dp in ROWS + ERRORS]
        open(os.path.join(args.out_dir, f"partial_whole_{part}_per_run.tsv"), 'w').write('\n'.join(rows) + '\n')
        print(f"\n== {part} ==\n" + '\n'.join(lines))


if __name__ == '__main__':
    main()
