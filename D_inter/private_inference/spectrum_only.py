"""The whole-graph Spectrum row on its own, for graphs too large for a dense
eigendecomposition (Vermont: 97,975 vertices). LOCALLY, `kmis` env:

    python spectrum_only.py --validate                                   # Oldenburg: counting vs dense
    python spectrum_only.py --release ../../results/private_inference/release_vermont_k7.pt \\
        --split-file ../../D_intra/data/splits/vermont_k7_s80.json

The graphs are syn_repeats_summary.py's exactly -- same release, assembly, configurations
and BS1/BS2 seeds -- but only graph_mmd's spectrum is computed, by eigenvalue counting
(graph_mmd.spectrum_hist_inertia). --validate also computes the dense histogram of every
graph and reports both MMDs, which must agree.

Writes spectrum_<graph>.tsv into --out-dir: Real, each configuration and BS1/BS2 as
"mean ± std" over seeds, like syn_summary.tsv's Spectrum row.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph_mmd  # noqa: E402
import syn_repeats_summary as S  # noqa: E402
import full_graph  # noqa: E402


def hists(args):
    A, dense = args
    t = time.time()
    h = graph_mmd.spectrum_hist_inertia(A)
    d = None
    if dense:
        e = graph_mmd.eigvalsh(graph_mmd.normalized_laplacian(A).toarray())
        d, _ = np.histogram(e, bins=200, range=(-1e-5, 2))
        d = d / d.sum()
    return h, d, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', default=os.path.join(S.OUT, 'release_oldenburg_k7.pt'))
    ap.add_argument('--split-file', default=None, help='unused; kept for symmetry with syn_repeats_summary')
    ap.add_argument('--samples-dir', default=S.SAMPLES)
    ap.add_argument('--configs', nargs='+', default=['B_v3_s80-B_deg_remb', 'Cft_v3_s80-C_deg_remb'])
    ap.add_argument('--graph-root', default=os.path.join(HERE, '..', '..', 'D_intra', 'extraction'))
    ap.add_argument('--validate', action='store_true', help='also the dense histogram (small graphs only)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--out-dir', default=os.path.join(S.RESULTS, 'remb_paste'))
    args = ap.parse_args()

    release = S._load(args.release)
    A_real = S.load_true_graph(release, os.path.abspath(args.graph_root))
    graph = release['graph']
    jobs = [('Real', None, A_real)]
    seeds = []
    for cfg in args.configs:
        files = sorted(S.glob.glob(os.path.join(
            args.samples_dir, f"blocks_{graph}_k{release['k']}_{cfg}_s*.pt")), key=S.seed_of)
        for f in files:
            s_ = S.seed_of(f)
            jobs.append((cfg, s_, S.assemble(release, *S.load_run(release, args.samples_dir, cfg, s_))))
            seeds.append(s_)
    rseeds = sorted(set(seeds)) or [0, 1, 2, 3, 4]
    r = full_graph.out_degree_profile(A_real, release)
    for s_ in rseeds:                                    # Huiping's seeds, as syn_repeats_summary
        jobs.append(('BS1', s_, S.assemble(release, *full_graph.bs1_blocks(release, 42 + s_))))
        jobs.append(('BS2', s_, S.assemble(release, *full_graph.bs2_blocks(release, r, 42 + s_))))
    print(f"{graph}: {A_real.shape[0]} nodes, {len(jobs)} graphs", flush=True)

    with ProcessPoolExecutor(args.workers) as ex:
        out = list(ex.map(hists, [(A, args.validate) for _, _, A in jobs]))
    real_h, real_d, _ = out[0]
    cols, worst = {}, 0.0
    for (name, s_, _), (h, d, dt) in zip(jobs, out):
        m = graph_mmd.mmd(h, real_h, graph_mmd.SIGMA['spec'])
        line = f"  {name:<24} s{s_}  counting {m:.6f}  ({dt:.0f}s)"
        if args.validate:
            md = graph_mmd.mmd(d, real_d, graph_mmd.SIGMA['spec'])
            worst = max(worst, abs(m - md))
            line += f"  dense {md:.6f}  bins differing {int(np.sum(np.abs(h - d) > 1e-12))}"
        print(line, flush=True)
        cols.setdefault(name, []).append(m)
    if args.validate:
        print(f"largest |counting - dense| MMD difference: {worst:.2e}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"spectrum_{graph}.tsv")
    names = list(cols)
    with open(path, 'w') as f:
        f.write('\t'.join(['Metric'] + names) + '\n')
        f.write('\t'.join(['Spectrum'] + [S.cell(cols[c], 4) for c in names]) + '\n')
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
