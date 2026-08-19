"""Verify the conditional D_inter model: generated cross blocks vs real held-out ones.

Run LOCALLY in the `kmis` env (needs only numpy/scipy/torch):

    python evaluate_pairs.py --generated generated_samples1.txt \
                             --pairs pairs_inter_luxembourg_osm_k7.pt        # _v2.pt under dataset=pair_lux_v2

Reports the metric from the brief,

    W_out-degree = W_1(P_out^gen, P_out^real),   d_{i->j}(u) = sum_v B_ij(u,v)

pooled over every node of every V_i, together with three reference points
without which the number cannot be read. All of them are evaluated at the SAME
sample size as the model -- see `matched_baselines` for why that matters:

  ZERO BASELINE     W_1 for a model that emits an empty cross block. The real
                    blocks average ~1.1 edges in ~240 slots, so "no edge
                    everywhere" is already a strong-looking answer. Any model
                    that does not beat this line comfortably has learned nothing.

  UNIFORM BASELINE  W_1 for a model that places the RIGHT number of edges
                    (w_ij is handed to it in the condition) at uniformly random
                    positions. Beating it means the model has learned WHERE
                    cross edges go, not just how many.

  NOISE FLOOR       W_1 for REAL pairs scored against the full real set, i.e.
                    what a PERFECT model scores at this sample size. W_1 is
                    inflated by finite-sample noise and does not reach 0. A
                    result inside this band is not a good result -- it is an
                    unresolvable one, and the honest reading is that the metric
                    cannot separate the model from perfect without more samples.

LIMITATION. generated_samples.txt records X and E but not the condition each
sample was drawn with, so this compares distributions only. It cannot check
per-sample whether the requested w_ij was honoured. For that, regenerate with
conditions logged alongside.
"""
import argparse
import os

import numpy as np
import torch
from scipy.stats import wasserstein_distance

SEED = 0


def parse_generated(path):
    """Parse generated_samples*.txt written by on_test_epoch_end.

    Format per sample:  N=<n> / "X:" / one row of node classes / "E:" / n rows.
    Node class 0 = side i, 1 = side j.
    """
    with open(path) as f:
        lines = f.read().splitlines()

    out, i = [], 0
    while i < len(lines):
        if not lines[i].startswith('N='):
            i += 1
            continue
        n = int(lines[i][2:])
        assert lines[i + 1].strip() == 'X:'
        side = np.array([int(v) for v in lines[i + 2].split()])
        assert lines[i + 3].strip() == 'E:'
        E = np.array([[int(v) for v in lines[i + 4 + r].split()] for r in range(n)])
        out.append((side, E))
        i += 4 + n
    return out


def out_degrees_from_dense(side, E):
    """d_{i->j}(u) for every u on side 0, including the zeros."""
    vi = np.flatnonzero(side == 0)
    vj = np.flatnonzero(side == 1)
    if len(vi) == 0 or len(vj) == 0:
        return None
    return (E[np.ix_(vi, vj)] == 1).sum(axis=1)


def _uniform_degrees(rec, rng):
    """d_{i->j} for a block with the right edge count placed at random."""
    n_i, n_j, w = int(rec['n_i']), int(rec['n_j']), int(rec['w'])
    slots = rng.integers(0, n_i * n_j, size=w)
    return np.bincount(slots // n_j, minlength=n_i)[:n_i]


def matched_baselines(recs, real, n_gen, rng, repeats=200):
    """Reference points for W_1, all drawn at the SAME sample size as the model.

    W_1 carries finite-sample noise, so a reference pooled over all len(recs)
    pairs is not comparable to a model scored on n_gen < len(recs) samples: the
    reference gets the quieter estimate and the model is charged for the
    difference. An earlier version of this script compared 512 generated pairs
    against baselines computed on all 1091 and reported a gap that does not
    exist. Each reference is therefore drawn on n_gen pairs, `repeats` times.

    Returns the uniform-placement and noise-floor spreads.
    """
    unif, floor = [], []
    for _ in range(repeats):
        idx = rng.choice(len(recs), min(n_gen, len(recs)), replace=False)
        unif.append(wasserstein_distance(
            np.concatenate([_uniform_degrees(recs[i], rng) for i in idx]), real))
        floor.append(wasserstein_distance(
            np.concatenate([out_degrees_from_record(recs[i]) for i in idx]), real))
    return np.array(unif), np.array(floor)


def out_degrees_from_record(rec):
    n_i = int(rec['n_i'])
    src, dst = rec['edge_index'][0], rec['edge_index'][1]
    cross = (src < n_i) & (dst >= n_i)
    return np.bincount(src[cross].numpy(), minlength=n_i)[:n_i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated', required=True, help='generated_samples*.txt')
    ap.add_argument('--pairs', required=True, help='the .pt from make_pair_blocks.py')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    args = ap.parse_args()

    try:
        blob = torch.load(args.pairs, weights_only=False)
    except TypeError:
        blob = torch.load(args.pairs)
    recs = blob['splits'][{'val': 1, 'test': 2}[args.split]]

    real = np.concatenate([out_degrees_from_record(r) for r in recs])
    real_counts = np.array([int(r['w']) for r in recs])

    samples = parse_generated(args.generated)
    gen_list, gen_counts = [], []
    for side, E in samples:
        d = out_degrees_from_dense(side, E)
        if d is None:
            continue
        gen_list.append(d)
        gen_counts.append(int(d.sum()))
    if not gen_list:
        raise SystemExit("no usable generated pairs -- every sample was one-sided")
    gen = np.concatenate(gen_list)

    # ---- baselines --------------------------------------------------------
    zero = np.zeros_like(real)
    w1_zero = wasserstein_distance(zero, real)

    rng = np.random.default_rng(SEED)
    n_gen = len(gen_list)
    unif_m, floor_m = matched_baselines(recs, real, n_gen, rng)

    w1 = wasserstein_distance(gen, real)

    print(f"{os.path.basename(args.generated)}: {len(gen_list)} generated pairs "
          f"vs {len(recs)} real {args.split} pairs\n")
    print("OUT-DEGREE DISTRIBUTION")
    print(f"  W_out-degree               {w1:.5f}     <- the metric")
    print(f"  zero-block baseline        {w1_zero:.5f}")
    print(f"  uniform at n={n_gen:<4}          {unif_m.mean():.5f} +/- "
          f"{unif_m.std():.5f}     <- the bar to beat")
    print(f"  noise floor at n={n_gen:<4}      {floor_m.mean():.5f} +/- "
          f"{floor_m.std():.5f}     <- what a PERFECT model scores")
    print(f"  mean out-degree   gen {gen.mean():.4f}   real {real.mean():.4f}")
    print(f"  max  out-degree   gen {gen.max()}        real {real.max()}")

    print("\nCROSS EDGES PER PAIR")
    gc = np.array(gen_counts)
    print(f"  gen  mean {gc.mean():.2f}  median {int(np.median(gc))}  "
          f"range {gc.min()}-{gc.max()}")
    print(f"  real mean {real_counts.mean():.2f}  median {int(np.median(real_counts))}  "
          f"range {real_counts.min()}-{real_counts.max()}")
    print(f"  W_1 on per-pair counts     "
          f"{wasserstein_distance(gc, real_counts):.5f}")
    print("  (w_ij is given in the condition, so a large gap here means the "
          "condition\n   is being ignored, not that the task is hard)")

    # Resolution first: below the floor the ordering against uniform is noise,
    # so reporting "beats/loses to uniform" there would be reading randomness.
    resolved = w1 > floor_m.mean() + 2 * floor_m.std()
    if not resolved:
        verdict = (f"W_1 {w1:.5f} is INSIDE the noise floor -- indistinguishable "
                   f"from a perfect model at n={n_gen}.\n"
                   f"         The metric cannot resolve this run; sample the full "
                   f"{len(recs)}-pair split to make it decisive.")
    elif w1 < unif_m.mean() - 2 * unif_m.std():
        verdict = (f"W_1 {w1:.5f} is OUTSIDE the noise floor and below uniform "
                   f"placement -- the model has learned WHERE edges go.")
    elif w1 > unif_m.mean() + 2 * unif_m.std():
        verdict = (f"W_1 {w1:.5f} is OUTSIDE the noise floor and WORSE than "
                   f"uniform placement.")
    else:
        verdict = (f"W_1 {w1:.5f} is OUTSIDE the noise floor but level with "
                   f"uniform placement -- no evidence it beats random placement.")
    print(f"\nverdict: {verdict}")


if __name__ == "__main__":
    main()
