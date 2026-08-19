"""Did sampling leave the observed context alone?

D_inter generates only the cross block B_ij. The within-supernode blocks L_i and
L_j are handed to the model as observed context: pair_mode re-imposes them after
every reverse step, and X is never noised, so a correct sampler must return them
bit-identical to the record it conditioned on.

Nothing else in the evaluation suite would notice if that failed. Every metric in
pair_stats.py and evaluate_pairs.py counts cross edges only, so a run that
silently corrupted L_i / L_j would still produce a clean-looking results table.

This script checks it directly. For each generated sample it extracts the two
within-side blocks, canonicalises them, and looks them up in the set of real
blocks from the split. A match is exact -- same n_i, n_j and the same adjacency
bits in the same node order -- so a single flipped within-side entry is caught.

generated_samples.txt does not record which record each sample was drawn from, so
matching is by content rather than by index. That is strictly stronger for this
purpose: a sample matches only if its context is byte-identical to SOME real one,
and the space of 45-node graphs is far too large for a corrupted block to collide
with a real one by accident.

    python check_context_integrity.py --generated generated_samples1.txt \
                                      --pairs data/pair_lux/raw/pairs_inter_luxembourg_osm_k7.pt
"""
import argparse
import collections
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_pairs import parse_generated  # noqa: E402


def blocks_from_record(rec):
    """Dense (L_i, L_j) for a real pair."""
    n_i, n_j = int(rec['n_i']), int(rec['n_j'])
    src, dst = rec['edge_index'][0].numpy(), rec['edge_index'][1].numpy()

    Li = np.zeros((n_i, n_i), np.uint8)
    m = (src < n_i) & (dst < n_i)
    Li[src[m], dst[m]] = 1

    Lj = np.zeros((n_j, n_j), np.uint8)
    m = (src >= n_i) & (dst >= n_i)
    Lj[src[m] - n_i, dst[m] - n_i] = 1
    return Li, Lj


def blocks_from_sample(side, E):
    vi = np.flatnonzero(side == 0)
    vj = np.flatnonzero(side == 1)
    return (E[np.ix_(vi, vi)].astype(np.uint8),
            E[np.ix_(vj, vj)].astype(np.uint8))


def key(Li, Lj):
    return (Li.shape[0], Lj.shape[0], Li.tobytes(), Lj.tobytes())


def degseq(A):
    """Order-independent fingerprint, to tell reordering apart from corruption."""
    return tuple(sorted(A.sum(1).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--generated', required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    args = ap.parse_args()

    try:
        blob = torch.load(args.pairs, weights_only=False)
    except TypeError:
        blob = torch.load(args.pairs)
    recs = blob['splits'][{'val': 1, 'test': 2}[args.split]]

    exact = collections.Counter()
    loose = collections.Counter()          # same shape + degree sequences
    shapes = collections.Counter()
    for rec in recs:
        Li, Lj = blocks_from_record(rec)
        exact[key(Li, Lj)] += 1
        loose[(Li.shape[0], Lj.shape[0], degseq(Li), degseq(Lj))] += 1
        shapes[(Li.shape[0], Lj.shape[0])] += 1

    samples = parse_generated(args.generated)
    n_exact = n_loose = n_shape = 0
    bad_x, asym, self_loop = [], [], []
    misses = []

    for idx, (side, E) in enumerate(samples):
        # X is never noised, so sides must be contiguous 0s then 1s.
        if not np.array_equal(side, np.sort(side)):
            bad_x.append(idx)
        if not np.array_equal(E, E.T):
            asym.append(idx)
        if np.any(np.diag(E)):
            self_loop.append(idx)

        Li, Lj = blocks_from_sample(side, E)
        if key(Li, Lj) in exact:
            n_exact += 1
        else:
            misses.append(idx)
            if (Li.shape[0], Lj.shape[0], degseq(Li), degseq(Lj)) in loose:
                n_loose += 1
            elif (Li.shape[0], Lj.shape[0]) in shapes:
                n_shape += 1

    n = len(samples)
    print(f"{os.path.basename(args.generated)}: {n} samples vs "
          f"{len(recs)} real {args.split} records\n")

    print("CONTEXT BLOCKS  L_i, L_j")
    print(f"  exact match to a real pair     {n_exact:>5} / {n}  ({100*n_exact/n:.1f}%)")
    if misses:
        print(f"  no exact match                 {len(misses):>5}")
        print(f"     of those, degree seqs match {n_loose:>5}   (node reordering, not corruption)")
        print(f"     of those, only shape match  {n_shape:>5}   (context CHANGED)")
        print(f"     of those, shape unseen      {len(misses)-n_loose-n_shape:>5}")
        print(f"  first few miss indices: {misses[:10]}")

    # ---- condition fidelity ------------------------------------------------
    # An exact context match identifies the record the sample was drawn from, so
    # the requested w_ij can be recovered even though generated_samples.txt never
    # recorded it. Where several records share a context they must also agree on
    # w_ij for the comparison to be well defined; those that do not are skipped.
    by_key = collections.defaultdict(list)
    for rec in recs:
        by_key[key(*blocks_from_record(rec))].append(int(rec['w']))

    diffs, untestable = [], 0
    for side, E in samples:
        ws = set(by_key.get(key(*blocks_from_sample(side, E)), []))
        if len(ws) != 1:
            untestable += 1
            continue
        vi = np.flatnonzero(side == 0)
        vj = np.flatnonzero(side == 1)
        diffs.append(int((E[np.ix_(vi, vj)] == 1).sum()) - ws.pop())

    if diffs:
        d = np.array(diffs)
        print(f"\nCONDITION FIDELITY   |B_ij| generated minus w_ij requested"
              f"   ({len(d)}/{n} testable, {untestable} skipped)")
        print(f"  exact                          {(d == 0).sum():>5} / {len(d)}"
              f"  ({100*(d == 0).mean():.1f}%)")
        for k in sorted(set(d.tolist())):
            if k:
                print(f"     {k:+d}                          {(d == k).sum():>5}"
                      f"  ({100*(d == k).mean():.1f}%)")
        print(f"  mean |error|                   {np.abs(d).mean():.4f}")

    print("\nSTRUCTURAL INVARIANTS")
    print(f"  X sorted (sides contiguous)    {'OK' if not bad_x else f'FAIL on {len(bad_x)}'}")
    print(f"  E symmetric                    {'OK' if not asym else f'FAIL on {len(asym)}'}")
    print(f"  no self loops                  {'OK' if not self_loop else f'FAIL on {len(self_loop)}'}")

    ok = (n_exact == n) and not (bad_x or asym or self_loop)
    print(f"\nVERDICT: {'PASS -- observed context survived sampling intact' if ok else 'FAIL -- see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
