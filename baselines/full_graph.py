"""BS1 (ER) and BS2 (SC-RG) for the whole release: every supernode and every pair.

Huiping's scripts (competitors/Baseline/) generate the TEST supernodes and pairs
of a dataset file. The assembled graph needs every block, so this module calls
her generator functions directly, block by block, with the same inputs:

    BS1  ER_intra.sample_gnm_adjacency      m_i edges among the n_i(n_i-1)/2 pairs
         ER_inter.sample_bipartite_edges    w_ij edges among the n_i n_j pairs
    BS2  SC_RG_intra.build_hop_aware_graph  hop layers from h_i, m_i edges
         SC_RG_inter.make_i_side_degree_sequence + sample_edges_from_i_degrees
                                            r_{i->j} out-degree profile, w_ij edges

n_i, rho_in (so m_i), h_i, the centre and w_ij come from the release. r_{i->j} is
not in the release: it is measured from the private graph's real cross block,
exactly as the dataset files store it (D_inter/dataprep/make_pair_blocks.py), so
BS2's cross blocks see a statistic D_inter is not given.

Returns (internal, blocks) in the form assemble_syn.assemble takes.
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "competitors", "Baseline"))
import ER_inter      # noqa: E402
import ER_intra      # noqa: E402
import SC_RG_inter   # noqa: E402
import SC_RG_intra   # noqa: E402

R_BINS = 5           # r_0 .. r_3, r_{>=4}, as make_pair_blocks.py


def _m(s):
    n = int(s["n"])
    return int(round(float(s["rho_in"]) * n * (n - 1) / 2.0))


def _edge_index(adj):
    """Her n x n 0/1 lists -> a (2, 2m) edge_index with both directions."""
    i, j = np.nonzero(np.asarray(adj, dtype=np.int8))
    return torch.as_tensor(np.vstack([i, j]), dtype=torch.long)


def _centre(s):
    return s["nodes"].tolist().index(int(s["centre"]))


def out_degree_profile(A_real, release):
    """r_{i->j} of every released pair, from the real cross block."""
    nodes = {int(s["sn"]): s["nodes"].numpy() for s in release["supernodes"]}
    r = {}
    for p in release["pairs"]:
        si, sj = int(p["sn_i"]), int(p["sn_j"])
        d = np.asarray(A_real[nodes[si]][:, nodes[sj]].sum(axis=1)).ravel().astype(int)
        r[(si, sj)] = (np.bincount(np.clip(d, 0, R_BINS - 1), minlength=R_BINS)
                       / len(d)).tolist()
    return r


def bs1_blocks(release, seed):
    rng = random.Random(seed)
    internal = {int(s["sn"]): _edge_index(ER_intra.sample_gnm_adjacency(int(s["n"]), _m(s), rng))
                for s in release["supernodes"]}
    blocks = {}
    for p in release["pairs"]:
        n_i, n_j, w = int(p["n_i"]), int(p["n_j"]), int(p["w"])
        blocks[(int(p["sn_i"]), int(p["sn_j"]))] = {
            "n_i": n_i, "n_j": n_j, "w_requested": w,
            "edge_index": ER_inter.sample_bipartite_edges(n_i, n_j, w, rng)}
    return internal, blocks


def bs2_blocks(release, r, seed):
    rng = random.Random(seed)
    internal = {}
    for s in release["supernodes"]:
        n, h = int(s["n"]), [float(v) for v in s["h"]]
        adj, _ = SC_RG_intra.build_hop_aware_graph(
            n=n, m=_m(s), centroid=_centre(s),
            layer_counts=SC_RG_intra.h_to_layer_counts(h=h, n=n), rng=rng)
        internal[int(s["sn"])] = _edge_index(adj)
    blocks = {}
    for p in release["pairs"]:
        key = (int(p["sn_i"]), int(p["sn_j"]))
        n_i, n_j, w = int(p["n_i"]), int(p["n_j"]), int(p["w"])
        deg, _ = SC_RG_inter.make_i_side_degree_sequence(n_i=n_i, n_j=n_j, w=w,
                                                         r=r[key], rng=rng)
        blocks[key] = {"n_i": n_i, "n_j": n_j, "w_requested": w,
                       "edge_index": SC_RG_inter.sample_edges_from_i_degrees(deg, n_j, rng)}
    return internal, blocks
