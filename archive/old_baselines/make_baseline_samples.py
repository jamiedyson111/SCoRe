"""Write BS1 / BS2 samples in the formats the D_intra and D_inter summaries read.

Run LOCALLY in the `kmis` env (CPU, a few minutes):

    python make_baseline_samples.py              # 5 seeds, every test split

Writes, for seeds 0..N-1:

    results/samples/D_intra/oldenburg/repeats/{BS1,BS2}_s<seed>.txt     Oldenburg test (160)
    results/samples/D_intra/oldenburg/repeats/{A_BS1,A_BS2}_s<seed>.txt Luxembourg test (882)
    results/samples/D_inter/transfer/{BS1,BS2}_s<seed>.pt               Oldenburg test (276)
    results/samples/D_inter/transfer/{A_BS1,A_BS2}_s<seed>.pt           Luxembourg test (1091)

which evaluation/repeat_summary.py (D_intra) and evaluation/transfer_summary.py
(D_inter) score exactly like a model's samples. The assembled-graph baselines
are built inside D_inter/private_inference/syn_repeats_summary.py instead, from
the same functions.

Every block gets its RECORD's own edge count -- round(rho_in * n(n-1)/2) for a
supernode, w_ij for a pair -- so both baselines score 100% on count fidelity by
construction; every other row measures placement alone.

E_h. A baseline has no hop condition, but its blocks can still be measured from a
node, so node 0 is marked as the centroid and E_h is reported: the hop error of a
generator that ignores h. For BS2 node 0 is the node with the largest expected
degree, as k-MIS picks its centres by degree rank; for BS1 every node is alike.
"""
import argparse
import os

import numpy as np
import torch

from uncoarsen import cross_block, internal_block, internal_edges, public_degrees

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
INTRA_PAIRS = os.path.join(ROOT, "D_intra", "data", "cluster_lux", "raw")
INTER_PAIRS = os.path.join(ROOT, "D_inter", "data", "pair_lux", "raw")
INTRA_OUT = os.path.join(ROOT, "results", "samples", "D_intra", "s80")
INTER_OUT = os.path.join(ROOT, "results", "samples", "D_inter", "s80")

# (prefix of the output name, D_intra pairs file, D_inter pairs file)
SPLITS = [("",   "pairs_B_lux2olde_k7_s80.pt", "pairs_inter_B_lux2olde_k7_s80.pt"),     # Oldenburg
          ("A_", "pairs_A_lux2olde_k7_s80.pt", "pairs_inter_luxembourg_osm_k7_s80.pt")]  # Luxembourg
CODE = {"BS1": 1, "BS2": 2}


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def degrees(pool, n, rng, bs):
    """Expected degrees for n nodes: None for BS1, a draw from the pool for BS2."""
    if bs == "BS1":
        return None
    return np.sort(rng.choice(pool, size=n, replace=True))[::-1]   # node 0 largest


def write_intra(path, test, rho_col, bs, pool, seed):
    rng = np.random.default_rng([seed, CODE[bs], 0])
    with open(path, "w") as f:
        for rec in test:
            n = int(rec["n"])
            rho = float(rec["y_raw"][rho_col])
            ei = internal_block(n, internal_edges(rho, n), rng, degrees(pool, n, rng, bs))
            A = np.zeros((n, n), dtype=int)
            A[ei[0], ei[1]] = 1
            x = np.zeros(n, dtype=int)
            x[0] = 1                                         # node 0 as the centroid
            f.write(f"N={n}\n")
            f.write(f"supernode={int(rec['sn'])} n_true={n} "
                    f"m_true={rec['edge_index'].shape[1] // 2}\n")
            f.write("cond: " + " ".join(f"{v:.4f}" for v in rec["y_raw"].tolist()) + "\n")
            f.write("X: \n" + " ".join(map(str, x)) + " \n")
            f.write("E: \n")
            for row in A:
                f.write(" ".join(map(str, row)) + " \n")
            f.write("\n")


def write_inter(path, pairs_file, test, bs, pool, seed):
    rng = np.random.default_rng([seed, CODE[bs], 1])
    blocks = []
    for idx, r in enumerate(test):
        n_i, n_j, w = int(r["n_i"]), int(r["n_j"]), int(r["w"])
        ei = cross_block(n_i, n_j, w, rng, degrees(pool, n_i, rng, bs),
                         degrees(pool, n_j, rng, bs))
        blocks.append({"idx": idx, "sn_i": int(r["sn_i"]), "sn_j": int(r["sn_j"]),
                       "n_i": n_i, "n_j": n_j, "w": w,
                       "edge_index": torch.as_tensor(ei, dtype=torch.long)})
    torch.save({"pairs_file": pairs_file, "checkpoint": f"baseline {bs}",
                "seed": seed, "blocks": blocks}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    pool = public_degrees("luxembourg")
    print(f"BS2 degree pool: Luxembourg, {len(pool)} nodes, mean {pool.mean():.3f}")
    os.makedirs(INTRA_OUT, exist_ok=True)
    os.makedirs(INTER_OUT, exist_ok=True)

    for prefix, intra_file, inter_file in SPLITS:
        blob = _load(os.path.join(INTRA_PAIRS, intra_file))
        test = blob["splits"][2]
        rho_col = list(blob["cond_names"]).index("rho_in")
        inter_test = _load(os.path.join(INTER_PAIRS, inter_file))["splits"][2]
        for bs in ("BS1", "BS2"):
            for seed in range(args.repeats):
                write_intra(os.path.join(INTRA_OUT, f"{prefix}{bs}_s{seed}.txt"),
                            test, rho_col, bs, pool, seed)
                write_inter(os.path.join(INTER_OUT, f"{prefix}{bs}_s{seed}.pt"),
                            inter_file, inter_test, bs, pool, seed)
            print(f"  {prefix}{bs}: {len(test)} supernodes, {len(inter_test)} pairs, "
                  f"seeds 0..{args.repeats - 1}")


if __name__ == "__main__":
    main()
