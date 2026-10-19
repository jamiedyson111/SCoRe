#!/usr/bin/env python3
"""
ER baseline for D_intra, tested on pairs_B_lux2olde_k7.pt


Input file format
-----------------
A dict with keys including:
    cond_names
    k
    splits

where:
    splits[0] = train
    splits[1] = validation
    splits[2] = test

Each test item contains:
    sn          original supernode id
    graph       graph name
    n           number of nodes
    edge_index  undirected edges stored in both directions
    y_raw       raw condition [n, rho_in, k, h1, ..., h7]
    centre      original centroid node id
    nodes       original node ids
    y           normalised condition

ER baseline
---------------
For each test supernode:
    1. Read n and m.
    2. Sample exactly m undirected edges uniformly without replacement
       from all C(n,2) node pairs.
    3. Ignore h1,...,h7 deliberately.
    4. Write output using the same block format as B_v3.txt

Output
------
One file per stochastic run:
    ER_samples_s0.txt
    ER_samples_s1.txt
    ...

The "cond:" line follows the supplied example and writes only:
    n rho_in k

The centroid indicator X marks the local position of the true centroid
inside item["nodes"] when possible. MC-SBM itself does not use it.
"""

import argparse
import random
from pathlib import Path

import torch


def load_test_split(path):
    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected dict file, got {type(obj)}"
        )

    if "splits" not in obj:
        raise ValueError("Input dict does not contain 'splits'.")

    splits = obj["splits"]
    if not isinstance(splits, (list, tuple)) or len(splits) < 3:
        raise ValueError(
            "Expected splits = [train, val, test]."
        )

    test_split = splits[2]

    if not isinstance(test_split, (list, tuple)):
        raise ValueError("Test split is not a list/tuple.")

    return obj, test_split


def infer_m(item):
    edge_index = item["edge_index"]

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Unexpected edge_index shape: {tuple(edge_index.shape)}"
        )

    directed_entries = edge_index.shape[1]

    if directed_entries % 2 != 0:
        raise ValueError(
            f"Supernode {item['sn']}: edge_index has "
            f"{directed_entries} entries, expected even count "
            "for an undirected graph stored in both directions."
        )

    return directed_entries // 2


def sample_gnm_adjacency(n, m, rng):
    """
    Uniform simple undirected G(n,m):
    sample exactly m distinct unordered pairs.
    """
    possible_edges = [
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
    ]

    if m > len(possible_edges):
        raise ValueError(
            f"Requested m={m}, but max possible is "
            f"{len(possible_edges)} for n={n}."
        )

    chosen_edges = rng.sample(possible_edges, m)

    adj = [[0] * n for _ in range(n)]

    for u, v in chosen_edges:
        adj[u][v] = 1
        adj[v][u] = 1

    return adj


def centroid_indicator(item):
    """
    Return X vector with 1 at the local index corresponding to item['centre'].
    If matching fails, fall back to node 0 for output compatibility.
    """
    n = int(item["n"])
    x = [0] * n

    centre = item.get("centre", None)
    nodes = item.get("nodes", None)

    if centre is not None and nodes is not None:
        node_list = nodes.detach().cpu().tolist()
        try:
            local_idx = node_list.index(int(centre))
            x[local_idx] = 1
            return x
        except ValueError:
            pass

    if n > 0:
        x[0] = 1

    return x


def raw_condition(item):
    """
    Return n, rho, k from y_raw where available.
    Falls back to recomputation.
    """
    n = int(item["n"])
    m = infer_m(item)

    if "y_raw" in item:
        y_raw = item["y_raw"].detach().cpu().tolist()
        if len(y_raw) >= 3:
            return float(y_raw[0]), float(y_raw[1]), float(y_raw[2])

    rho = 0.0 if n <= 1 else 2.0 * m / (n * (n - 1))
    return float(n), float(rho), 7.0


def write_block(f, item, adj):
    sn = item["sn"]
    n = int(item["n"])
    m = infer_m(item)
    n_cond, rho_cond, k_cond = raw_condition(item)
    x = centroid_indicator(item)

    f.write(f"N={n}\n")
    f.write(f"supernode={sn} n_true={n} m_true={m}\n")
    f.write(
        f"cond: {n_cond:.4f} {rho_cond:.4f} {k_cond:.4f}\n"
    )

    f.write("X: \n")
    f.write(" ".join(map(str, x)) + " \n")

    f.write("E: \n")
    for row in adj:
        f.write(" ".join(map(str, row)) + " \n")

    f.write("\n")


def validate_generated(adj, n, m):
    # shape
    if len(adj) != n or any(len(row) != n for row in adj):
        return False, "wrong adjacency shape"

    # diagonal + symmetry
    for i in range(n):
        if adj[i][i] != 0:
            return False, "self-loop found"
        for j in range(i + 1, n):
            if adj[i][j] != adj[j][i]:
                return False, "matrix is not symmetric"

    generated_m = sum(
        adj[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    )

    if generated_m != m:
        return False, f"expected m={m}, generated m={generated_m}"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description="ER D_intra baseline "
    )

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate every generated adjacency matrix."
    )

    args = parser.parse_args()

    obj, test_split = load_test_split(args.input)

    print(f"Graph config: {obj.get('config', 'unknown')}")
    print(f"Description: {obj.get('desc', 'unknown')}")
    print(f"k: {obj.get('k', 'unknown')}")
    print(f"Test supernodes: {len(test_split)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(args.num_samples):
        rng = random.Random(args.seed + sample_idx)

        output_path = (
            args.output_dir / f"ER_samples_s{sample_idx}.txt"
        )

        with output_path.open("w", encoding="utf-8") as f:
            for item in test_split:
                n = int(item["n"])
                m = infer_m(item)

                adj = sample_gnm_adjacency(n, m, rng)

                if args.check:
                    ok, msg = validate_generated(adj, n, m)
                    if not ok:
                        raise RuntimeError(
                            f"Supernode {item['sn']}: {msg}"
                        )

                write_block(f, item, adj)

        print(
            f"Wrote {len(test_split)} blocks -> {output_path}"
        )


if __name__ == "__main__":
    main()
