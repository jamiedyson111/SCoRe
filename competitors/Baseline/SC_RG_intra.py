#!/usr/bin/env python3
"""
SC-RG baseline for intra.

Tested on pairs_B_lux2olde_k7.pt format and B_v3.txt output format.

Input
-----
A dict containing:
    splits[0] = Luxembourg train
    splits[1] = Luxembourg validation
    splits[2] = Oldenburg test

Each test item is expected to contain:
    sn, n, edge_index, y_raw, centre, nodes

with:
    y_raw = [n, rho_in, k, h1, ..., hk]

Baseline
--------
For each Oldenburg test supernode:

1. Read n, m and h.
2. Convert h into integer hop-layer sizes.
3. Randomly assign non-centroid nodes to hop layers.
4. Create a mandatory radial backbone:
      centroid -> every node in V1
      each node in Vr (r >= 2) -> one random parent in V(r-1)
5. Add random extra edges until exactly m edges are reached.
   Extra edges are allowed only:
      - within the same hop layer, or
      - between adjacent hop layers.
   This preserves shortest-path distance from every node to the centroid.

Therefore the generated graph preserves:
    n, m, and h

but edge placement within those constraints is random.

Output
------
One text file per stochastic run:
    SCRG_samples_s0.txt
        SCRG_samples_s1.txt
    ...
"""

import argparse
import math
import random
from collections import deque
from pathlib import Path

import torch


# ---------------------------------------------------------------------
# Load test split
# ---------------------------------------------------------------------

def load_test_split(path):
    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict input, got {type(obj)}")

    if "splits" not in obj:
        raise ValueError("Input file does not contain 'splits'.")

    splits = obj["splits"]

    if not isinstance(splits, (list, tuple)) or len(splits) < 3:
        raise ValueError("Expected splits = [train, val, test].")

    test_split = splits[2]

    if not isinstance(test_split, (list, tuple)):
        raise ValueError("Test split is not a list/tuple.")

    return obj, test_split


# ---------------------------------------------------------------------
# Basic graph information
# ---------------------------------------------------------------------

def infer_m(item):
    edge_index = item["edge_index"]

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            f"Unexpected edge_index shape: {tuple(edge_index.shape)}"
        )

    directed_entries = edge_index.shape[1]

    if directed_entries % 2 != 0:
        raise ValueError(
            f"Supernode {item['sn']}: expected undirected edges stored "
            "in both directions, but edge_index has an odd number of entries."
        )

    return directed_entries // 2


def get_raw_condition(item):
    if "y_raw" not in item:
        raise ValueError(
            f"Supernode {item['sn']} has no y_raw field."
        )

    y_raw = item["y_raw"].detach().cpu().tolist()

    if len(y_raw) < 4:
        raise ValueError(
            f"Supernode {item['sn']}: y_raw is too short."
        )

    n_cond = float(y_raw[0])
    rho = float(y_raw[1])
    k = int(round(float(y_raw[2])))
    h = [float(v) for v in y_raw[3:3 + k]]

    if len(h) != k:
        raise ValueError(
            f"Supernode {item['sn']}: expected {k} hop values, got {len(h)}."
        )

    return n_cond, rho, k, h


def get_centroid_local_index(item):
    """
    Find the centroid's local index within item["nodes"].
    Falls back to 0 only if the mapping is unavailable.
    """
    centre = item.get("centre", None)
    nodes = item.get("nodes", None)

    if centre is not None and nodes is not None:
        node_list = nodes.detach().cpu().tolist()
        try:
            return node_list.index(int(centre))
        except ValueError:
            pass

    return 0


# ---------------------------------------------------------------------
# Convert hop proportions to integer layer counts
# ---------------------------------------------------------------------

def h_to_layer_counts(h, n):
    """
    Convert h_r proportions into integer counts summing to n-1.

    Uses a largest-remainder allocation so floating-point storage does not
    cause count inconsistencies.

    Example:
        n = 9
        h = [2/8, 2/8, 4/8]
        -> [2, 2, 4]
    """
    total_non_centroid = n - 1

    if total_non_centroid == 0:
        return [0] * len(h)

    raw = [max(0.0, p) * total_non_centroid for p in h]
    base = [int(math.floor(x + 1e-12)) for x in raw]

    remaining = total_non_centroid - sum(base)

    if remaining > 0:
        order = sorted(
            range(len(h)),
            key=lambda r: raw[r] - base[r],
            reverse=True,
        )
        for r in order[:remaining]:
            base[r] += 1

    elif remaining < 0:
        order = sorted(
            range(len(h)),
            key=lambda r: raw[r] - base[r],
        )
        needed = -remaining
        for r in order:
            if needed == 0:
                break
            if base[r] > 0:
                base[r] -= 1
                needed -= 1

    if sum(base) != total_non_centroid:
        raise ValueError(
            f"Could not convert h into counts summing to n-1={total_non_centroid}. "
            f"Got {base} with sum={sum(base)}."
        )

    return base


# ---------------------------------------------------------------------
# Hop-aware random generation
# ---------------------------------------------------------------------

def assign_nodes_to_layers(n, centroid, layer_counts, rng):
    """
    Randomly assign all non-centroid local node IDs to hop layers.
    Layer 0 contains only the centroid.
    """
    non_centroid_nodes = [v for v in range(n) if v != centroid]
    rng.shuffle(non_centroid_nodes)

    layers = [[centroid]]
    offset = 0

    for count in layer_counts:
        layer = non_centroid_nodes[offset:offset + count]
        layers.append(layer)
        offset += count

    if offset != n - 1:
        raise RuntimeError("Not all non-centroid nodes were assigned.")

    return layers


def build_hop_aware_graph(n, m, centroid, layer_counts, rng):
    """
    Generate a simple undirected graph preserving exact hop-layer counts
    and exact edge count m.
    """
    layers = assign_nodes_to_layers(
        n=n,
        centroid=centroid,
        layer_counts=layer_counts,
        rng=rng,
    )

    # Check that there are no empty intermediate layers before a non-empty
    # deeper layer. Such an h cannot describe valid shortest-path layers.
    seen_empty = False
    for r in range(1, len(layers)):
        if len(layers[r]) == 0:
            seen_empty = True
        elif seen_empty:
            raise ValueError(
                f"Invalid hop profile: layer {r} is non-empty after an empty layer."
            )

    edges = set()

    def add_edge(u, v):
        if u == v:
            return
        if u > v:
            u, v = v, u
        edges.add((u, v))

    # -------------------------------------------------------------
    # 1. Mandatory radial backbone
    # -------------------------------------------------------------

    # Every V1 node must be adjacent to centroid.
    if len(layers) > 1:
        for v in layers[1]:
            add_edge(centroid, v)

    # Every node in Vr, r >= 2, gets one parent in V(r-1).
    for r in range(2, len(layers)):
        previous = layers[r - 1]

        if layers[r] and not previous:
            raise ValueError(
                f"Cannot realise hop layer {r}: previous layer is empty."
            )

        for v in layers[r]:
            parent = rng.choice(previous)
            add_edge(parent, v)

    minimum_edges = len(edges)

    if m < minimum_edges:
        raise ValueError(
            f"Requested m={m}, but at least {minimum_edges} edges are needed "
            "to realise the requested hop structure."
        )

    # -------------------------------------------------------------
    # 2. Build all additional hop-preserving candidate edges
    # -------------------------------------------------------------

    candidates = []

    # Same-layer edges.
    for r in range(1, len(layers)):
        layer = layers[r]
        for a in range(len(layer)):
            for b in range(a + 1, len(layer)):
                u, v = layer[a], layer[b]
                edge = (u, v) if u < v else (v, u)
                if edge not in edges:
                    candidates.append(edge)

    # Adjacent-layer edges.
    # Includes V0-V1, although all such edges should already be mandatory.
    for r in range(len(layers) - 1):
        left = layers[r]
        right = layers[r + 1]

        for u in left:
            for v in right:
                edge = (u, v) if u < v else (v, u)
                if edge not in edges:
                    candidates.append(edge)

    extra_needed = m - len(edges)

    if extra_needed > len(candidates):
        raise ValueError(
            f"Requested m={m}, but only {len(edges) + len(candidates)} "
            "hop-preserving edges are possible for this h."
        )

    # Uniformly choose the remaining allowed edges.
    if extra_needed > 0:
        extra_edges = rng.sample(candidates, extra_needed)
        edges.update(extra_edges)

    # -------------------------------------------------------------
    # 3. Adjacency matrix
    # -------------------------------------------------------------

    adj = [[0] * n for _ in range(n)]

    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1

    return adj, layers


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def bfs_distances(adj, source):
    n = len(adj)
    dist = [-1] * n
    dist[source] = 0

    q = deque([source])

    while q:
        u = q.popleft()

        for v, connected in enumerate(adj[u]):
            if connected and dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    return dist


def realised_layer_counts(adj, centroid, k):
    dist = bfs_distances(adj, centroid)
    counts = [0] * k

    for v, d in enumerate(dist):
        if v == centroid:
            continue

        if d < 1 or d > k:
            raise ValueError(
                f"Generated node {v} has invalid centroid distance {d}."
            )

        counts[d - 1] += 1

    return counts


def validate_generated(adj, n, m, centroid, target_counts):
    # Matrix dimensions, symmetry and diagonal.
    if len(adj) != n or any(len(row) != n for row in adj):
        return False, "wrong adjacency shape"

    for i in range(n):
        if adj[i][i] != 0:
            return False, "self-loop found"

        for j in range(i + 1, n):
            if adj[i][j] != adj[j][i]:
                return False, "adjacency matrix is not symmetric"

    generated_m = sum(
        adj[i][j]
        for i in range(n)
        for j in range(i + 1, n)
    )

    if generated_m != m:
        return False, f"expected m={m}, generated m={generated_m}"

    actual_counts = realised_layer_counts(
        adj=adj,
        centroid=centroid,
        k=len(target_counts),
    )

    if actual_counts != target_counts:
        return False, (
            f"hop counts differ: target={target_counts}, "
            f"generated={actual_counts}"
        )

    return True, "ok"


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def write_block(f, item, adj, centroid):
    sn = item["sn"]
    n = int(item["n"])
    m = infer_m(item)

    n_cond, rho, k, _ = get_raw_condition(item)

    x = [0] * n
    x[centroid] = 1

    f.write(f"N={n}\n")
    f.write(f"supernode={sn} n_true={n} m_true={m}\n")
    f.write(f"cond: {n_cond:.4f} {rho:.4f} {float(k):.4f}\n")

    f.write("X: \n")
    f.write(" ".join(map(str, x)) + " \n")

    f.write("E: \n")
    for row in adj:
        f.write(" ".join(map(str, row)) + " \n")

    f.write("\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SC-RG random intra baseline"
    )

    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify n, m, symmetry, connectivity and exact hop counts."
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
            args.output_dir / f"scrg_samples_s{sample_idx}.txt"
        )

        with output_path.open("w", encoding="utf-8") as f:
            for item in test_split:
                n = int(item["n"])
                m = infer_m(item)

                _, _, k, h = get_raw_condition(item)
                target_counts = h_to_layer_counts(h=h, n=n)

                centroid = get_centroid_local_index(item)

                adj, _ = build_hop_aware_graph(
                    n=n,
                    m=m,
                    centroid=centroid,
                    layer_counts=target_counts,
                    rng=rng,
                )

                if args.check:
                    ok, msg = validate_generated(
                        adj=adj,
                        n=n,
                        m=m,
                        centroid=centroid,
                        target_counts=target_counts,
                    )

                    if not ok:
                        raise RuntimeError(
                            f"Supernode {item['sn']}: {msg}"
                        )

                write_block(
                    f=f,
                    item=item,
                    adj=adj,
                    centroid=centroid,
                )

        print(
            f"Wrote {len(test_split)} SC_RG blocks -> {output_path}"
        )


if __name__ == "__main__":
    main()
