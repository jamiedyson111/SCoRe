#!/usr/bin/env python3
"""
SC-RG inter-supernode baseline for Configuration B (Luxembourg -> Oldenburg).

This script is designed for:
    pairs_inter_B_lux2olde_k7.pt

The current test split contains:
    sn_i, sn_j, n_i, n_j, edge_index, w, r, y_raw, graph, y

Important:
    r is a 5-bin cross-degree distribution for the i-side nodes:
        [degree 0, degree 1, degree 2, degree 3, degree >= 4]

The current file stores ONLY this one-sided r. It does not store a separate
r_{j->i}. Therefore this baseline preserves:
    - n_i
    - n_j
    - w_ij
    - r_{i->j}

and leaves the j-side cross-degree distribution random.

For each pair:
    1. Convert r into integer counts of i-side nodes in each degree bin.
    2. Randomly assign i-side nodes to those degree bins.
    3. Construct an exact i-side cross-degree sequence.
    4. Randomly choose distinct neighbours in supernode j for each i-side node.
    5. Save exactly w_ij cross-edges.

The generated output matches the supplied pair_samples_s0.pt structure:
{
    "pairs_file": ...,
    "checkpoint": ...,
    "seed": ...,
    "blocks": [
        {
            "idx": ...,
            "sn_i": ...,
            "sn_j": ...,
            "n_i": ...,
            "n_j": ...,
            "w": ...,
            "edge_index": Tensor[2, w]
        },
        ...
    ]
}

The output edge_index uses LOCAL bipartite indices:
    edge_index[0] -> node index in supernode i, 0 <= u < n_i
    edge_index[1] -> node index in supernode j, 0 <= v < n_j

The true cross-edge locations in the input are NOT used for generation.
"""

import argparse
import math
import random
from pathlib import Path

import torch


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_test_split(path):
    # Compatible with PyTorch 1.10 on BEAR.
    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError(
            f"Expected input .pt to contain a dict, got {type(obj)}"
        )

    if "splits" not in obj:
        raise ValueError("Input file does not contain 'splits'.")

    splits = obj["splits"]

    if not isinstance(splits, (list, tuple)) or len(splits) < 3:
        raise ValueError("Expected splits = [train, val, test].")

    test_split = splits[2]

    if not isinstance(test_split, (list, tuple)):
        raise ValueError("The test split is not a list/tuple.")

    if len(test_split) > 0 and "r" not in test_split[0]:
        raise ValueError(
            "The test items do not contain 'r'. "
            "SC-RG inter requires the cross-degree synopsis r."
        )

    return obj, test_split


# ---------------------------------------------------------------------
# r -> integer bin counts
# ---------------------------------------------------------------------

def proportions_to_counts(r, n):
    """
    Convert a probability/proportion vector r into integer counts summing to n.

    Uses largest-remainder allocation.

    Example:
        n = 24
        r = [23/24, 1/24, 0, 0, 0]
        -> [23, 1, 0, 0, 0]
    """
    probs = [float(x) for x in r]

    if len(probs) != 5:
        raise ValueError(
            f"Expected r to have 5 bins [0,1,2,3,>=4], got {len(probs)}."
        )

    raw = [max(0.0, p) * n for p in probs]
    counts = [int(math.floor(x + 1e-12)) for x in raw]

    remaining = n - sum(counts)

    if remaining > 0:
        order = sorted(
            range(len(raw)),
            key=lambda k: raw[k] - counts[k],
            reverse=True,
        )
        for k in order[:remaining]:
            counts[k] += 1

    elif remaining < 0:
        order = sorted(
            range(len(raw)),
            key=lambda k: raw[k] - counts[k],
        )
        need = -remaining

        for k in order:
            while need > 0 and counts[k] > 0:
                counts[k] -= 1
                need -= 1
            if need == 0:
                break

    if sum(counts) != n:
        raise RuntimeError(
            f"Could not convert r to counts summing to n={n}. "
            f"Got counts={counts}, sum={sum(counts)}."
        )

    return counts


# ---------------------------------------------------------------------
# Construct an i-side degree sequence
# ---------------------------------------------------------------------

def make_i_side_degree_sequence(n_i, n_j, w, r, rng):
    """
    Construct a node-level i-side degree sequence consistent with:
        r = [P(d=0), P(d=1), P(d=2), P(d=3), P(d>=4)]
        total cross-edge count = w

    For bins 0,1,2,3 the degree is exact.
    For >=4 nodes, degree starts at 4 and any remaining stubs are distributed
    randomly across those nodes, subject to degree <= n_j.
    """
    counts = proportions_to_counts(r, n_i)

    c0, c1, c2, c3, c4p = counts

    degrees = (
        [0] * c0
        + [1] * c1
        + [2] * c2
        + [3] * c3
        + [4] * c4p
    )

    if len(degrees) != n_i:
        raise RuntimeError(
            f"Degree sequence length {len(degrees)} != n_i={n_i}."
        )

    minimum_sum = sum(degrees)

    if minimum_sum > w:
        raise ValueError(
            f"r is incompatible with w: minimum degree sum implied by r "
            f"is {minimum_sum}, but w={w}."
        )

    remaining = w - minimum_sum

    # Only >=4-bin nodes may absorb extra degree while preserving the bins.
    high_start = c0 + c1 + c2 + c3
    high_indices = list(range(high_start, n_i))

    if remaining > 0 and not high_indices:
        raise ValueError(
            f"r implies exact degree bins 0..3 with sum={minimum_sum}, "
            f"but w={w}. No >=4 node is available to absorb the "
            f"{remaining} remaining edge(s)."
        )

    while remaining > 0:
        candidates = [
            idx for idx in high_indices
            if degrees[idx] < n_j
        ]

        if not candidates:
            raise ValueError(
                "Cannot allocate remaining cross-degree stubs without "
                "exceeding n_j."
            )

        idx = rng.choice(candidates)
        degrees[idx] += 1
        remaining -= 1

    # Randomly assign the degree labels to actual i-side node indices.
    rng.shuffle(degrees)

    if sum(degrees) != w:
        raise RuntimeError(
            f"Constructed degree sum {sum(degrees)} != w={w}."
        )

    if any(d < 0 or d > n_j for d in degrees):
        raise RuntimeError(
            f"Invalid i-side degree sequence for n_j={n_j}: {degrees}"
        )

    return degrees, counts


# ---------------------------------------------------------------------
# Random bipartite edge generation
# ---------------------------------------------------------------------

def sample_edges_from_i_degrees(degrees_i, n_j, rng):
    """
    For each i-side node u, uniformly sample exactly d_u distinct neighbours
    from the n_j available j-side nodes.

    Different i-side nodes may share the same j-side neighbour, which is valid.
    The resulting graph is simple because a given (u,v) pair is never repeated.
    """
    rows_i = []
    rows_j = []

    for u, d in enumerate(degrees_i):
        if d == 0:
            continue

        neighbours = rng.sample(range(n_j), d)

        for v in neighbours:
            rows_i.append(u)
            rows_j.append(v)

    return torch.tensor(
        [rows_i, rows_j],
        dtype=torch.long,
    )


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def realised_r_counts(edge_index, n_i):
    deg = [0] * n_i

    for u in edge_index[0].tolist():
        deg[int(u)] += 1

    counts = [0, 0, 0, 0, 0]

    for d in deg:
        if d <= 3:
            counts[d] += 1
        else:
            counts[4] += 1

    return counts


def validate_block(block, target_counts):
    n_i = int(block["n_i"])
    n_j = int(block["n_j"])
    w = int(block["w"])
    edge_index = block["edge_index"]

    if not torch.is_tensor(edge_index):
        return False, "edge_index is not a tensor"

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return False, (
            f"unexpected edge_index shape {tuple(edge_index.shape)}"
        )

    if edge_index.shape[1] != w:
        return False, (
            f"expected exactly w={w} cross-edges, "
            f"generated {edge_index.shape[1]}"
        )

    if w > 0:
        if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= n_i:
            return False, "i-side node index out of range"

        if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= n_j:
            return False, "j-side node index out of range"

    pairs = list(
        zip(
            edge_index[0].tolist(),
            edge_index[1].tolist(),
        )
    )

    if len(pairs) != len(set(pairs)):
        return False, "duplicate cross-edge generated"

    generated_counts = realised_r_counts(
        edge_index=edge_index,
        n_i=n_i,
    )

    if generated_counts != target_counts:
        return False, (
            f"r-bin counts differ: target={target_counts}, "
            f"generated={generated_counts}"
        )

    return True, "ok"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "SC-RG inter baseline using w_ij and the stored "
            "i-side cross-degree synopsis r."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input pairs_inter_B_lux2olde_k7.pt file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for pair_samples_s*.pt.",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of stochastic runs. Default: 5.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed. Run s uses seed+s. Default: 42.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate exact w, valid indices, no duplicate edges, "
            "and exact i-side r-bin counts."
        ),
    )

    args = parser.parse_args()

    obj, test_split = load_test_split(args.input)

    print(f"Config: {obj.get('config', 'unknown')}")
    print(f"Source: {obj.get('public', obj.get('source', 'unknown'))}")
    print(f"Target: {obj.get('private', 'unknown')}")
    print(f"Test pairs: {len(test_split)}")

    # Check how r is stored.
    if len(test_split) > 0:
        example_r = test_split[0]["r"]
        print(f"r bins: {len(example_r)} -> [0, 1, 2, 3, >=4]")
        print(
            "Note: current input stores one-sided r only "
            "(i-side cross-degree distribution)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in range(args.num_samples):
        run_seed = args.seed + sample_idx
        rng = random.Random(run_seed)

        blocks = []

        for idx, item in enumerate(test_split):
            sn_i = int(item["sn_i"])
            sn_j = int(item["sn_j"])
            n_i = int(item["n_i"])
            n_j = int(item["n_j"])
            w = int(item["w"])
            r = item["r"]

            degrees_i, target_counts = make_i_side_degree_sequence(
                n_i=n_i,
                n_j=n_j,
                w=w,
                r=r,
                rng=rng,
            )

            edge_index = sample_edges_from_i_degrees(
                degrees_i=degrees_i,
                n_j=n_j,
                rng=rng,
            )

            block = {
                "idx": idx,
                "sn_i": sn_i,
                "sn_j": sn_j,
                "n_i": n_i,
                "n_j": n_j,
                "w": w,
                "edge_index": edge_index,
            }

            if args.check:
                ok, message = validate_block(
                    block=block,
                    target_counts=target_counts,
                )

                if not ok:
                    raise RuntimeError(
                        f"Pair idx={idx}, ({sn_i}, {sn_j}): {message}"
                    )

            blocks.append(block)

        output = {
            "pairs_file": str(args.input),
            "checkpoint": (
                "SC-RG inter baseline: exact w_ij + i-side r"
            ),
            "seed": run_seed,
            "blocks": blocks,
        }

        output_path = (
            args.output_dir / f"pair_samples_s{sample_idx}.pt"
        )

        torch.save(output, output_path)

        total_edges = sum(
            int(b["edge_index"].shape[1])
            for b in blocks
        )

        print(
            f"Run s{sample_idx}: seed={run_seed}, "
            f"pairs={len(blocks)}, cross-edges={total_edges} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
