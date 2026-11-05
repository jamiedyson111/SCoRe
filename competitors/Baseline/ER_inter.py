#!/usr/bin/env python3
"""
ER inter-supernode baseline for Configuration B (Luxembourg -> Oldenburg).

For each Oldenburg test supernode pair (i, j), uniformly sample exactly
w_ij distinct cross-edges from the n_i * n_j possible bipartite node pairs.

This is the inter-block counterpart of the fixed-edge Erdős-Rényi G(n,m)
baseline used for D_intra.

Input
-----
pairs_inter_B_lux2olde_k7.pt

Expected test item fields:
    sn_i, sn_j, n_i, n_j, w, ...

Output
------
pair_samples_s0.pt
pair_samples_s1.pt
...

Each output file matches the structure of the supplied D_inter sample:
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

Important
---------
The output edge_index uses LOCAL bipartite indices:
    row 0: node index in supernode i, 0 <= u < n_i
    row 1: node index in supernode j, 0 <= v < n_j

The true input edge locations are NOT used. Only n_i, n_j and w_ij are used.
"""

import argparse
import random
from pathlib import Path

import torch


def load_test_split(path):
    # Compatible with the PyTorch 1.10 module on BEAR.
    obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, dict):
        raise ValueError(f"Expected input .pt to contain a dict, got {type(obj)}")

    if "splits" not in obj:
        raise ValueError("Input file does not contain 'splits'.")

    splits = obj["splits"]

    if not isinstance(splits, (list, tuple)) or len(splits) < 3:
        raise ValueError("Expected splits = [train, val, test].")

    test_split = splits[2]

    if not isinstance(test_split, (list, tuple)):
        raise ValueError("The test split is not a list/tuple.")

    return obj, test_split


def sample_bipartite_edges(n_i, n_j, w, rng):
    """
    Uniformly sample exactly w distinct cross-edges from n_i * n_j
    possible bipartite pairs.

    Each possible edge (u,v) has:
        0 <= u < n_i
        0 <= v < n_j
    """
    total_possible = n_i * n_j

    if w < 0:
        raise ValueError(f"w must be non-negative, got {w}")

    if w > total_possible:
        raise ValueError(
            f"Cannot sample w={w} distinct edges from "
            f"n_i*n_j={total_possible} possible pairs."
        )

    if w == 0:
        return torch.empty((2, 0), dtype=torch.long)

    # Sampling integer IDs avoids constructing all n_i*n_j pairs in memory.
    chosen = rng.sample(range(total_possible), w)

    u = [edge_id // n_j for edge_id in chosen]
    v = [edge_id % n_j for edge_id in chosen]

    return torch.tensor([u, v], dtype=torch.long)


def validate_block(block):
    n_i = int(block["n_i"])
    n_j = int(block["n_j"])
    w = int(block["w"])
    edge_index = block["edge_index"]

    if not torch.is_tensor(edge_index):
        return False, "edge_index is not a tensor"

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        return False, f"unexpected edge_index shape {tuple(edge_index.shape)}"

    if edge_index.shape[1] != w:
        return False, (
            f"expected exactly w={w} cross-edges, "
            f"generated {edge_index.shape[1]}"
        )

    if w == 0:
        return True, "ok"

    if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= n_i:
        return False, "i-side node index out of range"

    if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= n_j:
        return False, "j-side node index out of range"

    pairs = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    if len(pairs) != len(set(pairs)):
        return False, "duplicate cross-edge generated"

    return True, "ok"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "ER fixed-edge random baseline for D_inter, "
            "Configuration B Luxembourg -> Oldenburg."
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
        help="Directory in which pair_samples_s*.pt will be stored.",
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
        help="Base random seed. Run s uses seed + s. Default: 42.",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate every generated bipartite block.",
    )

    args = parser.parse_args()

    obj, test_split = load_test_split(args.input)

    print(f"Config: {obj.get('config', 'unknown')}")
    print(f"Source: {obj.get('public', obj.get('source', 'unknown'))}")
    print(f"Target: {obj.get('private', 'unknown')}")
    print(f"Test pairs: {len(test_split)}")

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

            edge_index = sample_bipartite_edges(
                n_i=n_i,
                n_j=n_j,
                w=w,
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
                ok, message = validate_block(block)
                if not ok:
                    raise RuntimeError(
                        f"Pair idx={idx}, ({sn_i}, {sn_j}): {message}"
                    )

            blocks.append(block)

        output = {
            "pairs_file": str(args.input),
            "checkpoint": "ER fixed-edge bipartite random baseline",
            "seed": run_seed,
            "blocks": blocks,
        }

        output_path = args.output_dir / f"pair_samples_s{sample_idx}.pt"
        torch.save(output, output_path)

        total_edges = sum(int(b["edge_index"].shape[1]) for b in blocks)

        print(
            f"Run s{sample_idx}: seed={run_seed}, "
            f"pairs={len(blocks)}, cross-edges={total_edges} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()
