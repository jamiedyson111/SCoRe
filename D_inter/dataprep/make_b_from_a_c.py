"""Rebuild pairs_inter_B_<tag>.pt from the A and C files already on BlueBEAR.

    python make_b_from_a_c.py --tag dblp2coll_k3_s80_deg

B is A's training and validation pairs (the source graph, A's standardisation) with
C's test pairs (the target graph) re-standardised with A's statistics -- exactly what
make_partition_pairs.py writes (checked locally on dblp2coll: same pairs, max |dy| = 0),
without uploading the 1.7 GB file.
"""
import argparse
import os

import torch

RAW = "../data/pair_lux/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    L = lambda c: torch.load(os.path.join(RAW, f"pairs_inter_{c}_{args.tag}.pt"),  # noqa: E731
                             map_location="cpu", weights_only=False)
    A, C = L("A"), L("C")
    test = C["splits"][2]
    for r in test:
        r["y"] = (r["y_raw"] - A["cond_mean"]) / A["cond_std"]
    B = {k: v for k, v in A.items() if k not in ("splits", "graph")}
    B.update(config="B", splits=[A["splits"][0], A["splits"][1], test])
    out = os.path.join(RAW, f"pairs_inter_B_{args.tag}.pt")
    torch.save(B, out)
    print(f"wrote {out}: {[len(s) for s in B['splits']]}")


if __name__ == "__main__":
    main()
