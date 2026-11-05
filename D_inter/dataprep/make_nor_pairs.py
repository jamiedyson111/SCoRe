"""The no-r ablation: the internal-degree pair datasets with c_ij cut to [n_i, n_j, w_ij].

    python make_nor_pairs.py                      # after make_pair_blocks.py and make_transfer_pairs.py
    python make_nor_pairs.py --tag lux2verm       # Vermont's B and C (the Luxembourg file is shared)

Reads pairs_inter_{luxembourg_osm,B_lux2olde,C_lux2olde}_k7_s80_deg.pt and writes the
same files with a _nor suffix. Only the condition changes: r_{i->j} and r_{j->i} (the
last 10 columns) are dropped from y_raw, y, cond_mean / cond_std / cond_constant and
cond_names. Each column was standardised on its own, so n_i, n_j and w_ij keep exactly
their values; records, splits, X_i and the edge marginal are untouched, and the
records keep r / r_ji for scoring E_r. Comparing a _nor model with its _deg twin
measures what r adds.
"""
import argparse
import os

import torch

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "pair_lux", "raw")
TAG = "lux2olde"
KEEP = ["n_i", "n_j", "w_ij"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=TAG, help="the transfer pair (lux2olde, lux2verm)")
    tag = ap.parse_args().tag
    files = ["pairs_inter_luxembourg_osm_k7_s80_deg.pt", f"pairs_inter_B_{tag}_k7_s80_deg.pt",
             f"pairs_inter_C_{tag}_k7_s80_deg.pt"]
    for name in files:
        blob = torch.load(os.path.join(RAW, name))
        names = list(blob["cond_names"])
        assert names[:len(KEEP)] == KEEP, names
        d = len(KEEP)
        blob = dict(blob, cond_names=KEEP, cond_mean=blob["cond_mean"][:d],
                    cond_std=blob["cond_std"][:d], cond_constant=blob["cond_constant"][:d],
                    splits=[[dict(r, y_raw=r["y_raw"][:d], y=r["y"][:d]) for r in s]
                            for s in blob["splits"]])
        out = name.replace(".pt", "_nor.pt")
        torch.save(blob, os.path.join(RAW, out))
        print(f"  {out:<44} {[len(s) for s in blob['splits']]}  condition {KEEP} "
              f"(dropped {len(names) - d} r columns)")


if __name__ == "__main__":
    main()
