"""Target-only D_intra ("Ours-Target"): our conditioned model, trained from scratch on
the target graph's 80% alone, no pretraining on the source.

    python make_target_datasets.py --tag lux2olde_k7_s80 lux2verm_k7_s80 dblp2coll_k3_s80

Built from pairs_Cft_<tag>.pt, so the supernodes, records, split and test set are
exactly the fine-tune's (and B's): train = the target's training supernodes, test =
its test supernodes. Two things differ, so nothing comes from the source graph:

  - the condition is standardised with the TARGET training statistics, not A's;
  - there is no `edge_marginal`, so the noise process uses the target training
    data's own edge marginal (Cft keeps A's, to start from A's model).

Writes pairs_T_<tag>.pt; dataset config configs/dataset/cluster_T_<tag>.yaml.
"""
import argparse
import copy
import os

import torch

RAW = "../data/cluster_lux/raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True)
    args = ap.parse_args()

    for tag in args.tag:
        Cft = torch.load(os.path.join(RAW, f"pairs_Cft_{tag}.pt"))
        splits = [copy.deepcopy(s) for s in Cft["splits"]]
        tr = splits[0]
        if any(r["graph"] != Cft["private"] for r in tr):
            raise SystemExit(f"{tag}: Cft's training split holds non-target records")

        # kmis_pairs.standardise: train statistics only; constant features keep std 1
        Y = torch.stack([r["y_raw"] for r in tr])
        mean, std = Y.mean(0), Y.std(0)
        const = std < 1e-6
        std = torch.where(const, torch.ones_like(std), std)
        for split in splits:
            for r in split:
                r["y"] = (r["y_raw"] - mean) / std

        T = {k: v for k, v in Cft.items()
             if k not in ("splits", "edge_marginal", "finetune_of")}
        T.update(config="T", cond_mean=mean, cond_std=std, cond_constant=const,
                 desc=f"ours, trained on {Cft['private']} train only (no pretraining), "
                      f"test {Cft['private']}",
                 splits=splits)
        path = os.path.join(RAW, f"pairs_T_{tag}.pt")
        torch.save(T, path)
        print(f"  pairs_T_{tag}.pt  {len(splits[0])} train / {len(splits[1])} val / "
              f"{len(splits[2])} test  (target-train standardisation, own marginal)")


if __name__ == "__main__":
    main()
