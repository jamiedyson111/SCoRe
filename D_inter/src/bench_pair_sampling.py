"""Benchmark D_inter inference time per pair across the pair-size range (GPU).

    python3 bench_pair_sampling.py dataset=pair_B_dblp2coll_k3_s80_deg +experiment=pair_lux \
        general.test_only="'/abs/last.ckpt'" +bench_out=/abs/bench.json

For each combined-size bin (n_i + n_j) of the dataset's TEST pairs it takes the
LARGEST pairs of the bin (conservative), finds the largest batch that fits in GPU
memory, and times the reverse chain. Timing uses a shortened chain (T_SHORT steps,
scaled to the model's T): every step is the same forward pass, so the time per step
does not depend on T. One bin is also timed with the full chain to check the scaling.

Writes per-bin seconds per pair and the extrapolated total for the whole test split.
Real X_i, X_j as context (the cost does not depend on where the context comes from).
"""
import graph_tool as gt   # noqa: F401  MUST BE FIRST -- see sample_private_blocks.py
import json
import time

import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from src import utils                                              # noqa: F401
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

BINS = [0, 50, 100, 150, 200, 300, 400, 500, 600, 800, 1200]
BATCHES = [128, 64, 32, 16, 8, 4, 2, 1]
T_SHORT = 25


def run(model, recs, device, T):
    from datasets.pair_dataset import build_given, record_features
    model.T = T
    cond_y = torch.stack([r['y'] for r in recs]).float().to(device)
    given, n_nodes = build_given([record_features(r) for r in recs], device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    with torch.no_grad():
        model.sample_batch(batch_id=0, batch_size=len(recs), keep_chain=0, number_chain_steps=1,
                           save_final=0, num_nodes=n_nodes, cond_y=cond_y, given=given)
    torch.cuda.synchronize()
    return time.time() - t, torch.cuda.max_memory_allocated() / 2**30


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    from datasets.pair_dataset import PairGraphDataModule, PairDatasetInfos
    dm = PairGraphDataModule(cfg)
    infos = PairDatasetInfos(dm, cfg.dataset)
    extra = (ExtraFeatures(cfg.model.extra_features, dataset_info=infos)
             if cfg.model.extra_features is not None else DummyExtraFeatures())
    infos.compute_input_output_dims(datamodule=dm, extra_features=extra,
                                    domain_features=DummyExtraFeatures())
    device = 'cuda'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, map_location=torch.device(device), dataset_infos=infos,
        train_metrics=TrainAbstractMetricsDiscrete(), sampling_metrics=None,
        visualization_tools=None, extra_features=extra, domain_features=DummyExtraFeatures())
    model.eval().to(device)
    T_full = model.T

    test = dm.raw_blob()['splits'][2]
    size = np.array([int(r['n_i']) + int(r['n_j']) for r in test])
    order = np.argsort(size)
    print(f"{len(test)} test pairs; combined size median {np.median(size):.0f}, "
          f"p90 {np.percentile(size, 90):.0f}, max {size.max()}; T = {T_full}", flush=True)
    run(model, [test[order[0]]], device, 2)                       # warm-up

    out = {'n_pairs': len(test), 'T': T_full, 'gpu': torch.cuda.get_device_name(0), 'bins': []}
    total = 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        members = order[(size[order] > lo) & (size[order] <= hi)]
        if not len(members):
            continue
        per_pair = None
        for b in sorted({min(b, len(members)) for b in BATCHES}, reverse=True):
            recs = [test[i] for i in members[::-1][:b]]           # the largest of the bin
            try:
                sec, mem = run(model, recs, device, T_SHORT)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            per_pair = sec * T_full / T_SHORT / len(recs)
            break
        if per_pair is None:
            raise SystemExit(f"size {lo + 1}-{hi}: even one pair does not fit in memory")
        row = {'bin': f"{lo + 1}-{hi}", 'pairs': int(len(members)), 'max_size': int(size[members].max()),
               'batch': len(recs), 's_per_pair': per_pair, 'peak_gb': mem}
        total += per_pair * len(members)
        out['bins'].append(row)
        print(f"  size {lo + 1:>4}-{hi:<4} {len(members):>6} pairs  max {row['max_size']:>4}  "
              f"batch {len(recs):>3}  {per_pair:7.2f} s/pair  peak {mem:5.1f} GB", flush=True)

    # check the T scaling on the median bin with the full chain
    mid = out['bins'][len(out['bins']) // 2]
    lo, hi = (int(x) for x in mid['bin'].split('-'))
    members = order[(size[order] >= lo) & (size[order] <= hi)]
    recs = [test[i] for i in members[::-1][:mid['batch']]]
    sec, _ = run(model, recs, device, T_full)
    out['check'] = {'bin': mid['bin'], 'full_T_s_per_pair': sec / len(recs),
                    'scaled_s_per_pair': mid['s_per_pair']}
    print(f"  check (bin {mid['bin']}): full chain {sec / len(recs):.2f} s/pair vs scaled "
          f"{mid['s_per_pair']:.2f}", flush=True)
    out['total_gpu_h'] = total / 3600
    print(f"\nall {len(test)} test pairs, one run: {total / 3600:.1f} GPU-h", flush=True)
    json.dump(out, open(cfg.bench_out, 'w'), indent=1)


if __name__ == '__main__':
    main()
