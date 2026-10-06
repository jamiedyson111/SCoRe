"""Conditional sampling from a trained D_intra checkpoint.

For every supernode in the test split: take its condition c_i and its size n_i,
and generate a subgraph. This is the "feed in a supernode, get back its cluster"
step. Training is unchanged -- run main.py for that; this is generation only.

    python3 sample_conditional.py \
        dataset=cluster_lux +experiment=cluster_lux \
        general.test_only=/abs/path/to/checkpoints/cluster_lux/epoch=NNN.ckpt

Writes conditional_samples.txt in the hydra run directory, in the same format
main.py uses for generated_samples1.txt, plus the condition each graph was given.

REPEATS. +repeats=N samples the whole test split N times from the one loaded
checkpoint, with seeds +seed .. +seed+N-1 (default 0..N-1), and writes
conditional_samples_s<seed>.txt for each. evaluation/repeat_summary.py then
reports every metric as median and std across the N runs. This measures
SAMPLING noise only: all runs share one set of trained weights.

Sampling is seeded either way (seed 0 by default), so a run is reproducible on
the same hardware. Before 2026-10-06 it was unseeded.
"""
import random
import graph_tool as gt          # must precede torch; see coarse/cluster dataset notes
import os
import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

BATCH = 32


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    assert cfg.general.test_only, "pass general.test_only=/abs/path/to/checkpoint.ckpt"

    from datasets.cluster_dataset import (ClusterGraphDataModule, ClusterDatasetInfos,
                                          ClusterSamplingMetrics)
    from analysis.visualization import NonMolecularVisualization

    datamodule = ClusterGraphDataModule(cfg)
    dataset_infos = ClusterDatasetInfos(datamodule, cfg.dataset)
    extra_features = (ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                      if cfg.model.extra_features is not None else DummyExtraFeatures())
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                            extra_features=extra_features,
                                            domain_features=domain_features)

    # visualization_tools=None on purpose: the visualisation block inside
    # sample_batch calls self.print(), which needs an attached Lightning Trainer.
    # We call sample_batch directly, so passing a visualiser raises
    # "not attached to a `Trainer`". None skips the whole block.
    device = 'cuda' if (cfg.general.gpus > 0 and torch.cuda.is_available()) else 'cpu'

    # map_location is required to load a GPU-trained checkpoint on a CPU-only
    # machine; without it torch raises "Attempting to deserialize object on a
    # CUDA device but torch.cuda.is_available() is False".
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only,
        map_location=torch.device(device),
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        # sampling_metrics=None: constructing ClusterSamplingMetrics converts every
        # graph in all three splits to networkx (7,650+ here), which takes many
        # minutes. This script only generates -- it never scores -- so the object
        # is pure startup cost. Scoring is done separately by evaluate_clusters.py.
        sampling_metrics=None,
        visualization_tools=None,
        extra_features=extra_features,
        domain_features=domain_features)
    model.eval()
    model = model.to(device)

    blob = datamodule.raw_blob()
    test = blob['splits'][2]
    print(f"generating for {len(test)} test supernodes")
    print(f"condition ({len(blob['cond_names'])} dims): {blob['cond_names']}")

    repeats = int(cfg.get('repeats', 1))
    seed0 = int(cfg.get('seed', 0))
    for rep in range(repeats):
        seed = seed0 + rep
        out_path = ('conditional_samples.txt' if repeats == 1
                    else f'conditional_samples_s{seed}.txt')
        if repeats > 1:
            print(f"\nrepeat {rep + 1}/{repeats}  (seed {seed})")
        sample_split(model, test, device, seed, out_path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_split(model, test, device, seed, out_path):
    """One conditional sample per test supernode, written to out_path."""
    seed_all(seed)
    written = 0
    with open(out_path, 'w') as f:
        for start in range(0, len(test), BATCH):
            chunk = test[start:start + BATCH]

            # every graph in a batch is padded to the same n_max, so n_i is passed
            # per-graph and the node_mask does the trimming
            n_nodes = torch.tensor([int(r['n']) for r in chunk], device=device)
            cond_y = torch.stack([r['y'] for r in chunk]).float().to(device)

            # number_chain_steps must be >= 1 even with keep_chain=0: sample_batch
            # indexes chain_X[0] regardless, so 0 raises IndexError.
            samples = model.sample_batch(batch_id=start, batch_size=len(chunk),
                                         keep_chain=0, number_chain_steps=1,
                                         save_final=0, num_nodes=n_nodes,
                                         cond_y=cond_y)

            for rec, (node_types, edge_types) in zip(chunk, samples):
                raw = (rec['y_raw'].tolist() if 'y_raw' in rec else rec['y'].tolist())
                f.write(f"N={node_types.shape[0]}\n")
                f.write(f"supernode={rec['sn']} n_true={int(rec['n'])} "
                        f"m_true={rec['edge_index'].shape[1] // 2}\n")
                f.write("cond: " + " ".join(f"{v:.4f}" for v in raw) + "\n")
                f.write("X: \n" + " ".join(str(v) for v in node_types.tolist()) + " \n")
                f.write("E: \n")
                for row in edge_types:
                    f.write(" ".join(str(v) for v in row.tolist()) + " \n")
                f.write("\n")
                written += 1
            print(f"  {written}/{len(test)}", flush=True)

    print(f"wrote {os.path.abspath(out_path)}")


if __name__ == '__main__':
    main()
