"""Record-aligned sampling of D_inter's test split: one B_ij per test pair.

Runs in the `digress` env, on a GPU. For every pair in the dataset's TEST split,
the real L_i and L_j are given as context, the pair's own standardised condition
y as c_ij, and the model generates the cross block:

    python3 sample_pairs_conditional.py \
        dataset=pair_B_lux2olde_k7 +experiment=pair_lux \
        general.test_only="'/abs/path/checkpoints/pair_lux/epoch=179.ckpt'" \
        +repeats=5

main.py's test mode is not used for this because it draws conditions at random,
with replacement, and does not record which pair a sample was for -- fine for a
distributional check, useless for scoring each test pair against its own real
block, which is what the A/B/C comparison needs.

+repeats=N samples the split N times with seeds +seed .. +seed+N-1 (default
0..N-1) from the one loaded checkpoint and writes pair_samples_s<seed>.pt into the
hydra run directory:

    {pairs_file, checkpoint, seed, blocks: [{idx, sn_i, sn_j, n_i, n_j, w, edge_index}]}

idx is the pair's position in the test split; edge_index is the generated cross
block in local ids (row: i side 0..n_i-1, col: j side 0..n_j-1).
evaluation/transfer_summary.py scores the files.

The `given` encoding mirrors PairDatasetInfos.sample_conditioning and
private_inference/sample_private_blocks.py:build_given entry for entry:
no-edge [1,0], edge [0,1], diagonal and padding [0,0]. Cross entries start at
no-edge and are overwritten with noise by sample_batch, so are never read.
"""
import graph_tool as gt   # noqa: F401  MUST BE FIRST -- see sample_private_blocks.py
import os
import random
import warnings

import numpy as np
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from src import utils                                              # noqa: F401
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

warnings.filterwarnings("ignore", category=PossibleUserWarning)

BATCH = 32


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_given(chunk, device):
    """(X, E) observed context for a batch of test records, padded to the batch max.

    The internal blocks come from each record's own edge_index: an edge is
    context exactly when both endpoints are on the same side.
    """
    n_nodes = torch.tensor([r['n_i'] + r['n_j'] for r in chunk], device=device)
    n_max = int(n_nodes.max())
    bs = len(chunk)

    X = torch.zeros(bs, n_max, 2, device=device)
    E = torch.zeros(bs, n_max, n_max, 2, device=device)
    for b, r in enumerate(chunk):
        n_i, n = r['n_i'], r['n_i'] + r['n_j']
        X[b, :n_i, 0] = 1.
        X[b, n_i:n, 1] = 1.
        E[b, :n, :n, 0] = 1.                           # no-edge everywhere first
        src, dst = r['edge_index'].to(device)
        same_side = (src < n_i) == (dst < n_i)
        src, dst = src[same_side], dst[same_side]
        E[b, src, dst, 0] = 0.
        E[b, src, dst, 1] = 1.
        E[b, torch.arange(n), torch.arange(n), :] = 0.
    return {'X': X, 'E': E}, n_nodes


def sample_split(model, test, device, seed, meta, out_path):
    """One cross block per test record, written to out_path."""
    seed_all(seed)
    # Sorted by combined size so each batch pads close to its own maximum.
    order = sorted(range(len(test)), key=lambda i: test[i]['n_i'] + test[i]['n_j'])
    blocks, exact = [None] * len(test), 0
    with torch.no_grad():
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            chunk = [test[i] for i in idx]
            cond_y = torch.stack([r['y'] for r in chunk]).float().to(device)
            given, n_nodes = build_given(chunk, device)
            samples = model.sample_batch(batch_id=start, batch_size=len(chunk),
                                         keep_chain=0, number_chain_steps=1,
                                         save_final=0, num_nodes=n_nodes,
                                         cond_y=cond_y, given=given)
            for i, r, (side, edge_types) in zip(idx, chunk, samples):
                vi = torch.nonzero(side == 0, as_tuple=True)[0]
                vj = torch.nonzero(side == 1, as_tuple=True)[0]
                B = (edge_types[vi][:, vj] == 1)
                s, d = torch.nonzero(B, as_tuple=True)
                blocks[i] = {'idx': i, 'sn_i': int(r['sn_i']), 'sn_j': int(r['sn_j']),
                             'n_i': int(r['n_i']), 'n_j': int(r['n_j']), 'w': int(r['w']),
                             'edge_index': torch.stack([s, d]).long().cpu()}
                exact += int(int(B.sum()) == int(r['w']))
            print(f"  {min(start + BATCH, len(test))}/{len(test)}", flush=True)
    torch.save(dict(meta, seed=seed, blocks=blocks), out_path)
    print(f"  |B_ij| exactly w_ij: {exact}/{len(test)} ({100.0 * exact / len(test):.1f}%)")
    print(f"wrote {out_path}")


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    assert cfg.general.test_only, 'pass general.test_only="\'/abs/path/to/ckpt\'"'
    from datasets.pair_dataset import PairGraphDataModule, PairDatasetInfos

    datamodule = PairGraphDataModule(cfg)
    dataset_infos = PairDatasetInfos(datamodule, cfg.dataset)
    extra_features = (ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                      if cfg.model.extra_features is not None else DummyExtraFeatures())
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                            extra_features=extra_features,
                                            domain_features=domain_features)

    device = 'cuda' if (cfg.general.gpus > 0 and torch.cuda.is_available()) else 'cpu'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, map_location=torch.device(device),
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=None, visualization_tools=None,
        extra_features=extra_features, domain_features=domain_features)
    model.eval()
    model = model.to(device)

    blob = datamodule.raw_blob()
    test = blob['splits'][2]
    # Smoke tests only: the first N test pairs. transfer_summary.py rejects a
    # file that does not cover the whole split, so this cannot leak into a table.
    if cfg.get('limit'):
        test = test[:int(cfg.limit)]
    print(f"{cfg.dataset.pairs_file}: generating {len(test)} cross blocks")
    print(f"condition ({len(blob['cond_names'])} dims): {list(blob['cond_names'])}")

    out_dir = HydraConfig.get().runtime.output_dir
    meta = {'pairs_file': str(cfg.dataset.pairs_file),
            'checkpoint': str(cfg.general.test_only)}
    repeats = int(cfg.get('repeats', 1))
    seed0 = int(cfg.get('seed', 0))
    for rep in range(repeats):
        seed = seed0 + rep
        print(f"\nrepeat {rep + 1}/{repeats}  (seed {seed})")
        sample_split(model, test, device, seed, meta,
                     os.path.join(out_dir, f'pair_samples_s{seed}.pt'))


if __name__ == '__main__':
    main()
