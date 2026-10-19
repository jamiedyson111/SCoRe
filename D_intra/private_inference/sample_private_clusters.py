"""STAGE 1 of private inference: generate every supernode's internal block.

Runs in the `digress` env, on a GPU. Reads the release from stage 0 and produces
one generated adjacency L_i per supernode:

    python3 sample_private_clusters.py \
        dataset=cluster_lux +experiment=cluster_lux \
        general.test_only="'/abs/path/checkpoints/cluster_lux/epoch=NNN.ckpt'" \
        +release=/abs/path/release_oldenburg_k7.pt \
        +out=/abs/path/clusters_oldenburg_k7.pt

Note the LITERAL single quotes inside the double quotes on test_only. Hydra's
override grammar cannot parse the second '=' in "epoch=NNN.ckpt"; the inner
quotes reach Hydra, which strips them itself. Shell quoting alone does not help.

This is sample_conditional.py with one thing changed: conditions come from the
release rather than from the dataset's own test split. The dataset is still
loaded, because the checkpoint needs its dataset_infos to rebuild the model AND
because its blob carries the standardisation the model was trained under.

STANDARDISATION IS THE EASIEST THING TO GET SILENTLY WRONG HERE. The condition
must be scaled by the TRAINING graph's mean and std, not the private graph's --
those numbers are part of the model, not part of the data being described. The
divisor for a constant feature such as k is a clamped 1e-6, so a k that differs
from the trained one does not degrade gracefully, it explodes by six orders of
magnitude. Hence the hard assert rather than a warning.

REPEATS. +tag=CFG +repeats=N samples the whole release N times, seeds +seed ..
+seed+N-1 (default 0..N-1), from the one loaded checkpoint, and writes
clusters_<graph>_k<k>_<CFG>_s<seed>.pt beside the release for each -- the input
of sample_private_blocks.py's own +clusters_tag. Without +tag it writes one file,
+out or clusters_<graph>_k<k>.pt, at seed +seed (0 by default).

THE CONDITION IS SELECTED BY NAME. The release carries the full [n, rho_in, k,
h1..h7]; each checkpoint's dataset says which of those columns it was trained on,
so a no-hop model reads [n, rho_in, k] out of the same release. v3 and the
*_centroid models also need dataset.mark_centroid=true (and v3
model.hop_embed_dim=1) -- then sample_batch fixes node 0 as the centroid, exactly
as in sample_conditional.py. run_private_clusters_repeats.sh sets both.

SUPERNODES BELOW MIN_N NEED NO MODEL. n = 1 has no internal slot at all, and
n = 2 has exactly one, whose occupancy the released rho_in determines outright.
Training dropped these, so the model has never seen them; filling them
arithmetically is both correct and closer to the release than a guess.
"""
import graph_tool as gt          # must precede torch; see cluster dataset notes
import os
import pathlib
import random
import sys
import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

ROOT = pathlib.Path(__file__).resolve().parents[1]
# sample_conditional.py lives in src/ and mixes `from src import utils` with
# `from datasets... import`, so both the project root and src/ must be importable.
for p in (str(ROOT), str(ROOT / 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import utils                                              # noqa: E402,F401
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete  # noqa: E402
from diffusion_model_discrete import DiscreteDenoisingDiffusion    # noqa: E402
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # noqa: E402

warnings.filterwarnings("ignore", category=PossibleUserWarning)

BATCH = 32


def _load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def condition_columns(blob, release_names):
    """Positions in the release's y_raw_intra of the columns the model was trained on."""
    names, have = list(blob['cond_names']), list(release_names)
    missing = [nm for nm in names if nm not in have]
    if missing:
        raise SystemExit(f"the model was trained on {names}, but the release has no "
                         f"{missing} (it supplies {have})")
    return [have.index(nm) for nm in names]


def standardise(y_raw, blob):
    """Apply the TRAINING statistics carried by the checkpoint's own dataset."""
    return (y_raw - blob['cond_mean']) / blob['cond_std']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def small_block(n, rho_in):
    """Internal edges of a supernode too small for the model to have been trained on.

    m = rho * n(n-1)/2 inverts the density exactly, and for n < 3 there is at
    most one slot, so this is determined rather than approximated.
    """
    slots = n * (n - 1) // 2
    m = int(round(rho_in * slots))
    if n < 2 or m < 1:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.long)     # the only slot


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    assert cfg.general.test_only, 'pass general.test_only="\'/abs/path/to/ckpt\'"'
    assert cfg.get('release'), "pass +release=/abs/path/to/release_<graph>_k<k>.pt"

    from datasets.cluster_dataset import ClusterGraphDataModule, ClusterDatasetInfos
    from analysis.visualization import NonMolecularVisualization      # noqa: F401

    release = _load(cfg.release)
    repeats = int(cfg.get('repeats', 1))
    seed0 = int(cfg.get('seed', 0))
    stem = os.path.join(os.path.dirname(os.path.abspath(cfg.release)),
                        f"clusters_{release['graph']}_k{release['k']}")
    if cfg.get('tag'):
        jobs = [(seed0 + r, f"{stem}_{cfg.tag}_s{seed0 + r}.pt") for r in range(repeats)]
    else:
        assert repeats == 1, "+repeats needs +tag, which names the output files"
        jobs = [(seed0, cfg.get('out') or f"{stem}.pt")]

    datamodule = ClusterGraphDataModule(cfg)
    dataset_infos = ClusterDatasetInfos(datamodule, cfg.dataset)
    extra_features = (ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                      if cfg.model.extra_features is not None else DummyExtraFeatures())
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                            extra_features=extra_features,
                                            domain_features=domain_features)

    blob = datamodule.raw_blob()
    # blob['k'] is the k the model TRAINED at -- the public graph's. A transfer
    # dataset coarsens the private graph separately and records that as
    # 'k_private', so a private graph at k=4 against a Luxembourg-k=7 checkpoint is
    # the configuration this pipeline exists to run, not an error.
    #
    # What would actually break the forward pass is dividing an unseen k by a
    # near-zero training std, so check that instead of checking k itself.
    # kmis_pairs.standardise() leaves constant features on their raw scale
    # (std := 1) precisely so an unseen k stays a shift of a few units rather
    # than ~1e6.
    k_train = int(blob['k'])
    k_rel = int(release['k'])
    k_priv = int(blob.get('k_private', k_train))
    if k_rel != k_priv:
        raise SystemExit(
            f"release was coarsened at k={k_rel} but {cfg.dataset.pairs_file} "
            f"holds out a private graph coarsened at k={k_priv}. Build the "
            f"release with --k {k_priv}, or point at the matching dataset.")
    if k_rel != k_train:
        names = list(blob['cond_names'])
        ki = names.index('k')
        k_std = float(blob['cond_std'][ki])
        if k_std < 1e-3:
            raise SystemExit(
                f"release k={k_rel} != training k={k_train}, and the k feature "
                f"was standardised by std={k_std:.3g}. Dividing by that scales "
                f"the mismatch by ~{1 / k_std:.0g} and destroys the forward "
                f"pass. This dataset cannot carry a k it never saw vary.")
        z = (k_rel - float(blob['cond_mean'][ki])) / k_std
        print(f"note: release k={k_rel}, checkpoint trained at k={k_train}; "
              f"the k feature arrives at {z:+.2f} sigma, where training was "
              f"always {(k_train - float(blob['cond_mean'][ki])) / k_std:+.2f}. "
              f"Out of distribution by design -- see NOTES.md, 2026-08-14.")

    device = 'cuda' if (cfg.general.gpus > 0 and torch.cuda.is_available()) else 'cpu'
    # visualization_tools=None: the visualisation block calls self.print(), which
    # needs an attached Trainer, and we call sample_batch directly.
    # sampling_metrics=None: building them converts every graph in all splits to
    # networkx, which costs minutes and is never used here.
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, map_location=torch.device(device),
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=None, visualization_tools=None,
        extra_features=extra_features, domain_features=domain_features)
    model.eval()
    model = model.to(device)

    sns = release['supernodes']
    modelled = [s for s in sns if s['modelled']]
    tiny = [s for s in sns if not s['modelled']]
    print(f"{release['graph']} k={release['k']}: {len(sns)} supernodes "
          f"({len(modelled)} generated, {len(tiny)} filled arithmetically)")
    print(f"condition ({len(blob['cond_names'])} dims): {list(blob['cond_names'])}")
    print(f"centroid node type: {'node 0' if dataset_infos.fixed_X else 'none'}")
    assert 'y_raw_intra' in sns[0], (
        "this release predates the hop-histogram condition and carries no "
        "y_raw_intra. Re-run stage 0 (make_private_release.py) to rebuild it.")
    cols = condition_columns(blob, release['intra_cond_names'])
    # Smoke tests only: the first N supernodes. The output is missing the rest,
    # so it is marked and assemble_syn / syn_repeats_summary refuse it.
    limit = int(cfg.get('limit') or 0)
    if limit:
        modelled, tiny = modelled[:limit], []
        print(f"LIMIT: {limit} supernodes only -- for testing, do not assemble")

    for seed, out_path in jobs:
        print(f"\nseed {seed} -> {out_path}")
        seed_all(seed)
        out = sample_release(model, modelled, tiny, blob, cols, device)
        result = {
            'graph': release['graph'], 'k': release['k'],
            'checkpoint': str(cfg.general.test_only),
            'dataset': str(cfg.dataset.pairs_file),
            'tag': cfg.get('tag'), 'seed': seed,
            'centroid_marked': bool(dataset_infos.fixed_X),
            'limit': limit or None,
            'clusters': out,
        }
        torch.save(result, out_path)

        gen_m = sum(v['edge_index'].shape[1] // 2 for v in out.values())
        rel_m = sum(int(s['m_in_true']) for s in sns)
        print(f"  generated {gen_m} internal edges, release describes {rel_m} "
              f"({100.0 * gen_m / max(rel_m, 1):.1f}%)")
        print(f"wrote {out_path}")


def sample_release(model, modelled, tiny, blob, cols, device):
    """One internal block per supernode of the release: {sn: {n, modelled, edge_index}}."""
    out = {}
    for s in tiny:
        out[int(s['sn'])] = {'n': int(s['n']), 'modelled': False,
                             'edge_index': small_block(int(s['n']), float(s['rho_in']))}

    # Sort by size so each batch pads to roughly its own n, rather than to the
    # largest supernode in the graph.
    modelled = sorted(modelled, key=lambda s: int(s['n']))
    done = 0
    with torch.no_grad():
        for start in range(0, len(modelled), BATCH):
            chunk = modelled[start:start + BATCH]

            # Stage 0 assembled y_raw with the same build_condition() the
            # training records used, so nothing here re-derives the layout.
            # Rebuilding it from n/rho_in would silently drop the h1..hk hop
            # histogram the current checkpoints are also conditioned on.
            y_raw = torch.stack([s['y_raw_intra'].float()[cols] for s in chunk])
            cond_y = standardise(y_raw, blob).float().to(device)
            n_nodes = torch.tensor([int(s['n']) for s in chunk], device=device)

            # number_chain_steps must be >= 1 even with keep_chain=0: sample_batch
            # indexes chain_X[0] regardless, so 0 raises IndexError.
            samples = model.sample_batch(batch_id=start, batch_size=len(chunk),
                                         keep_chain=0, number_chain_steps=1,
                                         save_final=0, num_nodes=n_nodes,
                                         cond_y=cond_y)

            for s, (_, edge_types) in zip(chunk, samples):
                E = (edge_types == 1)
                src, dst = torch.nonzero(E, as_tuple=True)
                out[int(s['sn'])] = {
                    'n': int(s['n']), 'modelled': True,
                    'edge_index': torch.stack([src, dst]).long(),
                }
            done += len(chunk)
            print(f"  {done}/{len(modelled)}", flush=True)
    return out


if __name__ == '__main__':
    main()
