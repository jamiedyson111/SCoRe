"""STAGE 2 of private inference: generate the cross block for every connected pair.

Runs in the `digress` env, on a GPU. Reads the stage-0 release and the stage-1
generated internal blocks, and produces one B_ij per connected supernode pair:

    python3 sample_private_blocks.py \
        dataset=pair_lux +experiment=pair_lux \
        general.test_only="'/abs/path/checkpoints/pair_lux/epoch=179.ckpt'" \
        +release=/abs/path/release_oldenburg_k7.pt \
        +clusters=/abs/path/clusters_oldenburg_k7.pt \
        +out=/abs/path/blocks_oldenburg_k7.pt

Note the LITERAL single quotes inside the double quotes on test_only -- Hydra's
override grammar cannot parse the second '=' in "epoch=179.ckpt".

To validate this stage independently of D_intra, pass the true internal blocks
that stage 0 writes alongside the release:

    +clusters=/abs/path/clusters_oldenburg_k7_true.pt

With real context the run should reproduce the public-run numbers -- around 96%
of blocks with |B_ij| exactly w_ij -- so a bad result points here rather than at
stage 1's checkpoint. The file is read like any other clusters file; nothing from
the extraction stack (h5py, torch_sparse) is imported, since the GPU environment
does not have it.

WHAT IS OBSERVED AND WHAT IS GENERATED. The model is given the two internal
blocks and the sides, and produces only the cross block. sample_batch re-imposes
the observed part after every reverse step, so L_i and L_j come back bit-identical
-- see the `given` handling in diffusion_model_discrete.sample_batch.

THE `given` ENCODING MUST MATCH TRAINING EXACTLY. PairDatasetInfos.sample_conditioning
is the reference implementation and build_given below mirrors it entry for entry:
no-edge is [1,0], the diagonal is [0,0], padding is [0,0]. Only the non-cross
entries are ever read back, but a mismatch would be a silent train/sample skew --
the kind that produces plausible output and wrong numbers.

rho_i COMES FROM THE RELEASE, not from the generated L_i. The brief defines
c_ij from the released supernode conditions, and the released value is what the
model was trained against. The realised density of the generated block will drift
from it; that drift is reported rather than fed back in.
"""
import graph_tool as gt   # noqa: F401  MUST BE FIRST -- see below
# graph_tool loads the conda environment's libstdc++. Without it the system
# /lib64/libstdc++.so.6 wins, and the first thing to need a newer one dies:
#
#   ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found
#                (required by .../rdkit/rdBase.so)
#
# rdkit gets pulled in here even though nothing in this file uses it, because
# unpickling the checkpoint imports src/analysis/visualization.py -- the saved
# hyperparameters reference its class. Passing visualization_tools=None does not
# help: the override happens after torch.load has already resolved the pickle.
# src/main.py opens with the same import for the same reason.
import os
import pathlib
import sys
import warnings

import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.utilities.warnings import PossibleUserWarning

ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def build_given(chunk, internal, device):
    """(X, E) observed context for a batch of pairs, padded to the batch maximum.

    Mirrors PairDatasetInfos.sample_conditioning in D_inter/src/datasets/pair_dataset.py.
    Cross entries are left at no-edge; sample_batch overwrites them with noise
    because `keep` is False there, so their initial value is never read.
    """
    n_nodes = torch.tensor([p['n_i'] + p['n_j'] for p in chunk], device=device)
    n_max = int(n_nodes.max())
    bs = len(chunk)

    X = torch.zeros(bs, n_max, 2, device=device)
    E = torch.zeros(bs, n_max, n_max, 2, device=device)
    for b, p in enumerate(chunk):
        n_i, n_j = p['n_i'], p['n_j']
        n = n_i + n_j
        X[b, :n_i, 0] = 1.
        X[b, n_i:n, 1] = 1.

        E[b, :n, :n, 0] = 1.                       # no-edge everywhere first
        for offset, sn in ((0, p['sn_i']), (n_i, p['sn_j'])):
            ei = internal[int(sn)]
            if ei.numel():
                src, dst = ei[0] + offset, ei[1] + offset
                E[b, src, dst, 0] = 0.
                E[b, src, dst, 1] = 1.
        E[b, torch.arange(n), torch.arange(n), :] = 0.

    return {'X': X, 'E': E}, n_nodes


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    assert cfg.general.test_only, 'pass general.test_only="\'/abs/path/to/ckpt\'"'
    assert cfg.get('release'), "pass +release=/abs/path/to/release_<graph>_k<k>.pt"

    from datasets.pair_dataset import PairGraphDataModule, PairDatasetInfos

    release = _load(cfg.release)

    assert cfg.get('clusters'), "pass +clusters=/abs/path/to/clusters_<graph>_k<k>.pt"
    blob_c = _load(cfg.clusters)
    assert blob_c['graph'] == release['graph'] and blob_c['k'] == release['k'], (
        f"clusters are for {blob_c['graph']} k={blob_c['k']} but the release is "
        f"{release['graph']} k={release['k']}")
    internal = {int(sn): v['edge_index'] for sn, v in blob_c['clusters'].items()}

    # The validation run passes clusters_<graph>_k<k>_true.pt, the real internal
    # blocks written by stage 0. Reading them from a file rather than rebuilding
    # them here keeps the extraction stack -- h5py, torch_sparse -- out of the
    # GPU environment, which does not have it.
    real_context = blob_c.get('checkpoint') is None
    if real_context:
        print("USING THE PRIVATE GRAPH'S REAL INTERNAL BLOCKS (validation mode)")

    out_path = cfg.get('out') or os.path.join(
        os.path.dirname(cfg.release), f"blocks_{release['graph']}_k{release['k']}.pt")

    datamodule = PairGraphDataModule(cfg)
    dataset_infos = PairDatasetInfos(datamodule, cfg.dataset)
    extra_features = (ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
                      if cfg.model.extra_features is not None else DummyExtraFeatures())
    domain_features = DummyExtraFeatures()
    dataset_infos.compute_input_output_dims(datamodule=datamodule,
                                            extra_features=extra_features,
                                            domain_features=domain_features)

    blob = datamodule.raw_blob()
    assert list(blob['cond_names']) == list(release['inter_cond_names']), (
        f"condition layout mismatch: model trained on {list(blob['cond_names'])}, "
        f"release supplies {list(release['inter_cond_names'])}")
    # D_inter trains on one graph at one k, so the k column is constant and its
    # training std was clamped to ~1e-6. That makes k an unusable channel rather
    # than a wrong one: the model has only ever seen this slot at exactly 0, and
    # feeding a k it never saw would divide the difference by 1e-6 and scale it
    # by a million. There is no value that means "k=4" to this model, so pass
    # the one value it has semantics for and say so.
    _names = list(blob['cond_names'])
    _ki = _names.index('k')
    k_std = float(blob['cond_std'][_ki])
    k_train, k_rel = int(blob['k']), int(release['k'])
    k_feat = float(k_rel)
    if k_rel != k_train:
        if k_std < 1e-3:
            k_feat = float(k_train)
            print(f"note: release k={k_rel}, checkpoint trained at k={k_train} "
                  f"with std={k_std:.3g} -- the k column never varied in "
                  f"training and carries no signal. Feeding k={k_train} so it "
                  f"standardises to 0, the only value this model has seen. "
                  f"D_inter is therefore blind to the coarsening scale; the "
                  f"pair's own n_i, n_j, w_ij still describe it.")
        else:
            z = (k_rel - float(blob['cond_mean'][_ki])) / k_std
            print(f"note: release k={k_rel} against training k={k_train}; "
                  f"the k feature arrives at {z:+.2f} sigma.")

    device = 'cuda' if (cfg.general.gpus > 0 and torch.cuda.is_available()) else 'cpu'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, map_location=torch.device(device),
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=None, visualization_tools=None,
        extra_features=extra_features, domain_features=domain_features)
    model.eval()
    model = model.to(device)

    rho_of = {int(s['sn']): float(s['rho_in']) for s in release['supernodes']}
    n_of = {int(s['sn']): int(s['n']) for s in release['supernodes']}

    # n_i / n_j must describe the blocks actually being placed. With generated
    # context those are the sizes stage 1 produced, which equal the released n by
    # construction -- assert rather than assume, since a mismatch would misalign
    # every index below.
    # r_{i->j} is a statistic OF THE BLOCK BEING GENERATED, so unlike n/rho/w it
    # cannot be recovered from the supernode-level release -- stage 0 has to
    # publish it per connected pair. If it is missing the cond_names assert above
    # has already fired; this second check only catches a release that renamed
    # the columns without filling them.
    n_r = len(blob['cond_names']) - 6
    pairs = []
    for p in release['pairs']:
        si, sj = int(p['sn_i']), int(p['sn_j'])
        assert n_of[si] == int(p['n_i']) and n_of[sj] == int(p['n_j'])
        r = torch.as_tensor(p['r'], dtype=torch.float).view(-1) if n_r else torch.zeros(0)
        assert r.numel() == n_r, (
            f"pair ({si},{sj}) carries r of width {r.numel()}, model wants {n_r}")
        pairs.append({'sn_i': si, 'sn_j': sj, 'n_i': n_of[si], 'n_j': n_of[sj],
                      'w': int(p['w']), 'r': r})

    # Validation-only: score fidelity on a random subsample instead of every pair.
    # Luxembourg has 10902 connected pairs against Oldenburg's 552, which at ~3 min
    # a batch is far past any sensible wall clock, and a few hundred pairs already
    # pins the fidelity rate to well under a percentage point. Never use this for a
    # run you intend to assemble: the omitted pairs become empty blocks in L^syn.
    limit = cfg.get('limit')
    if limit and int(limit) < len(release['pairs']):
        idx = np.random.default_rng(int(cfg.get('limit_seed', 0))).choice(
            len(release['pairs']), int(limit), replace=False)
        release = dict(release, pairs=[release['pairs'][int(i)] for i in idx])
        print(f"LIMIT: scoring a random {limit} of the connected pairs. "
              f"This output is for validation only -- do not assemble it.")

    max_combined = cfg.get('max_combined')
    if max_combined:
        keep = [p for p in pairs if p['n_i'] + p['n_j'] <= int(max_combined)]
        if len(keep) < len(pairs):
            print(f"skipping {len(pairs) - len(keep)} pairs above "
                  f"--max-combined={max_combined}; their blocks will be left empty")
        pairs = keep

    trained_cap = int(release['max_combined_trained'])
    over = sum(1 for p in pairs if p['n_i'] + p['n_j'] > trained_cap)
    if over:
        print(f"NOTE: {over} pairs exceed the combined size {trained_cap} seen in "
              f"training. The model is size-agnostic, so this is a memory question, "
              f"not a correctness one -- use +max_combined=N if it does not fit.")

    print(f"{release['graph']} k={release['k']}: generating {len(pairs)} cross blocks")
    print(f"condition ({len(blob['cond_names'])} dims): {list(blob['cond_names'])}")

    # Sort by combined size so each batch pads close to its own maximum.
    pairs = sorted(pairs, key=lambda p: p['n_i'] + p['n_j'])

    out, exact, done = {}, 0, 0
    with torch.no_grad():
        for start in range(0, len(pairs), BATCH):
            chunk = pairs[start:start + BATCH]

            y_raw = torch.stack([
                torch.cat([torch.tensor([float(p['n_i']), rho_of[p['sn_i']],
                                         float(p['n_j']), rho_of[p['sn_j']],
                                         float(p['w']), k_feat]),
                           p['r']]) for p in chunk])
            cond_y = ((y_raw - blob['cond_mean']) / blob['cond_std']).float().to(device)

            given, n_nodes = build_given(chunk, internal, device)
            samples = model.sample_batch(batch_id=start, batch_size=len(chunk),
                                         keep_chain=0, number_chain_steps=1,
                                         save_final=0, num_nodes=n_nodes,
                                         cond_y=cond_y, given=given)

            for p, (side, edge_types) in zip(chunk, samples):
                vi = torch.nonzero(side == 0, as_tuple=True)[0]
                vj = torch.nonzero(side == 1, as_tuple=True)[0]
                B = (edge_types[vi][:, vj] == 1)
                src, dst = torch.nonzero(B, as_tuple=True)
                out[(p['sn_i'], p['sn_j'])] = {
                    'n_i': p['n_i'], 'n_j': p['n_j'], 'w_requested': p['w'],
                    'edge_index': torch.stack([src, dst]).long(),   # local, i-side x j-side
                }
                exact += int(int(B.sum()) == p['w'])
            done += len(chunk)
            print(f"  {done}/{len(pairs)}", flush=True)

    result = {
        'graph': release['graph'], 'k': release['k'],
        'checkpoint': str(cfg.general.test_only),
        'real_context': real_context,
        'blocks': out,
    }
    torch.save(result, out_path)

    gen_w = sum(v['edge_index'].shape[1] for v in out.values())
    req_w = sum(p['w'] for p in pairs)
    print(f"\n  condition fidelity: {exact}/{len(pairs)} blocks have |B_ij| exactly "
          f"w_ij ({100.0 * exact / max(len(pairs), 1):.1f}%)")
    print(f"  cross edges: generated {gen_w}, requested {req_w}")
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
