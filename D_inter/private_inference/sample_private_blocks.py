"""STAGE 2 of private inference: generate the cross block for every connected pair.

Runs in the `digress` env, on a GPU. Reads the stage-0 release and the stage-1
generated internal blocks, and produces one B_ij per connected supernode pair:

    python3 sample_private_blocks.py \
        dataset=pair_B_lux2olde_k7_s80_deg +experiment=pair_lux \
        general.test_only="'/abs/path/checkpoints/pair_lux_s80_deg/last.ckpt'" \
        +release=/abs/path/release_oldenburg_k7.pt \
        +clusters=/abs/path/clusters_oldenburg_k7.pt \
        +out=/abs/path/blocks_oldenburg_k7.pt

Note the LITERAL single quotes inside the double quotes on test_only -- Hydra's
override grammar cannot parse the second '=' in "epoch=179.ckpt".

To validate this stage independently of D_intra, pass the true internal blocks
that stage 0 writes alongside the release:

    +clusters=/abs/path/clusters_oldenburg_k7_true.pt

With real context the run measures D_inter alone, so a bad result points here
rather than at stage 1's checkpoint. The file is read like any other clusters file; nothing from
the extraction stack (h5py, torch_sparse) is imported, since the GPU environment
does not have it.

REPEATS. +clusters_tag=ICFG +tag=NAME +repeats=N runs seeds +seed .. +seed+N-1
(default 0..N-1) from one loaded checkpoint. Seed s reads stage 1's
clusters_<graph>_k<k>_<ICFG>_s<s>.pt from beside the release and writes
blocks_<graph>_k<k>_<NAME>_s<s>.pt there, so each repeat is a whole synthetic
graph, internal and cross blocks sampled together.
run_private_blocks_repeats.sh sets these.

WHAT IS OBSERVED AND WHAT IS GENERATED. The model is given the sides and the node
features X^_i, X^_j -- each node's normalised internal degree d_in(v) / (n_i - 1)
in the GENERATED L^_i, L^_j of the clusters file -- and produces only the cross
block. Training used the true X_i, X_j; this train--inference shift is the one the
paper evaluates. The context is built by datasets/pair_dataset.py:build_given, the
same function training's sample_conditioning uses, so the encodings cannot drift.

THE CONDITION c_ij = [n_i, n_j, w_ij, r_{i->j}, r_{j->i}] comes from the release
(make_private_release.py publishes both r profiles per connected pair), and is
standardised with the pairs file's cond_mean / cond_std, as in training.
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
import random
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
import costlog                                                     # noqa: E402

warnings.filterwarnings("ignore", category=PossibleUserWarning)

BATCH = 32


def _load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # +sample_batch=N overrides the batch (default 32, which the Oldenburg runs used).
    global BATCH
    BATCH = int(cfg.get('sample_batch', BATCH))
    assert cfg.general.test_only, 'pass general.test_only="\'/abs/path/to/ckpt\'"'
    assert cfg.get('release'), "pass +release=/abs/path/to/release_<graph>_k<k>.pt"

    from datasets.pair_dataset import PairGraphDataModule, PairDatasetInfos

    release = _load(cfg.release)

    # One (seed, clusters file, output file) per run. With +clusters_tag the
    # context is stage 1's repeated output, seed for seed: run s places its
    # cross blocks between the internal blocks of clusters_..._<TAG>_s<s>.pt.
    repeats = int(cfg.get('repeats', 1))
    seed0 = int(cfg.get('seed', 0))
    rdir = os.path.dirname(os.path.abspath(cfg.release))
    gk = f"{release['graph']}_k{release['k']}"
    if cfg.get('clusters_tag'):
        assert cfg.get('tag'), "+clusters_tag needs +tag, which names the output files"
        jobs = [(seed0 + r,
                 os.path.join(rdir, f"clusters_{gk}_{cfg.clusters_tag}_s{seed0 + r}.pt"),
                 os.path.join(rdir, f"blocks_{gk}_{cfg.tag}_s{seed0 + r}.pt"))
                for r in range(repeats)]
    else:
        assert cfg.get('clusters'), ("pass +clusters=/abs/path/to/clusters_<graph>_k<k>.pt, "
                                     "or +clusters_tag=CFG +tag=NAME for repeats")
        assert repeats == 1, "+repeats needs +clusters_tag"
        jobs = [(seed0, cfg.clusters,
                 cfg.get('out') or os.path.join(rdir, f"blocks_{gk}.pt"))]
    for _, cpath, _ in jobs:
        if not os.path.isfile(cpath):
            raise SystemExit(f"clusters file not found: {cpath}")

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
    device = 'cuda' if (cfg.general.gpus > 0 and torch.cuda.is_available()) else 'cpu'
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(
        cfg.general.test_only, map_location=torch.device(device),
        dataset_infos=dataset_infos,
        train_metrics=TrainAbstractMetricsDiscrete(),
        sampling_metrics=None, visualization_tools=None,
        extra_features=extra_features, domain_features=domain_features)
    model.eval()
    model = model.to(device)

    n_of = {int(s['sn']): int(s['n']) for s in release['supernodes']}

    # n_i / n_j must describe the blocks actually being placed. With generated
    # context those are the sizes stage 1 produced, which equal the released n by
    # construction -- assert rather than assume, since a mismatch would misalign
    # every index below.
    # r_{i->j}, r_{j->i} are statistics OF THE BLOCK BEING GENERATED, so unlike n
    # and w they cannot be recovered from the supernode-level release -- stage 0
    # publishes them per connected pair. The cond_names assert above catches a
    # release without them; this one a release that named them but left them short.
    n_r = (len(blob['cond_names']) - 3) // 2
    pairs = []
    for p in release['pairs']:
        si, sj = int(p['sn_i']), int(p['sn_j'])
        assert n_of[si] == int(p['n_i']) and n_of[sj] == int(p['n_j'])
        r_ij = torch.as_tensor(p['r_ij'], dtype=torch.float).view(-1)
        r_ji = torch.as_tensor(p['r_ji'], dtype=torch.float).view(-1)
        assert r_ij.numel() == r_ji.numel() == n_r, (
            f"pair ({si},{sj}) carries r of width {r_ij.numel()}/{r_ji.numel()}, model wants {n_r}")
        pairs.append({'sn_i': si, 'sn_j': sj, 'n_i': n_of[si], 'n_j': n_of[sj],
                      'w': int(p['w']), 'r_ij': r_ij, 'r_ji': r_ji})

    # Validation-only: score fidelity on a random subsample instead of every pair.
    # Luxembourg has 10902 connected pairs against Oldenburg's 552, which at ~3 min
    # a batch is far past any sensible wall clock, and a few hundred pairs already
    # pins the fidelity rate to well under a percentage point. Never use this for a
    # run you intend to assemble: the omitted pairs become empty blocks in L^syn.
    # (Until 2026-10-12 this subsampled release['pairs'] AFTER `pairs` had been
    # built from it, so it printed the warning and then generated every pair.)
    limit = int(cfg.get('limit') or 0)
    if limit and limit < len(pairs):
        idx = np.random.default_rng(int(cfg.get('limit_seed', 0))).choice(
            len(pairs), limit, replace=False)
        pairs = [pairs[int(i)] for i in idx]
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

    runs = []
    for seed, cpath, out_path in jobs:
        blob_c = _load(cpath)
        assert blob_c['graph'] == release['graph'] and blob_c['k'] == release['k'], (
            f"clusters are for {blob_c['graph']} k={blob_c['k']} but the release is "
            f"{release['graph']} k={release['k']}")
        internal = {int(sn): v['edge_index'] for sn, v in blob_c['clusters'].items()}
        # The validation run passes clusters_<graph>_k<k>_true.pt, the real internal
        # blocks written by stage 0. Reading them from a file rather than rebuilding
        # them here keeps the extraction stack -- h5py, torch_sparse -- out of the
        # GPU environment, which does not have it.
        real_context = blob_c.get('checkpoint') is None
        print(f"\nseed {seed}: context {cpath}")
        if real_context:
            print("USING THE PRIVATE GRAPH'S REAL INTERNAL BLOCKS (validation mode)")

        seed_all(seed)
        with costlog.Measure() as m:
            out, exact = sample_pairs(model, pairs, internal, blob, device)
        runs.append({'seed': seed, **m.record})

        result = {
            'graph': release['graph'], 'k': release['k'],
            'checkpoint': str(cfg.general.test_only),
            'dataset': str(cfg.dataset.pairs_file),
            'clusters': str(cpath), 'tag': cfg.get('tag'), 'seed': seed,
            'real_context': real_context,
            'limit': limit or None,
            'blocks': out,
        }
        torch.save(result, out_path)

        gen_w = sum(v['edge_index'].shape[1] for v in out.values())
        req_w = sum(p['w'] for p in pairs)
        print(f"  condition fidelity: {exact}/{len(pairs)} blocks have |B_ij| exactly "
              f"w_ij ({100.0 * exact / max(len(pairs), 1):.1f}%)")
        print(f"  cross edges: generated {gen_w}, requested {req_w}")
        print(f"wrote {out_path}")
    # Whole-graph inference, D_inter's part: every connected pair, per seed
    # -> blocks_<graph>_k<k>_<tag>_cost.json beside the outputs (costlog.py)
    costlog.write(os.path.join(rdir, f"blocks_{gk}_{cfg.get('tag') or 'single'}_cost.json"), {
        **costlog.safe_context(cfg, phase='full_graph_inter', what='D_inter, every connected pair',
                          n_items=len(pairs), sample_batch=BATCH,
                          limit=limit or None), 'runs': runs})


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_pairs(model, pairs, internal, blob, device):
    """One cross block per pair, around the node features of the given internal blocks."""
    from datasets.pair_dataset import build_given, internal_degree
    feat = {}                                   # supernode -> x^, computed once
    for p in pairs:
        for sn, n in ((p['sn_i'], p['n_i']), (p['sn_j'], p['n_j'])):
            if sn not in feat:
                feat[sn] = internal_degree(internal[sn], n)
    out, exact, done = {}, 0, 0
    with torch.no_grad():
        for start in range(0, len(pairs), BATCH):
            chunk = pairs[start:start + BATCH]
            y_raw = torch.stack([
                torch.cat([torch.tensor([float(p['n_i']), float(p['n_j']), float(p['w'])]),
                           p['r_ij'], p['r_ji']]) for p in chunk])
            cond_y = ((y_raw - blob['cond_mean']) / blob['cond_std']).float().to(device)
            given, n_nodes = build_given([(feat[p['sn_i']], feat[p['sn_j']]) for p in chunk],
                                         device)
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
    return out, exact


if __name__ == '__main__':
    main()
