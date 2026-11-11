"""Record-aligned sampling of D_inter's test split: one B_ij per test pair.

Runs in the `digress` env, on a GPU. For every pair in the dataset's TEST split,
the node features X_i, X_j (normalised internal degree) are given as context, the
pair's own standardised condition y as c_ij, and the model generates the cross
block:

    python3 sample_pairs_conditional.py \
        dataset=pair_B_lux2olde_k7_s80_deg +experiment=pair_lux \
        general.test_only="'/abs/path/checkpoints/pair_lux_s80_deg/last.ckpt'" \
        +repeats=5 \
        +clusters_dir=/abs/D_intra/private_inference/out +clusters_tag=B_v3_s80

WHERE X_i, X_j COME FROM (the paper: true at training, generated at inference):
  generated  +clusters_dir and +clusters_tag given. Seed s computes X^_i, X^_j from
             the generated internal blocks L^_i, L^_j of D_intra's whole-graph run
             clusters_<graph>_k<k>_<clusters_tag>_s<s>.pt -- the same files the
             full-graph inference uses, so supernode ids are the release's, which
             are the pairs file's (checked: every size must match). Configs B, C.
  real       neither given: X_i, X_j from the record's own real L_i, L_j. Config A
             (D_intra has no generated L^ for Luxembourg's training supernodes,
             which half of every A test pair is), and the B / C ablation that
             measures the train--inference shift.

main.py's test mode is not used for this because it draws conditions at random,
with replacement, and does not record which pair a sample was for -- fine for a
distributional check, useless for scoring each test pair against its own real
block, which is what the A/B/C comparison needs.

+repeats=N samples the split N times with seeds +seed .. +seed+N-1 (default
0..N-1) from the one loaded checkpoint and writes pair_samples_s<seed>.pt into the
hydra run directory:

    {pairs_file, checkpoint, seed, context, clusters,
     blocks: [{idx, sn_i, sn_j, n_i, n_j, w, edge_index}]}

idx is the pair's position in the test split; edge_index is the generated cross
block in local ids (row: i side 0..n_i-1, col: j side 0..n_j-1).
evaluation/transfer_summary.py scores the files.

The observed context is built by datasets/pair_dataset.py:build_given, the one
function training's sample_conditioning and the full-graph sampler also use.
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
import costlog

warnings.filterwarnings("ignore", category=PossibleUserWarning)

BATCH = 32


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def generated_features(test, path):
    """[(x^_i, x^_j)] per test record, from the generated internal blocks in `path`."""
    from datasets.pair_dataset import internal_degree
    clusters = _load(path)['clusters']
    feats = []
    for r in test:
        pair = []
        for sn, n in ((int(r['sn_i']), int(r['n_i'])), (int(r['sn_j']), int(r['n_j']))):
            c = clusters[sn]
            if int(c['n']) != n:
                raise SystemExit(f"{path}: supernode {sn} has {c['n']} nodes, the pairs "
                                 f"file says {n} -- not the same coarsening")
            pair.append(internal_degree(c['edge_index'], n))
        feats.append(tuple(pair))
    return feats


# +pair_budget=N: batches packed while batch x (n_i + n_j)^2 <= N, up to BATCH pairs
# (Collab benchmark, A100 80 GB: 64 x 300^2 keeps every size range fast and in memory).
PAIR_BUDGET = None
POS = None          # each record's position in the full test split (+subset_file)
# +partial_file=/abs/path.pt: the blocks done so far are saved there every SAVE_EVERY
# seconds, and a rerun with the same file skips them, so a failed or timed-out job
# loses at most SAVE_EVERY of work instead of everything.
PARTIAL = None
SAVE_EVERY = 600


def batches(test, order):
    """(first position, record indices) per batch, in size order."""
    if not PAIR_BUDGET:
        return [(s, order[s:s + BATCH]) for s in range(0, len(order), BATCH)]
    out, s = [], 0
    while s < len(order):
        e = s + 1
        while (e < len(order) and e - s < BATCH and
               (e - s + 1) * (test[order[e]]['n_i'] + test[order[e]]['n_j']) ** 2 <= PAIR_BUDGET):
            e += 1
        out.append((s, order[s:e]))
        s = e
    return out


def hybrid_features(test, samples_path):
    """(x_i, x_j) per record: GENERATED internal degrees for the supernodes D_intra
    generated (its conditional_samples file -- the test supernodes), REAL ones for every
    other supernode (training supernodes keep their original structure)."""
    from datasets.pair_dataset import internal_degree, record_features
    gen, lines, i = {}, open(samples_path).read().splitlines(), 0
    while i < len(lines):
        if not lines[i].startswith('N='):
            i += 1
            continue
        n = int(lines[i][2:])
        sn = int(lines[i + 1].split()[0].split('=')[1])
        while not lines[i].startswith('E'):
            i += 1
        A = torch.tensor([[int(v) for v in lines[i + 1 + r].split()] for r in range(n)]) == 1
        gen[sn] = internal_degree(torch.nonzero(A | A.T).T.contiguous(), n)
        i += 1 + n
    feats, used = [], 0
    for r in test:
        real = record_features(r)
        pair = []
        for side, sn, n in ((0, int(r['sn_i']), int(r['n_i'])), (1, int(r['sn_j']), int(r['n_j']))):
            if sn in gen:
                if len(gen[sn]) != n:
                    raise SystemExit(f"{samples_path}: supernode {sn} has {len(gen[sn])} nodes, pair says {n}")
                pair.append(gen[sn])
                used += 1
            else:
                pair.append(real[side])
        feats.append(tuple(pair))
    print(f"  context: {len(gen)} generated supernodes; {used} of {2 * len(test)} pair sides generated, "
          f"the rest real")
    return feats


def sample_split(model, test, feats, device, seed, meta, out_path):
    """One cross block per test record, written to out_path. feats: (x_i, x_j) each."""
    from datasets.pair_dataset import build_given
    import time
    seed_all(seed)
    # Sorted by combined size so each batch pads close to its own maximum.
    order = sorted(range(len(test)), key=lambda i: test[i]['n_i'] + test[i]['n_j'])
    blocks, exact = [None] * len(test), 0
    if PARTIAL and os.path.exists(PARTIAL):
        prev = torch.load(PARTIAL, map_location='cpu', weights_only=False)
        if prev.get('pairs_file') != meta.get('pairs_file') or prev.get('checkpoint') != meta.get('checkpoint'):
            raise SystemExit(f"{PARTIAL} was written for another dataset or checkpoint")
        at = {p: i for i, p in enumerate(POS)}
        for b in prev['blocks']:
            if b is not None and b['idx'] in at:
                blocks[at[b['idx']]] = b
                exact += int(b['edge_index'].shape[1] == b['w'])
        order = [i for i in order if blocks[i] is None]
        print(f"  resuming from {PARTIAL}: {len(test) - len(order)} blocks already done", flush=True)
    last_save = time.time()
    with torch.no_grad():
        for start, idx in batches(test, order):
            chunk = [test[i] for i in idx]
            cond_y = torch.stack([r['y'] for r in chunk]).float().to(device)
            given, n_nodes = build_given([feats[i] for i in idx], device)
            samples = model.sample_batch(batch_id=start, batch_size=len(chunk),
                                         keep_chain=0, number_chain_steps=1,
                                         save_final=0, num_nodes=n_nodes,
                                         cond_y=cond_y, given=given)
            for i, r, (side, edge_types) in zip(idx, chunk, samples):
                vi = torch.nonzero(side == 0, as_tuple=True)[0]
                vj = torch.nonzero(side == 1, as_tuple=True)[0]
                B = (edge_types[vi][:, vj] == 1)
                s, d = torch.nonzero(B, as_tuple=True)
                blocks[i] = {'idx': POS[i], 'sn_i': int(r['sn_i']), 'sn_j': int(r['sn_j']),
                             'n_i': int(r['n_i']), 'n_j': int(r['n_j']), 'w': int(r['w']),
                             'edge_index': torch.stack([s, d]).long().cpu()}
                exact += int(int(B.sum()) == int(r['w']))
            print(f"  {len(test) - len(order) + start + len(idx)}/{len(test)}", flush=True)
            if PARTIAL and time.time() - last_save > SAVE_EVERY:
                torch.save(dict(meta, seed=seed, blocks=blocks), PARTIAL + '.tmp')
                os.replace(PARTIAL + '.tmp', PARTIAL)
                last_save = time.time()
    torch.save(dict(meta, seed=seed, blocks=blocks), out_path)
    if PARTIAL:
        torch.save(dict(meta, seed=seed, blocks=blocks), PARTIAL)
    print(f"  |B_ij| exactly w_ij: {exact}/{len(test)} ({100.0 * exact / len(test):.1f}%)")
    print(f"wrote {out_path}")


@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    # +sample_batch=N overrides the batch (default 32, which the Oldenburg runs used).
    global BATCH, PAIR_BUDGET, POS, PARTIAL
    PARTIAL = cfg.get('partial_file')
    BATCH = int(cfg.get('sample_batch', BATCH))
    PAIR_BUDGET = cfg.get('pair_budget')
    assert cfg.general.test_only, 'pass general.test_only="\'/abs/path/to/ckpt\'"'
    from datasets.pair_dataset import PairGraphDataModule, PairDatasetInfos, record_features

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
    # +subset_file=test_groups_d_inter.json: only the pairs of its "groups" (baselines/
    # make_test_groups.py); each block keeps its idx in the full test split.
    pos = list(range(len(test)))
    if cfg.get('subset_file'):
        import json
        groups = json.load(open(cfg.subset_file))['groups']
        pos = sorted({int(p['idx']) for g in groups for p in g})
        for g in groups:
            for p in g:
                r = test[int(p['idx'])]
                if (int(r['sn_i']), int(r['sn_j'])) != (int(p['sn_i']), int(p['sn_j'])):
                    raise SystemExit(f"{cfg.subset_file}: pair {p} does not match the test split")
        test = [test[i] for i in pos]
        print(f"subset_file: {len(test)} test pairs ({cfg.subset_file})")
    POS = pos
    print(f"{cfg.dataset.pairs_file}: generating {len(test)} cross blocks")
    print(f"condition ({len(blob['cond_names'])} dims): {list(blob['cond_names'])}")

    # Context: generated (B, C) or real (A, and the shift ablation). See the top.
    generated = bool(cfg.get('clusters_tag'))
    assert generated == bool(cfg.get('clusters_dir')), \
        "+clusters_dir and +clusters_tag go together (generated context) or not at all (real)"
    gk = f"{blob.get('private', blob.get('graph'))}_k{blob['k']}"
    print("context: " + (f"GENERATED X^_i, X^_j from {cfg.clusters_dir}/clusters_{gk}_"
                         f"{cfg.clusters_tag}_s<seed>.pt" if generated else
                         "REAL X_i, X_j (from the true L_i, L_j)"))
    real_feats = None if generated else [record_features(r) for r in test]
    # +gen_samples=<D_intra conditional_samples file>: generated X for the supernodes in
    # it, real X for the rest (partial reconstruction: training supernodes unchanged).
    if cfg.get('gen_samples'):
        assert not generated, "+gen_samples replaces +clusters_dir/+clusters_tag"
        real_feats = hybrid_features(test, cfg.gen_samples)
        print(f"context: GENERATED X^ for the supernodes of {cfg.gen_samples}, REAL X otherwise")

    out_dir = HydraConfig.get().runtime.output_dir
    meta = {'pairs_file': str(cfg.dataset.pairs_file),
            'checkpoint': str(cfg.general.test_only),
            'context': ('generated' if generated else
                        'gen_samples (generated test supernodes, real others)' if cfg.get('gen_samples')
                        else 'real'),
            'gen_samples': cfg.get('gen_samples')}
    repeats = int(cfg.get('repeats', 1))
    seed0 = int(cfg.get('seed', 0))
    runs = []
    for rep in range(repeats):
        seed = seed0 + rep
        print(f"\nrepeat {rep + 1}/{repeats}  (seed {seed})")
        feats, cpath = real_feats, None
        if generated:
            cpath = os.path.join(cfg.clusters_dir, f"clusters_{gk}_{cfg.clusters_tag}_s{seed}.pt")
            print(f"  context {cpath}")
            feats = generated_features(test, cpath)
        with costlog.Measure() as m:
            sample_split(model, test, feats, device, seed, dict(meta, clusters=cpath),
                         os.path.join(out_dir, f'pair_samples_s{seed}.pt'))
        runs.append({'seed': seed, **m.record})
    # Runtime and peak GPU memory of each repeat -> cost_sample.json (costlog.py)
    costlog.write(os.path.join(out_dir, 'cost_sample.json'), {
        **costlog.safe_context(cfg, phase='sample',
                               what=f"D_inter, test pairs ({meta['context']} X_i, X_j)",
                               n_items=len(test), sample_batch=BATCH), 'runs': runs})


if __name__ == '__main__':
    main()
