"""Supernode PAIRS as a partially-observed conditional DiGress dataset (D_inter).

Reads the file produced by D_inter/dataprep/make_pair_blocks.py.

Each example is one pair of supernodes, stored as the single combined graph

        H_ij = [ L_i    B_ij  ]
               [ B_ij^T L_j   ]

    X          (n, 2)   THE SIDE.  [1,0] = node is in V_i, [0,1] = node is in V_j
    edge_attr  (E, 2)   class 0 = "no edge", class 1 = edge
    y          (1, 11)  THE CONDITION c_ij = [n_i, rho_i, n_j, rho_j, w_ij, k,
                        r0, r1, r2, r3, r4+]  -- the V2 condition. r_{i->j} is
                        the out-degree profile of the block: r_d = fraction of
                        V_i nodes with exactly d edges into V_j. Fixed width 5
                        whatever n_i and n_j are. Built entirely in
                        dataprep/make_pair_blocks.py; nothing here depends on
                        the width, so a re-binned r needs no change below.

Only B_ij is generated. L_i and L_j are observed context: they are never noised,
never predicted, and contribute to no loss.

WHY THE SIDE LIVES IN X. The model needs a mask saying "this entry of E is a
cross entry". Deriving it from node ORDER would work here (V_i is numbered
first) but is an invisible invariant that breaks silently if anything ever
permutes nodes. Putting the side in X makes the mask a function of the data:
an entry is cross exactly when its two endpoints have different classes. It is
also real signal -- the model cannot sensibly predict a cross block without
knowing which side each node is on.

The price is that X must not be diffused, or the noised X_t would scramble the
sides and the mask with them. The model holds X fixed whenever
`dataset_infos.pair_mode` is set, which is what makes the mask exact at every
training step and at every step of the reverse process.
"""
import os
import pathlib

import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def _load_blob(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:                       # torch < 2.0 has no weights_only kwarg
        return torch.load(path)


def _out_degrees(edge_index, n_i, n_j):
    """Out-degrees d_{i->j}(u) = sum_v B_ij(u,v), for every u in V_i.

    Nodes of V_i are local ids [0, n_i), nodes of V_j are [n_i, n_i + n_j).
    Returns a length-n_i array, including the zeros -- a node with no cross edge
    is part of the distribution, not absent from it.
    """
    src, dst = edge_index[0], edge_index[1]
    cross = (src < n_i) & (dst >= n_i)
    return np.bincount(src[cross].numpy(), minlength=n_i)[:n_i]


class PairGraphDataset(InMemoryDataset):
    def __init__(self, split, root, pairs_file, transform=None, pre_transform=None,
                 pre_filter=None):
        self.split = split
        self.pairs_file = pairs_file
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.pairs_file]

    @property
    def processed_file_names(self):
        return [f"{os.path.splitext(self.pairs_file)[0]}_{self.split}.pt"]

    def download(self):
        raise FileNotFoundError(
            f"{os.path.join(self.raw_dir, self.pairs_file)} not found. "
            f"Produce it with D_inter/dataprep/make_pair_blocks.py.")

    def process(self):
        blob = _load_blob(os.path.join(self.raw_dir, self.pairs_file))
        records = blob['splits'][{'train': 0, 'val': 1, 'test': 2}[self.split]]

        data_list = []
        for rec in records:
            n_i, n_j = int(rec['n_i']), int(rec['n_j'])
            n = n_i + n_j
            edge_index = rec['edge_index'].long()

            X = torch.zeros(n, 2, dtype=torch.float)
            X[:n_i, 0] = 1.                     # side i
            X[n_i:, 1] = 1.                     # side j

            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1

            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr,
                y=rec['y'].view(1, -1).float(),               # <-- the condition
                n_i=torch.tensor([n_i], dtype=torch.long),
                w=torch.tensor([int(rec['w'])], dtype=torch.long),
                n_nodes=n * torch.ones(1, dtype=torch.long))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])


class PairGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root_path = os.path.join(base_path, self.datadir)
        self.pairs_file = cfg.dataset.pairs_file

        datasets = {s: PairGraphDataset(split=s, root=self.root_path,
                                        pairs_file=self.pairs_file)
                    for s in ('train', 'val', 'test')}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

    def raw_blob(self):
        return _load_blob(os.path.join(self.root_path, 'raw', self.pairs_file))

    def edge_counts(self):
        """Edge-class marginal over the CROSS BLOCK ONLY.

        This is what seeds the marginal transition matrix, and therefore the
        limit distribution the reverse process starts from at t = T. The
        whole-graph marginal would be wrong: the internal blocks are an order of
        magnitude denser than B_ij, and they are never noised, so they have no
        business influencing the noise model. Getting this wrong is quiet --
        training looks fine and only sampling degrades.
        """
        d = torch.zeros(2, dtype=torch.float)
        for data in self.train_dataloader():
            _, counts = torch.unique(data.batch, return_counts=True)
            n_i = data.n_i.view(-1).to(counts.device)
            n_j = counts - n_i
            # E is symmetric and DiGress counts ordered pairs, so both triangles.
            slots = 2.0 * float((n_i * n_j).sum())
            edges = 2.0 * float(data.w.sum())
            d[0] += slots - edges
            d[1] += edges
        assert d[0] > 0 and d[1] > 0, "degenerate cross-block marginal"
        return d / d.sum()


class PairDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'pair_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()      # side i / side j split
        self.edge_types = self.datamodule.edge_counts()     # CROSS BLOCK ONLY

        # Read by the model. Turns on: X held fixed, noise confined to the cross
        # block, and both the training CE and the validation NLL restricted to it.
        self.pair_mode = True

        blob = datamodule.raw_blob()
        self._val_records = blob['splits'][1]
        self._test_records = blob['splits'][2]

        super().complete_infos(self.n_nodes, self.node_types)

    def sample_conditioning(self, batch_size, device, test=False):
        """Real (condition, sizes, observed context) drawn from val/test.

        Generation here is partially observed: L_i, L_j and the sides are given
        and only B_ij is produced, so a batch cannot be conditioned on y alone.
        Everything pair-specific stays here rather than in the shared model --
        the model only asks for this hook and re-imposes whatever it returns.
        """
        recs = self._test_records if test else self._val_records
        idx = torch.randint(0, len(recs), (batch_size,))
        chosen = [recs[int(i)] for i in idx]

        n_nodes = torch.tensor([int(r['n_i']) + int(r['n_j']) for r in chosen],
                               device=device)
        n_max = int(n_nodes.max())

        X = torch.zeros(batch_size, n_max, 2, device=device)
        E = torch.zeros(batch_size, n_max, n_max, 2, device=device)
        for b, r in enumerate(chosen):
            n_i, n_j = int(r['n_i']), int(r['n_j'])
            n = n_i + n_j
            X[b, :n_i, 0] = 1.
            X[b, n_i:n, 1] = 1.

            # Match utils.to_dense exactly: no-edge is [1,0], the diagonal is
            # [0,0], padding is [0,0]. Only the non-cross entries are ever read,
            # but a mismatch here would be a silent train/sample skew.
            E[b, :n, :n, 0] = 1.
            ei = r['edge_index']
            side = torch.zeros(n, dtype=torch.bool)
            side[n_i:] = True
            internal = side[ei[0]] == side[ei[1]]
            src, dst = ei[0][internal], ei[1][internal]
            E[b, src, dst, 0] = 0.
            E[b, src, dst, 1] = 1.
            E[b, torch.arange(n), torch.arange(n), :] = 0.

        y = torch.stack([r['y'] for r in chosen]).float().to(device)
        return {'y': y, 'num_nodes': n_nodes, 'given': {'X': X, 'E': E}}


def PairSamplingMetrics(datamodule):
    """W_1 between generated and real out-degree distributions.

    d_{i->j}(u) = sum_{v in V_j} B_ij(u,v), pooled over every node of every V_i
    in the batch, compared against the same pooled quantity on the real held-out
    pairs. Lower is better.

    A factory for symmetry with the D_intra metrics, and so that importing this
    module never pulls in spectre_utils -> graph_tool.
    """
    from scipy.stats import wasserstein_distance

    class _PairSamplingMetrics:
        def __init__(self, dm):
            blob = dm.raw_blob()
            self.real = {}
            for split, name in ((1, 'val'), (2, 'test')):
                recs = blob['splits'][split]
                pooled = np.concatenate([
                    _out_degrees(r['edge_index'], int(r['n_i']), int(r['n_j']))
                    for r in recs])
                self.real[name] = pooled

            # Reference point. W_1 for a model that places the RIGHT number of
            # cross edges but in uniformly random positions -- it gets w_ij from
            # the condition for free, so anything above this line has learned
            # nothing about where the edges go. Without it the raw W_1 has no scale.
            rng = np.random.default_rng(0)
            recs = blob['splits'][1]
            baseline = []
            for r in recs:
                n_i, n_j, w = int(r['n_i']), int(r['n_j']), int(r['w'])
                slots = rng.integers(0, n_i * n_j, size=w)
                baseline.append(np.bincount(slots // n_j, minlength=n_i)[:n_i])
            self.uniform_baseline = wasserstein_distance(
                np.concatenate(baseline), self.real['val'])

        def reset(self):
            pass

        def forward(self, generated_graphs, name, current_epoch, val_counter,
                    test=False, local_rank=0):
            pooled, n_edges, n_pairs = [], 0, 0
            for atom_types, edge_types in generated_graphs:
                side = atom_types.numpy()
                E = edge_types.numpy()
                vi = np.flatnonzero(side == 0)
                vj = np.flatnonzero(side == 1)
                if len(vi) == 0 or len(vj) == 0:
                    continue
                B = (E[np.ix_(vi, vj)] == 1)
                pooled.append(B.sum(axis=1))
                n_edges += int(B.sum())
                n_pairs += 1

            split = 'test' if test else 'val'
            if not pooled:
                print(f"[{split}] no usable generated pairs -- skipping W_out-degree")
                return

            gen = np.concatenate(pooled)
            w1 = wasserstein_distance(gen, self.real[split])

            print(f"[{split}] W_out-degree {w1:.5f}   "
                  f"(uniform-placement baseline {self.uniform_baseline:.5f})")
            print(f"[{split}]   generated {n_edges / max(n_pairs, 1):.2f} cross edges/pair, "
                  f"mean out-degree {gen.mean():.4f} vs real {self.real[split].mean():.4f}")

            try:
                import wandb
                if wandb.run:
                    wandb.log({f"{split}/W_out_degree": w1,
                               f"{split}/gen_edges_per_pair": n_edges / max(n_pairs, 1),
                               f"{split}/gen_mean_out_degree": float(gen.mean())},
                              commit=False)
            except ImportError:
                pass

        __call__ = forward

    return _PairSamplingMetrics(datamodule)
