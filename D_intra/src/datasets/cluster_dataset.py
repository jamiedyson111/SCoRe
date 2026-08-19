"""Intra-cluster (supernode) subgraphs as a CONDITIONAL DiGress dataset.

Reads the file produced by k-mis-pool/make_pairs.py. Imports no K-MIS code.

Each example is one supernode:
    X          (n, 1)   one dummy node class -- the fine graph has no node types
    edge_attr  (E, 2)   class 0 = "no edge" (reserved), class 1 = edge
    y          (1, d)   THE CONDITION -- this is what makes it conditional

y is the only difference from the unconditional setup. DiGress never noises y
(see apply_noise) and never predicts it (lambda_train[1] = 0), so it acts purely
as a clean instruction read by the model at every denoising step.
"""
import os
import pathlib

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


def _load_blob(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:                       # torch < 2.0 has no weights_only kwarg
        return torch.load(path)


class ClusterGraphDataset(InMemoryDataset):
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
            f"Produce it with k-mis-pool/make_pairs.py and upload it there.")

    def process(self):
        blob = _load_blob(os.path.join(self.raw_dir, self.pairs_file))
        records = blob['splits'][{'train': 0, 'val': 1, 'test': 2}[self.split]]

        data_list = []
        for rec in records:
            n = int(rec['n'])
            edge_index = rec['edge_index'].long()

            X = torch.ones(n, 1, dtype=torch.float)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1

            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr,
                y=rec['y'].view(1, -1).float(),          # <-- the condition
                n_nodes=n * torch.ones(1, dtype=torch.long))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)          # exactly ONCE

        torch.save(self.collate(data_list), self.processed_paths[0])


class ClusterGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        self.root_path = os.path.join(base_path, self.datadir)
        self.pairs_file = cfg.dataset.pairs_file

        datasets = {s: ClusterGraphDataset(split=s, root=self.root_path,
                                           pairs_file=self.pairs_file)
                    for s in ('train', 'val', 'test')}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

    def raw_blob(self):
        """The source file, for the conditional sampler (conditions + normalisation)."""
        return _load_blob(os.path.join(self.root_path, 'raw', self.pairs_file))


class ClusterDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'cluster_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # one dummy class
        self.edge_types = self.datamodule.edge_counts()

        # Bank of real (condition, size) pairs from the TRAIN split. sample_batch
        # draws from this whenever it is called without an explicit condition --
        # which is what DiGress's periodic validation/test sampling does. Without
        # it, a conditionally-trained model crashes there on a y-width mismatch.
        train = datamodule.train_dataset
        self.cond_bank = {
            'y': torch.cat([train[i].y for i in range(len(train))]).float(),
            'n': torch.tensor([int(train[i].x.size(0)) for i in range(len(train))]),
        }

        super().complete_infos(self.n_nodes, self.node_types)


def ClusterSamplingMetrics(datamodule):
    """Structural + spectral comparison against held-out real cluster subgraphs.

    A factory rather than a class so importing this module does not pull in
    spectre_utils -> graph_tool, which fails if torch is imported first.
    """
    try:
        from analysis.spectre_utils import SpectreSamplingMetrics
    except ImportError:
        from src.analysis.spectre_utils import SpectreSamplingMetrics

    class _ClusterSamplingMetrics(SpectreSamplingMetrics):
        def __init__(self, dm):
            super().__init__(datamodule=dm, compute_emd=False,
                             metrics_list=['degree', 'clustering', 'orbit', 'spectre'])

    return _ClusterSamplingMetrics(datamodule)
