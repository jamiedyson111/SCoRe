"""Generic unweighted, undirected graph dataset for DiGress.

Use this when your graphs have NO node types and NO edge types -- the model just
learns their connectivity. It is the same setup as the SPECTRE and coarse_sbm
datasets, but it reads the most universal possible input so you can plug in any
dataset later without touching the model:

    a single .pt file holding a Python list of dense adjacency matrices
    (each an n_i x n_i torch tensor or numpy array; sizes may differ per graph).

Each graph is encoded exactly as the untyped datasets do:

    X          (n, 1)   one dummy node class -- node diffusion is effectively a no-op
    edge_attr  (E, 2)   class 0 = "no edge" (reserved, never emitted), class 1 = edge

To use it later:
    1. build a list `adjs = [A_0, A_1, ...]` of symmetric 0/1 adjacency matrices,
    2. `torch.save(adjs, 'data/generic/raw/graphs.pt')`,
    3. run with `dataset=generic` (see configs/dataset/generic.yaml).
The train/val/test split is computed deterministically here, so you only supply
the one file.
"""
import os
import pathlib

import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class GenericGraphDataset(InMemoryDataset):
    # fractions of the full set; the remainder after val+test goes to train
    val_frac = 0.1
    test_frac = 0.1
    split_seed = 0

    def __init__(self, split, root, graph_file, transform=None, pre_transform=None,
                 pre_filter=None):
        self.split = split
        self.graph_file = graph_file
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.graph_file]

    @property
    def processed_file_names(self):
        # keyed on the source file so different graph sets cannot collide
        return [f"{os.path.splitext(self.graph_file)[0]}_{self.split}.pt"]

    def download(self):
        raise FileNotFoundError(
            f"{os.path.join(self.raw_dir, self.graph_file)} not found. Save a list of "
            f"adjacency matrices there with torch.save(list_of_adjs, path).")

    def _split_indices(self, n_graphs):
        """Deterministic disjoint train/val/test indices over range(n_graphs)."""
        g = torch.Generator().manual_seed(self.split_seed)
        perm = torch.randperm(n_graphs, generator=g)
        n_val = int(round(n_graphs * self.val_frac))
        n_test = int(round(n_graphs * self.test_frac))
        # keep val/test non-empty on small datasets, but never starve train
        if n_graphs >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        val_idx = perm[:n_val]
        test_idx = perm[n_val:n_val + n_test]
        train_idx = perm[n_val + n_test:]
        return {'train': train_idx, 'val': val_idx, 'test': test_idx}[self.split]

    def process(self):
        path = os.path.join(self.raw_dir, self.graph_file)
        try:
            raw = torch.load(path, weights_only=False)
        except TypeError:                       # torch < 2.0 has no weights_only kwarg
            raw = torch.load(path)

        indices = self._split_indices(len(raw))

        data_list = []
        for i in indices.tolist():
            adj = torch.as_tensor(raw[i], dtype=torch.float)
            # enforce unweighted + undirected + no self-loops, regardless of input
            adj = ((adj + adj.t()) > 0).float()
            adj.fill_diagonal_(0)

            n = adj.shape[-1]
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)   # both directions

            X = torch.ones(n, 1, dtype=torch.float)                      # single dummy class
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1                                          # class 0 stays free

            data = torch_geometric.data.Data(
                x=X, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.zeros([1, 0]).float(),
                n_nodes=n * torch.ones(1, dtype=torch.long))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)          # append exactly ONCE

        torch.save(self.collate(data_list), self.processed_paths[0])


class GenericGraphDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        graph_file = cfg.dataset.graph_file

        datasets = {s: GenericGraphDataset(split=s, root=root_path, graph_file=graph_file)
                    for s in ('train', 'val', 'test')}

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class GenericDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'generic_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # one dummy class
        self.edge_types = self.datamodule.edge_counts()   # [P(no edge), P(edge)]
        super().complete_infos(self.n_nodes, self.node_types)


def GenericSamplingMetrics(datamodule):
    """Structure/spectrum comparison against held-out real graphs.

    Uses only family-agnostic metrics (degree, clustering, orbit, spectre) -- no
    sbm/planar validity checks, which assume a particular graph family. A factory
    rather than a class so importing this module does not pull in
    spectre_utils -> graph_tool (which must be imported before torch).
    """
    try:                                    # main.py puts src/ itself on the path
        from analysis.spectre_utils import SpectreSamplingMetrics
    except ImportError:                     # ...but the package is importable as src.* too
        from src.analysis.spectre_utils import SpectreSamplingMetrics

    class _GenericSamplingMetrics(SpectreSamplingMetrics):
        def __init__(self, dm):
            super().__init__(datamodule=dm, compute_emd=False,
                             metrics_list=['degree', 'clustering', 'orbit', 'spectre'])

    return _GenericSamplingMetrics(datamodule)
