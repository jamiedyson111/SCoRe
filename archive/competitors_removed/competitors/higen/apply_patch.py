"""Two small, idempotent patches to a HiGen checkout (github.com/Karami-m/HiGen_main).

    python apply_patch.py --repo $REPO

1. utils/dataset/dataset_preprocessing.py: a 'LuxRegions2' dataset -- the
   Luxembourg regions from data/LuxRegions2/lux_regions_nx.pkl, coarsened into a
   2-level hierarchy exactly as the repo's own FIRSTMM2 (its largest graphs, up to
   ~5000 nodes, the closest match to Oldenburg): same splice_out_inds.
2. higen_runner.py, test(): HiGen draws each sample's totals (leaf nodes, leaf
   edges) from the training graphs' empirical distribution, so its size cannot
   be chosen. With test.target_num_nodes set, every drawn total is rescaled to
   that many nodes, edges in proportion -- the edges-per-node ratio still comes
   from the TRAINING graph drawn (public Luxembourg), never from Oldenburg. This
   gives HiGen the same information ICLR [2] gets: the node count only.
3. utils/dataset/graph_corsening.py, best_spectral_partition: the released code
   implements only spectral-clustering coarsening (SC0; the Louvain option in
   its configs raises "not implemented"). SC0 tries EVERY cluster count K in
   [0.7 sqrt(n), 1.3 sqrt(n)] -- about 40 values at n = 4000 -- with 1000 k-means
   restarts each on a dense matrix: one to two hours per region, 100-200 h for
   the 120 regions. Here: 5 evenly spaced K over the same range, 10 restarts,
   sparse affinity, fixed seed. Same method, same range, same selection by
   modularity; a coarser search.
"""
import argparse
import os

MARK = "# competitors/higen/apply_patch.py"

BRANCH = f'''    elif graph_type == 'LuxRegions2':   {MARK}
        with open(os.path.join(data_dir, 'LuxRegions2', 'lux_regions_nx.pkl'), 'rb') as f:
            graphs = pickle.load(f)
        kwargs.update({{'splice_out_inds': {{2: [], 3: [1], 4: [0, 2], 5: [0, 2, 3],
                                             6: [0, 2, 3, 4], 7: [0, 2, 3, 4, 5]}}}})
        graphs_dataset = get_hierarchical_graph(
            graphs,
            graph_type=graph_type,
            max_num_level=max_num_level,
            orders=node_orders,
            **kwargs
        )

'''

RESCALE = f'''            _T = self.test_conf.get('target_num_nodes', None)   {MARK}
            if _T:
                stat_HG_gen = [(l_, int(_T), int(round(float(s_) * _T / float(n_))),
                                int(round(float(m_) * _T / float(n_))))
                               for l_, n_, s_, m_ in stat_HG_gen]
'''


SC_OLD = """    for K in range(k_min, k_max + 1):
        # Perform spectral clustering algorithm
        adj_mat = nx.to_numpy_matrix(G)
        sc = SpectralClustering(K, affinity='precomputed', n_init=1000)
"""
SC_NEW = f"""    _grid = sorted(set(np.linspace(k_min, k_max, 5).astype(int).tolist()))   {MARK}
    adj_mat = nx.to_scipy_sparse_matrix(G)
    for K in _grid:
        # Perform spectral clustering algorithm
        sc = SpectralClustering(K, affinity='precomputed', n_init=10, random_state=0)
"""


def patch(path, anchor, insert, before=True, replace=False):
    s = open(path).read()
    if MARK in s and insert.strip().splitlines()[0] in s:
        print(f"  already patched: {path}")
        return
    assert s.count(anchor) == 1, f"anchor not found once in {path}: {anchor!r}"
    s = s.replace(anchor, insert if replace else (insert + anchor if before else anchor + insert))
    open(path, "w").write(s)
    print(f"  patched: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    r = ap.parse_args().repo
    dp = os.path.join(r, "utils", "dataset", "dataset_preprocessing.py")
    patch(dp, "    'FIRSTMM2': 'db',\n", f"    'LuxRegions2': 'lx',   {MARK}\n", before=False)
    patch(dp, "    elif graph_type in ['FIRSTMM2', 'FIRSTMM', 'FIRSTMM_1level']:\n", BRANCH)
    patch(os.path.join(r, "utils", "dataset", "graph_corsening.py"), SC_OLD, SC_NEW, replace=True)
    patch(os.path.join(r, "higen_runner.py"),
          "            stat_HG_gen = self.stat_HG_pmf.sample2(n_sample=self.num_test_gen, leaf_edge_weight=1)\n",
          RESCALE, before=False)


if __name__ == "__main__":
    main()
