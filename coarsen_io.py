"""Shared graph loading and portable coarsening-result format."""
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

DATA_ROOT = '/rds/projects/c/chenhp-graphsyn/D_intra/extraction/datasets'
DATA_FILES = {
    'oldenburg': 'Brinkhoff/oldenburg/raw/OL.cedge',
    'luxembourg': 'DIMACS10/luxembourg_osm/raw/luxembourg_osm.mat',
    'minnesota': 'Gleich/minnesota/raw/minnesota.mat',
}
BASE = Path(__file__).resolve().parent


def load_graph(data='oldenburg', data_root=DATA_ROOT):
    path = (Path(data_root) / DATA_FILES[data]) if data in DATA_FILES else Path(data).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Input graph not found: {path}')
    if path.suffix.lower() == '.npz':
        # Preprocessed sparse CSR graphs (e.g., SNAP com-Orkut).
        A = sp.load_npz(path).tocsr()
    elif path.suffix.lower() == '.mat':
        obj = loadmat(path)
        mat = obj['Problem']['A'][0, 0] if 'Problem' in obj else obj['A']
        A = sp.csr_matrix(mat)
    elif path.suffix.lower() in ('.cedge', '.edgelist', '.edges', '.txt', '.csv'):
        if path.suffix.lower() == '.cedge':
            edges = np.loadtxt(path, usecols=(1, 2), dtype=np.int64, ndmin=2)
        else:
            edges = np.loadtxt(path, usecols=(0, 1), delimiter=',' if path.suffix.lower()=='.csv' else None,
                               dtype=np.int64, ndmin=2)
        if not len(edges) or np.any(edges < 0):
            raise ValueError('Edge list must contain nonnegative vertex IDs and at least one edge')
        A = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                          shape=(int(edges.max())+1,)*2).tocsr()
    else:
        raise ValueError(f'Unsupported input format: {path.suffix}')
    A = A.maximum(A.T).tocsr()
    A.data[:] = 1
    A.setdiag(0)
    A.eliminate_zeros()
    A.sort_indices()
    return A, path.resolve()


def save_partition(A, assignment, *, data, method, source, parameters, elapsed_seconds):
    """Save fine adjacency + assignment + weighted coarse graph in one .npz file."""
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if len(assignment) != A.shape[0]:
        raise ValueError('Assignment size differs from graph size')
    _, assignment = np.unique(assignment, return_inverse=True)
    n = len(assignment)
    M = int(assignment.max()) + 1
    B = sp.csr_matrix((np.ones(n, dtype=np.int64), (np.arange(n), assignment)), shape=(n, M))
    coarse = (B.T @ A @ B).tocsr()  # off-diagonal entries count fine cross-edges
    # Diagonal contains twice each internal fine edge (undirected adjacency).
    tag = Path(str(data)).stem if str(data) not in DATA_FILES else str(data)
    safe_tag = ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in tag)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = BASE / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f'{safe_tag}_{method}_{stamp}.npz'
    meta = {'dataset': str(data), 'method': method, 'source_path': str(source),
            'parameters': parameters, 'elapsed_seconds': elapsed_seconds,
            'n': n, 'M': M, 'format_version': 1}
    np.savez_compressed(dest,
                        assignment=assignment,
                        adj_indptr=A.indptr, adj_indices=A.indices, adj_data=A.data,
                        coarse_indptr=coarse.indptr, coarse_indices=coarse.indices,
                        coarse_data=coarse.data,
                        metadata=np.array(json.dumps(meta)))
    print(f'n={n}, M={M}, saved: {dest}')
    return dest


def read_partition(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        n, M = int(meta['n']), int(meta['M'])
        A = sp.csr_matrix((z['adj_data'], z['adj_indices'], z['adj_indptr']), shape=(n, n))
        C = sp.csr_matrix((z['coarse_data'], z['coarse_indices'], z['coarse_indptr']), shape=(M, M))
        assignment = z['assignment'].astype(np.int64)
    if len(assignment) != n or np.any(assignment < 0) or np.any(assignment >= M):
        raise ValueError('Invalid node assignment')
    return A, C, assignment, meta
