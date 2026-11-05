"""Clustering, orbit and spectrum discrepancies between two graphs, as in DiGress.

The paper's full-graph Clustering, Orbit and Spectrum columns (Tables VI, VII):
the statistics and kernels of DiGress's src/analysis/spectre_utils.py with its
defaults (the GRAN form of the GraphRNN metrics), reimplemented on a scipy sparse
adjacency so that scoring needs neither networkx nor graph_tool:

  Clustering  100-bin histogram of the local clustering coefficients on [0, 1]
              (c_v = 2 t_v / (d_v (d_v - 1)), 0 below degree 2 -- the paper's eq. 8),
              Gaussian-TV kernel, sigma 0.1
  Orbit       ORCA's 4-node graphlet-orbit counts per vertex, summed and divided by
              the vertex count (15 numbers), Gaussian-TV kernel, sigma 30
  Spectrum    200-bin histogram of the normalised-Laplacian eigenvalues on
              [-1e-5, 2], Gaussian-TV kernel, sigma 1

Each is the squared MMD between one generated graph and the real one,
k(x,x) + k(y,y) - 2 k(x,y) = 2 - 2 k(x,y), so 0 is identical and 2 the maximum;
callers average it over seeds. The spectrum needs a dense eigendecomposition and is
left out (NaN) above SPECTRUM_MAX_N vertices.
"""
import os
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh

HERE = os.path.dirname(os.path.abspath(__file__))
ORCA = os.path.join(HERE, '..', 'src', 'analysis', 'orca', 'orca')
SPECTRUM_MAX_N = 20000


def gaussian_tv(x, y, sigma):
    """spectre_utils' kernel: exp(-TV(x, y)^2 / (2 sigma^2)), TV = |x - y|_1 / 2."""
    d = np.abs(x - y).sum() / 2.0
    return np.exp(-d * d / (2 * sigma * sigma))


def mmd(x, y, sigma):
    """Squared MMD between two single samples under gaussian_tv."""
    return float(2.0 - 2.0 * gaussian_tv(x, y, sigma))


def local_clustering(A):
    """Each vertex's local clustering coefficient (0 below degree 2)."""
    A = sp.csr_matrix(A, dtype=np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    t = np.asarray((A @ A).multiply(A).sum(1)).ravel() / 2.0
    denom = np.where(deg >= 2, deg * (deg - 1) / 2.0, 1.0)
    return np.where(deg >= 2, t / denom, 0.0)


def clustering_hist(A):
    h, _ = np.histogram(local_clustering(A), bins=100, range=(0.0, 1.0))
    return h / max(h.sum(), 1)


def orbit_profile(A):
    """ORCA node orbits of graphlets up to 4 nodes: the mean count per vertex (15)."""
    A = sp.triu(sp.csr_matrix(A), k=1).tocoo()
    n = A.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'graph.in')
        with open(src, 'w') as f:
            f.write(f"{n} {A.nnz}\n")
            f.write(''.join(f"{u} {v}\n" for u, v in zip(A.row, A.col)))
        # 'std' as spectre_utils.orca does; ORCA crashes writing to an output file
        out = subprocess.check_output([ORCA, 'node', '4', src, 'std']).decode()
    out = out[out.find('orbit counts:') + len('orbit counts:'):]
    counts = np.array([l.split() for l in out.strip().splitlines()], dtype=np.float64)
    if counts.shape != (n, 15):
        raise RuntimeError(f"ORCA returned {counts.shape}, expected ({n}, 15)")
    return counts.sum(0) / n


def spectrum_hist(A):
    """Histogram of the normalised-Laplacian spectrum, or None when too large."""
    n = A.shape[0]
    if n > SPECTRUM_MAX_N:
        return None
    A = sp.csr_matrix(A, dtype=np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    D = sp.diags(inv)
    L = D @ (sp.diags(deg) - A) @ D            # networkx's normalized_laplacian_matrix
    eigs = eigvalsh(L.toarray())
    h, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2))
    return h / h.sum()


def profiles(A):
    """The three statistics of one graph."""
    return {'clus': clustering_hist(A), 'orbit': orbit_profile(A), 'spec': spectrum_hist(A)}


SIGMA = {'clus': 0.1, 'orbit': 30.0, 'spec': 1.0}


def discrepancies(p, p_real):
    """{'clus_mmd', 'orbit_mmd', 'spec_mmd'} of one graph's profiles against the real one's."""
    return {f'{k}_mmd': (float('nan') if p[k] is None or p_real[k] is None
                         else mmd(p[k], p_real[k], SIGMA[k])) for k in SIGMA}
