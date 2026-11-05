"""Verify the conditional D_intra model: generated clusters vs real held-out ones.

Run LOCALLY in the `kmis` env (needs only numpy/scipy/networkx):

    python evaluate_clusters.py --generated conditional_samples.txt

Answers two separate questions. Keep them separate when reporting.

  1. ARE THE CLUSTERS REALISTIC?
     Generated vs the real test clusters on connectivity, tree fraction, edge
     count, degree distribution and normalized-Laplacian spectrum. An
     Erdos-Renyi graph with matched node and edge counts is reported alongside
     as a floor -- without it, a small number means nothing, because sparse
     graphs of this size resemble each other anyway.

  2. IS THE CONDITION ACTUALLY USED?
     The condition is [n, rho_in, k]. n is FORCED (passed as num_nodes), so
     matching it proves nothing. k is constant, so it cannot carry information.
     That leaves rho_in as the only feature the model could learn from -- and on
     Luxembourg rho_in is ~2/n because road clusters are trees, so it is nearly
     redundant with n.

     The test therefore is a PARTIAL correlation: does generated density track
     the requested rho_in *after* removing everything n already explains? If that
     is ~0, the model is conditioning on size alone and the extra features are
     doing nothing.

     This is a weaker test than regenerating with shuffled conditions, which
     needs a second BlueBEAR run. It is the strongest available from one file.
"""
import argparse

import numpy as np
import networkx as nx
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import wasserstein_distance

P_EIG = 10
SEED = 0


# --------------------------------------------------------------------------
def parse_generated(path):
    """Parse conditional_samples.txt written by sample_conditional.py."""
    with open(path) as f:
        lines = f.read().splitlines()

    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip().startswith("N="):
            i += 1
            continue
        n = int(lines[i].strip()[2:])
        i += 1

        meta = {}
        if i < len(lines) and lines[i].strip().startswith("supernode="):
            for tok in lines[i].split():
                k, _, v = tok.partition("=")
                meta[k] = int(v)
            i += 1

        cond = None
        if i < len(lines) and lines[i].strip().startswith("cond:"):
            cond = np.fromstring(lines[i].split(":", 1)[1], sep=" ")
            i += 1

        while i < len(lines) and not lines[i].strip().startswith("X"):
            i += 1
        # The node-class row. With the centroid marked (v3, *_centroid) node 0 is
        # class 1 and every other node class 0; with one node type it is all 0.
        x = (np.fromstring(lines[i + 1], dtype=int, sep=" ")
             if i + 1 < len(lines) else np.zeros(0, dtype=int))
        i += 2                                    # skip "X: " and the node-class row
        while i < len(lines) and not lines[i].strip().startswith("E"):
            i += 1
        i += 1                                    # skip "E: "

        A = np.zeros((n, n))
        for r in range(n):
            if i + r >= len(lines):
                break
            v = np.fromstring(lines[i + r], dtype=float, sep=" ")
            A[r, :len(v)] = v
        i += n

        A = (A == 1).astype(float)                # class 1 = edge
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)
        out.append({"A": A, "n": n, "cond": cond, "x": x, **meta})
    return out


def centroid_index(x):
    """The node the node-class row marks as the centroid, or None unless exactly one is.

    The models fix the centroid at node 0; other generators (Huiping's ER / SC_RG)
    mark the k-MIS centre at its own position in the supernode.
    """
    marked = np.flatnonzero(np.asarray(x) == 1)
    return int(marked[0]) if len(marked) == 1 else None


def hop_hist(A, centre, k):
    """h(r), r = 1..k: fraction of the non-centre nodes at shortest-path distance r.

    The definition extraction/kmis_pairs.hop_histogram uses for the condition,
    computed here from a dense block without importing the torch_sparse stack.
    Nodes further than k, or unreachable, fall in no ring, so a disconnected or
    stretched block has sum(h) < 1 and that shortfall shows up in E_h.
    """
    n = A.shape[0]
    h = np.zeros(k)
    if n <= 1:
        return h
    d = shortest_path(csr_matrix(A), unweighted=True, directed=False, indices=centre)
    for r in range(1, k + 1):
        h[r - 1] = np.count_nonzero(d == r)
    return h / (n - 1)


def dense_adj(edge_index, n):
    A = np.zeros((n, n))
    if edge_index.numel():
        ei = edge_index.numpy()
        A[ei[0], ei[1]] = 1.0
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def norm_lap_eigs(A):
    """Eigenvalues of I - D^-1/2 A D^-1/2, ascending. Isolated nodes get a 0 row."""
    d = A.sum(1)
    nz = d > 0
    dinv = np.zeros_like(d)
    dinv[nz] = d[nz] ** -0.5
    L = -(dinv[:, None] * A * dinv[None, :])
    L[np.arange(len(d)), np.arange(len(d))] = nz.astype(float)
    return np.sort(np.linalg.eigvalsh(L))


def er_like(A, rng):
    """Erdos-Renyi graph with the same node and edge count -- the floor baseline."""
    n = A.shape[0]
    m = min(int(A.sum() // 2), n * (n - 1) // 2)
    iu = np.triu_indices(n, 1)
    B = np.zeros((n, n))
    if m:
        pick = rng.choice(len(iu[0]), size=m, replace=False)
        B[iu[0][pick], iu[1][pick]] = 1.0
    return np.maximum(B, B.T)


def describe(A):
    n = A.shape[0]
    m = int(A.sum() // 2)
    G = nx.from_numpy_array(A)
    comp = nx.number_connected_components(G)
    return {
        "n": n,
        "m": m,
        "density": 2.0 * m / (n * (n - 1)) if n > 1 else 0.0,
        "connected": comp == 1,
        "n_components": comp,
        # a forest has no independent cycles: m - n + components == 0
        "is_forest": (m - n + comp) == 0,
        "degrees": np.asarray([d for _, d in G.degree()], dtype=float),
        "eigs": norm_lap_eigs(A),
    }


def agg(stats):
    return {
        "connected": 100.0 * np.mean([s["connected"] for s in stats]),
        "forest": 100.0 * np.mean([s["is_forest"] for s in stats]),
        "mean_m": np.mean([s["m"] for s in stats]),
        "mean_density": np.mean([s["density"] for s in stats]),
        "mean_deg": np.mean(np.concatenate([s["degrees"] for s in stats])),
        "components": np.mean([s["n_components"] for s in stats]),
    }


def dist_to_real(stats, real_stats):
    """Distribution distances against the real held-out clusters."""
    gd = np.concatenate([s["degrees"] for s in stats])
    rd = np.concatenate([s["degrees"] for s in real_stats])
    deg_w = wasserstein_distance(gd, rd)

    ref = [s["eigs"] for s in real_stats]
    spec = float(np.median([np.median([wasserstein_distance(s["eigs"], r) for r in ref])
                            for s in stats]))
    return deg_w, spec


def partial_corr(x, y, z):
    """corr(x, y) with the linear effect of z removed from both."""
    def resid(a, b):
        A = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    rx, ry = resid(x, z), resid(y, z)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="conditional_samples.txt")
    ap.add_argument("--pairs", default="pairs_luxembourg_osm_k7.pt")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    gen = parse_generated(args.generated)
    if not gen:
        raise SystemExit(f"no graphs parsed from {args.generated}")

    try:
        blob = torch.load(args.pairs, weights_only=False)
    except TypeError:
        blob = torch.load(args.pairs)
    test = {int(r["sn"]): r for r in blob["splits"][2]}
    names = blob["cond_names"]

    print(f"parsed {len(gen)} generated / {len(test)} real test clusters")
    print(f"condition: {names}\n")

    # pair each generated cluster with the real one it was conditioned on
    pairs, unmatched = [], 0
    for g in gen:
        r = test.get(g.get("supernode"))
        if r is None:
            unmatched += 1
            continue
        pairs.append((g, r))
    if unmatched:
        print(f"WARNING: {unmatched} generated clusters had no matching supernode\n")

    g_stats = [describe(g["A"]) for g, _ in pairs]
    r_stats = [describe(dense_adj(r["edge_index"], int(r["n"]))) for _, r in pairs]
    e_stats = [describe(er_like(g["A"], rng)) for g, _ in pairs]

    # ---- 1. realism -------------------------------------------------------
    G, R, E = agg(g_stats), agg(r_stats), agg(e_stats)
    gdw, gsp = dist_to_real(g_stats, r_stats)
    edw, esp = dist_to_real(e_stats, r_stats)

    print("1. ARE THE CLUSTERS REALISTIC?")
    print(f"{'':<24}{'generated':>12}{'real':>12}{'random ER':>12}")
    print("-" * 60)
    print(f"{'connected %':<24}{G['connected']:>12.1f}{R['connected']:>12.1f}{E['connected']:>12.1f}")
    print(f"{'forest (no cycles) %':<24}{G['forest']:>12.1f}{R['forest']:>12.1f}{E['forest']:>12.1f}")
    print(f"{'components (mean)':<24}{G['components']:>12.2f}{R['components']:>12.2f}{E['components']:>12.2f}")
    print(f"{'edges (mean)':<24}{G['mean_m']:>12.2f}{R['mean_m']:>12.2f}{E['mean_m']:>12.2f}")
    print(f"{'density (mean)':<24}{G['mean_density']:>12.4f}{R['mean_density']:>12.4f}{E['mean_density']:>12.4f}")
    print(f"{'node degree (mean)':<24}{G['mean_deg']:>12.2f}{R['mean_deg']:>12.2f}{E['mean_deg']:>12.2f}")
    print()
    print(f"{'degree dist. vs real':<24}{gdw:>12.4f}{0.0:>12.4f}{edw:>12.4f}   (lower better)")
    print(f"{'spectrum vs real':<24}{gsp:>12.4f}{'-':>12}{esp:>12.4f}   (lower better)")
    print()
    print("  The 'random ER' column is the floor. Generated must beat it clearly,")
    print("  or a good-looking number means nothing.")

    # ---- 2. is the condition used? ---------------------------------------
    n_req = np.array([float(r["n"]) for _, r in pairs])
    n_gen = np.array([float(g["n"]) for g, _ in pairs])
    size_ok = 100.0 * np.mean(n_req == n_gen)

    rho_req = np.array([float(g["cond"][names.index("rho_in")]) for g, _ in pairs]) \
        if "rho_in" in names else None
    rho_gen = np.array([s["density"] for s in g_stats])
    rho_true = np.array([s["density"] for s in r_stats])

    print("\n2. IS THE CONDITION ACTUALLY USED?")
    print("-" * 60)
    print(f"  requested node count honoured: {size_ok:.1f}%")
    print("    (n is passed as num_nodes and forced, so this is a sanity check,")
    print("     NOT evidence that the model conditions on anything)")

    if rho_req is not None:
        r_raw = float(np.corrcoef(rho_gen, rho_req)[0, 1])
        # Controlling for n LINEARLY is not enough: rho_in ~ 2/n is hyperbolic, so
        # a linear fit in n leaves most of n's explanatory power in the residual
        # and the partial correlation comes out spuriously high. Regress out 2/n
        # -- the density a tree on n nodes must have -- which is the actual
        # confound. On Luxembourg these two controls give ~0.93 vs ~0.07.
        r_lin = partial_corr(rho_gen, rho_req, n_req)
        r_par = partial_corr(rho_gen, rho_req, 2.0 / n_req)
        r_nvs = float(np.corrcoef(rho_req, 2.0 / n_req)[0, 1])
        ex_req, ex_gen = rho_req - 2.0 / n_req, rho_gen - 2.0 / n_req
        r_exc = float(np.corrcoef(ex_gen, ex_req)[0, 1])
        print()
        print(f"  corr(generated density, requested rho_in)      {r_raw:>7.3f}")
        print(f"  corr(requested rho_in, 2/n) in this data       {r_nvs:>7.3f}")
        print(f"  partial, controlling for n linearly            {r_lin:>7.3f}   (MISLEADING)")
        print(f"  partial, controlling for 2/n                   {r_par:>7.3f}   <-- the test")
        print(f"  corr of EXCESS density beyond a tree           {r_exc:>7.3f}   <-- cleanest")
        print()
        if not np.isnan(r_par):
            if abs(r_par) < 0.15:
                print("  VERDICT: ~0. The model conditions on SIZE alone. rho_in adds nothing")
                print("  beyond what n already implies. Expected on a road network, where")
                print("  clusters are trees and rho_in is therefore pinned to 2/n.")
            elif abs(r_par) < 0.4:
                print("  VERDICT: weak but non-zero. rho_in carries some signal beyond n.")
            else:
                print("  VERDICT: the model genuinely uses rho_in independently of n.")

    print()
    print(f"  for reference, real clusters:  mean density {rho_true.mean():.4f}")
    print(f"                 generated:      mean density {rho_gen.mean():.4f}")

    # ---- 3. per-cluster accuracy vs a trivial baseline --------------------
    m_gen = np.array([s["m"] for s in g_stats], dtype=float)
    m_true = np.array([s["m"] for s in r_stats], dtype=float)
    # A tree on n nodes has exactly n-1 edges. Since ~89% of real clusters are
    # trees, "m = n-1" is a one-line rule that uses ONLY the node count. If the
    # model cannot beat it, the model is not adding anything on edge count.
    m_naive = n_req - 1.0

    print("\n3. PER-CLUSTER EDGE COUNT (vs a rule that only knows n)")
    print("-" * 60)
    print(f"{'':<20}{'mean |err|':>12}{'corr':>10}{'within +/-1':>14}")
    for label, m in (("m = n-1 (naive)", m_naive), ("DiGress", m_gen)):
        print(f"{label:<20}{np.abs(m - m_true).mean():>12.2f}"
              f"{np.corrcoef(m, m_true)[0, 1]:>10.3f}"
              f"{100.0 * np.mean(np.abs(m - m_true) <= 1):>13.1f}%")
    if np.abs(m_naive - m_true).mean() <= np.abs(m_gen - m_true).mean():
        print("\n  The naive rule MATCHES OR BEATS the model. On edge count specifically,")
        print("  the model adds nothing over 'output a tree with n nodes'.")


if __name__ == "__main__":
    main()
