"""D_inter A / B / C datasets with REDUCED, w_ij-STRATIFIED training sets (Huiping, 9 Nov 2026).

For graphs whose every-pair dataset is too large to train on (LiveJournal -> YouTube):

    python make_stratified_pairs.py --public livejournal --public-npz LJ.npz \\
        --private youtube --private-npz YT.npz --n-public 103850 --n-private 80000

Standalone (no k-mis-pool import), so it runs on BlueBEAR. Pairs are SELECTED from the
coarse graph first and only the selected ones are extracted. Record format, orientation
draw, condition [n_i, n_j, w_ij, r_ij(5), r_ji(5)] and standardisation as
make_partition_pairs.py.

Selection, per graph (pairs with both supernodes n >= 3; seed 0):
  train  both supernodes in D_intra's training split. Ranked by the cross DENSITY
         w_ij / (n_i + n_j) (Huiping, 9 Nov: ranking by w_ij alone favours large pairs;
         ties in random order), cut into 5 equal-count groups; n/5 drawn from each,
         spread over 5 size bins (quintiles of n_i + n_j) in the pool's proportions, so
         the size distribution is kept. DIVERSE supernodes: pairs are taken in random
         order in rounds: round L takes pairs whose supernodes are both used fewer than L
         times so far (train and val together), so unused supernodes go first. --max-combined (default 800) drops pairs with n_i + n_j above it from
         train and val only: at batch 2 a 1,000-node pair needs ~70 GB of an 80 GB A100.
  val    a SEPARATE set (no pair in train), drawn from the same 5 groups in the same
         proportion and under the same cap: --val-frac of the training size.
  test   private: 5 non-overlapping groups of --test-group pairs drawn uniformly from the
         pairs touching a test supernode (no size cap), saved for the competitors as
         test_groups_d_inter.json; public: --public-test pairs, A's test slot only.
r BINS from the real cross-degree distribution: 0 | 1 | then three bins splitting the
public training pairs' cross degrees >= 2 into terciles (overridable with --r-edges).
Writes pairs_inter_{A,B,C}_<tag>_deg.pt, <tag>_checks.tsv (n_i, n_j, w_ij of every pool
vs its selection) and <tag>_test_groups_d_inter.json, into data/pair_lux/raw/.
"""
import argparse
import copy
import json
import pathlib

import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import ks_2samp

HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "pair_lux" / "raw"
SPLIT_DIR = HERE.parents[1] / "D_intra" / "data" / "splits"
MIN_N, SEED, STRATA = 3, 0, 5


def load_partition(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
        n = int(meta["n"])
        A = sp.csr_matrix((np.ones(len(z["adj_indices"])), z["adj_indices"], z["adj_indptr"]),
                          shape=(n, n))
        cluster = z["assignment"].astype(np.int64)
    return A, cluster, int(meta["parameters"]["k"]), meta


def coarse_pairs(A, cluster, graph, k):
    """Every connected pair (i < j) with both n >= MIN_N: arrays i, j, w, n_i, n_j, test."""
    n, c = A.shape[0], int(cluster.max()) + 1
    P = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, c))
    C = sp.triu(P.T @ A @ P, k=1).tocoo()
    size = np.bincount(cluster, minlength=c)
    split = json.load(open(SPLIT_DIR / f"{graph}_k{k}_s80.json"))
    n_of = {int(a): b for a, b in split["n"].items()}
    bad = [s for s in list(n_of)[:2000] if size[s] != n_of[s]]
    if bad:
        raise SystemExit(f"{graph}: supernode sizes disagree with the split file ({len(bad)})")
    is_test = np.zeros(c, bool)
    is_test[split["test"]] = True
    keep = (size[C.row] >= MIN_N) & (size[C.col] >= MIN_N) & (C.data > 0)
    i, j, w = C.row[keep], C.col[keep], C.data[keep].astype(np.int64)
    return dict(i=i, j=j, w=w, n_i=size[i], n_j=size[j], test=is_test[i] | is_test[j])


def quotas(total, counts):
    """total split in proportion to counts (largest remainder)."""
    raw = total * np.asarray(counts, float) / max(sum(counts), 1)
    q = np.floor(raw).astype(int)
    q[np.argsort(-(raw - q))[:total - q.sum()]] += 1
    return q


def stratified(pool, idx, n_train, n_val, rng):
    """(train, val, cap) indices into pool.

    Ranked by w_ij / (n_i + n_j), 5 equal-count groups, n/5 from each; within a group
    the draw is proportional to the pool over 5 size bins (quintiles of n_i + n_j), so
    the selection keeps the pool's size distribution. DIVERSE supernodes: each cell is
    filled in rounds, round L taking (in random order) pairs whose two supernodes are
    both used fewer than L times so far, so unused supernodes go first; the largest-size
    cells are filled first. Returns the largest use of any supernode as `cap`."""
    idx = rng.permutation(idx)
    comb = pool["n_i"][idx] + pool["n_j"][idx]
    dens = pool["w"][idx] / comb
    order = np.argsort(dens, kind="stable")                          # ties in random order
    sbin = np.digitize(comb, np.quantile(comb, [0.2, 0.4, 0.6, 0.8]), right=True)
    cells = []                                                       # (size bin, members, nt, nv)
    for g, pos in enumerate(np.array_split(order, STRATA)):
        nt_g = n_train // STRATA + (g < n_train % STRATA)
        nv_g = n_val // STRATA + (g < n_val % STRATA)
        counts = [np.sum(sbin[pos] == b) for b in range(5)]
        for b, nt, nv in zip(range(5), quotas(nt_g, counts), quotas(nv_g, counts)):
            cells.append((b, rng.permutation(idx[pos[sbin[pos] == b]]), int(nt), int(nv)))
    cells.sort(key=lambda c: -c[0])
    used = np.zeros(int(max(pool["i"].max(), pool["j"].max())) + 1, np.int64)
    train, val = [], []
    for b, members, nt, nv in cells:
        need, got = nt + nv, []
        ai, aj = pool["i"][members].tolist(), pool["j"][members].tolist()
        taken = np.zeros(len(members), bool)
        level = 0
        while len(got) < need:                      # round L: both supernodes used < L times
            level += 1
            if level > 10000:
                raise SystemExit(f"cell (size bin {b}) cannot be filled")
            for k in range(len(members)):
                if len(got) == need:
                    break
                if not taken[k] and used[ai[k]] < level and used[aj[k]] < level:
                    taken[k] = True
                    got.append(members[k])
                    used[ai[k]] += 1
                    used[aj[k]] += 1
        train.append(np.array(got[:nt], dtype=np.int64))
        val.append(np.array(got[nt:], dtype=np.int64))
    return np.concatenate(train), np.concatenate(val), int(used.max())


def extract(A, cluster, pool, idx, rng, graph):
    """Records for pool pairs idx (make_partition_pairs' format, r filled in later)."""
    order = np.argsort(cluster, kind="stable")
    starts = np.searchsorted(cluster[order], np.arange(cluster.max() + 2))
    mem = lambda s: order[starts[s]:starts[s + 1]]
    recs = []
    for t in idx:
        ci, cj = int(pool["i"][t]), int(pool["j"][t])
        if rng.random() < 0.5:                                      # ORIENTATION
            ci, cj = cj, ci
        mi, mj = mem(ci), mem(cj)
        n_i, n_j = len(mi), len(mj)
        Ri, Rj = A[mi], A[mj]
        L_i, L_j, B = Ri[:, mi].tocoo(), Rj[:, mj].tocoo(), Ri[:, mj].tocoo()
        src = np.concatenate([L_i.row, L_j.row + n_i, B.row, B.col + n_i])
        dst = np.concatenate([L_i.col, L_j.col + n_i, B.col + n_i, B.row])
        recs.append({
            "sn_i": ci, "sn_j": cj, "n_i": n_i, "n_j": n_j, "graph": graph,
            "edge_index": torch.tensor(np.vstack([src, dst]), dtype=torch.long),
            "w": int(B.nnz),
            "rho_i": 2.0 * L_i.nnz / 2 / (n_i * (n_i - 1)), "rho_j": 2.0 * L_j.nnz / 2 / (n_j * (n_j - 1)),
            "_d_i": np.bincount(B.row, minlength=n_i), "_d_j": np.bincount(B.col, minlength=n_j),
        })
        assert recs[-1]["w"] == int(pool["w"][t])
    return recs


def r_edges_from(recs):
    d = np.concatenate([np.concatenate([r["_d_i"], r["_d_j"]]) for r in recs])
    big = d[d >= 2]
    t1, t2 = (int(np.ceil(q)) for q in np.quantile(big, [1 / 3, 2 / 3]))
    t1 = max(t1, 3)
    t2 = max(t2, t1 + 1)
    return [0, 1, 2, t1, t2], d


def finish(recs, edges):
    hist = lambda d: np.bincount(np.digitize(d, edges[1:]), minlength=len(edges)) / len(d)
    for r in recs:
        ri, rj = hist(r.pop("_d_i")), hist(r.pop("_d_j"))
        r["r"], r["r_ji"] = torch.tensor(ri, dtype=torch.float), torch.tensor(rj, dtype=torch.float)
        r["y_raw"] = torch.tensor([r["n_i"], r["n_j"], r["w"], *ri, *rj], dtype=torch.float)
    return recs


def standardise(train, splits):
    """make_pair_blocks.standardise: n_i, n_j, w_ij standardised on train; r left raw."""
    Y = torch.stack([r["y_raw"] for r in train])
    mean, std = Y.mean(0), Y.std(0)
    raw = torch.tensor([False] * 3 + [True] * (Y.shape[1] - 3))
    constant = std < 1e-6
    mean = torch.where(raw, torch.zeros_like(mean), mean)
    std = torch.where(raw | constant, torch.ones_like(std), std)
    for split in splits:
        for r in split:
            r["y"] = (r["y_raw"] - mean) / std
    return mean, std, constant


def cross_marginal(recs):
    slots = 2.0 * sum(int(r["n_i"]) * int(r["n_j"]) for r in recs)
    edges = 2.0 * sum(int(r["w"]) for r in recs)
    return torch.tensor([slots - edges, edges]) / slots


def check_rows(graph, what, pool, pool_idx, sel_idx):
    rows = []
    get = {"n_i": lambda x: pool["n_i"][x], "n_j": lambda x: pool["n_j"][x], "w_ij": lambda x: pool["w"][x],
           "n_i+n_j": lambda x: pool["n_i"][x] + pool["n_j"][x],
           "w_ij/(n_i+n_j)": lambda x: pool["w"][x] / (pool["n_i"][x] + pool["n_j"][x])}
    for col, f in get.items():
        a, b = f(pool_idx), f(sel_idx)
        q = lambda x: "/".join(f"{v:.4g}" for v in np.quantile(x, [0.1, 0.5, 0.9, 0.99]))
        ks = ks_2samp(a, b)
        rows.append(f"{graph}\t{what}\t{col}\t{len(pool_idx)}\t{len(sel_idx)}\t"
                    f"{a.mean():.4g}\t{b.mean():.4g}\t{q(a)}\t{q(b)}\t{ks.statistic:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True)
    ap.add_argument("--public-npz", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--private-npz", required=True)
    ap.add_argument("--n-public", type=int, default=103850, help="public training pairs (DBLP's A: 103,851)")
    ap.add_argument("--n-private", type=int, default=80000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-combined", type=int, default=800)
    ap.add_argument("--test-group", type=int, default=1000)
    ap.add_argument("--public-test", type=int, default=256)
    ap.add_argument("--r-edges", type=int, nargs=5, default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    loaded = {}
    for role, g, path in (("public", args.public, args.public_npz), ("private", args.private, args.private_npz)):
        A, cluster, k, meta = load_partition(path)
        print(f"{g}: {A.shape[0]} nodes, {int(A.nnz / 2)} edges, k={k}, "
              f"method {meta.get('method', meta.get('parameters'))}", flush=True)
        pool = coarse_pairs(A, cluster, g, k)
        tr_pool = np.flatnonzero(~pool["test"])
        te_pool = np.flatnonzero(pool["test"])
        capped = tr_pool[pool["n_i"][tr_pool] + pool["n_j"][tr_pool] <= args.max_combined]
        print(f"  {len(pool['w'])} pairs (both n >= {MIN_N}): {len(tr_pool)} train-eligible "
              f"({len(tr_pool) - len(capped)} above {args.max_combined} nodes dropped), "
              f"{len(te_pool)} touching a test supernode", flush=True)
        loaded[role] = dict(graph=g, A=A, cluster=cluster, k=k, pool=pool, tr_pool=tr_pool,
                            capped=capped, te_pool=te_pool)
    if loaded["public"]["k"] != loaded["private"]["k"]:
        raise SystemExit("k differs")
    k = loaded["public"]["k"]

    checks, sel = [], {}
    for role, n_train in (("public", args.n_public), ("private", args.n_private)):
        L = loaded[role]
        tr, va, cap = stratified(L["pool"], L["capped"], n_train, round(n_train * args.val_frac), rng)
        sel_sn = np.concatenate([L["pool"]["i"][tr], L["pool"]["j"][tr]])
        rand = rng.choice(L["capped"], len(tr), replace=False)
        rand_sn = np.concatenate([L["pool"]["i"][rand], L["pool"]["j"][rand]])
        comb = lambda s: (L["pool"]["n_i"][s] + L["pool"]["n_j"][s]).astype(float)
        print(f"  {L['graph']} train+val: max {cap} uses per supernode; train: {len(np.unique(sel_sn))} distinct "
              f"supernodes (max use {np.bincount(sel_sn).max()}); a plain random draw of the same "
              f"size: {len(np.unique(rand_sn))} (max use {np.bincount(rand_sn).max()})", flush=True)
        print(f"  {L['graph']} train: mean (n_i+n_j)^2 = {np.mean(comb(tr) ** 2):.0f} "
              f"(pool {np.mean(comb(L['capped']) ** 2):.0f}; training cost per pair ~ this)", flush=True)
        if role == "private":
            te = rng.choice(L["te_pool"], 5 * args.test_group, replace=False)
        else:
            te = rng.choice(L["te_pool"], args.public_test, replace=False)
        sel[role] = (tr, va, te)
        checks += check_rows(L["graph"], "train", L["pool"], L["tr_pool"], tr)
        checks += check_rows(L["graph"], "val", L["pool"], L["tr_pool"], va)
        checks += check_rows(L["graph"], "test", L["pool"], L["te_pool"], te)
        print(f"  {L['graph']}: selected {len(tr)} train / {len(va)} val / {len(te)} test", flush=True)

    recs = {}
    for role in ("public", "private"):
        L = loaded[role]
        recs[role] = [extract(L["A"], L["cluster"], L["pool"], s, rng, L["graph"]) for s in sel[role]]
        print(f"  {L['graph']}: extracted", flush=True)

    edges, d = r_edges_from(recs["public"][0])
    if args.r_edges:
        edges = list(args.r_edges)
    names = [str(a) if b - a == 1 else f"{a}-{b - 1}" for a, b in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    frac = np.bincount(np.digitize(d, edges[1:]), minlength=5) / len(d)
    print(f"\nr bins {names} (cross out-degree on {args.public} train, node fractions "
          f"{frac.round(4).tolist()}; degrees >= 2: max {d.max()})")
    for role in recs:
        for s in recs[role]:
            finish(s, edges)

    tag = args.tag or f"{args.public[:4]}2{args.private[:4]}_k{k}_s80_strat"
    cond_names = ["n_i", "n_j", "w_ij"] + [f"rij{b}" for b in names] + [f"rji{b}" for b in names]
    common = {"k": k, "cond_names": cond_names, "r_edges": edges, "public": args.public,
              "private": args.private, "max_combined": args.max_combined, "source": tag,
              "split_by": "supernode (s80, shared with D_intra); selection stratified on w_ij/(n_i+n_j), diverse supernodes",
              "selection": vars(args)}
    pub_tr, pub_va, pub_te = recs["public"]
    pri_tr, pri_va, pri_te = recs["private"]
    A_splits = [pub_tr, pub_va, pub_te]
    mean, std, constant = standardise(pub_tr, A_splits)
    stats = {"cond_mean": mean, "cond_std": std, "cond_constant": constant}

    def a_std(rs):
        rs = copy.deepcopy(rs)
        for r in rs:
            r["y"] = (r["y_raw"] - mean) / std
        return rs

    te_std = a_std(pri_te)
    blobs = [dict(common, config="A", graph=args.public, **stats, splits=A_splits),
             dict(common, config="B", **stats, splits=[pub_tr, pub_va, te_std]),
             dict(common, config="C", **stats, edge_marginal=cross_marginal(pub_tr), finetune_of="A",
                  splits=[a_std(pri_tr), a_std(pri_va), te_std])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for blob in blobs:
        path = OUT_DIR / f"pairs_inter_{blob['config']}_{tag}_deg.pt"
        torch.save(blob, path)
        print(f"  {path.name}: " + " / ".join(str(len(s)) for s in blob["splits"])
              + f" ({path.stat().st_size / 1e6:.0f} MB)", flush=True)

    g = args.test_group
    groups = [[{"idx": i, "sn_i": int(r["sn_i"]), "sn_j": int(r["sn_j"])}
               for i, r in enumerate(te_std[q * g:(q + 1) * g], start=q * g)] for q in range(5)]
    gpath = OUT_DIR / f"{tag}_test_groups_d_inter.json"
    json.dump({"graph": args.private, "what": "D_inter test pairs (at least one test supernode)",
               "seed": SEED, "non_overlapping": True, "of": int(len(loaded["private"]["te_pool"])),
               "pairs_file": f"pairs_inter_C_{tag}_deg.pt", "r_edges": edges, "groups": groups},
              open(gpath, "w"))
    cpath = OUT_DIR / f"{tag}_checks.tsv"
    open(cpath, "w").write("graph\tset\tvariable\tpool\tselected\tpool mean\tselected mean\t"
                           "pool q10/50/90/99\tselected q10/50/90/99\tKS statistic\n" + "\n".join(checks) + "\n")
    print(f"  {gpath.name}, {cpath.name}\n")
    print(open(cpath).read())


if __name__ == "__main__":
    main()
