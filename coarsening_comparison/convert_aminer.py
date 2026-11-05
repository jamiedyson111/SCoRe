"""AMiner DBLP-Citation-network (V12+ JSON Lines, zipped) -> undirected .mtx, streamed.

    python -I convert_aminer.py DBLP-Citation-network-V19.zip dblp_v19.mtx

Nodes: every paper (its "id"); edges: id -- r for every r in "references" that is
itself a paper in the file (references outside the dataset are dropped). Edges are
made undirected, duplicates merged, self-citations dropped; papers with no
in-dataset citation stay as isolated nodes, as for CitationCS.

One pass: only "id" and "references" are pulled out of each line, with str.find
(a JSON string cannot contain an unescaped '"references": [', so this is exact);
--check N compares that against json.loads on the first N lines.
"""
import argparse
import json
import zipfile

import numpy as np

REF = '"references": ['


def fields(line):
    """(id, [reference ids]) of one JSON line, without parsing the rest."""
    a = line.index('"id": "') + 7
    pid = line[a:line.index('"', a)]
    r = line.find(REF)
    if r < 0:
        return pid, []
    body = line[r + len(REF):line.index("]", r)]
    return pid, [s.strip().strip('"') for s in body.split(",") if s.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip")
    ap.add_argument("out")
    ap.add_argument("--check", type=int, default=20000)
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.zip)
    member = next(n for n in zf.namelist() if n.endswith(".jsonl") or n.endswith(".json"))
    index, is_paper = {}, []
    src, dst, cs, cd = [], [], [], []
    n_lines = n_refs = 0
    with zf.open(member) as f:
        for raw in f:
            line = raw.decode("utf-8")
            if not line.strip():
                continue
            pid, refs = fields(line)
            if n_lines < args.check:
                rec = json.loads(line)
                assert (rec["id"], rec.get("references") or []) == (pid, refs), n_lines
            i = index.setdefault(pid, len(index))
            if i == len(is_paper):
                is_paper.append(True)
            else:
                is_paper[i] = True
            for r in refs:
                j = index.setdefault(r, len(index))
                if j == len(is_paper):
                    is_paper.append(False)
                cs.append(i)
                cd.append(j)
            n_refs += len(refs)
            n_lines += 1
            if len(cs) > 5_000_000:
                src.append(np.array(cs, np.int64)); dst.append(np.array(cd, np.int64))
                cs, cd = [], []
            if n_lines % 1_000_000 == 0:
                print(f"  {n_lines:,} papers, {n_refs:,} references", flush=True)
    src.append(np.array(cs, np.int64)); dst.append(np.array(cd, np.int64))
    u, v = np.concatenate(src), np.concatenate(dst)
    is_paper = np.array(is_paper)
    del index

    papers = np.flatnonzero(is_paper)
    new = np.full(len(is_paper), -1, np.int64)
    new[papers] = np.arange(len(papers))
    n = len(papers)
    print(f"{n_lines:,} lines, {n:,} distinct papers, {n_refs:,} references, "
          f"{int((~is_paper[v]).sum()):,} to papers outside the file")
    keep = is_paper[v]
    u, v = new[u[keep]], new[v[keep]]
    keep = u != v
    print(f"{int((~keep).sum()):,} self-citations dropped")
    lo, hi = np.minimum(u[keep], v[keep]), np.maximum(u[keep], v[keep])
    pairs = np.unique(lo * n + hi)
    i, j = pairs % n, pairs // n
    with open(args.out, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"% {member} (AMiner): citations made undirected, duplicates merged\n")
        f.write(f"{n} {n} {len(pairs)}\n")
        np.savetxt(f, np.column_stack([j + 1, i + 1]), fmt="%d")
    print(f"wrote {args.out}: {n:,} nodes, {len(pairs):,} undirected edges")


if __name__ == "__main__":
    main()
