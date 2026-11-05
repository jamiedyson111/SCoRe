"""graph-tool binary file (.gt, e.g. a Netzschleuder download, decompressed) -> undirected .mtx.

    zstd -dc VT.gt.zst > VT.gt
    python -I convert_gt.py VT.gt VT.mtx

Reads only the header, the graph-level string properties (printed, e.g. Netzschleuder's
name / description) and the adjacency; vertex and edge properties are not needed.
Format: https://graph-tool.skewed.de/static/doc/gt_format.html
Edges are made undirected, duplicates merged, self-loops dropped; every vertex is kept.
"""
import struct
import sys

import numpy as np


def read_gt(path):
    buf = open(path, "rb").read()
    if buf[:6] != b"\xe2\x9b\xbe gt":
        raise SystemExit(f"{path}: not a graph-tool binary file")
    if buf[6] != 1:
        raise SystemExit(f"{path}: gt format version {buf[6]}, expected 1")
    e = "<" if buf[7] == 0 else ">"
    pos = 8

    def u64():
        nonlocal pos
        v = struct.unpack_from(e + "Q", buf, pos)[0]
        pos += 8
        return v

    clen = u64()
    comment = buf[pos:pos + clen].decode("utf-8", "replace")
    pos += clen
    directed = bool(buf[pos])
    pos += 1
    n = u64()
    width = 1 if n <= 2 ** 8 else 2 if n <= 2 ** 16 else 4 if n <= 2 ** 32 else 8
    dt = np.dtype({1: "u1", 2: "u2", 4: "u4", 8: "u8"}[width]).newbyteorder(e)
    src, dst = [], []
    for v in range(n):
        k = u64()
        if k:
            dst.append(np.frombuffer(buf, dt, k, pos).astype(np.int64))
            src.append(np.full(k, v, np.int64))
            pos += k * width
    u = np.concatenate(src) if src else np.zeros(0, np.int64)
    w = np.concatenate(dst) if dst else np.zeros(0, np.int64)

    props = {}
    nprops = u64()
    for _ in range(nprops):                     # graph properties come first; stop after them
        kind = buf[pos]
        if kind != 0:
            break
        pos += 1
        nlen = u64()
        name = buf[pos:pos + nlen].decode()
        pos += nlen
        vtype = buf[pos]
        pos += 1
        if vtype != 6:                          # only string graph properties are decoded
            break
        slen = u64()
        props[name] = buf[pos:pos + slen].decode("utf-8", "replace")
        pos += slen
    return n, directed, u, w, comment, props


def write_mtx(path, n, u, w, note):
    keep = u != w
    lo, hi = np.minimum(u[keep], w[keep]), np.maximum(u[keep], w[keep])
    pairs = np.unique(lo * n + hi)
    i, j = pairs % n, pairs // n                # lower triangle: row > col
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate pattern symmetric\n")
        f.write(f"% {note}\n")
        f.write(f"{n} {n} {len(pairs)}\n")
        np.savetxt(f, np.column_stack([j + 1, i + 1]), fmt="%d")
    return len(pairs)


def main():
    src, out = sys.argv[1], sys.argv[2]
    n, directed, u, w, comment, props = read_gt(src)
    print(f"comment: {comment}")
    for k, v in props.items():
        print(f"graph property {k}: {v[:300]}")
    print(f"{n} vertices, {len(u)} stored edges, directed={directed}")
    name = props.get("name", src)
    m = write_mtx(out, n, u, w, f"{name} (graph-tool file): edges made undirected, duplicates merged")
    print(f"wrote {out}: {n} nodes, {m} undirected edges")


if __name__ == "__main__":
    main()
