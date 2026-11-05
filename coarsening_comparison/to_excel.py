"""coarsening_stats.py's TSV (one column per k) -> rows per k in the spreadsheet's column order:
#components, #superNodes, n_i>=3, mean n_i, median n_i, max n_i, 95th pct n_i,
#superEdges, #connected pairs, median m_i, mean m_i.   python -I to_excel.py in.tsv out.tsv"""
import sys

ORDER = ["#components", "#supernodes M", "#supernodes with n_i >= 3", "n_i mean", "n_i median",
         "n_i max", "n_i 95th percentile", "#super edges (edges between supernodes)",
         "#connected supernode pairs", "m_i median", "m_i mean"]
rows = {r[0]: r[1:] for r in (l.rstrip("\n").split("\t") for l in open(sys.argv[1]))}
assert all(v == "yes" for v in rows["check: sum m_i + super edges = m"])
out = ["\t".join(rows[q][c] for q in ORDER) for c in range(len(rows["k"]))]
open(sys.argv[2], "w").write("\n".join(out) + "\n")
print("\n".join(f"k={k}\t{o}" for k, o in zip(rows["k"], out)))
