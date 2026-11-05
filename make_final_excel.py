"""The final tables (make_final_tables.sh's TSVs) as one Excel workbook, one sheet each.

    python -I make_final_excel.py <results dir> <out.xlsx>            # Oldenburg
    python -I make_final_excel.py <results dir> <out.xlsx> vermont    # Vermont

Cells stay as the TSVs give them ("mean ± std" text, as in the existing sheets); the
column headers are renamed to the short names used in the paper. Each sheet is also
written as a paste-ready TSV (short headers, no footnote) to <out>_paste/, for
`iconv -f UTF-8 -t UTF-16LE <file> | clip.exe`.
"""
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

SHORT = {"A v3 (hops + centroid)": "A v3",
         "B (no hops + centroid)": "B (no h)", "B v3 (hops + centroid)": "B v3",
         "C (no h, fine-tuned)": "C (no h)", "C v3 (fine-tuned)": "C v3",
         "DiGress (Oldenburg 80/20)": "DiGress (Olde 80/20)",
         "DiGress (private graph only, 80/20)": "DiGress (Olde 80/20)",
         "A (Lux -> Lux)": "A", "B (Lux -> Olde)": "B", "C (fine-tuned)": "C",
         "C (no r, fine-tuned)": "C (no r)"}

SHEETS = [
    ("D_intra Luxembourg", "D_intra/tables/final/repeats_summary_A.tsv",
     "D_intra on the Luxembourg test supernodes (version 3: hops + centroid)."),
    ("D_intra Oldenburg", "D_intra/tables/final/repeats_summary.tsv",
     "D_intra on the 63 Oldenburg test supernodes. B: the Luxembourg model. C: the Luxembourg "
     "model fine-tuned on the 254 Oldenburg training supernodes. No h: the condition without "
     "the hop histogram (centroid still marked). DiGress: plain DiGress, no condition, "
     "trained on Oldenburg alone (80% train / 20% test), each graph at its test supernode's size."),
    ("D_inter Luxembourg", "D_inter/tables/final/pairs_summary_A_final.tsv",
     "D_inter (internal-degree X_i) on the Luxembourg test pairs, true X_i."),
    ("D_inter Oldenburg", "D_inter/tables/final/pairs_summary_final.tsv",
     "D_inter on the 187 Oldenburg test pairs, X_i computed from D_intra's generated "
     "supernodes (B from D_intra B, C from D_intra C). C: the Luxembourg model fine-tuned on "
     "the 362 Oldenburg training pairs. No r: condition [n_i, n_j, w_ij] without the "
     "out-degree profiles r_i->j, r_j->i."),
    ("Full graph", "D_inter/tables/syn_final/syn_re_summary.tsv",
     "The whole assembled Oldenburg graph (6105 nodes): every supernode from D_intra, every "
     "cross block from D_inter. Each error is computed per seed, then mean ± std over 5 seeds. "
     "RE = |generated - real| / real, in %. degW: degree-distribution W1 distance."),
    ("Runtime & memory", "costs/runtime_memory_final.tsv",
     "Training, test-set sampling and whole-graph inference costs of the final models. Peak GPU "
     "alloc is what the model needs; reserved / device memory grows to fill the card (40 or 80 GB)."),
]

VERMONT_SHEETS = [
    SHEETS[0],
    ("D_intra Vermont", "D_intra/tables/vermont/repeats_summary.tsv",
     "D_intra on the 1104 Vermont test supernodes. B: the Luxembourg model. C: the Luxembourg "
     "model fine-tuned on the 4418 Vermont training supernodes. No h: the condition without "
     "the hop histogram (centroid still marked). DiGress: plain DiGress, no condition, "
     "trained on Vermont alone (80% train / 20% test), each graph at its test supernode's size."),
    SHEETS[2],
    ("D_inter Vermont", "D_inter/tables/vermont/pairs_summary_vermont.tsv",
     "D_inter on the 2881 Vermont test pairs, X_i computed from D_intra's generated "
     "supernodes (B from D_intra B, C from D_intra C). C: the Luxembourg model fine-tuned on "
     "the 5069 Vermont training pairs. No r: condition [n_i, n_j, w_ij] without the "
     "out-degree profiles r_i->j, r_j->i."),
    ("Full graph", "D_inter/tables/syn_vermont/syn_re_summary.tsv",
     "The whole assembled Vermont graph (97,975 nodes, 756 components): every supernode from "
     "D_intra, every cross block from D_inter. Each error is computed per seed, then mean ± std "
     "over 5 seeds. RE = |generated - real| / real, in %. degW: degree-distribution W1 distance."),
    ("Runtime & memory", "costs/runtime_memory_vermont.tsv", SHEETS[5][2]),
]


def read(path):
    return [line.rstrip("\n").split("\t") for line in open(path, encoding="utf-8")]


def main():
    results, out = sys.argv[1], sys.argv[2]
    vermont = len(sys.argv) > 3 and sys.argv[3] == "vermont"
    sheets = VERMONT_SHEETS if vermont else SHEETS
    if vermont:
        SHORT["DiGress (private graph only, 80/20)"] = "DiGress (VT 80/20)"
    paste_dir = os.path.splitext(out)[0] + "_paste"
    os.makedirs(paste_dir, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    head_fill = PatternFill("solid", fgColor="F2C14E")
    for name, rel, note in sheets:
        path = os.path.join(results, rel)
        if not os.path.exists(path):
            print(f"  {name}: {rel} missing, skipped")
            continue
        ws = wb.create_sheet(name)
        ws.append([name])
        ws["A1"].font = Font(bold=True, size=13)
        ws.append([note])
        ws["A2"].alignment = Alignment(wrap_text=False)
        ws.append([])
        rows = read(path)
        for i, r in enumerate(rows):
            if i == 0:
                r = [SHORT.get(c, c) for c in r]
            ws.append(r)
            if i == 0:
                for cell in ws[ws.max_row]:
                    cell.font = Font(bold=True)
                    cell.fill = head_fill
            elif r and r[0]:
                ws.cell(ws.max_row, 1).font = Font(bold=True)
        width = max(len(r) for r in rows)
        ws.column_dimensions["A"].width = 28
        for c in range(2, width + 1):
            ws.column_dimensions[get_column_letter(c)].width = 22
        blanks = [i for i, r in enumerate(rows) if not any(r)]
        table = rows[:blanks[-1]] if blanks else rows          # drop the closing footnote
        table = [[SHORT.get(c, c) for c in table[0]]] + table[1:]
        dst = os.path.join(paste_dir, name.lower().replace(" & ", "_").replace(" ", "_") + ".tsv")
        open(dst, "w", encoding="utf-8").write("\n".join("\t".join(r) for r in table) + "\n")
        print(f"  {name}: {len(rows) - 1} rows from {rel}")
    wb.save(out)
    print(f"wrote {out}\npaste-ready TSVs in {paste_dir}/")


if __name__ == "__main__":
    main()
