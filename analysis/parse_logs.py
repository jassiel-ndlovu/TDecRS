"""Collect the best-epoch test (and validation) metrics from every Azure run log.

Writes analysis/tier1_runs.csv: one row per (job, model, dcr_tag, seed) with
P/R/N/MRR at 10, 20, 50, 100 for test, plus validation Recall@10 at the
selected epoch.
"""
import csv
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, os.pardir, "results-azure")
KS = [10, 20, 50, 100]
METRICS = ["Precision", "Recall", "NDCG", "MRR"]
LINE = re.compile(r"Precision: ([\d.-]+) Recall: ([\d.-]+) NDCG: ([\d.-]+) MRR: ([\d.-]+)")


def parse_line(s):
    m = LINE.search(s)
    vals = [list(map(float, g.split("-"))) for g in m.groups()]
    return {"%s@%d" % (name, k): v for name, row in zip(METRICS, vals) for k, v in zip(KS, row)}


rows = []
for path in sorted(glob.glob(os.path.join(ROOT, "results-*", "log", "*_s[0-9]*.txt"))):
    text = open(path, encoding="utf-8", errors="replace").read()
    head = re.search(r"# job: (\S+)\s+dcr_tag: (\S+)\s+seed: (\d+)", text)
    model = re.search(r"model='(\w+)'", text).group(1)
    end = text.find("End. Best epoch")
    if end < 0:
        continue
    best = int(re.search(r"End. Best epoch (\d+)", text).group(1))
    test = parse_line(text[end:])
    # validation line printed for the selected epoch
    blk = re.search(r"Runing Epoch %03d .*?\n.*?\n\[Valid\]: (.*?)\n" % best, text, re.S)
    val = parse_line(blk.group(1)) if blk else {}
    tag = head.group(2)
    tag = "static" if tag.startswith("<") else tag
    row = {"job": head.group(1), "model": model, "tag": tag, "seed": int(head.group(3)),
           "best_epoch": best, "valid_Recall@10": val.get("Recall@10", "")}
    row.update(test)
    rows.append(row)

out = os.path.join(HERE, "tier1_runs.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "runs ->", out)
