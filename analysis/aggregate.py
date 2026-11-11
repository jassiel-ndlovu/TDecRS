"""Mean +- std over seeds and paired tests on tier1_runs.csv (dcrseedsA + baseseeds)."""
import csv
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, "tier1_runs.csv"))))
rows = [r for r in rows if r["job"] in ("dcrseedsA", "baseseeds")]
METRICS = ["%s@%d" % (m, k) for m in ("Recall", "NDCG", "Precision", "MRR") for k in (10, 20, 50, 100)]

cells = {}
for r in rows:
    key = r["model"] if r["tag"] == "static" else "DCR " + r["tag"]
    cells.setdefault(key, {})[int(r["seed"])] = r

order = ["FM", "NFM", "DecFM", "DecNFM", "DCR g0.5_k0_static", "DCR g0.3_k0_static"]
print("seeds per cell:", {k: sorted(cells[k]) for k in order})
summary = {}
for k in order:
    summary[k] = {m: (np.mean([float(cells[k][s][m]) for s in range(1, 6)]),
                      np.std([float(cells[k][s][m]) for s in range(1, 6)], ddof=1)) for m in METRICS + ["valid_Recall@10"]}
    print(k, " ".join("%s=%.4f±%.4f" % (m, *summary[k][m]) for m in ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "valid_Recall@10"]))

print("\nPaired over seeds (same seed = same init/sampling stream), vs DecFM:")
tests = []
for k in ["FM", "DCR g0.5_k0_static", "DCR g0.3_k0_static"]:
    for m in METRICS:
        a = np.array([float(cells[k][s][m]) for s in range(1, 6)])
        b = np.array([float(cells["DecFM"][s][m]) for s in range(1, 6)])
        d = a - b
        t = stats.ttest_rel(a, b)
        tests.append((k, m, d.mean(), d.std(ddof=1), 100 * d.mean() / b.mean(), t.pvalue, int((d > 0).sum())))
        print("%-20s %-13s diff=%+.4f±%.4f (%+.1f%%) p=%.2g wins=%d/5" % tests[-1])

import json
json.dump({"summary": {k: {m: list(v) for m, v in d.items()} for k, d in summary.items()},
           "tests": tests}, open(os.path.join(HERE, "tier1_summary.json"), "w"), indent=1)
