"""E2b: the kappa/gamma frontier on ML-1M, seed 1 -- accuracy against calibration.

The 12 grid cells of dcrgrid/dcrgrid2 (seed 1). Their checkpoints did not all
survive: dcrgrid saved all 7 of its cells to one untagged file (the last to finish
was g0.9_k0_windowed; checked against its log in score_all output), dcrgrid2 kept
its 5 cells. The kappa = inf static corner is DecRS seed 1 (dcrseedsA). So 7 cells
are re-scored from checkpoints (R@20, N@20, C_KL@20, coverage, Gini); the other 5
have accuracy from their training logs only and no C_KL. Every cell is ranked with the
Eq. 11 blend, as in training.

    python analysis/eval_package/e2b_frontier.py
    python analysis/eval_package/e2b_plot.py         # needs matplotlib
Output: analysis/eval_package/out/e2b_frontier.csv, e2b_frontier.json
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT_DIR, REPO, Data, beyond, load_top, metrics  # noqa: E402

import numpy as np  # noqa: E402

LOGS = os.path.join(REPO, "results-azure")
# (log file, top-100 variant or None, kappa, gamma, population prior)
CELLS = [
    ("results-dcrgrid/log/DecFM_static_s1.txt", "DecRS_s1", "inf", "-", "static"),
    ("results-dcrgrid2/log/DecFM_popnull_s1.txt", "grid_g0.9_kinf_windowed", "inf", "0.9", "windowed"),
    ("results-dcrgrid/log/DecFM_g0.9_k10_static_s1.txt", None, "10", "0.9", "static"),
    ("results-dcrgrid/log/DecFM_g0.7_k10_static_s1.txt", None, "10", "0.7", "static"),
    ("results-dcrgrid2/log/DecFM_g0.5_k10_static_s1.txt", "grid_g0.5_k10_static", "10", "0.5", "static"),
    ("results-dcrgrid2/log/DecFM_g0.3_k10_static_s1.txt", "grid_g0.3_k10_static", "10", "0.3", "static"),
    ("results-dcrgrid/log/DecFM_g1_k0_static_s1.txt", None, "0", "1", "static"),
    ("results-dcrgrid/log/DecFM_g0.9_k0_static_s1.txt", None, "0", "0.9", "static"),
    ("results-dcrgrid/log/DecFM_g0.9_k0_windowed_s1.txt", "grid_g0.9_k0_windowed", "0", "0.9", "windowed"),
    ("results-dcrgrid/log/DecFM_g0.7_k0_static_s1.txt", None, "0", "0.7", "static"),
    ("results-dcrgrid2/log/DecFM_g0.5_k0_static_s1.txt", "grid_g0.5_k0_static", "0", "0.5", "static"),
    ("results-dcrgrid2/log/DecFM_g0.3_k0_static_s1.txt", "grid_g0.3_k0_static", "0", "0.3", "static"),
]


def logged(path):
    """Best-epoch test metrics: the [Test] line that follows 'End. Best epoch'."""
    lines = open(os.path.join(LOGS, path)).read().splitlines()
    j = max(i for i, l in enumerate(lines) if l.startswith("End. Best epoch"))
    t = lines[j + 1]
    r = [float(x) for x in re.search(r"Recall: ([\d.\-]+)", t).group(1).split("-")]
    n = [float(x) for x in re.search(r"NDCG: ([\d.\-]+)", t).group(1).split("-")]
    v = [l for l in lines[:j] if l.startswith("[Valid]")][-1]
    vr = [float(x) for x in re.search(r"Recall: ([\d.\-]+)", v).group(1).split("-")]
    return {"log_R@10": r[0], "log_R@20": r[1], "log_N@10": n[0], "log_N@20": n[1],
            "log_valid_R@10": vr[0], "best_epoch": int(lines[j].split()[-1])}


def main():
    D = Data("ml_1m")
    rows = []
    for log, var, kappa, gamma, pop in CELLS:
        row = {"kappa": kappa, "gamma": gamma, "prior": pop, "label": "(%s, %s)" % (
            "inf" if kappa == "inf" else kappa, gamma), **logged(log), "rescored": var is not None}
        if var:
            top = load_top("ml_1m", var)
            m = metrics(D, top)
            row.update({k: float(m[k].mean()) for k in ("R@10", "R@20", "N@10", "N@20", "C_KL@20")})
            row.update(beyond(D, top))
            row["matches_log_R@10"] = abs(round(row["R@10"], 4) - row["log_R@10"]) < 6e-4
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    cols = ["label", "kappa", "gamma", "prior", "rescored", "R@10", "R@20", "N@10", "N@20", "C_KL@20",
            "Coverage@20", "Gini@20", "log_R@10", "log_R@20", "log_N@10", "log_N@20", "log_valid_R@10",
            "best_epoch", "matches_log_R@10"]
    with open(os.path.join(OUT_DIR, "e2b_frontier.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    rs = [r for r in rows if r["rescored"]]
    rho = float(np.corrcoef([r["R@20"] for r in rs], [r["C_KL@20"] for r in rs])[0, 1])
    json.dump({"cells": rows, "pearson_R20_CKL20_rescored": rho}, open(os.path.join(OUT_DIR, "e2b_frontier.json"),
                                                                       "w"), indent=1)
    print("Pearson(R@20, C_KL@20) over re-scored cells: %.3f" % rho)


if __name__ == "__main__":
    main()
