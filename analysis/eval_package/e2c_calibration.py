"""E2c: is "DCR is worse calibrated" an artefact of calibrating against the long-run mix?

C_KL@20 (Steck 2018, 0.01 smoothing) of each model's top-20 against three target
genre mixes p_u:
  train   the user's whole training history (the usual C_KL; favours static models)
  recent  d_u^(gamma=0.3), the recency-weighted mix DCR/FMconf receive as input
  test    the genre mix of the user's test items (ORACLE: not available at ranking time)
for FM, DecRS, FMconf (raw and blended) and DCR, seed-averaged, on both datasets.

Then, per user, the correlation of the C_KL change DCR - DecRS (seed-averaged) with
the recency gap ||d_u^0.3 - d_u^1||_1 (how far recent taste sits from the long-run
mix; d_u^1 is the running mean, gamma = 1, kappa = 0).

    python analysis/eval_package/e2c_calibration.py
Output: analysis/eval_package/out/e2c_calibration.json, e2c_calibration.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DS, LABEL, OUT_DIR, Data, ld, load_top, metrics, user_stats  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

TARGETS = ("train", "recent", "test")


def main():
    out = {"datasets": {}}
    rows = []
    for ds in DS:
        D = Data(ds)
        seeds = DS[ds]["seeds"]
        c03 = ld(os.path.join(DS[ds]["dcr"], "user_confounder_g0.3_k0_static.npy"))
        f1 = os.path.join(DS[ds]["dcr"], "user_confounder_g1_k0_static.npy")
        c1 = ld(f1) if os.path.exists(f1) else D.recent_mix(1.0, 0.0)
        rec = np.stack([np.asarray(c03[u], float) for u in D.users])
        rec1 = np.stack([np.asarray(c1[u], float) for u in D.users])
        tg = {"recent": rec, "test": D.test_g}
        cells = {"FM": ["FM_s%d" % s for s in seeds]}
        if ds == "ml_1m":
            cells["FM (authors)"] = ["FM_A"]
        for c, pfx in (("DecRS", "DecRS"), ("FMconf raw", "FMconf_raw"), ("FMconf blend", "FMconf_blend"),
                       ("DCR", "DCR_g0.3")):
            cells[c] = ["%s_s%d" % (pfx, s) for s in seeds]
        per = {}
        res = {"models": {}}
        for c, names in cells.items():
            ms = [metrics(D, load_top(ds, n), ks=(20,), targets=tg) for n in names]
            per[c] = {}
            for t in TARGETS:
                key = "C_KL@20" if t == "train" else "C_KL@20|" + t
                per[c][t] = np.mean([m[key] for m in ms], 0)
            per[c]["R@20"] = np.mean([m["R@20"] for m in ms], 0)
            res["models"][c] = {t: {"mean": float(np.mean([m["C_KL@20" if t == "train" else "C_KL@20|" + t].mean()
                                                           for m in ms])),
                                    "sd": float(np.std([m["C_KL@20" if t == "train" else "C_KL@20|" + t].mean()
                                                        for m in ms], ddof=1)) if len(ms) > 1 else 0.0}
                                for t in TARGETS}
            rows.append({"dataset": LABEL[ds], "model": c, "n_runs": len(names),
                         **{"C_KL@20 vs " + t: round(res["models"][c][t]["mean"], 4) for t in TARGETS},
                         "R@20": round(float(per[c]["R@20"].mean()), 4)})
            print("%-6s %-14s" % (LABEL[ds], c), "  ".join("%s %.4f" % (t, res["models"][c][t]["mean"])
                                                       for t in TARGETS))
        res["tests"] = {}
        for x, y in (("DCR", "DecRS"), ("DCR", "FMconf blend"), ("DecRS", "FM")):
            res["tests"]["%s vs %s" % (x, y)] = {t: user_stats(per[x][t], per[y][t]) for t in TARGETS}
        gap = np.abs(rec - rec1).sum(1)
        res["recency_gap"] = {"mean": float(gap.mean()), "median": float(np.median(gap))}
        res["corr_dCKL_gap"] = {}
        for x, y in (("DCR", "DecRS"), ("DCR", "FMconf blend")):
            for t in TARGETS:
                d = per[x][t] - per[y][t]
                res["corr_dCKL_gap"]["%s - %s | %s" % (x, y, t)] = {
                    "spearman": float(stats.spearmanr(gap, d).correlation),
                    "pearson": float(np.corrcoef(gap, d)[0, 1]),
                    "dCKL_by_gap_quartile": [float(d[(gap >= lo) & (gap <= hi)].mean()) for lo, hi in
                                             zip(np.quantile(gap, [0, .25, .5, .75]),
                                                 np.quantile(gap, [.25, .5, .75, 1]))]}
        for k, v in res["corr_dCKL_gap"].items():
            print("   corr(gap, dC_KL) %-32s spearman %+.3f  by gap quartile %s" % (
                k, v["spearman"], " ".join("%+.3f" % q for q in v["dCKL_by_gap_quartile"])))
        out["datasets"][ds] = res
    json.dump(out, open(os.path.join(OUT_DIR, "e2c_calibration.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "e2c_calibration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
