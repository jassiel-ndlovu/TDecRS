"""E2e: H2 drift z-quartiles, and the null-calibrated Eq. 11 weight (inference only).

Pre-registered H2: the DCR - DecRS gain rises monotonically
across quartiles of the null-calibrated drift z-score (drift_z.py). Users are
split into z-quartiles; within each, the seed-averaged per-user R@20 / N@20 gain
of DCR over DecRS (and, as the information-matched check, over blended FMconf)
is reported with a Wilcoxon p, plus a seed-level paired t-test on the quartile means.
The trend is summarised by Spearman(z, per-user gain) and by whether the four
quartile gains are monotone increasing. Raw-KL quartiles are shown for contrast.

Null-calibrated lambda: Eq. 11 with kl_u replaced by z_u,
    lam_u = ((z_u - min z) / (max z - min z)) ** 0.2,
applied to the unchanged DecFM (DecRSz) and DCR (DCRz) checkpoints -- no retraining.

    python analysis/eval_package/e2e_h2.py
Output: analysis/eval_package/out/e2e_h2.json, e2e_quartiles.csv, e2e_zblend.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DS, LABEL, OUT_DIR, Data, load_top, metrics, seed_stats, user_stats  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

KEYS = ("R@10", "R@20", "N@10", "N@20", "C_KL@20")


def main():
    out = {"datasets": {}}
    qrows, zrows = [], []
    for ds in DS:
        D = Data(ds)
        seeds = DS[ds]["seeds"]
        z = np.load(os.path.join(OUT_DIR, "drift_z_%s.npz" % ds))
        zd = dict(zip(z["users"].tolist(), z["z"]))
        kd = dict(zip(z["users"].tolist(), z["kl"]))
        zu = np.array([zd[u] for u in D.users])
        ku = np.array([kd[u] for u in D.users])
        cells = {c: ["%s_s%d" % (c, s) for s in seeds] for c in
                 ("DecRS", "DCR_g0.3", "FMconf_blend", "DecRSz", "DCRz")}
        per = {c: [metrics(D, load_top(ds, n)) for n in v] for c, v in cells.items()}
        res = {"z_summary": {"share_z_gt_1.96": float((zu > 1.96).mean()), "median": float(np.median(zu)),
                             "spearman_z_kl": float(stats.spearmanr(zu, ku).correlation)},
               "quartiles": {}, "trend": {}, "zblend": {}}
        for score_name, sc in (("z", zu), ("raw_kl", ku)):
            qs = np.quantile(sc, [0.25, 0.5, 0.75])
            lab = np.digitize(sc, qs, right=True)
            res["quartiles"][score_name] = {}
            for ref in ("DecRS", "FMconf_blend"):
                rows_ = []
                for m in ("R@20", "N@20"):
                    gains = []
                    for q in range(4):
                        msk = lab == q
                        a = [r[m][msk].mean() for r in per["DCR_g0.3"]]
                        b = [r[m][msk].mean() for r in per[ref]]
                        ua = np.mean([r[m][msk] for r in per["DCR_g0.3"]], 0)
                        ub = np.mean([r[m][msk] for r in per[ref]], 0)
                        s = seed_stats(a, b)
                        u = user_stats(ua, ub)
                        gains.append(s["rel_pct"])
                        row = {"dataset": LABEL[ds], "split": score_name, "reference": ref, "metric": m,
                               "quartile": "Q%d" % (q + 1), "n_users": int(msk.sum()),
                               "range": "[%.2f, %.2f]" % (sc[msk].min(), sc[msk].max()),
                               "gain_pct": round(s["rel_pct"], 2), "ci95_lo": round(s["ci95"][0], 2),
                               "ci95_hi": round(s["ci95"][1], 2), "p_seed_t": round(s["p_t"], 4),
                               "p_wilcoxon": u["p_wilcoxon"]}
                        rows_.append(row)
                        qrows.append(row)
                    gain_u = (np.mean([r[m] for r in per["DCR_g0.3"]], 0) - np.mean([r[m] for r in per[ref]], 0))
                    res["trend"]["%s|%s|%s" % (score_name, ref, m)] = {
                        "quartile_gain_pct": gains,
                        "monotone_increasing": bool(all(np.diff(gains) > 0)),
                        "spearman_score_gain": float(stats.spearmanr(sc, gain_u).correlation),
                        "spearman_p": float(stats.spearmanr(sc, gain_u).pvalue)}
                res["quartiles"][score_name][ref] = rows_
        for k, v in res["trend"].items():
            print("%-6s %-30s quartile gains %s  monotone=%s  spearman %+.3f (p=%.2g)" % (
                LABEL[ds], k, " ".join("%+.1f%%" % g for g in v["quartile_gain_pct"]), v["monotone_increasing"],
                v["spearman_score_gain"], v["spearman_p"]))
        for x, y in (("DecRSz", "DecRS"), ("DCRz", "DCR_g0.3")):
            res["zblend"]["%s vs %s" % (x, y)] = {}
            for k in KEYS:
                s = seed_stats([r[k].mean() for r in per[x]], [r[k].mean() for r in per[y]])
                u = user_stats(np.mean([r[k] for r in per[x]], 0), np.mean([r[k] for r in per[y]], 0))
                res["zblend"]["%s vs %s" % (x, y)][k] = {"seed": s, "user": u}
                zrows.append({"dataset": LABEL[ds], "comparison": "%s vs %s" % (x, y), "metric": k,
                              "x": round(s["x_mean"], 4), "y": round(s["y_mean"], 4),
                              "rel_pct": round(s["rel_pct"], 2), "ci95_lo": round(s["ci95"][0], 2),
                              "ci95_hi": round(s["ci95"][1], 2), "p_seed_t": round(s["p_t"], 4),
                              "p_wilcoxon": u["p_wilcoxon"], "equivalent_seed_tost": s["equivalent"]})
                print("%-6s %-18s %-8s %.4f vs %.4f  %+.2f%% [%+.2f, %+.2f]  Wilcoxon p=%.2g" % (
                    LABEL[ds], "%s vs %s" % (x, y), k, s["x_mean"], s["y_mean"], s["rel_pct"], s["ci95"][0],
                    s["ci95"][1], u["p_wilcoxon"]))
        out["datasets"][ds] = res
    json.dump(out, open(os.path.join(OUT_DIR, "e2e_h2.json"), "w"), indent=1)
    for name, rows in (("e2e_quartiles.csv", qrows), ("e2e_zblend.csv", zrows)):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
