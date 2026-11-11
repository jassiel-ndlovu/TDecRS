"""E2f: equivalence tests, seed-level 95% CIs and per-user Wilcoxon for the headline comparisons.

PRE-STATED MARGIN. The equivalence margin is +-2% relative,
fixed before any number below was computed. It is
written to the output and printed before the first comparison is evaluated.

For every comparison x vs y, dataset and metric:
  seed level  per-seed relative difference 100*(x_s/y_s - 1), paired by seed:
              mean, t-based 95% CI, TOST (two one-sided t-tests, df = n_seeds - 1)
  user level  per-user metrics averaged over seeds: Wilcoxon signed-rank p,
              bootstrap 95% CI of the relative difference, and user-level TOST
              (equivalent iff the bootstrap 90% CI lies inside +-2%)
Verdict (seed level): "x better"/"x worse" if the 95% CI excludes 0, "equivalent"
if TOST p < 0.05, both can hold; otherwise "inconclusive".

Comparisons and their status:
  DecRS vs FM           pre-registered (RQ1, replication)
  DCR vs DecRS          pre-registered (H1)
  DCR vs FMconf_blend   post hoc control (Tier 1.5, designed after the Tier 1 result)
  DecRS vs ENS          post hoc control (E2a)

FM: seeds 1-5 (ML-1M) and 1-3 (ML-10M), re-scored, paired with DecRS by seed.
DecRS vs ENS: ML-1M pairs DecRS_s with FM_s in the same slot (seed by seed); on
ML-10M only seeds 2, 3 have an analogue, so the ensemble mean is held fixed there.

    python analysis/eval_package/e2f_stats.py
Output: analysis/eval_package/out/e2f_stats.json, e2f_stats.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DS, LABEL, OUT_DIR, TOST_MARGIN, Data, load_top, metrics, seed_stats, user_stats  # noqa: E402

import numpy as np  # noqa: E402

KEYS = ("R@10", "R@20", "N@10", "N@20", "C_KL@20")
STATUS = {"DecRS_vs_FM": "pre-registered (RQ1)", "DCR_vs_DecRS": "pre-registered (H1)",
          "DCR_vs_FMconf_blend": "post hoc control (Tier 1.5)", "DecRS_vs_ENS": "post hoc control (E2a)"}


def verdict(s):
    if "ci95" not in s:
        return "n/a"
    lo, hi = s["ci95"]
    v = []
    if lo > 0:
        v.append("x higher")
    elif hi < 0:
        v.append("x lower")
    if s.get("equivalent"):
        v.append("equivalent")
    return " & ".join(v) or "inconclusive"


def main():
    pre = {"margin_pct_relative": TOST_MARGIN, "alpha": 0.05,
           "statement": "Equivalence margin +-%.0f%% relative, fixed in execution-plan.md E2f before computing." %
                        TOST_MARGIN}
    print(pre["statement"], flush=True)
    out = {"pre_registration": pre, "status": STATUS, "datasets": {}}
    rows = []
    for ds in DS:
        D = Data(ds)
        seeds = DS[ds]["seeds"]
        names = {c: ["%s_s%d" % (c, s) for s in seeds] for c in ("DecRS", "DCR_g0.3", "FMconf_blend")}
        if ds == "ml_1m":
            fm_ckpt = ["FM_s%d" % s for s in seeds]
            ens = ["ENS_FM_s%d+FM_A" % s for s in seeds]
        else:
            fm_ckpt = ["FM_s%d" % s for s in seeds]
            ens = ["ENS_FM_s2+FM_s1", "ENS_FM_s3+FM_s1"]
        per = {n: metrics(D, load_top(ds, n)) for n in sum(names.values(), []) + fm_ckpt + ens}
        seedv = {c: {k: [float(per[n][k].mean()) for n in v] for k in KEYS} for c, v in names.items()}
        useru = {c: {k: np.mean([per[n][k] for n in v], 0) for k in KEYS} for c, v in names.items()}
        useru["FM"] = {k: np.mean([per[n][k] for n in fm_ckpt], 0) for k in KEYS}
        useru["ENS"] = {k: np.mean([per[n][k] for n in ens], 0) for k in KEYS}
        seedv["FM"] = {k: [float(per[n][k].mean()) for n in fm_ckpt] for k in KEYS}
        if ds == "ml_1m":   # matched ensemble seed by seed
            seedv["ENS"] = {k: [float(per[n][k].mean()) for n in ens] for k in KEYS}
        else:
            ens_m = {k: float(np.mean([per[n][k].mean() for n in ens])) for k in KEYS}
            seedv["ENS"] = {k: [ens_m[k]] * len(seeds) for k in KEYS}
        res = {}
        for comp, (x, y) in {"DecRS_vs_FM": ("DecRS", "FM"), "DCR_vs_DecRS": ("DCR_g0.3", "DecRS"),
                             "DCR_vs_FMconf_blend": ("DCR_g0.3", "FMconf_blend"),
                             "DecRS_vs_ENS": ("DecRS", "ENS")}.items():
            res[comp] = {}
            for k in KEYS:
                s = seed_stats(seedv[x][k], seedv[y][k])
                u = user_stats(useru[x][k], useru[y][k])
                res[comp][k] = {"seed": s, "user": u, "verdict": verdict(s)}
                rows.append({"dataset": LABEL[ds], "comparison": comp.replace("_vs_", " vs "),
                             "status": STATUS[comp], "metric": k, "rel_pct": round(s["rel_pct"], 2),
                             "ci95_lo": round(s["ci95"][0], 2), "ci95_hi": round(s["ci95"][1], 2),
                             "p_tost_seed": round(s["p_tost"], 4), "p_wilcoxon_user": u["p_wilcoxon"],
                             "user_equivalent": u["equivalent"], "verdict": res[comp][k]["verdict"]})
                print("%-6s %-20s %-8s %+6.2f%% [%+.2f, %+.2f]  TOST p=%.3f  Wilcoxon p=%.2g  -> %s" % (
                    LABEL[ds], comp, k, s["rel_pct"], s["ci95"][0], s["ci95"][1], s["p_tost"],
                    u["p_wilcoxon"], res[comp][k]["verdict"]), flush=True)
        out["datasets"][ds] = {"comparisons": res, "seed_values": seedv}
    json.dump(out, open(os.path.join(OUT_DIR, "e2f_stats.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "e2f_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
