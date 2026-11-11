"""Q3 scoring: Goodreads at 300 epochs, under the rule fixed before the runs.

Seed 1 only, per-user paired bootstrap (B=2000), +-2% margin, verdict class unadjusted, for the five
pre-stated comparisons; 200-epoch arrays come from the audit cache, 300-epoch ones from gr300/topk_goodreads.
Primary metric R@20 (the Goodreads tables' metric); R@10, N@20 and C_KL@20 are shown for completeness.
"""
import json
import os
import sys

import numpy as np

AN = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.insert(0, os.path.join(AN, "eval_package"))
sys.path.insert(0, os.path.join(AN, "audit"))
from common import Data, per_user, ckl
from stats import compare, verdict

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.join(HERE, "gr300", "topk_goodreads")
D = Data("goodreads")
z = np.load(os.path.join(AN, "audit", "cache", "goodreads.npz"))

# label -> (audit-cache label, 300-epoch top-k file)
LAB = {"FM": ("FM", "FM"), "DecRS": ("DecRS", "DecRS"), "TDecRS": ("TDecRS", "DCR_g0.94"),
       "FMconf_blend": ("FMconf_blend", "FMconf_blend"), "DecRS-W": ("DecRS-W", "DecRS-W"),
       "DecRS-pool": ("DecRS-pool", "DecRS-pool"), "ENS": ("ENS", "ENS_FM_s2+FM_s1")}
METRICS = ["R@20", "R@10", "N@20", "C_KL@20"]
PAIRS = [("DecRS", "FM"), ("TDecRS", "FMconf_blend"), ("DecRS-W", "DecRS-pool"),
         ("DecRS-W", "FMconf_blend"), ("DecRS", "ENS")]


def m300(stem):
    f = np.load(os.path.join(TOP, stem + ".npz"))
    top = dict(zip(f["users"].tolist(), f["top"]))
    out = {}
    for k in (10, 20):
        R, N, Q = per_user(D, top, k)
        out["R@%d" % k], out["N@%d" % k] = R, N
        if k == 20:
            out["C_KL@20"] = ckl(D.his_g, Q)
    return out


new = {lab: m300(stem) for lab, (_, stem) in LAB.items()}
old = {lab: {k: z["%s|1|%s" % (c, k)] for k in METRICS} for lab, (c, _) in LAB.items()}

res = {"means": {}, "pairs": {}}
for lab in LAB:
    res["means"][lab] = {k: {"200": float(old[lab][k].mean()), "300": float(new[lab][k].mean())} for k in METRICS}
print("%-13s" % "model" + "".join("%18s" % k for k in METRICS))
for lab in LAB:
    print("%-13s" % lab + "".join("   %.4f -> %.4f" % (old[lab][k].mean(), new[lab][k].mean()) for k in METRICS))

robust = True
for a, b in PAIRS:
    key = "%s_vs_%s" % (a, b)
    res["pairs"][key] = {}
    for k in METRICS:
        r2, r3 = compare(old[a][k], old[b][k]), compare(new[a][k], new[b][k])
        v2, v3 = verdict(r2), verdict(r3)
        same = (np.sign(r2["d"]) == np.sign(r3["d"])) and v2.split(" ")[0] == v3.split(" ")[0]
        if k == "R@20":
            robust &= bool(same)
        res["pairs"][key][k] = {"200": {"d": r2["d"], "ci95": r2["ci95"], "ci90": r2["ci90"], "p": r2["p"],
                                        "p_tost": r2["p_tost"], "verdict": v2},
                                "300": {"d": r3["d"], "ci95": r3["ci95"], "ci90": r3["ci90"], "p": r3["p"],
                                        "p_tost": r3["p_tost"], "verdict": v3}, "same_class": bool(same)}
        print("%-26s %-8s 200: %+5.1f%% [%+5.1f,%+5.1f] %-14s 300: %+5.1f%% [%+5.1f,%+5.1f] %-14s %s" % (
            key, k, r2["d"], *r2["ci95"], v2[:14], r3["d"], *r3["ci95"], v3[:14], "same" if same else "CHANGED"))
res["robust_R@20"] = robust
print("robust on R@20:", robust)
json.dump(res, open(os.path.join(HERE, "q3_results.json"), "w"), indent=1)
