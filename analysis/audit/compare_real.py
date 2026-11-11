"""Unified paired comparisons for the real datasets (PROTOCOL.md). Reads audit/cache/<ds>.npz.

Families (Holm within each): one per dataset x metric, over the comparisons listed in PAIRS.
Output: audit/out/real.json and audit/out/real.txt
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from stats import compare, family, fmt  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

PAIRS = {
    "ml_1m": [("TDecRS", "DecRS"), ("TDecRS", "FMconf_blend"), ("DecRS", "FM_A"), ("DecRS", "FM"),
              ("ENS", "FM_A"), ("DecRS", "ENS"), ("FMconf_blend", "FM_A"), ("FMconf_blend", "DecRS"),
              ("DecRS-R", "DecRS"), ("DecRS-R", "FMconf_blend"), ("DecRS-R", "TDecRS"),
              ("DecFM_raw", "FM_A"), ("FMconf_raw", "FM_A"), ("TDecRS", "FM_A"),
              ("ENS", "FM"), ("DecFM_raw", "FM"), ("FMconf_raw", "FM"), ("TDecRS", "ENS")],
    "ml_10m": [("TDecRS", "DecRS"), ("TDecRS", "FMconf_blend"), ("DecRS", "FM"), ("ENS", "FM"), ("DecRS", "ENS"),
               ("FMconf_blend", "DecRS"), ("DecRS-R", "DecRS"), ("DecRS-R", "FMconf_blend"),
               ("DecFM_raw", "FM"), ("FMconf_raw", "FM"), ("TDecRS", "ENS"), ("x_DecRS_t2", "DecRS"), ("x_TDecRS_t2", "TDecRS")],
    "goodreads": [("DecRS", "FM"), ("TDecRS", "DecRS"), ("TDecRS", "FMconf_blend"), ("ENS", "FM"), ("DecRS", "ENS"),
                  ("TDecRS", "ENS"), ("FMconf_blend", "ENS"), ("DecRS-W", "DecRS-pool"), ("DecRS-W", "DecRS"),
                  ("DecRS-W", "FMconf_blend"), ("DecRS-W", "ENS"), ("DecRS-pool", "DecRS"), ("DecRS-pool", "ENS"),
                  ("DecRS-R", "DecRS"), ("FMconf_raw", "FM"), ("FMconf_blend", "DecRS"), ("DecRS-R", "FMconf_blend")],
    "kuairand": [("DecRS-W", "DecRS"), ("DecRS-W", "DecRS-pool"), ("DecRS-W", "FMconf_blend"), ("DecRS-W", "ENS"),
                 ("TDecRS", "DecRS"), ("TDecRS", "DecRS-pool"), ("TDecRS", "FMconf_blend"), ("TDecRS", "ENS"),
                 ("DecRS", "FM"), ("ENS", "FM"), ("DecRS", "ENS"), ("DecRS", "FMconf_blend"), ("DecRS-pool", "DecRS"),
                 ("DecRS-R", "FMconf_blend"), ("FMconf_raw", "FM"), ("FMconf_blend", "FM")],
}
# the report's 2-seed FM+FM (seeds 2,3 in the Dec* slot, partner FM_s1): reproduced separately
ALT = {"goodreads": [("DecRS", "ENS_alt"), ("TDecRS", "ENS_alt"), ("FMconf_blend", "ENS_alt"),
                     ("DecRS-W", "ENS_alt"), ("ENS_alt", "FM")],
       "kuairand": []}
METRICS = {"ml_1m": ["R@20", "R@10", "N@20", "C_KL@20"], "ml_10m": ["R@20", "R@10", "N@20", "C_KL@20"],
           "goodreads": ["R@20", "R@10", "N@10", "N@20", "C_KL@20"],
           "kuairand": ["AUC", "R@20", "R@10", "N@20", "C_KL@20"]}


def load(ds):
    z = np.load(os.path.join(HERE, "cache", ds + ".npz"))
    m = {}
    for k in z.files:
        if k.startswith("_"):
            continue
        lab, s, met = k.split("|")
        m.setdefault(lab, {}).setdefault(met, {})[int(s)] = z[k]
    return m


def stack(m, a, b, met, seeds=None):
    sa, sb = m[a][met], m[b][met]
    common = sorted(set(sa) & set(sb)) if seeds is None else seeds
    if len(set(sa)) == 1 and len(sb) > 1:              # a single-seed reference (FM_A) paired with every seed
        sa = {s: next(iter(sa.values())) for s in sb}
        common = sorted(sb)
    if len(set(sb)) == 1 and len(sa) > 1 and not set(sb) & set(sa) - {1}:
        pass
    return np.stack([sa[s] for s in common]), np.stack([sb[s] for s in common]), common


def run(ds):
    m = load(ds)
    res = {"means": {}, "comparisons": {}}
    for lab, mets in m.items():
        res["means"][lab] = {met: {"mean": float(np.mean([v.mean() for v in by.values()])),
                                   "per_seed": {s: float(v.mean()) for s, v in sorted(by.items())}}
                             for met, by in mets.items()}
    for met in METRICS[ds]:
        absolute = (met == "AUC")
        margin = 0.003 if absolute else 2.0
        rows = []
        for a, b in PAIRS[ds] + ALT.get(ds, []):
            if a not in m or b not in m or met not in m[a]:
                continue
            A, B, seeds = stack(m, a, b, met)
            r = compare(A, B, margin=margin, absolute=absolute)
            r["seeds"] = seeds
            rows.append(("%s vs %s" % (a, b), r))
        main = [x for x in rows if "ENS_alt" not in x[0] and not x[0].startswith("x_")]
        family(main)
        family([x for x in rows if x not in main])
        res["comparisons"][met] = {lab: r for lab, r in rows}
    return res


if __name__ == "__main__":
    dss = sys.argv[1:] or ["ml_1m", "ml_10m", "goodreads", "kuairand"]
    allres = {}
    p = os.path.join(OUT, "real.json")
    if os.path.exists(p):
        allres = json.load(open(p))
    lines = []
    for ds in dss:
        allres[ds] = run(ds)
    for ds, res in allres.items():
        lines.append("=" * 100 + "\n" + ds)
        for lab, mets in res["means"].items():
            lines.append("  %-14s " % lab + "  ".join("%s %.4f (%s)" % (k, v["mean"], "/".join(
                "%.4f" % x for x in v["per_seed"].values())) for k, v in mets.items() if k != "Cov@20")
                + "  Cov@20 %.3f" % mets.get("Cov@20", {"mean": np.nan})["mean"])
        for met, rows in res["comparisons"].items():
            lines.append("-- " + met)
            for lab, r in rows.items():
                lines.append("  %-30s %s   per-seed %s" % (lab, fmt(r, 2), "/".join(
                    ("%+.4f" if r["absolute"] else "%+.1f") % x for x in r["per_seed"])))
    json.dump(allres, open(p, "w"), indent=1)
    open(os.path.join(OUT, "real.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
