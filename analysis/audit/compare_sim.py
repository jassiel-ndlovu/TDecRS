"""Simulator audit: re-score the frozen P1-P4 rows and the regime table (PROTOCOL.md).

1. as registered: sim_eval.rel_diff (two-level seed x user bootstrap) + sim_eval.verdict, fresh RNG
2. unified: two-component t interval (stats.compare), same frozen verdict rules applied to its CIs,
   plus Holm within each prediction (P1, P2, P3, P4) on p and p_tost
3. regime table: every model vs FM per world on R_int, unified, Holm over the 48 cells
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, AN)
import sim_eval as SE  # noqa: E402
from stats import compare, holm  # noqa: E402

SEEDS = [1, 2, 3]
z = [np.load(os.path.join(AN, "sim", "sim_eval_per_user_seed%d.npz" % s)) for s in SEEDS]


def arr(world, model, metric):
    return np.stack([zz["%s|%s|%s" % (world, model, metric)] for zz in z])


def frozen_rule(op, d, ci95, ci90, m=2.0):
    """sim_eval.verdict on percent units."""
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    excl0 = lo95 > 0 or hi95 < 0
    if op == "~":
        if lo90 >= -m and hi90 <= m:
            return "confirmed"
        return "refuted" if (abs(d) > m and excl0) else "inconclusive"
    if op == ">":
        if d > m and lo95 > 0:
            return "confirmed"
        return "refuted" if (d < -m and hi95 < 0) else "inconclusive"
    if op == ">=":
        return "refuted" if (d < -m and hi95 < 0) else "confirmed"


def holm_rule(op, r, p, pt, m=2.0):
    """The frozen rule with Holm-adjusted significance: 'CI excludes 0' -> p_holm < .05, TOST -> p_tost_holm < .05."""
    d = r["d"]
    if op == "~":
        if pt < 0.05:
            return "confirmed"
        return "refuted" if (abs(d) > m and p < 0.05) else "inconclusive"
    if op == ">":
        if d > m and p < 0.05:
            return "confirmed"
        return "refuted" if (d < -m and p < 0.05) else "inconclusive"
    if op == ">=":
        return "refuted" if (d < -m and p < 0.05) else "confirmed"


js = json.load(open(os.path.join(AN, "sim", "sim_eval.json")))
rng = np.random.RandomState(0)
rows = []
for r0 in js["predictions"]:
    a, op, b = r0["rel"].split(" ")
    A, B = arr(r0["world"], a, r0["metric"]), arr(r0["world"], b, r0["metric"])
    pt, ci = SE.rel_diff(A, B, rng)
    u = compare(A, B)
    rows.append({"pred": r0["pred"].split(":")[0], "world": r0["world"], "rel": r0["rel"], "metric": r0["metric"],
                 "op": op, "stored": r0["verdict"], "stored_diff": 100 * r0["diff"],
                 "frozen": SE.verdict(op, pt, ci), "frozen_diff": 100 * pt,
                 "unified": frozen_rule(op, u["d"], u["ci95"], u["ci90"]), "u": u})
for P in ("P1", "P2", "P3", "P4"):
    fam = [r for r in rows if r["pred"] == P]
    ph, pth = holm([r["u"]["p"] for r in fam]), holm([r["u"]["p_tost"] for r in fam])
    for r, x, y in zip(fam, ph, pth):
        r["unified_holm"] = holm_rule(r["op"], r["u"], x, y)

out = ["Simulator predictions: confirmed / refuted / inconclusive"]
for P in ("P1", "P2", "P3", "P4"):
    fam = [r for r in rows if r["pred"] == P]
    line = "  %s n=%d" % (P, len(fam))
    for key in ("stored", "frozen", "unified", "unified_holm"):
        c = Counter(r[key] for r in fam)
        line += "   %s %d/%d/%d" % (key, c["confirmed"], c["refuted"], c["inconclusive"])
    out.append(line)
flips = [r for r in rows if r["unified_holm"] != r["stored"]]
out.append("\nRows whose verdict changes (stored -> unified+Holm): %d" % len(flips))
for r in flips:
    u = r["u"]
    out.append("  %-3s %-14s %-6s %-30s %+.1f%% [%+.1f,%+.1f] seeds %s  %s -> %s" % (
        r["pred"], r["world"][4:], r["metric"], r["rel"], u["d"], *u["ci95"],
        "/".join("%+.1f" % x for x in u["per_seed"]), r["stored"], r["unified_holm"]))

# regime table
models = ["DecRS", "windowed", "FMconf", "FMconf_blend", "TDecRS0", "DecRS-R"]
cells = []
for w in SE.WORLDS:
    for m in models:
        cells.append((w, m, compare(arr(w, m, "R_int"), arr(w, "FM", "R_int"))))
ph = holm([c[2]["p"] for c in cells])
out.append("\nRegime table: interventional R@20, %% vs FM (3 seeds; * = Holm p<.05 over 48 cells; stored value)")
out.append("%-14s" % "world" + "".join("%22s" % m for m in models))
tab = {}
for i, w in enumerate(SE.WORLDS):
    s = "%-14s" % w[4:]
    for j, m in enumerate(models):
        k = i * len(models) + j
        r = cells[k][2]
        r["p_holm"] = ph[k]
        st = 100 * js["summary"][w][m]["adv_int_vs_FM"]
        s += "%22s" % ("%+.1f%s [%+.1f,%+.1f] (%+.1f)" % (r["d"], "*" if ph[k] < 0.05 else " ", *r["ci95"], st))[:22]
        tab["%s|%s" % (w, m)] = r
    out.append(s)
nonsig = [(w, m, r["d"], r["p_holm"]) for (w, m, r) in cells if r["p_holm"] >= 0.05]
out.append("\nCells NOT significant after Holm: " + "; ".join("%s %s %+.1f%% (p_holm %.2f)" % (w[4:], m, d, p)
                                                            for w, m, d, p in nonsig))
os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
json.dump({"predictions": rows, "regime": tab}, open(os.path.join(HERE, "out", "sim.json"), "w"), indent=1, default=float)
open(os.path.join(HERE, "out", "sim.txt"), "w").write("\n".join(out))
print("\n".join(out))
