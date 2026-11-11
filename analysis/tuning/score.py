"""Score the tuned winners under the frozen protocol (PROTOCOL.md, "Scoring" and "Scoring record").

Alpha is selected on validation R@10 at seed 1 first and written to alpha_selected.json; test metrics are
read after that. Each comparison is scored tuned and at D (seeds 1-3, same Holm family). Goodreads also gets
the 300-epoch seed-1 D reference (per user, no Holm, as Q3).

    python analysis/tuning/score.py
Output: analysis/tuning/alpha_selected.json, tuning_results.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AN, "eval_package"))
sys.path.insert(0, os.path.join(AN, "audit"))
from common import Data, per_user, ckl  # noqa: E402
from stats import compare, family, verdict  # noqa: E402

METRICS = ["R@20", "R@10", "N@20", "C_KL@20"]
SEEDS = (1, 2, 3)
COMMON = [("DecRS", "FM"), ("TDecRS", "FMconf_blend"), ("DecRS", "ENS"), ("TDecRS", "DecRS"), ("FMconf_blend", "DecRS")]
PAIRS = {"ml_1m": COMMON,
         "goodreads": COMMON + [("DecRS-W", "DecRS-pool"), ("DecRS-W", "FMconf_blend")]}
# audit-cache label -> rerank class
CLS = {"DecRS": "DecRS", "TDecRS": "TDecRS", "FMconf_blend": "FMconf", "ENS": "ENS",
       "DecRS-W": "windowed", "DecRS-pool": "pooled"}
D300 = {"FM": "FM", "DecRS": "DecRS", "TDecRS": "DCR_g0.94", "FMconf_blend": "FMconf_blend",
        "DecRS-W": "DecRS-W", "DecRS-pool": "DecRS-pool", "ENS": "ENS_FM_s2+FM_s1"}


def metrics(D, f):
    z = np.load(f)
    top = dict(zip(z["users"].tolist(), z["top"]))
    out = {}
    for k in (10, 20):
        R, N, Q = per_user(D, top, k)
        out["R@%d" % k], out["N@%d" % k] = R, N
        if k == 20:
            out["C_KL@20"] = ckl(D.his_g, Q)
    return out


def cls_of(v):
    return v.split(" ")[0]


def keep(r):
    return {k: r[k] for k in ("d", "ci95", "p", "p_tost", "per_seed", "a", "b", "S", "seeds_same_sign") if k in r} | {
        k: r[k] for k in ("p_holm", "p_tost_holm", "verdict") if k in r}


# ---- alpha, validation only ----
sel = {}
for ds in PAIRS:
    v = json.load(open(os.path.join(HERE, "rerank_%s.json" % ds)))["valid_R@10"]
    own = {}
    for c in [c for c in ("DecRS", "TDecRS", "FMconf", "ENS", "windowed", "pooled") if "%s_a0.2_s1" % c in v]:
        a, b = v["%s_a0.2_s1" % c], v["%s_a0.5_s1" % c]
        own[c] = {"valid_a0.2": a, "valid_a0.5": b, "alpha": 0.5 if b > a else 0.2}
    chosen = {c: own[c]["alpha"] for c in own}
    for c in ("windowed", "pooled"):
        if c in chosen:
            chosen[c] = chosen["DecRS"]        # Scoring record, point 1
    sel[ds] = {"own": own, "chosen": chosen}
    print(ds, {c: (round(x["valid_a0.2"], 4), round(x["valid_a0.5"], 4), chosen[c]) for c, x in own.items()})
json.dump(sel, open(os.path.join(HERE, "alpha_selected.json"), "w"), indent=1)

# ---- test ----
res = {"alpha": sel, "means": {}, "pairs": {}}
for ds, pairs in PAIRS.items():
    D = Data(ds)
    TK = os.path.join(HERE, "topk", ds)
    tuned = {}
    for lab, c in CLS.items():
        if c in sel[ds]["chosen"]:
            tuned[lab] = {s: metrics(D, os.path.join(TK, "%s_a%g_s%d.npz" % (c, sel[ds]["chosen"][c], s))) for s in SEEDS}
    tuned["FM"] = {s: metrics(D, os.path.join(TK, "FM_raw_s%d.npz" % s)) for s in SEEDS}
    tuned["FMconf_raw"] = {s: metrics(D, os.path.join(TK, "FMconf_raw_s%d.npz" % s)) for s in SEEDS}
    z = np.load(os.path.join(AN, "audit", "cache", ds + ".npz"))
    dcell = {lab: {s: {k: z["%s|%d|%s" % (lab, s, k)] for k in METRICS} for s in SEEDS} for lab in tuned}
    d300 = None
    if ds == "goodreads":
        d300 = {lab: metrics(D, os.path.join(AN, "topk_goodreads_300ep", stem + ".npz")) for lab, stem in D300.items()}

    def arr(M, lab, k):
        return np.stack([M[lab][s][k] for s in SEEDS])

    res["means"][ds] = {lab: {k: {"tuned": float(arr(tuned, lab, k).mean()), "D": float(arr(dcell, lab, k).mean()),
                                  "tuned_per_seed": arr(tuned, lab, k).mean(1).tolist(),
                                  "D300_s1": float(d300[lab][k].mean()) if d300 and lab in d300 else None}
                            for k in METRICS} for lab in tuned}
    res["pairs"][ds] = {}
    for k in METRICS:
        T = [("%s - %s" % p, compare(arr(tuned, p[0], k), arr(tuned, p[1], k))) for p in pairs]
        O = [("%s - %s" % p, compare(arr(dcell, p[0], k), arr(dcell, p[1], k))) for p in pairs]
        family(T), family(O)
        row = {}
        for (lab, t), (_, o), p in zip(T, O, pairs):
            same = bool(np.sign(t["d"]) == np.sign(o["d"]) and cls_of(t["verdict"]) == cls_of(o["verdict"]))
            row[lab] = {"tuned": keep(t), "D": keep(o), "robust": same}
            if d300:
                r3 = compare(d300[p[0]][k], d300[p[1]][k])
                r3["verdict"] = verdict(r3)
                row[lab]["D300_s1"] = keep(r3)
                row[lab]["robust_vs_D300"] = bool(np.sign(t["d"]) == np.sign(r3["d"])
                                                  and cls_of(t["verdict"]) == cls_of(r3["verdict"]))
            print("%-9s %-8s %-28s D %+6.1f [%+5.1f,%+5.1f] %-13s tuned %+6.1f [%+5.1f,%+5.1f] p_h=%.3f %-13s %s%s" % (
                ds, k, lab, o["d"], *o["ci95"], cls_of(o["verdict"]), t["d"], *t["ci95"], t["p_holm"],
                cls_of(t["verdict"]), "same" if same else "CHANGED",
                ("  | 300s1 %+5.1f %s" % (r3["d"], cls_of(r3["verdict"]))) if d300 else ""), flush=True)
        res["pairs"][ds][k] = row

# ---- predictions, on R@20 ----
P = res["pairs"]
sig = lambda ds, lab: P[ds]["R@20"][lab]["tuned"]["p_holm"] < 0.05
td = "TDecRS - FMconf_blend"
de = "DecRS - ENS"
hp = {"HP1": {ds: {"significant": sig(ds, td), "d": P[ds]["R@20"][td]["tuned"]["d"]} for ds in PAIRS},
      "HP2": {ds: {"significant_positive": sig(ds, de) and P[ds]["R@20"][de]["tuned"]["d"] > 0,
                   "d": P[ds]["R@20"][de]["tuned"]["d"]} for ds in PAIRS},
      "HP3": {lab: {"significant": sig("ml_1m", lab), "d": P["ml_1m"]["R@20"][lab]["tuned"]["d"]}
              for lab in ("TDecRS - DecRS", "FMconf_blend - DecRS")}}
labs = ["%s - %s" % c for c in COMMON]
hp4 = {}
for ds, ref in (("ml_1m", "D"), ("goodreads", "D"), ("goodreads", "D300_s1")):
    rows = [P[ds]["R@20"][lab] for lab in labs]
    hp4["%s vs %s" % (ds, ref)] = {
        "mean_abs_change_pp": float(np.mean([abs(r["tuned"]["d"] - r[ref]["d"]) for r in rows])),
        "verdict_changes": int(sum(cls_of(r["tuned"]["verdict"]) != cls_of(r[ref]["verdict"]) for r in rows))}
hp["HP4"] = hp4
res["predictions"] = hp
res["robust_R@20"] = {ds: {lab: P[ds]["R@20"][lab]["robust"] for lab in P[ds]["R@20"]} for ds in PAIRS}
print(json.dumps(hp, indent=1))
json.dump(res, open(os.path.join(HERE, "tuning_results.json"), "w"), indent=1)
