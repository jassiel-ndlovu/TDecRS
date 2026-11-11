"""E2a: is DecRS's gain over FM just a two-model Eq. 11 ensemble?

Every Dec* ranking is  lam_u * sigmoid(DecFM) + (1 - lam_u) * sigmoid(FM)  (Eq. 11).
The control puts an independently trained FM in DecFM's slot, with the same
lam_u (raw KL, alpha = 0.2) and the same partner FM:

  ML-1M   partner = the authors' released FM scores (as every ML-1M Dec* run).
          FM seeds 1-5 (fmseeds 1-4 retrained with tagged names, logs identical to the
          lost baseseeds 1-4; seed 5 = the surviving baseseeds file), each put in the
          slot = 5 matched ensembles. "All members" adds the 5 trained pairs
          FM_s -> FM_s+1 (partner a trained seed) and three sweep FMs (other lr /
          weight decay).
  ML-10M  partner = FM seed 1 (as every ML-10M Dec* run); FM seeds 2 and 3 in the
          slot are the exact analogues of DecRS seeds 2 and 3; all 6 ordered pairs
          of FM seeds 1-3 are reported.

"Share of DecRS's gain explained" = (ENS - FM) / (DecRS - FM), FM = trained-FM mean.
Raw DecFM (lam = 1) is also reported.

    python analysis/eval_package/e2a_ensemble.py
Output: analysis/eval_package/out/e2a_ensemble.json, e2a_ensemble.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, DS, LABEL, OUT_DIR, Data, load_top, metrics, user_stats  # noqa: E402

import numpy as np  # noqa: E402

KEYS = ("R@10", "R@20", "N@10", "N@20", "C_KL@20")


def logged_fm():
    """ML-1M FM seeds 1-5, best-epoch test metrics from the training logs."""
    rows = [r for r in csv.DictReader(open(os.path.join(ANALYSIS, "tier1_runs.csv")))
            if r["job"] == "baseseeds" and r["model"] == "FM"]
    m = {"R@10": "Recall@10", "R@20": "Recall@20", "N@10": "NDCG@10", "N@20": "NDCG@20"}
    return {k: [float(r[v]) for r in sorted(rows, key=lambda r: int(r["seed"]))] for k, v in m.items()}


def main():
    out = {"alpha": 0.2, "datasets": {}}
    rows = []
    for ds in DS:
        D = Data(ds)
        seeds = DS[ds]["seeds"]
        if ds == "ml_1m":
            fm_members = ["FM_s%d" % s for s in seeds]
            ens = {"ENS_FM_s%d+FM_A" % s: "FM_s%d" % s for s in seeds}
            extra = {"ENS_FM_s%d+FM_s%d" % (s, s % 5 + 1): "FM_s%d" % s for s in seeds}
            extra.update({"ENS_FM_sw1+FM_A": "FM_sw1", "ENS_FM_sw2+FM_A": "FM_sw2", "ENS_FM_sw3+FM_A": "FM_sw3"})
            singles = ["FM_A"] + fm_members + ["FM_sw1", "FM_sw2", "FM_sw3"]
        else:
            fm_members = ["FM_s%d" % s for s in seeds]
            ens = {"ENS_FM_s2+FM_s1": "FM_s2", "ENS_FM_s3+FM_s1": "FM_s3"}
            extra = {"ENS_FM_s%d+FM_s%d" % (a, b): "FM_s%d" % a for a in seeds for b in seeds
                     if a != b and b != 1}
            singles = fm_members
        groups = {"DecRS": ["DecRS_s%d" % s for s in seeds], "DecFM_raw": ["DecFM_raw_s%d" % s for s in seeds]}
        per = {}
        for n in singles + list(ens) + list(extra) + groups["DecRS"] + groups["DecFM_raw"]:
            per[n] = metrics(D, load_top(ds, n))
        res = {"models": {n: {k: float(per[n][k].mean()) for k in KEYS} for n in per}}
        # trained-FM reference
        fm_ref = {k: float(np.mean([res["models"][f][k] for f in fm_members])) for k in KEYS}
        if ds == "ml_1m":
            lg = logged_fm()
            res["FM_logged_5seeds"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": v}
                                       for k, v in lg.items()}
        res["FM_reference"] = fm_ref
        summ = {}
        for g, names in (("DecRS", groups["DecRS"]), ("DecFM_raw", groups["DecFM_raw"]),
                         ("ENS (matched)", list(ens)), ("ENS (all pairs)", list(ens) + list(extra))):
            summ[g] = {k: {"mean": float(np.mean([res["models"][n][k] for n in names])),
                           "sd": float(np.std([res["models"][n][k] for n in names], ddof=1))
                           if len(names) > 1 else 0.0, "n": len(names)} for k in KEYS}
        res["summary"] = summ
        share = {}
        for k in KEYS:
            dec, en, f = summ["DecRS"][k]["mean"], summ["ENS (matched)"][k]["mean"], fm_ref[k]
            share[k] = {"DecRS_vs_FM_pct": 100 * (dec / f - 1), "ENS_vs_FM_pct": 100 * (en / f - 1),
                        "DecRS_vs_ENS_pct": 100 * (dec / en - 1),
                        "share_explained": (en - f) / (dec - f) if dec != f else float("nan")}
        res["gain_decomposition"] = share
        # per-user: seed-averaged DecRS vs matched-ensemble average, and vs each ensemble member's own FM
        dec_u = {k: np.mean([per[n][k] for n in groups["DecRS"]], 0) for k in KEYS}
        ens_u = {k: np.mean([per[n][k] for n in ens], 0) for k in KEYS}
        fm_u = {k: np.mean([per[n][k] for n in fm_members], 0) for k in KEYS}
        raw_u = {k: np.mean([per[n][k] for n in groups["DecFM_raw"]], 0) for k in KEYS}
        res["user_tests"] = {
            "DecRS_vs_ENS": {k: user_stats(dec_u[k], ens_u[k]) for k in KEYS},
            "ENS_vs_FM": {k: user_stats(ens_u[k], fm_u[k]) for k in KEYS},
            "DecRS_vs_FM": {k: user_stats(dec_u[k], fm_u[k]) for k in KEYS},
            "DecFM_raw_vs_FM": {k: user_stats(raw_u[k], fm_u[k]) for k in KEYS},
        }
        # exact slot analogues, seed by seed (ML-10M seed 1's analogue would be FM_s1 with itself)
        res["slot_matched"] = {s: {k: 100 * (res["models"]["DecRS_s%d" % s][k] /
                                             res["models"][e][k] - 1) for k in KEYS}
                               for s, e in ((int(e.split("+")[0][8:]), e) for e in ens)}
        out["datasets"][ds] = res
        for n, v in res["models"].items():
            rows.append({"dataset": LABEL[ds], "variant": n, **{k: round(v[k], 5) for k in KEYS}})
        print("\n== %s  (FM reference: %s)" % (LABEL[ds], {k: round(v, 4) for k, v in fm_ref.items()}))
        for g, v in summ.items():
            print("  %-16s" % g, "  ".join("%s %.4f" % (k, v[k]["mean"]) for k in KEYS))
        for k in KEYS:
            s = share[k]
            print("  %-8s DecRS vs FM %+.1f%%  ENS vs FM %+.1f%%  DecRS vs ENS %+.1f%%  share %.2f  "
                  "(user p DecRS vs ENS %.2g)" % (k, s["DecRS_vs_FM_pct"], s["ENS_vs_FM_pct"],
                                                  s["DecRS_vs_ENS_pct"], s["share_explained"],
                                                  res["user_tests"]["DecRS_vs_ENS"][k]["p_wilcoxon"]))
    json.dump(out, open(os.path.join(OUT_DIR, "e2a_ensemble.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "e2a_ensemble.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
