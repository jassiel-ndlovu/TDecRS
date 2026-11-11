"""D2: power of the headline comparisons on R@20.

For each comparison x vs y and dataset, from the existing top-100 lists:
  user level  d_u = x_u - y_u, with x_u, y_u the per-user R@20 averaged over the seeds
              both arms share; n = evaluated users. MDE at 80% power, alpha = .05 two-sided:
              (z_.975 + z_.8) * sd(d) / sqrt(n), as % of mean(y). 90% CI half-width:
              z_.95 * sd(d) / sqrt(n), same units. This ignores training-seed variance,
              so it is the most optimistic case.
  seed level  r_s = 100 (x_s / y_s - 1), paired by seed; n = seeds. MDE: the mean r at which
              a one-sample t-test (df = n - 1) has 80% power, from the noncentral t (exact).
              90% CI half-width: t_.95,n-1 * sd(r) / sqrt(n).
Reading (per level): "equivalent" = 90% CI inside +-2%, "difference" = 95%-style CI
excluding 0 is approximated by the 90% CI here (one-sided 5%), "underpowered" = neither.
The seed-level CI is centred on the mean per-seed relative difference.
A null result is informative (an equivalence result at the +-2% TOST margin is
reachable) only when the 90% half-width is below 2% minus |observed|; the MDE says
what effect the design could have found.

Arms:
  TDecRS        DCR, Eq. 11 blend: g0.3 (ML-1M, ML-10M), g0.94 = gamma* (Goodreads)
  FMconf_blend  FMconf with the same confounder, Eq. 11 blend
  DecRS-R       DecFMR, Eq. 11 blend (analysis/topk_decrsr*, analysis/topk_goodreads)
  FM+FM         E2a control, FM seed s in DecFM's slot: ML-1M partner = authors' FM, seeds
                1-5; ML-10M and Goodreads partner = FM seed 1, seeds 2-3 only (n = 2)

    python analysis/eval_package/diag/d2_power.py
Output: analysis/eval_package/diag/out/d2_power.json, d2_power.csv
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import ANALYSIS, LABEL, TOPK_DIR, TOST_MARGIN, Data, per_user  # noqa: E402

import numpy as np  # noqa: E402
from scipy import optimize, stats  # noqa: E402

OUT = os.path.join(HERE, "out")
EP = lambda ds, n: os.path.join(TOPK_DIR, ds, n + ".npz")          # noqa: E731
GR = lambda n: os.path.join(ANALYSIS, "topk_goodreads", n + ".npz")  # noqa: E731


def arms(ds):
    """arm -> {seed: path}"""
    if ds == "goodreads":
        sx = lambda s: "" if s == 1 else "_s%d" % s                   # noqa: E731
        a = {k: {s: GR(f + sx(s)) for s in (1, 2, 3)} for k, f in
             (("TDecRS", "DCR_g0.94"), ("FMconf_blend", "FMconf_blend"), ("DecRS", "DecRS"),
              ("FM", "FM"), ("DecRS-R", "DecRS-R"))}
        a["FM+FM"] = {s: os.path.join(HERE, "topk", "goodreads", "ENS_FM_s%d+FM_s1.npz" % s) for s in (2, 3)}
        return a
    seeds = (1, 2, 3, 4, 5) if ds == "ml_1m" else (1, 2, 3)
    a = {k: {s: EP(ds, "%s_s%d" % (f, s)) for s in seeds} for k, f in
         (("TDecRS", "DCR_g0.3"), ("FMconf_blend", "FMconf_blend"), ("DecRS", "DecRS"), ("FM", "FM"))}
    rd = "topk_decrsr" if ds == "ml_1m" else "topk_decrsr_ml10m"
    a["DecRS-R"] = {s: os.path.join(ANALYSIS, rd, "DecRSR_s%d.npz" % s) for s in seeds}
    a["FM+FM"] = ({s: EP(ds, "ENS_FM_s%d+FM_A" % s) for s in seeds} if ds == "ml_1m" else
                  {s: EP(ds, "ENS_FM_s%d+FM_s1" % s) for s in (2, 3)})
    return a


COMPS = (("TDecRS", "FMconf_blend"), ("TDecRS", "DecRS"), ("DecRS", "FM"), ("DecRS", "FM+FM"),
         ("DecRS-R", "FMconf_blend"), ("TDecRS", "FM+FM"), ("FMconf_blend", "FM+FM"))


def r20(D, path):
    z = np.load(path)
    top = dict(zip(z["users"].tolist(), z["top"]))
    return per_user(D, top, 20)[0]


def seed_mde(sd, n, power=0.8, alpha=0.05):
    """Smallest mean effect a one-sample two-sided t-test (n obs) detects with the given power."""
    if n < 2 or not sd > 0:
        return float("nan")
    df, tc = n - 1, stats.t.ppf(1 - alpha / 2, n - 1)
    f = lambda d: (1 - stats.nct.cdf(tc, df, d * np.sqrt(n) / sd)) - power   # noqa: E731
    return float(optimize.brentq(f, 1e-9, 1e4 * sd))


def main():
    z80, z975, z95 = stats.norm.ppf(0.8), stats.norm.ppf(0.975), stats.norm.ppf(0.95)
    rows, out = [], {"margin_pct": TOST_MARGIN, "metric": "R@20", "power": 0.8, "alpha": 0.05, "rows": []}
    for ds in ("ml_1m", "ml_10m", "goodreads"):
        D = Data(ds)
        A = arms(ds)
        cache = {}
        get = lambda p: cache.setdefault(p, r20(D, p))                  # noqa: E731
        for x, y in COMPS:
            seeds = sorted(set(A[x]) & set(A[y]))
            X = np.stack([get(A[x][s]) for s in seeds])
            Y = np.stack([get(A[y][s]) for s in seeds])
            xu, yu = X.mean(0), Y.mean(0)
            d = xu - yu
            n = len(d)
            se_u = d.std(ddof=1) / np.sqrt(n)
            obs = 100 * (xu.mean() / yu.mean() - 1)
            r = 100 * (X.mean(1) / Y.mean(1) - 1)
            ns = len(seeds)
            sd_s = float(r.std(ddof=1)) if ns > 1 else float("nan")
            hw_s = float(stats.t.ppf(0.95, ns - 1) * sd_s / np.sqrt(ns)) if ns > 1 else float("nan")
            row = {"dataset": LABEL[ds], "comparison": "%s vs %s" % (x, y), "seeds": seeds,
                   "n_users": n, "n_seeds": ns, "obs_rel_pct": float(obs),
                   "per_seed_rel_pct": [float(v) for v in r],
                   "user_sd_abs": float(d.std(ddof=1)), "y_mean": float(yu.mean()),
                   "user_mde_pct": float(100 * (z975 + z80) * se_u / yu.mean()),
                   "user_hw90_pct": float(100 * z95 * se_u / yu.mean()),
                   "seed_sd_pct": sd_s, "seed_mde_pct": seed_mde(sd_s, ns), "seed_hw90_pct": hw_s}
            row["seed_mean_rel_pct"] = float(r.mean())
            for lv, ctr in (("user", obs), ("seed", float(r.mean()))):
                hw = row[lv + "_hw90_pct"]
                lo, hi = ctr - hw, ctr + hw
                eq, df_ = -TOST_MARGIN < lo and hi < TOST_MARGIN, lo > 0 or hi < 0
                row[lv + "_reading"] = ("equivalent+difference" if eq and df_ else "equivalent" if eq
                                        else "difference" if df_ else "underpowered")
                if not hw == hw:
                    row[lv + "_reading"] = "n/a"
            out["rows"].append(row)
            rows.append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                         if k not in ("per_seed_rel_pct", "seeds")})
            print("%-9s %-26s obs %+6.2f%% | user n=%5d MDE %5.2f%% hw90 %5.2f%% %-21s | seed n=%d mean %+6.2f%% "
                  "sd %5.2f MDE %6.2f%% hw90 %6.2f%% %s" % (
                      LABEL[ds], row["comparison"], obs, n, row["user_mde_pct"], row["user_hw90_pct"],
                      row["user_reading"], ns, r.mean(), sd_s, row["seed_mde_pct"], hw_s, row["seed_reading"]),
                  flush=True)
    json.dump(out, open(os.path.join(OUT, "d2_power.json"), "w"), indent=1)
    with open(os.path.join(OUT, "d2_power.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
