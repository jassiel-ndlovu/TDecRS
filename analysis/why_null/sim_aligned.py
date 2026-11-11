"""Why-null Prop. 3 on the simulator: is exposure confounding aligned with preference?

Per user, over the unseen items at test time:
  s_int(i) = logit p_true(i)             (interventional click propensity, truth/)
  b(i)     = beta <d_hist, g(i)>         (the genre part of the logging policy's log-exposure)
  bp(i)    = a log pop_i                 (the popularity part)
  s_obs    = log p_true + b + bp         (log of exposure x click, the observational propensity)
Reports corr(b, s_int), kappa = sd(b)/sd(s_int), Spearman(s_obs, s_int), the top-20 overlap of
the two rankings, and the interventional loss of ranking by s_obs instead of s_int (expected
true clicks in the top 20). Also the Gaussian prediction corr = (1+rho k)/sqrt(1+2 rho k+k^2).

    python sim_aligned.py sim_bH_qL_s0 sim_b0_qL_s0 ...
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.getcwd()
sys.path.insert(0, os.path.join(REPO, "tools"))
import simulate_drift as sd  # noqa: E402

cat = sd.load_catalogue()
G, pop, items = cat["G"], cat["pop"], cat["items"]
pos = {int(i): j for j, i in enumerate(items)}
out = {}
for world in sys.argv[1:]:
    d = os.path.join(REPO, "data", world)
    m = json.load(open(os.path.join(d, "manifest.json")))
    P = m["params"]
    beta, a, c_hist = P["beta"], P["a"], P["c_hist"]
    pis = np.load(os.path.join(d, "truth", "pi_t.npy"))
    pt = np.load(os.path.join(d, "truth", "p_true_test.npy")).astype(np.float64)
    hist = {}
    for u, i in np.load(os.path.join(d, "training_list.npy"), allow_pickle=True).tolist():
        hist.setdefault(u, []).append(pos[int(i)])
    for u, xs in np.load(os.path.join(d, "validation_dict.npy"), allow_pickle=True).item().items():
        hist.setdefault(u, []).extend(pos[int(i)] for i in xs)
    T = pis.shape[0]
    rows = {k: [] for k in ("rho_b", "kappa", "rho_bp", "kappa_p", "spear", "spear_genre_only", "gauss_pred",
                            "overlap", "overlap_genre_only", "loss", "loss_genre_only", "rho_b_pref_genre")}
    lp = a * np.log(pop)
    for r in range(pt.shape[0]):
        u = r + 1
        h = hist[u]
        dh = (G[h].sum(0) + c_hist * pis[T - 2]) / (len(h) + c_hist)
        cand = pt[r] > 0
        p = np.clip(pt[r, cand], 1e-12, 1 - 1e-12)
        s_int = np.log(p / (1 - p))
        b = beta * (G[cand] @ dh)
        bp = lp[cand]
        s_obs = np.log(p) + b + bp
        s_obs_g = np.log(p) + b
        if beta > 0:
            rho = np.corrcoef(b, s_int)[0, 1]
            k = b.std() / s_int.std()
            rows["rho_b"].append(rho); rows["kappa"].append(k)
            rows["gauss_pred"].append((1 + rho * k) / np.sqrt(1 + 2 * rho * k + k * k))
        rows["rho_bp"].append(np.corrcoef(bp, s_int)[0, 1]); rows["kappa_p"].append(bp.std() / s_int.std())
        rows["spear"].append(spearmanr(s_obs, s_int)[0])
        rows["spear_genre_only"].append(spearmanr(s_obs_g, s_int)[0])
        ti = np.argpartition(-s_int, 20)[:20]
        best = p[ti].sum()
        for key, s in (("", s_obs), ("_genre_only", s_obs_g)):
            to = np.argpartition(-s, 20)[:20]
            rows["overlap" + key].append(len(np.intersect1d(ti, to)) / 20)
            rows["loss" + key].append(1 - p[to].sum() / best)
    out[world] = {"beta": beta, **{k: (float(np.mean(v)) if v else None) for k, v in rows.items()},
                  "n_users": int(pt.shape[0])}
    print(world, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[world].items()}),
          flush=True)
json.dump(out, open(os.path.join(REPO, "analysis", "why_null_sim_aligned.json"), "w"), indent=1)
