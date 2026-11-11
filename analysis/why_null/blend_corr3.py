"""Gaussian score model for the Eq. 11 blend (why-null Prop. 2), ML-1M.

Per user, each model's logits over the unmasked items are mapped to normal scores
z = Phi^-1((rank - 0.5)/N). Model: z_k = a_k * 1[test] + e_k, e ~ N(0, 1), corr r.
Measured: a_k (mean z of test items), r (corr of z over non-test items).
Predicted blend: signal lam*a_x + (1-lam)*a_y, sd sqrt(lam^2+(1-lam)^2+2lam(1-lam)r).
R@20 is predicted by simulating that Gaussian model per user and compared with the
actual Eq. 11 blend (sigmoid scores, real lam_u).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "eval_package"))
import score_all as S  # noqa: E402
from common import CODE, DS, Data, eq11_lambda, ld  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.special import ndtri  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402

torch.set_num_threads(4)
ds = "ml_1m"
c = DS[ds]
src = S.sources(ds)
NAMES = ["FM_s1", "FM_s2", "DecFM_s1", "FMconf_s1"]
PAIRS = [("FM_s1", "FM_A"), ("DecFM_s1", "FM_A"), ("FMconf_s1", "FM_A"), ("FM_s1", "FM_s2")]

uf, itf, _ = data_utils.map_features(os.path.join(c["data"], "user_feature_file.npy"),
                                     os.path.join(c["data"], "item_feature_file.npy"))
train = {}
for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist():
    train.setdefault(u, []).append(i)
valid = ld(os.path.join(c["data"], "validation_dict.npy"))
test = ld(os.path.join(c["data"], "testing_dict.npy"))
cat = ld(os.path.join(c["data"], "item_category.npy"))
urf, ukl, aif, aifv = evaluate.pre_ranking(uf, itf, cat, c["G"], train, valid, test)
users = list(test.keys())
item_idx = np.array(list(itf.keys()))
pos = {it: p for p, it in enumerate(item_idx)}
lam = eq11_lambda(ukl, users)

models, fast = {}, {}
for n in NAMES:
    m = torch.load(src[n][0], map_location="cpu", weights_only=False)
    m.eval()
    if hasattr(m, "confounder_prior"):
        m.confounder_prior = m.confounder_prior.cpu()
    models[n] = m
    fast[n] = S.FastFM(m, aif, aifv)
conf = ld(os.path.join(c["dcr"], "user_confounder_g0.3_k0_static.npy"))
fmA = ld(os.path.join(CODE, "inference", "FM_user_item_scores.npy"))


def cf(n, ub):
    if src[n][1] is None:
        return None
    return torch.tensor(np.stack([np.asarray(conf[u], np.float32) for u in ub]))



users = [users[k] for k in np.random.default_rng(3).choice(len(users), 2000, replace=False)]
lamd = dict(zip(list(test.keys()), lam))
acc = {p: {"zx": [], "zy": [], "r": [], "N": [], "nt": [], "lam": [], "hx": [], "hy": [], "hb": []} for p in PAIRS}
with torch.no_grad():
    for b in range(0, len(users), 250):
        ub = users[b:b + 250]
        mask = torch.stack([urf[u][2] for u in ub]).numpy()
        P = {n: fast[n](urf, ub, cf(n, ub)).numpy() for n in NAMES}
        P["FM_A"] = np.stack([np.asarray(fmA[u], np.float32) for u in ub])
        for j, u in enumerate(ub):
            cand = np.nonzero(mask[j] == 0)[0]
            N = len(cand)
            is_t = np.isin(cand, np.array([pos[i] for i in test[u] if i in pos]))
            nt = int(is_t.sum())
            if nt == 0:
                continue
            Z, R = {}, {}
            for n in P:
                x = P[n][j, cand]
                rk = np.empty(N); rk[np.argsort(x)] = np.arange(1, N + 1)
                Z[n] = ndtri((rk - 0.5) / N); R[n] = x
            for (x, y) in PAIRS:
                A = acc[(x, y)]; l = lamd[u]
                A["zx"].append(Z[x][is_t]); A["zy"].append(Z[y][is_t])
                A["r"].append(np.corrcoef(Z[x][~is_t], Z[y][~is_t])[0, 1])
                A["N"].append(N); A["nt"].append(nt); A["lam"].append(l)
                s = l / (1 + np.exp(-R[x])) + (1 - l) / (1 + np.exp(-R[y]))
                for key, sc in (("hb", s), ("hx", R[x]), ("hy", R[y])):
                    A[key].append(is_t[np.argpartition(-sc, 20)[:20]].sum() / nt)
        print(b + len(ub), flush=True)


def model_recall(m, tau, scale, N, nt, seed=1):
    """Heterogeneous Gaussian model: test item i has signal mu_i ~ N(m, tau^2) shared by
    both models; each model adds unit noise. scale[u] multiplies the signal (blend SNR gain)."""
    st = np.random.default_rng(seed)
    h = 0.0
    for k in range(len(N)):
        z = st.standard_normal(N[k])
        z[:nt[k]] += scale[k] * (m + tau * st.standard_normal(nt[k]))
        h += (np.argpartition(-z, 20)[:20] < nt[k]).sum() / nt[k]
    return h / len(N)


def calib(target, tau, N, nt):
    lo, hi = -1.0, 4.0
    one = np.ones(len(N))
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if model_recall(mid, tau, one, N, nt) < target else (lo, mid)
    return 0.5 * (lo + hi)


out = {}
for p in PAIRS:
    A = acc[p]
    zx, zy = np.concatenate(A["zx"]), np.concatenate(A["zy"])
    r = np.array(A["r"]); N = np.array(A["N"]); nt = np.array(A["nt"]); L = np.array(A["lam"])
    cov_t = np.cov(zx, zy)[0, 1]
    tau2 = max(cov_t - r.mean(), 0.0)          # shared-signal variance on test items
    tau = np.sqrt(tau2)
    hx, hy, hb = np.mean(A["hx"]), np.mean(A["hy"]), np.mean(A["hb"])
    mx, my = calib(hx, tau, N, nt), calib(hy, tau, N, nt)
    sd = np.sqrt(L ** 2 + (1 - L) ** 2 + 2 * L * (1 - L) * r)
    # blend signal lam*mu_x + (1-lam)*mu_y with mu_k = m_k + shared deviation
    mb = (L * mx + (1 - L) * my)
    st = np.random.default_rng(1); h = 0.0
    for k in range(len(N)):
        z = st.standard_normal(N[k])
        z[:nt[k]] += (mb[k] + tau * st.standard_normal(nt[k])) / sd[k]
        h += (np.argpartition(-z, 20)[:20] < nt[k]).sum() / nt[k]
    pb = h / len(N)
    o = {"n_users": len(N), "r_noise_mean": r.mean(), "var_test_x": zx.var(), "var_test_y": zy.var(),
         "cov_test": cov_t, "tau2": tau2, "m_x": mx, "m_y": my, "lam_mean": L.mean(),
         "snr_factor_mean": float(np.mean(1 / sd)),
         "R20_x": hx, "R20_y": hy, "R20_blend_actual": hb, "R20_blend_pred": pb,
         "gain_actual_vs_partner": hb / hy - 1, "gain_pred_vs_partner": pb / hy - 1,
         "gain_actual_vs_mean": hb / ((hx + hy) / 2) - 1, "gain_pred_vs_mean": pb / ((hx + hy) / 2) - 1}
    out["%s+%s" % p] = o
    print(p, json.dumps({k: round(float(v), 4) for k, v in o.items()}), flush=True)
prev = json.load(open(os.path.join("analysis", "why_null_blend_corr_ml_1m.json")))
prev["heterogeneous_model_2000_users"] = out
json.dump(prev, open(os.path.join("analysis", "why_null_blend_corr_ml_1m.json"), "w"), indent=1, default=float)
