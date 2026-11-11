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
NAMES = ["FM_s1", "FM_s2", "DecFM_s1", "FMconf_s1", "DCR_s1"]
PAIRS = [("FM_s1", "FM_A"), ("FM_s2", "FM_A"), ("DecFM_s1", "FM_A"), ("FMconf_s1", "FM_A"),
         ("DCR_s1", "FM_A"), ("FM_s1", "FM_s2")]

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


rng = np.random.default_rng(0)
acc = {p: {"a_x": [], "a_y": [], "r": [], "hit_act": [], "hit_pred": [], "hit_x": [], "hit_y": [],
           "pred_x": [], "pred_y": [], "n": []} for p in PAIRS}
single_z = {}
CB = 256
with torch.no_grad():
    for b in range(0, len(users), CB):
        ub = users[b:b + CB]
        mask = torch.stack([urf[u][2] for u in ub]).numpy()
        P = {n: fast[n](urf, ub, cf(n, ub)).numpy() for n in NAMES}
        P["FM_A"] = np.stack([np.asarray(fmA[u], np.float32) for u in ub])
        for j, u in enumerate(ub):
            keep = mask[j] == 0
            cand = np.nonzero(keep)[0]
            N = len(cand)
            tpos = np.array([pos[i] for i in test[u] if i in pos])
            is_t = np.isin(cand, tpos)
            nt = is_t.sum()
            if nt == 0:
                continue
            Z, R = {}, {}
            for n in P:
                x = P[n][j, cand]
                rk = np.empty(N)
                rk[np.argsort(x)] = np.arange(1, N + 1)
                Z[n] = ndtri((rk - 0.5) / N)
                R[n] = x
            for (x, y) in PAIRS:
                A = acc[(x, y)]
                l = lam[b + j]
                zx, zy = Z[x], Z[y]
                ax, ay = zx[is_t].mean(), zy[is_t].mean()
                r = np.corrcoef(zx[~is_t], zy[~is_t])[0, 1]
                s = l / (1 + np.exp(-R[x])) + (1 - l) / (1 + np.exp(-R[y]))
                top = np.argpartition(-s, 20)[:20]
                A["hit_act"].append(is_t[top].sum() / nt)
                A["hit_x"].append(is_t[np.argpartition(-R[x], 20)[:20]].sum() / nt)
                A["hit_y"].append(is_t[np.argpartition(-R[y], 20)[:20]].sum() / nt)
                A["a_x"].append(ax); A["a_y"].append(ay); A["r"].append(r); A["n"].append((N, nt, l))
        print(b + len(ub), flush=True)


def sim_recall(a_eff, N, nt, reps=4):
    """R@20 of a Gaussian model: nt relevant items with mean a_eff, N - nt with mean 0, unit sd."""
    h = 0.0
    for _ in range(reps):
        z = rng.standard_normal(N)
        z[:nt] += a_eff
        top = np.argpartition(-z, 20)[:20]
        h += (top < nt).sum() / nt
    return h / reps


np.savez(os.path.join(os.environ.get("SP", os.path.join("analysis", "why_null")), "blend_corr_users.npz"),
         **{"%s+%s|%s" % (p[0], p[1], k): np.array(v) for p in PAIRS for k, v in acc[p].items() if v})


def calibrate(target, N, nt, lo=0.5, hi=4.0):
    """Shared signal a such that the Gaussian model's mean R@20 equals the observed one."""
    st = np.random.default_rng(1)
    noise = [st.standard_normal(n) for n in N]

    def rec(a):
        h = 0.0
        for z0, n, t in zip(noise, N, nt):
            z = z0.copy(); z[:t] += a
            h += (np.argpartition(-z, 20)[:20] < t).sum() / t
        return h / len(N)
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rec(mid) < target else (lo, mid)
    return 0.5 * (lo + hi), rec


out = {}
for p in PAIRS:
    A = acc[p]
    Nn = np.array([n[0] for n in A["n"]]); Nt = np.array([n[1] for n in A["n"]]); Lm = np.array([n[2] for n in A["n"]])
    rr = np.array(A["r"])
    sub = np.random.default_rng(2).choice(len(Nn), 1500, replace=False)
    ax_c, rec = calibrate(np.mean(A["hit_x"]), Nn[sub], Nt[sub])
    ay_c, _ = calibrate(np.mean(A["hit_y"]), Nn[sub], Nt[sub])
    sd = np.sqrt(Lm[sub] ** 2 + (1 - Lm[sub]) ** 2 + 2 * Lm[sub] * (1 - Lm[sub]) * rr[sub])
    st = np.random.default_rng(1)
    hb = 0.0
    for k, (n, t) in enumerate(zip(Nn[sub], Nt[sub])):
        z = st.standard_normal(n); z[:t] += (Lm[sub][k] * ax_c + (1 - Lm[sub][k]) * ay_c) / sd[k]
        hb += (np.argpartition(-z, 20)[:20] < t).sum() / t
    hb /= len(sub)
    out_cal = {"a_x_cal": ax_c, "a_y_cal": ay_c, "pred_R20_blend_cal": hb,
               "gain_pred_cal_vs_mean_single": hb / ((np.mean(A["hit_x"]) + np.mean(A["hit_y"])) / 2) - 1,
               "gain_pred_cal_vs_partner": hb / np.mean(A["hit_y"]) - 1,
               "snr_factor_mean": float(np.mean(1 / sd))}
    print(p, "calibrated", json.dumps({k: round(float(v), 4) for k, v in out_cal.items()}), flush=True)
    A = acc[p]
    ax, ay, r = np.mean(A["a_x"]), np.mean(A["a_y"]), np.array(A["r"])
    px, py, pb = [], [], []
    for k, (N, nt, l) in enumerate(A["n"]):
        # pooled signal, per-user noise correlation and lambda
        px.append(sim_recall(ax, N, nt)); py.append(sim_recall(ay, N, nt))
        sd = np.sqrt(l * l + (1 - l) ** 2 + 2 * l * (1 - l) * r[k])
        pb.append(sim_recall((l * ax + (1 - l) * ay) / sd, N, nt))
    act_x, act_y, act_b = np.mean(A["hit_x"]), np.mean(A["hit_y"]), np.mean(A["hit_act"])
    pxm, pym, pbm = np.mean(px), np.mean(py), np.mean(pb)
    out["%s+%s" % p] = {
        "a_x": ax, "a_y": ay, "r_mean": float(r.mean()), "r_median": float(np.median(r)),
        "lam_mean": float(np.mean([n[2] for n in A["n"]])),
        "R20_x": act_x, "R20_y": act_y, "R20_blend": act_b,
        "gain_actual_vs_mean_single": act_b / ((act_x + act_y) / 2) - 1,
        "gain_actual_vs_partner": act_b / act_y - 1,
        "pred_R20_x": pxm, "pred_R20_y": pym, "pred_R20_blend": pbm,
        "gain_pred_vs_mean_single": pbm / ((pxm + pym) / 2) - 1,
        "gain_pred_vs_partner": pbm / pym - 1,
    }
    out["%s+%s" % p].update(out_cal)
    print(p, json.dumps({k: round(float(v), 4) for k, v in out["%s+%s" % p].items()}), flush=True)
json.dump(out, open(os.path.join("analysis", "why_null_blend_corr_ml_1m.json"), "w"), indent=1, default=float)
