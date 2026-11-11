"""E4 KuaiRand: exposure-confounding probe and a D1-style genre-mix ceiling (CPU only).

Reads data/kuairand/ and data/kuairand_temporal/{per_user,exposure}/ (tools/build_kuairand.py).

PROBE: policy (standard log) vs uniformly random exposure, both on 4/22-5/08, kept users only.
  1. TV(policy genre mix of exposures, random genre mix): population, and per user with a
     split-half random-vs-random noise floor at the same sample sizes.
  2. Alignment (as N3's rho_b): per user, over genres with >= MIN_R random impressions,
     Spearman( log over-exposure E_u(g) = policy share / random share,
               random-exposure click rate CTR_u(g) ).
     Also the population version across genres, and the same Spearman for the user's
     TRAINING click mix d_u(g) (the DecRS confounder) instead of policy exposure.
  3. Click rate under policy vs random, same users and days.

CEILING (D1 method, analysis/eval_package/diag/d1_headroom.py): re-rank a base with
  s' = s + w * log(0.01 + <m_u, g_i>), w on a 0-20 grid tuned on validation.
  Base = log(1 + training clicks) (popularity; an FM does not fit the CPU budget).
  Tests: observational (standard log 5/05-5/08) and interventional (random-log clicks, same days).
  Validation: observational = standard 5/02-5/04; interventional = random-log clicks 4/22-5/04.
  m_u rows: (a) static training click mix, (b) recency gamma 0.7 and gamma* 0.98 (block EWMA),
  (w) the last training window's population mix (DecRS-windowed's test-time prior),
  (P) oracle: the population genre mix of the test positives, (c) half-B oracle: users with
  >= 2 test positives, test split into A/B at random, m_u from A, scored on B with A masked.
  CIs: paired per-user bootstrap (1,000) of the ratio of mean R@20, in % vs the base.

  AUC protocol (interventional, unbiased): rank each user's own random impressions on the test days,
  per-user AUC of clicked vs unclicked; w tuned on the 4/22-5/04 random impressions.

    python analysis/kuairand_probe.py              -> analysis/kuairand_probe.json
    python analysis/kuairand_probe.py kuairand auc  (re-run only the probe and the AUC protocol)
"""
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
DS = sys.argv[1] if len(sys.argv) > 1 else "kuairand"
DATA = os.path.join(REPO, "data", DS)
TMP = os.path.join(REPO, "data", DS + "_temporal")
D_RAND_VALID = (1650556800, 1651680000)   # 2022-04-22 00:00 to 2022-05-05 00:00, UTC+8
MIN_R = 3
K = 20
rng = np.random.RandomState(0)

ic = np.load(os.path.join(DATA, "item_category.npy"), allow_pickle=True).item()
G = len(np.load(os.path.join(DATA, "confounder_prior.npy")))
NI = max(ic) + 1
Gi = np.zeros((NI, G))
for i, cs in ic.items():
    Gi[i, cs] = 1.0 / len(cs)
train = np.load(os.path.join(DATA, "training_list.npy"))
tts = np.load(os.path.join(TMP, "per_user", "training_timestamps.npy"))
NU = train[:, 0].max() + 1
valid = np.load(os.path.join(DATA, "validation_dict.npy"), allow_pickle=True).item()
test = np.load(os.path.join(DATA, "testing_dict.npy"), allow_pickle=True).item()
test_r = np.load(os.path.join(DATA, "testing_dict_rand.npy"), allow_pickle=True).item()
std = np.load(os.path.join(TMP, "exposure", "standard.npy"))
rnd = np.load(os.path.join(TMP, "exposure", "random.npy"))
out = {"dataset": DS}


def mix(users, items, n=NU, w=None):
    m = np.zeros((n, G))
    np.add.at(m, users, Gi[items] * (1.0 if w is None else w[:, None]))
    return m


def norm(m):
    s = m.sum(-1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


# ------------------------------------------------------------------ probe ----
lo = rnd[:, 2].min()
pol = std[std[:, 2] >= lo]
P_pol, P_rnd = mix(pol[:, 0], pol[:, 1]), mix(rnd[:, 0], rnd[:, 1])
pp, pr = norm(P_pol.sum(0)), norm(P_rnd.sum(0))
tvpop = 0.5 * np.abs(pp - pr).sum()
cat = norm(Gi[1:].sum(0))
n_pol = np.bincount(pol[:, 0], minlength=NU)
n_rnd = np.bincount(rnd[:, 0], minlength=NU)
ok = (n_pol >= 20) & (n_rnd >= 20)
tv_u = 0.5 * np.abs(norm(P_pol) - norm(P_rnd)).sum(1)[ok]
# noise floor: two disjoint random halves of the user's own random exposures, sized like the comparison
half = rng.rand(len(rnd)) < 0.5
A, B = mix(rnd[half, 0], rnd[half, 1]), mix(rnd[~half, 0], rnd[~half, 1])
nA, nB = A.sum(1), B.sum(1)
ok2 = ok & (nA >= 10) & (nB >= 10)
tv_null = 0.5 * np.abs(norm(A) - norm(B)).sum(1)[ok2]
tv_u2 = 0.5 * np.abs(norm(P_pol) - norm(B)).sum(1)[ok2]   # policy vs the same half-size random sample

# alignment
C_rnd = mix(rnd[:, 0], rnd[:, 1], w=rnd[:, 3].astype(float))      # random clicks per genre (mass)
d_tr = norm(mix(train[:, 0], train[:, 1]))
rho_exp, rho_d = [], []
for u in np.where(ok)[0]:
    g = P_rnd[u] >= MIN_R
    if g.sum() < 5:
        continue
    ctr = C_rnd[u, g] / P_rnd[u, g]
    if np.ptp(ctr) == 0:
        continue
    E = np.log((P_pol[u, g] / P_pol[u].sum() + 1e-4) / (P_rnd[u, g] / P_rnd[u].sum()))
    rho_exp.append(stats.spearmanr(E, ctr)[0])
    rho_d.append(stats.spearmanr(d_tr[u, g], ctr)[0])
rho_exp, rho_d = np.array(rho_exp), np.array(rho_d)
rho_exp, rho_d = rho_exp[np.isfinite(rho_exp)], rho_d[np.isfinite(rho_d)]
ctr_g = C_rnd.sum(0) / P_rnd.sum(0)
E_g = np.log(pp / pr)
out["probe"] = {
    "window": "2022-04-22..05-08, kept users",
    "impressions": {"policy": int(len(pol)), "random": int(len(rnd))},
    "tv_population_policy_vs_random": float(tvpop),
    "tv_population_random_vs_catalogue": float(0.5 * np.abs(pr - cat).sum()),
    "tv_user_policy_vs_random": {"n": int(ok.sum()), "median": float(np.median(tv_u)), "mean": float(tv_u.mean())},
    "tv_user_halfsize": {"n": int(ok2.sum()), "policy_vs_randomhalf_median": float(np.median(tv_u2)),
                         "random_vs_random_null_median": float(np.median(tv_null)),
                         "excess_median": float(np.median(tv_u2) - np.median(tv_null))},
    "alignment_user_spearman_overexposure_vs_randomCTR": {
        "n_users": int(len(rho_exp)), "median": float(np.median(rho_exp)), "mean": float(rho_exp.mean()),
        "share_positive": float((rho_exp > 0).mean()), "min_random_impr_per_genre": MIN_R},
    "alignment_user_spearman_trainmix_vs_randomCTR": {
        "n_users": int(len(rho_d)), "median": float(np.median(rho_d)), "mean": float(rho_d.mean()),
        "share_positive": float((rho_d > 0).mean())},
    "alignment_population_spearman_overexposure_vs_randomCTR": dict(zip(
        ("rho", "p"), map(float, stats.spearmanr(E_g, ctr_g)))),
    "ctr": {"policy_same_days": float(pol[:, 3].mean()), "random": float(rnd[:, 3].mean()),
            "long_view_policy": float(pol[:, 4].mean()), "long_view_random": float(rnd[:, 4].mean())},
    "genre_table": [{"g": int(g), "policy_share": float(pp[g]), "random_share": float(pr[g]),
                     "random_ctr": float(ctr_g[g])} for g in np.argsort(-pp)],
}
print(json.dumps({k: v for k, v in out["probe"].items() if k != "genre_table"}, indent=1))

# ---------------------------------------------------------------- ceiling ----
pop = np.bincount(train[:, 1], minlength=NI).astype(float)
base = np.log1p(pop)
base[0] = -np.inf
train_items = {}
for u, i in train:
    train_items.setdefault(int(u), []).append(int(i))
rv = rnd[(rnd[:, 2] >= D_RAND_VALID[0]) & (rnd[:, 2] < D_RAND_VALID[1]) & (rnd[:, 3] == 1)]
valid_r = {}
for u, i in rv[:, :2]:
    if i not in train_items.get(int(u), []):
        valid_r.setdefault(int(u), set()).add(int(i))
valid_r = {u: sorted(v) for u, v in valid_r.items() if u in train_items}


def recency(gamma):
    m = np.zeros((NU, G))
    order = np.lexsort((tts, train[:, 0]))
    last_u, last_t = -1, None
    for k in order:
        u, i, t = train[k, 0], train[k, 1], tts[k]
        if u != last_u:
            last_u, last_t = u, t
        elif t != last_t:
            m[u] *= gamma
            last_t = t
        m[u] += Gi[i]
    return norm(m)


wp = np.load(os.path.join(TMP, "global", "prior_by_window_consumed.npy"))
M = {"(a) static d^(1)": d_tr, "(b) recent g0.7": recency(0.7), "(b) recent g*0.98": recency(0.98),
     "(w) last-window population": np.tile(wp[-1], (NU, 1))}


def topk(users, m_u, w, mask_extra=None):
    """users: list; m_u: (NU, G) or None; returns {u: top-K items}."""
    res = {}
    for b in range(0, len(users), 512):
        ub = np.array(users[b:b + 512])
        s = np.tile(base, (len(ub), 1))
        if m_u is not None and w:
            s = s + w * np.log(0.01 + m_u[ub] @ Gi.T)
        for j, u in enumerate(ub):
            s[j, train_items.get(int(u), [])] = -np.inf
            if mask_extra is not None:
                s[j, mask_extra.get(int(u), [])] = -np.inf
        top = np.argpartition(-s, K, 1)[:, :K]
        for j, u in enumerate(ub):
            res[int(u)] = set(top[j].tolist())
    return res


def recall(top, gt, users):
    return np.array([len(top[u] & set(gt[u])) / len(gt[u]) for u in users])


def boot(x, y):
    idx = rng.randint(0, len(x), (1000, len(x)))
    r = x[idx].mean(1) / y[idx].mean(1) * 100 - 100
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


W = [0, 0.25, 0.5, 1, 2, 3, 5, 10, 20]


# ---- exposure-restricted interventional protocol: rank each user's RANDOM impressions only ----
def auc_rows():
    """Per-user AUC of clicked vs unclicked random impressions (test days; w tuned on the 4/22-5/04
    random impressions). This is the unbiased protocol: every candidate was exposed uniformly at random."""
    WF = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]

    def per_user(rows):
        d = {}
        for u, i, _t, c, _l in rows:
            if i not in train_items.get(int(u), []):
                d.setdefault(int(u), []).append((int(i), int(c)))
        return {u: np.array(v) for u, v in d.items() if u in train_items and 0 < np.array(v)[:, 1].sum() < len(v)}

    te = per_user(rnd[rnd[:, 2] >= D_RAND_VALID[1]])
    va = per_user(rnd[(rnd[:, 2] >= D_RAND_VALID[0]) & (rnd[:, 2] < D_RAND_VALID[1])])

    def aucs(d, m, w):
        r = []
        for u, v in d.items():
            s = base[v[:, 0]] + (w * np.log(0.01 + Gi[v[:, 0]] @ m[u]) if w else 0)
            r.append(stats.mannwhitneyu(s[v[:, 1] == 1], s[v[:, 1] == 0]).statistic / (v[:, 1].sum() * (len(v) - v[:, 1].sum())))
        return np.array(r)

    b = aucs(te, M["(a) static d^(1)"], 0)
    res = {"n_users": len(te), "impressions_per_user_median": float(np.median([len(v) for v in te.values()])),
           "base_popularity_AUC": float(b.mean())}
    hb_mix = np.zeros((NU, G))
    for u, v in te.items():          # oracle: the genre mix of the user's own clicked random test items
        hb_mix[u] = norm(Gi[v[v[:, 1] == 1, 0]].sum(0))
    for nm, m in list(M.items()) + [("(c) oracle own test-click mix (leaky upper bound)", hb_mix)]:
        wv = 0 if nm.startswith("(c)") else max(WF, key=lambda w: aucs(va, m, w).mean())
        tb = {w: aucs(te, m, w) for w in WF}
        wt = max(WF, key=lambda w: tb[w].mean())
        r = tb[wv]
        res[nm] = {"w_valid": wv, "AUC": float(r.mean()), "dAUC": float(r.mean() - b.mean()),
                   "ci95_dAUC": [float(x) for x in np.percentile(
                       [(r[ix] - b[ix]).mean() for ix in rng.randint(0, len(r), (1000, len(r)))], [2.5, 97.5])],
                   "w_test_best": wt, "dAUC_test_best": float(tb[wt].mean() - b.mean())}
        print("AUC", nm, res[nm], flush=True)
    return res


if "auc" in sys.argv[2:]:
    prev = json.load(open(os.path.join(HERE, DS + "_probe.json")))
    prev["probe"] = out["probe"]
    prev["interventional_auc"] = auc_rows()
    json.dump(prev, open(os.path.join(HERE, DS + "_probe.json"), "w"), indent=1)
    sys.exit(0)
ceil = {}
for proto, gt, vgt, vmask in (("observational", test, valid, None), ("interventional", test_r, valid_r, None)):
    us = sorted(u for u in gt if u in train_items)
    vus = sorted(u for u in vgt if vgt[u] and u in train_items)
    tmask = {u: valid.get(u, []) for u in us}
    b = recall(topk(us, None, 0, tmask), gt, us)
    rows = {"base (popularity)": {"R@20": float(b.mean()), "w": 0}}
    popor = norm(mix(np.repeat(us, [len(gt[u]) for u in us]), np.concatenate([gt[u] for u in us])).sum(0))
    cand = dict(M, **{"(P) oracle test-period population mix": np.tile(popor, (NU, 1))})
    for nm, m in cand.items():
        vb = {w: recall(topk(vus, m, w), vgt, vus).mean() for w in W}
        wv = max(W, key=lambda w: vb[w])
        r = recall(topk(us, m, wv, tmask), gt, us)
        tb = {w: recall(topk(us, m, w, tmask), gt, us).mean() for w in W}
        wt = max(W, key=lambda w: tb[w])
        rows[nm] = {"w_valid": wv, "R@20": float(r.mean()), "rel_pct": float(r.mean() / b.mean() * 100 - 100),
                    "ci95": boot(r, b), "w_test_best": wt, "rel_pct_test_best": float(tb[wt] / b.mean() * 100 - 100)}
        print(proto, nm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[nm].items()}, flush=True)
    # half-B oracle
    hb = [u for u in us if len(gt[u]) >= 2]
    Aset, Bgt = {}, {}
    for u in hb:
        xs = np.array(gt[u])
        a = rng.rand(len(xs)) < 0.5
        if a.all() or (~a).all():
            a[0], a[-1] = True, False
        Aset[u], Bgt[u] = xs[a].tolist(), xs[~a].tolist()
    mA = np.zeros((NU, G))
    for u in hb:
        mA[u] = norm(Gi[Aset[u]].sum(0))
    maskAB = {u: list(tmask[u]) + Aset[u] for u in hb}
    bB = recall(topk(hb, None, 0, maskAB), Bgt, hb)
    hrows = {"n_users": len(hb), "base R@20 on B": float(bB.mean())}
    for nm, m in (("(a) static d^(1)", M["(a) static d^(1)"]), ("(c) oracle mix from A", mA)):
        tb = {w: recall(topk(hb, m, w, maskAB), Bgt, hb) for w in W}
        wt = max(W, key=lambda w: tb[w].mean())
        hrows[nm] = {"w_B_best": wt, "rel_pct": float(tb[wt].mean() / bB.mean() * 100 - 100), "ci95": boot(tb[wt], bB)}
        print(proto, "half-B", nm, hrows[nm], flush=True)
    rows["half-B"] = hrows
    ceil[proto] = {"n_users": len(us), "rows": rows}
out["ceiling"] = ceil
out["interventional_auc"] = auc_rows()
json.dump(out, open(os.path.join(HERE, DS + "_probe.json"), "w"), indent=1)
