"""E4 KuaiRec: exposure-confounding probe and genre-mix ceiling on the fully observed small matrix (CPU).

Reads data/kuairec/ and data/kuairec_temporal/exposure/{big,small}.npy (tools/build_kuairec.py).
For the 1,309 kept small-matrix users:
  policy exposure  = the user's big-matrix impressions (the recommender's choice)
  random exposure  = the small block, every small video shown (fully observed, so no exposure bias)
  1. TV(policy genre mix, small-block catalogue mix): population and per user (noise floor: a random
     subsample of the small block the size of the user's policy impressions).
  2. Alignment: per user, Spearman over genres( log over-exposure, small-block positive rate ),
     and the same for the user's training positive mix d_u.
  3. Positive rate (watch_ratio >= 2): policy vs small block.
Ceiling: rank each user's small-block videos (training items excluded) by
  base = log(1 + training positives) + w * log(0.01 + <m_u, g_i>); per-user AUC and R@20 within the block.
  m_u: static training mix; oracle = the genre mix of a random half A of the user's small positives,
  scored on the other half B (A excluded). w tuned on the oracle-free half for static, on B for the oracle.

    python analysis/kuairec_probe.py  -> analysis/kuairec_probe.json
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(REPO, "data", "kuairec")
EXP = os.path.join(REPO, "data", "kuairec_temporal", "exposure")
rng = np.random.RandomState(0)

ic = np.load(os.path.join(DATA, "item_category.npy"), allow_pickle=True).item()
G = len(np.load(os.path.join(DATA, "confounder_prior.npy")))
NI = max(ic) + 1
Gi = np.zeros((NI, G))
for i, cs in ic.items():
    Gi[i, cs] = 1.0 / len(cs)
train = np.load(os.path.join(DATA, "training_list.npy"))
NU = train[:, 0].max() + 1
big = np.load(os.path.join(EXP, "big.npy"))
small = np.load(os.path.join(EXP, "small.npy"))


def norm(m):
    s = m.sum(-1, keepdims=True)
    return np.divide(m, s, out=np.zeros_like(m), where=s > 0)


def mix(u, i, w=None):
    m = np.zeros((NU, G))
    np.add.at(m, u, Gi[i] * (1.0 if w is None else w[:, None]))
    return m


su = np.unique(small[:, 0])
sitems = np.unique(small[:, 1])
catS = norm(Gi[sitems].sum(0))
bs = big[np.isin(big[:, 0], su)]
Pp = mix(bs[:, 0], bs[:, 1])
Ps, Cs = mix(small[:, 0], small[:, 1]), mix(small[:, 0], small[:, 1], small[:, 3].astype(float))
d_tr = norm(mix(train[:, 0], train[:, 1]))
tv_pop = 0.5 * np.abs(norm(Pp[su].sum(0)) - catS).sum()
npol = Pp[su].sum(1)
tv_u = 0.5 * np.abs(norm(Pp[su]) - catS).sum(1)
null = []
for k, u in enumerate(su):          # same-size uniform draws from the small block
    n = int(round(npol[k]))
    if n < 5:
        continue
    null.append(0.5 * np.abs(norm(Gi[rng.choice(sitems, n)].sum(0)) - catS).sum())
rho_e, rho_d = [], []
for u in su:
    ctr = Cs[u] / np.maximum(Ps[u], 1e-9)
    g = Ps[u] >= 5
    if g.sum() < 5 or Pp[u].sum() < 20:
        continue
    E = np.log((Pp[u, g] / Pp[u].sum() + 1e-4) / catS[g])
    rho_e.append(stats.spearmanr(E, ctr[g])[0])
    rho_d.append(stats.spearmanr(d_tr[u, g], ctr[g])[0])
rho_e, rho_d = np.array(rho_e), np.array(rho_d)
rho_e, rho_d = rho_e[np.isfinite(rho_e)], rho_d[np.isfinite(rho_d)]
ctr_g = Cs[su].sum(0) / Ps[su].sum(0)
Eg = np.log(norm(Pp[su].sum(0)) / catS)
ok = np.isfinite(Eg) & (catS > 0)
out = {"probe": {
    "users": int(len(su)), "small_items": int(len(sitems)),
    "tv_population_policy_vs_small_catalogue": float(tv_pop),
    "tv_user": {"median": float(np.median(tv_u)), "null_median_same_size": float(np.median(null))},
    "alignment_user_spearman_overexposure_vs_smallrate": {"n": int(len(rho_e)), "median": float(np.median(rho_e)),
                                                          "share_positive": float((rho_e > 0).mean())},
    "alignment_user_spearman_trainmix_vs_smallrate": {"n": int(len(rho_d)), "median": float(np.median(rho_d)),
                                                      "share_positive": float((rho_d > 0).mean())},
    "alignment_population_spearman": dict(zip(("rho", "p"), map(float, stats.spearmanr(Eg[ok], ctr_g[ok])))),
    "positive_rate": {"policy_big_these_users": float(bs[:, 3].mean()), "small_block": float(small[:, 3].mean())}}}
print(json.dumps(out["probe"], indent=1))

pop = np.bincount(train[:, 1], minlength=NI).astype(float)
base = np.log1p(pop)
titems = {}
for u, i in train:
    titems.setdefault(int(u), set()).add(int(i))
blocks = {}
for u, i, _t, p in small:
    if i not in titems.get(int(u), ()):
        blocks.setdefault(int(u), []).append((int(i), int(p)))
blocks = {u: np.array(v) for u, v in blocks.items() if u in titems and v and 2 <= np.array(v)[:, 1].sum()}
W = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]


def score(items, m, w):
    return base[items] + (w * np.log(0.01 + Gi[items] @ m) if w else 0)


def metrics(m_of, w, part):
    auc, r20 = [], []
    for u, v in blocks.items():
        keep = part[u]["keep"]
        it, lab = v[keep, 0], v[keep, 1]
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        s = score(it, m_of(u), w)
        auc.append(stats.mannwhitneyu(s[lab == 1], s[lab == 0]).statistic / (lab.sum() * (len(lab) - lab.sum())))
        top = np.argsort(-s)[:20]
        r20.append(lab[top].sum() / lab.sum())
    return np.array(auc), np.array(r20)


part, mA = {}, {}
for u, v in blocks.items():
    pos = np.where(v[:, 1] == 1)[0]
    a = pos[rng.rand(len(pos)) < 0.5]
    keep = np.ones(len(v), bool)
    keep[a] = False
    part[u] = {"keep": keep}
    mA[u] = norm(Gi[v[a, 0]].sum(0)) if len(a) else d_tr[u]
b_auc, b_r = metrics(lambda u: d_tr[u], 0, part)
cel = {"n_users": len(blocks), "base_AUC": float(b_auc.mean()), "base_R@20": float(b_r.mean())}
for nm, f in (("(a) static d^(1)", lambda u: d_tr[u]), ("(c) oracle mix from half A", lambda u: mA[u])):
    res = {w: metrics(f, w, part) for w in W}
    wa = max(W, key=lambda w: res[w][0].mean())
    wr = max(W, key=lambda w: res[w][1].mean())
    cel[nm] = {"w_best_AUC": wa, "dAUC": float(res[wa][0].mean() - b_auc.mean()),
               "w_best_R@20": wr, "R@20_rel_pct": float(res[wr][1].mean() / b_r.mean() * 100 - 100),
               "R@20_ci95": [float(x) for x in np.percentile(
                   [res[wr][1][ix].mean() / b_r[ix].mean() * 100 - 100 for ix in rng.randint(0, len(b_r), (1000, len(b_r)))],
                   [2.5, 97.5])]}
    print(nm, cel[nm], flush=True)
out["ceiling_small_block_halfB"] = cel
json.dump(out, open(os.path.join(HERE, "kuairec_probe.json"), "w"), indent=1)
