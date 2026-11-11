"""Shared loaders, metrics and the Eq. 11 blend for the CPU evaluation package (E2).

Everything here follows the existing scorers exactly:
  * ranking path   -- analysis/rescore.py, analysis/ml10m_eval.py (training items
                      masked with -999, users scored in batches of 16)
  * Eq. 11 blend   -- code/ml-1m/evaluate.Ranking:
                      s = lam_u * sigmoid(x) + (1 - lam_u) * sigmoid(fm),
                      lam_u = ((kl_u - min kl) / (max kl - min kl)) ** alpha, alpha = 0.2
  * metrics        -- analysis/segments.py per_user (Recall, NDCG; C_KL with
                      Steck's 0.01 smoothing against a target genre mix p)

Top-100 lists produced by score_all.py live in analysis/eval_package/topk/<ds>/.
"""
import math
import os
import sys

os.environ.setdefault("DECRS_DEVICE", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(ANALYSIS)
CODE = os.path.join(REPO, "code", "ml-1m")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import numpy as np  # noqa: E402

ALPHA = 0.2          # Eq. 11 exponent used by every Dec* run
TOPK_DIR = os.path.join(HERE, "topk")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(OUT_DIR, exist_ok=True)

DS = {
    "ml_1m": {"G": 18, "data": os.path.join(REPO, "data", "ml_1m"),
              "dcr": os.path.join(REPO, "data", "ml_1m_dcr"),
              "temporal": os.path.join(REPO, "data", "ml_1m_temporal", "per_user"),
              "seeds": [1, 2, 3, 4, 5]},
    "ml_10m": {"G": 19, "data": os.path.join(REPO, "data", "ml_10m"),
               "dcr": os.path.join(REPO, "data", "ml_10m_dcr"),
               "temporal": os.path.join(REPO, "data", "ml_10m_temporal", "per_user"),
               "seeds": [1, 2, 3]},
}
LABEL = {"ml_1m": "ML-1M", "ml_10m": "ML-10M", "goodreads": "Goodreads"}
# datasets outside the E2 loops (the diag/ scripts use them)
DS_ALL = {**DS, "goodreads": {"G": 10, "data": os.path.join(REPO, "data", "goodreads"),
                              "dcr": os.path.join(REPO, "data", "goodreads_dcr"),
                              "temporal": os.path.join(REPO, "data", "goodreads_temporal", "per_user"),
                              "seeds": [1, 2, 3]}}


def ld(p):
    z = np.load(p, allow_pickle=True)
    return z.item() if z.dtype == object and z.shape == () else z


class Data:
    """Train/valid/test dicts, genres and per-user genre mixes for one dataset."""

    def __init__(self, ds):
        c = DS_ALL[ds]
        self.ds, self.G = ds, c["G"]
        self.train = {}
        for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist():
            self.train.setdefault(u, []).append(i)
        self.valid = ld(os.path.join(c["data"], "validation_dict.npy"))
        self.test = ld(os.path.join(c["data"], "testing_dict.npy"))
        self.cat = ld(os.path.join(c["data"], "item_category.npy"))
        self.all_users = list(self.test.keys())          # ranking order of rescore.py
        self.users = [u for u in self.all_users if len(self.test[u])]
        self.n_items = len(self.cat)
        Gm = np.zeros((max(self.cat) + 1, self.G))
        for i, cs in self.cat.items():
            cs = [cs] if np.isscalar(cs) else list(cs)
            for x in cs:
                Gm[i, x] += 1.0 / len(cs)
        self.Gm = Gm
        self.his_g = np.stack([self.gvec(self.train[u]) for u in self.users])
        self.test_g = np.stack([self.gvec(self.test[u]) for u in self.users])

    def gvec(self, items):
        v = self.Gm[np.asarray(list(items), int)].sum(0) if len(items) else np.zeros(self.G)
        return v / v.sum() if v.sum() else v

    def kl_scores(self):
        """DecRS's raw drift score kl_u (evaluate.kl_score_calc), all ranked users."""
        import evaluate
        return {u: evaluate.kl_score_calc(self.train[u], self.valid[u], self.cat, self.G)
                for u in self.all_users}

    def recent_mix(self, gamma=0.3, kappa=0.0):
        """d_u^(gamma) after the whole training history (tools/dcr_confounder.py)."""
        from dcr_confounder import block_confounders, static_pop
        c = DS_ALL[self.ds]
        tl = ld(os.path.join(c["temporal"], "training_list.npy"))
        ts = ld(os.path.join(c["temporal"], "training_timestamps.npy"))
        pop = ld(os.path.join(c["data"], "confounder_prior.npy")).astype(np.float64)
        rows = {}
        for (u, i), t in zip(tl.tolist(), ts.tolist()):
            rows.setdefault(u, ([], []))
            rows[u][0].append(t)
            rows[u][1].append(i)
        out = {}
        for u, (t, it) in rows.items():
            order = np.argsort(t, kind="stable")
            _, d = block_confounders([t[j] for j in order], [it[j] for j in order], self.Gm, static_pop(pop),
                                     gamma, kappa)
            out[u] = d
        return out


def eq11_lambda(score, users, alpha=ALPHA):
    """Eq. 11 weight from any per-user drift score (dict over the ranked users)."""
    v = np.array([score[u] for u in users], float)
    lo, hi = v.min(), v.max()
    return ((v - lo) / (hi - lo)) ** alpha


def load_top(ds, name, with_scores=False):
    z = np.load(os.path.join(TOPK_DIR, ds, name + ".npz"))
    top = dict(zip(z["users"].tolist(), z["top"]))
    if with_scores:
        return top, dict(zip(z["users"].tolist(), z["score"]))
    return top


def has_top(ds, name):
    return os.path.exists(os.path.join(TOPK_DIR, ds, name + ".npz"))


def ckl(p, q, a=0.01):
    """Row-wise C_KL(p || (1-a) q + a p), Steck (2018), as evaluate.calibration."""
    q = (1 - a) * q + a * p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(np.where(p > 0, p, 1) / np.where(q > 0, q, 1)), 0.0)
    return t.sum(-1)


def per_user(D, top, k):
    """Recall@k, NDCG@k and the top-k genre mix for every evaluated user."""
    R, N, Q = np.zeros(len(D.users)), np.zeros(len(D.users)), np.zeros((len(D.users), D.G))
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    for j, u in enumerate(D.users):
        t = np.asarray(top[u][:k])
        gt = D.test[u]
        hits = np.isin(t, gt).astype(float)
        R[j] = hits.sum() / len(gt)
        N[j] = (hits * disc).sum() / disc[:min(len(gt), k)].sum()
        Q[j] = D.Gm[t].sum(0) / k
    return R, N, Q


def metrics(D, top, ks=(10, 20), targets=None):
    """Per-user arrays: R@k, N@k, C_KL@20 against the training mix (and optional targets)."""
    out = {}
    for k in ks:
        R, N, Q = per_user(D, top, k)
        out["R@%d" % k], out["N@%d" % k] = R, N
        if k == 20:
            out["C_KL@20"] = ckl(D.his_g, Q)
            for tn, P in (targets or {}).items():
                out["C_KL@20|" + tn] = ckl(P, Q)
    return out


def beyond(D, top, k=20):
    cnt = np.zeros(max(D.cat) + 1)
    for u in D.users:
        np.add.at(cnt, np.asarray(top[u][:k]), 1)
    x = np.sort(np.concatenate([cnt[list(D.cat)], []]))
    n = len(x)
    gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))
    return {"Coverage@%d" % k: float((x > 0).mean()), "Gini@%d" % k: gini}


def rel(a, b):
    return 100.0 * (np.mean(a) / np.mean(b) - 1.0)


# ------------------------------------------------------------------ statistics ----
TOST_MARGIN = 2.0     # +-2 % relative, fixed before any number was computed
N_BOOT = 2000


def seed_stats(x, y, margin=TOST_MARGIN):
    """Seed-paired relative difference 100*(x_s/y_s - 1): mean, t-based 95% CI and TOST.

    x, y: per-seed metric means, paired by seed (or unpaired equal-length lists).
    TOST: two one-sided t-tests of H0 rel <= -margin and H0 rel >= +margin (df = n-1);
    p_tost = max of the two; equivalence at 5% iff the 90% CI lies inside +-margin.
    """
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = 100 * (x / y - 1)
    n = len(d)
    m = float(d.mean())
    out = {"per_seed_rel_pct": d.tolist(), "rel_pct": m, "n_seeds": n,
           "x_mean": float(x.mean()), "y_mean": float(y.mean())}
    if n < 2:
        return out
    se = d.std(ddof=1) / np.sqrt(n)
    t95, t90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    out["ci95"] = [m - t95 * se, m + t95 * se] if se > 0 else [m, m]
    out["ci90"] = [m - t90 * se, m + t90 * se] if se > 0 else [m, m]
    if se > 0:
        p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)
        p_hi = stats.t.cdf((m - margin) / se, n - 1)
        out["p_t"] = float(2 * (1 - stats.t.cdf(abs(m) / se, n - 1)))
    else:
        p_lo = p_hi = 0.0 if abs(m) < margin else 1.0
        out["p_t"] = 0.0 if m else 1.0
    out["p_tost"] = float(max(p_lo, p_hi))
    out["equivalent"] = bool(out["p_tost"] < 0.05)
    return out


def user_stats(a, b, margin=TOST_MARGIN, seed=0):
    """Per-user paired comparison: Wilcoxon p, bootstrap CI of 100*(mean a / mean b - 1), user-level TOST."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    rel_ = rel(a, b)
    p = float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue) if np.any(a != b) else 1.0
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), (N_BOOT, len(a)))
    bs = 100 * (a[idx].mean(1) / b[idx].mean(1) - 1)
    ci90 = np.percentile(bs, [5, 95])
    return {"rel_pct": float(rel_), "p_wilcoxon": p, "ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
            "ci90": ci90.tolist(), "equivalent": bool(ci90[0] > -margin and ci90[1] < margin),
            "better": int((a > b).sum()), "worse": int((a < b).sum())}
