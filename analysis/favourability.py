"""Favourability screen for time-aware deconfounding (TDecRS / DCR).

CPU-only statistics over a dataset's user x timestamp-block genre-mix data,
each tied to one result of the theory chapter (T2-T6):

    population_shift      T2   ||dbar_t - dbar||_1 and TV across time windows
    variogram / fit_*     T3   V(h) = E||y_{t+h} - y_t||^2 -> sigma_eps^2, sigma_eta^2, q, rho_x
    gamma_star / recency  T3   gamma* = 1 - k*,  k* = (-q + sqrt(q^2 + 4q)) / 2
    dirichlet_mle         T4   alpha_0 by Minka's fixed point, so kappa* = alpha_0
    calibration_cost      T5   KL(d^1 || d^gamma)  vs  0.5 * chi^2(d^1 || d^gamma)
    mediator_lipschitz    T2   L_u of the DecFM mediator from a checkpoint
    favourability         T6   everything above, plus a predicted regime

Blocks are built exactly as tools/dcr_confounder.py builds them: a user's
interactions sharing one timestamp form one block, and g(i) puts mass
1/|genres(i)| on each of item i's genres. One block's genre MASS is
sum_{i in block} g(i), which sums to the block's item count n.

Importable API (the simulator and the dataset screens use it):

    blocks = load_blocks("ml_1m")                         # from data/<ds>_temporal/per_user
    blocks = blocks_from_arrays(users, items, times, G)   # from any interaction arrays
    blocks = blocks_from_mass(users, times, mass)         # from per-interaction genre mass
    summary = favourability(blocks)                       # dict, JSON-serialisable
    print(summary_line(summary))

CLI:
    python analysis/favourability.py --dataset ml_1m  --checkpoint <DecFM .pth> --out analysis/favourability_ml_1m.json
    python analysis/favourability.py --dataset ml_10m --checkpoint <DecFM .pth> --out analysis/favourability_ml_10m.json
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.special import digamma, gammaln

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir))

# Pre-registered regime thresholds, fixed on 2026-11-01 before the simulator and
# new datasets were run. Don't tune them on those results.
S_HIGH_TV = 0.05          # population drift is "high" if max_t TV(dbar_t, dbar) >= 0.05
Q_HIGH = 0.05             # individual drift is "high" if q >= 0.05 ...
GAIN_HIGH = 0.10          # ... and EWMA(gamma*) cuts one-step MSE by >= 10% vs the running mean


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
@dataclass
class GenreBlocks:
    """Ragged user x block genre-mix data. Blocks are sorted by (user, time).

    users    (U,)   user ids, in block order
    offsets  (U+1,) blocks of users[j] are rows offsets[j]:offsets[j+1]
    ts       (B,)   block timestamp (any monotone unit; seconds for MovieLens)
    mass     (B,K)  sum of g(i) over the block's items; rows sum to n
    n        (B,)   items in the block
    sq       (B,)   sum of ||g(i)||^2 over the block's items (for the multinomial noise check)
    prior    (K,)   the static population prior DecRS uses (catalogue mix), or None
    """
    users: np.ndarray
    offsets: np.ndarray
    ts: np.ndarray
    mass: np.ndarray
    n: np.ndarray
    sq: np.ndarray
    prior: np.ndarray = None

    @property
    def K(self):
        return self.mass.shape[1]

    @property
    def block_user(self):
        """(B,) position j of each block's user in self.users."""
        return np.repeat(np.arange(len(self.users)), np.diff(self.offsets))

    def mix(self):
        """(B,K) normalised block genre mix y_t."""
        return self.mass / self.n[:, None]

    def user_counts(self):
        """(U,K) total genre mass per user and (U,) item count."""
        c = np.add.reduceat(self.mass, self.offsets[:-1], axis=0)
        n = np.add.reduceat(self.n, self.offsets[:-1])
        return c, n


def genre_matrix(item_category, num_groups):
    """g(i): mass 1/|genres(i)| on each genre, indexed by raw item id (as tools/dcr_confounder.py)."""
    G = np.zeros((max(item_category) + 1, num_groups))
    for i, cats in item_category.items():
        for c in cats:
            G[i, c] = 1.0 / len(cats)
    return G


def blocks_from_mass(users, times, mass, prior=None):
    """Blocks from per-interaction genre mass rows (N,K). Rows of one user that
    share a timestamp are summed into one block."""
    users = np.asarray(users)
    times = np.asarray(times)
    mass = np.asarray(mass, dtype=np.float64)
    order = np.lexsort((times, users))
    u, t, m = users[order], times[order], mass[order]
    new = np.ones(len(u), dtype=bool)
    new[1:] = (u[1:] != u[:-1]) | (t[1:] != t[:-1])
    starts = np.flatnonzero(new)
    bmass = np.add.reduceat(m, starts, axis=0)
    bn = np.diff(np.append(starts, len(u)))
    bsq = np.add.reduceat((m ** 2).sum(1), starts)
    bu, bt = u[starts], t[starts]
    ustart = np.flatnonzero(np.r_[True, bu[1:] != bu[:-1]])
    return GenreBlocks(users=bu[ustart], offsets=np.append(ustart, len(bu)),
                       ts=bt, mass=bmass, n=bn.astype(np.int64), sq=bsq,
                       prior=None if prior is None else np.asarray(prior, dtype=np.float64))


def blocks_from_arrays(users, items, times, G, prior=None):
    """Blocks from interaction arrays and a genre matrix G (items x K)."""
    return blocks_from_mass(users, times, G[np.asarray(items)], prior=prior)


def dataset_paths(dataset, repo=REPO):
    d = os.path.join(repo, "data", dataset)
    return {"src": d, "temporal": os.path.join(repo, "data", dataset + "_temporal"),
            "per_user": os.path.join(repo, "data", dataset + "_temporal", "per_user")}


def load_blocks(dataset, repo=REPO, split="train"):
    """Training-history blocks for a DecRS-format dataset with a _temporal/per_user split."""
    p = dataset_paths(dataset, repo)
    ic = np.load(os.path.join(p["src"], "item_category.npy"), allow_pickle=True).item()
    prior = np.load(os.path.join(p["src"], "confounder_prior.npy")).astype(np.float64)
    G = genre_matrix(ic, len(prior))
    if split != "train":
        raise ValueError("only the training history is used by the screen")
    tr = np.load(os.path.join(p["per_user"], "training_list.npy"), allow_pickle=True).astype(np.int64)
    ts = np.load(os.path.join(p["per_user"], "training_timestamps.npy"), allow_pickle=True).astype(np.int64)
    return blocks_from_arrays(tr[:, 0], tr[:, 1], ts, G, prior=prior)


# ----------------------------------------------------------------------------
# T2: population shift
# ----------------------------------------------------------------------------
def quantile_bounds(ts, weights, n_windows):
    """Window edges with equal interaction counts."""
    order = np.argsort(ts)
    cw = np.cumsum(weights[order]) / weights.sum()
    cuts = [ts[order][np.searchsorted(cw, k / n_windows)] for k in range(1, n_windows)]
    return np.array([ts.min()] + cuts + [ts.max() + 1])


def _kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.stack([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab


def population_shift(blocks, n_windows=8, edges=None, n_cells=16, seed=0):
    """T2 statistics.

    genre-marginal:  dbar_t = consumed genre mix of the interactions in window t,
                     dbar = the whole-history consumed mix. Reports ||dbar_t - dbar||_1
                     and TV = half of it (TV between the two genre marginals).
    confounder cells: P(d|t) is the law of the user confounder d_u among window t's
                     interactions. d_u (whole-history mix) is quantised into n_cells
                     k-means cells and TV is taken between cell histograms. A
                     coarsening can only LOWER TV, so this is a lower estimate of
                     TV(P(d|t), P(d)); the genre-marginal one is what the plug-in
                     mediator sees.
    edges: window edges (len n_windows+1) in timestamp units; default equal-count.
    """
    if edges is None:
        edges = quantile_bounds(blocks.ts, blocks.n.astype(float), n_windows)
    edges = np.asarray(edges)
    W = len(edges) - 1
    win = np.clip(np.searchsorted(edges, blocks.ts, side="right") - 1, 0, W - 1)
    dbar = blocks.mass.sum(0) / blocks.n.sum()
    c, nu = blocks.user_counts()
    du = c / nu[:, None]
    lab = _kmeans(du, min(n_cells, len(du)), seed=seed)[blocks.block_user]
    cell_all = np.bincount(lab, weights=blocks.n, minlength=n_cells) / blocks.n.sum()
    rows = []
    for t in range(W):
        m = win == t
        if not m.any():
            continue
        dt = blocks.mass[m].sum(0) / blocks.n[m].sum()
        cell_t = np.bincount(lab[m], weights=blocks.n[m], minlength=n_cells) / blocks.n[m].sum()
        l1 = float(np.abs(dt - dbar).sum())
        rows.append({"window": t, "start": int(edges[t]), "end": int(edges[t + 1]),
                     "interactions": int(blocks.n[m].sum()),
                     "users": int(len(np.unique(blocks.block_user[m]))),
                     "l1": l1, "tv": l1 / 2,
                     "l1_vs_catalogue": None if blocks.prior is None else float(np.abs(dt - blocks.prior).sum()),
                     "tv_cells": float(0.5 * np.abs(cell_t - cell_all).sum())})
    l1s = np.array([r["l1"] for r in rows])
    out = {"windows": rows, "n_windows": len(rows),
           "max_l1": float(l1s.max()), "mean_l1": float(l1s.mean()),
           "max_tv": float(l1s.max() / 2),
           "max_tv_cells": float(max(r["tv_cells"] for r in rows)),
           "n_cells": int(n_cells)}
    if blocks.prior is not None:
        out["l1_consumed_vs_catalogue"] = float(np.abs(dbar - blocks.prior).sum())
    out["dbar_consumed"] = dbar.tolist()
    out["window_shift_vectors"] = [(blocks.mass[win == t].sum(0) / blocks.n[win == t].sum() - dbar).tolist()
                                   for t in range(W) if (win == t).any()]
    return out


# ----------------------------------------------------------------------------
# T3: variogram, local-level / AR(1) fit, optimal decay
# ----------------------------------------------------------------------------
def detrend_by_window(blocks, n_windows=20, edges=None):
    """(B,K) block mixes minus the population mix of their time window.
    Common (population) drift otherwise reads as individual drift in the
    variogram (simulator: true q 0.002 fitted as 0.02 under population drift)."""
    if edges is None:
        edges = quantile_bounds(blocks.ts, blocks.n.astype(float), n_windows)
    W = len(edges) - 1
    win = np.clip(np.searchsorted(edges, blocks.ts, side="right") - 1, 0, W - 1)
    pop = np.zeros((W, blocks.K))
    for k in range(blocks.K):
        pop[:, k] = np.bincount(win, weights=blocks.mass[:, k], minlength=W)
    pop /= np.maximum(np.bincount(win, weights=blocks.n, minlength=W), 1)[:, None]
    return blocks.mix() - pop[win]


def variogram(blocks, max_lag=20, weighting="pairs", min_n=1, detrend=None):
    """Empirical V(h) = E||y_{t+h} - y_t||^2 over block lags h = 1..max_lag,
    within users. y_t is the normalised block mix; ||.||^2 sums over genres, so
    every variance below is a total over the K coordinates.

    weighting = "pairs": every within-user pair counts once (heavy users weigh more);
                "users": per-user mean at each lag, then the mean over users.
    min_n: drop blocks with fewer items (keeps every block by default, as DCR does).
    detrend: None (the pre-registered statistic) or a number of equal-count time
             windows whose population mix is subtracted first (detrend_by_window).
    Also returns the multinomial noise prediction for comparison with the intercept.
    """
    keep = blocks.n >= min_n
    y = blocks.mix() if detrend is None else detrend_by_window(blocks, n_windows=int(detrend))
    bu = blocks.block_user
    # multinomial (iid within user) noise: E||y - theta||^2 = (E||g||^2 - ||theta||^2) / n
    c, nu = blocks.user_counts()
    theta = c / nu[:, None]
    g2 = np.add.reduceat(blocks.sq, blocks.offsets[:-1]) / nu
    per_item = np.maximum(g2 - (theta ** 2).sum(1), 0)
    eps_mult = per_item[bu] / blocks.n
    lags, V, npairs, nusers = [], [], [], []
    for h in range(1, max_lag + 1):
        a = np.arange(len(bu) - h)
        b = a + h
        ok = (bu[a] == bu[b]) & keep[a] & keep[b]
        if not ok.any():
            break
        a, b = a[ok], b[ok]
        d2 = ((y[b] - y[a]) ** 2).sum(1)
        if weighting == "users":
            s = np.bincount(bu[a], weights=d2, minlength=len(blocks.users))
            k = np.bincount(bu[a], minlength=len(blocks.users))
            v = float((s[k > 0] / k[k > 0]).mean())
        else:
            v = float(d2.mean())
        lags.append(h); V.append(v); npairs.append(int(ok.sum()))
        nusers.append(int(len(np.unique(bu[a]))))
    return {"lags": lags, "V": V, "npairs": npairs, "nusers": nusers, "weighting": weighting,
            "detrend_windows": detrend,
            "sigma_eps2_multinomial": float(eps_mult[keep].mean()),
            "block_n_mean": float(blocks.n[keep].mean()),
            "block_n_median": float(np.median(blocks.n[keep]))}


def fit_variogram(lags, V, weights=None, fit_lags=None):
    """Fit the two variograms of T3 by weighted least squares.

    random walk (local level):  V(h) = 2 sigma_eps^2 + h sigma_eta^2
    AR(1) around a user mean:   V(h) = 2 sigma_eps^2 + 2 sigma_x^2 (1 - rho^h)
    For AR(1), the innovation variance is sigma_x^2 (1 - rho^2) and its
    noise-to-signal ratio is q_ar = sigma_x^2 (1 - rho^2) / sigma_eps^2.
    fit_lags: use only h <= fit_lags for the random-walk line (it bends for AR(1)).
    """
    h = np.asarray(lags, float)
    v = np.asarray(V, float)
    w = np.ones_like(h) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    m = h <= (fit_lags or h.max())
    X = np.c_[np.ones(m.sum()), h[m]]
    sw = np.sqrt(w[m])
    beta, *_ = np.linalg.lstsq(X * sw[:, None], v[m] * sw, rcond=None)
    s_eps2 = max(beta[0] / 2, 1e-12)
    s_eta2 = max(beta[1], 0.0)
    rw_sse = float((w[m] * (v[m] - X @ beta) ** 2).sum())
    rw = {"sigma_eps2": float(s_eps2), "sigma_eta2": float(s_eta2),
          "q": float(s_eta2 / s_eps2), "wsse": rw_sse, "fit_lags": int(h[m].max())}
    # AR(1): grid over rho, linear LS for (2 s_eps2, 2 s_x2) at each rho
    best = None
    for rho in np.linspace(0.0, 0.9999, 2000):
        Xa = np.c_[np.ones_like(h), 1 - rho ** h]
        sw = np.sqrt(w)
        bb, *_ = np.linalg.lstsq(Xa * sw[:, None], v * sw, rcond=None)
        if bb[0] < 0 or bb[1] < 0:
            continue
        sse = float((w * (v - Xa @ bb) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, rho, bb)
    sse, rho, bb = best
    s_eps2_a = bb[0] / 2
    s_x2 = bb[1] / 2
    eta_a = s_x2 * (1 - rho ** 2)
    ar = {"sigma_eps2": float(s_eps2_a), "sigma_x2": float(s_x2), "rho_x": float(rho),
          "sigma_eta2": float(eta_a), "q": float(eta_a / max(s_eps2_a, 1e-12)), "wsse": sse}
    # same-lag full-range SSE for the RW line, for a fair comparison
    rw["wsse_all_lags"] = float((w * (v - (beta[0] + beta[1] * h)) ** 2).sum())
    return {"random_walk": rw, "ar1": ar,
            "preferred": "ar1" if sse < 0.5 * rw["wsse_all_lags"] else "random_walk"}


def kalman_gain(q):
    """Steady-state gain of the local-level model: k* = (-q + sqrt(q^2 + 4q)) / 2."""
    q = float(q)
    return 0.0 if q <= 0 else (-q + np.sqrt(q * q + 4 * q)) / 2


def gamma_star(q):
    """Optimal infinite-history EWMA decay for the local-level model: gamma* = 1 - k*."""
    return 1.0 - kalman_gain(q)


def ewma_weights(gamma, T):
    """Normalised DCR weights on lags j = 0..T-1 (j=0 the most recent block)."""
    w = gamma ** np.arange(T, dtype=float) if gamma > 0 else np.r_[1.0, np.zeros(T - 1)]
    return w / w.sum()


def predictor_mse(w, sigma_eps2, sigma_eta2=0.0, rho_x=None, sigma_x2=None):
    """One-step-ahead MSE of xhat = sum_j w_j y_{T-j} (sum w = 1) for x_{T+1}.

    random walk (rho_x None): sigma_eps2 * sum w_j^2 + sigma_eta2 * sum_m W_m^2,
        W_m = sum_{j >= m} w_j  (exact; T3 in the theory chapter).
    AR(1) (rho_x, sigma_x2): sigma_eps2 * sum w^2 + sigma_x2 * (1 - 2 sum_j w_j rho^{j+1}
        + sum_{j,l} w_j w_l rho^{|j-l|}) (stationary, mean unknown so sum w = 1).
    """
    w = np.asarray(w, float)
    if rho_x is None:
        W = np.cumsum(w[::-1])[::-1]
        return float(sigma_eps2 * (w ** 2).sum() + sigma_eta2 * (W ** 2).sum())
    j = np.arange(len(w))
    R = rho_x ** np.abs(j[:, None] - j[None, :])
    return float(sigma_eps2 * (w ** 2).sum()
                 + sigma_x2 * (1 - 2 * (w * rho_x ** (j + 1)).sum() + w @ R @ w))


def recency_gain(fit, T, gammas=None):
    """T3 predictions at history length T (blocks): optimal gamma, MSE gain over the
    running mean (gamma = 1), and the MSE at the decays actually run."""
    gammas = np.round(np.linspace(0.0, 1.0, 101), 2) if gammas is None else np.asarray(gammas)
    out = {"T": int(T)}
    for name in ("random_walk", "ar1"):
        f = fit[name]
        if name == "random_walk":
            mse = lambda g: predictor_mse(ewma_weights(g, T), f["sigma_eps2"], f["sigma_eta2"])
        else:
            mse = lambda g: predictor_mse(ewma_weights(g, T), f["sigma_eps2"],
                                          rho_x=f["rho_x"], sigma_x2=f["sigma_x2"])
        curve = np.array([mse(g) for g in gammas])
        gb = float(gammas[curve.argmin()])
        rm = mse(1.0)
        out[name] = {"gamma_opt_finite_T": gb, "mse_opt": float(curve.min()), "mse_running_mean": rm,
                     "gain_vs_running_mean": float(1 - curve.min() / rm),
                     "mse_at": {("%g" % g): mse(g) for g in (0.3, 0.5, 0.7, 0.9, 1.0)},
                     "gamma_star_steady_state": gamma_star(f["q"]) if name == "random_walk" else None}
    return out


def variogram_by_time(blocks, bins_days=(0, 1, 7, 30, 90, 180, 365, 730, 1460, 1e9),
                      max_blocks=300, seed=0, sec_per_unit=86400.0):
    """V as a function of calendar lag (for long-span data, where a block lag is
    not a fixed amount of time). Users with more than max_blocks blocks are
    subsampled to max_blocks blocks."""
    rng = np.random.default_rng(seed)
    y = blocks.mix()
    edges = np.asarray(bins_days, float)
    s = np.zeros(len(edges) - 1)
    k = np.zeros(len(edges) - 1)
    for j in range(len(blocks.users)):
        a, b = blocks.offsets[j], blocks.offsets[j + 1]
        idx = np.arange(a, b)
        if len(idx) < 2:
            continue
        if len(idx) > max_blocks:
            idx = np.sort(rng.choice(idx, max_blocks, replace=False))
        t = blocks.ts[idx] / sec_per_unit
        Y = y[idx]
        iu, ju = np.triu_indices(len(idx), 1)
        lag = t[ju] - t[iu]
        d2 = ((Y[ju] - Y[iu]) ** 2).sum(1)
        bi = np.clip(np.searchsorted(edges, lag, side="right") - 1, 0, len(edges) - 2)
        s += np.bincount(bi, weights=d2, minlength=len(edges) - 1)
        k += np.bincount(bi, minlength=len(edges) - 1)
    return {"bins_days": [float(e) for e in edges],
            "V": [float(x) if c else None for x, c in zip(s / np.maximum(k, 1), k)],
            "npairs": k.astype(int).tolist()}


# ----------------------------------------------------------------------------
# T4: Dirichlet-multinomial shrinkage
# ----------------------------------------------------------------------------
def dm_loglik(C, alpha):
    """Dirichlet-multinomial log-likelihood (multinomial coefficient dropped;
    real-valued counts are allowed, as the fractional genre mass needs)."""
    n = C.sum(1)
    a0 = alpha.sum()
    return float((gammaln(a0) - gammaln(n + a0)).sum()
                 + (gammaln(C + alpha) - gammaln(alpha)).sum())


def dirichlet_mle(C, pi=None, iters=2000, tol=1e-9):
    """Minka (2000) fixed point for Dirichlet-multinomial counts C (U,K).

    pi None : free alpha;   alpha_k <- alpha_k sum_u[psi(c_uk+alpha_k)-psi(alpha_k)] / sum_u[psi(n_u+a0)-psi(a0)]
    pi given: alpha = s*pi, precision only;
              s <- s sum_u sum_k pi_k[psi(c_uk+s pi_k)-psi(s pi_k)] / sum_u[psi(n_u+s)-psi(s)]
    Returns alpha_0, alpha, the mean pi and the log-likelihood.
    """
    C = np.asarray(C, float)
    n = C.sum(1)
    p = C / n[:, None]
    m = p.mean(0)
    # moment-matching start
    v = p.var(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        s0 = np.nanmedian((m * (1 - m) / np.maximum(v, 1e-12)) - 1)
    s = float(np.clip(s0, 0.1, 1e4))
    if pi is None:
        alpha = m * s
        for it in range(iters):
            a0 = alpha.sum()
            num = (digamma(C + alpha) - digamma(alpha)).sum(0)
            den = (digamma(n + a0) - digamma(a0)).sum()
            new = np.maximum(alpha * num / den, 1e-10)
            if np.abs(new - alpha).max() < tol * max(1.0, alpha.max()):
                alpha = new
                break
            alpha = new
    else:
        pi = np.maximum(np.asarray(pi, float), 1e-8)
        pi = pi / pi.sum()
        for it in range(iters):
            num = (pi * (digamma(C + s * pi) - digamma(s * pi))).sum()
            den = (digamma(n + s) - digamma(s)).sum()
            new = s * num / den
            if abs(new - s) < tol * max(1.0, s):
                s = new
                break
            s = new
        alpha = s * pi
    return {"alpha0": float(alpha.sum()), "alpha": alpha.tolist(), "pi": (alpha / alpha.sum()).tolist(),
            "loglik": dm_loglik(C, alpha), "iterations": it + 1}


def shrinkage_summary(blocks, alpha0, kappas=(10,)):
    """T4 predictions: who is short enough for shrinkage to matter.

    users       : the n_u distribution and the share with n_u < alpha0
    training rows: the share of rows whose causal history n^(<tau) < alpha0 (the
                   rows where TDecRS's rho = n/(n+kappa) actually shrinks)
    rho          : mean shrinkage weight over training rows for kappa = alpha0 and each kappa run
    """
    c, nu = blocks.user_counts()
    # n^(<tau) for every block (rows of a block share it), weighted by the block's rows
    cum = np.cumsum(blocks.n)
    start = np.r_[0, cum[:-1]]
    first = blocks.offsets[:-1]
    n_before = start - np.repeat(start[first], np.diff(blocks.offsets))
    w = blocks.n.astype(float)
    def mean_rho(k):
        return float((w * n_before / (n_before + k)).sum() / w.sum())
    q = np.percentile(nu, [5, 25, 50, 75, 95])
    return {"alpha0": float(alpha0),
            "n_user_quantiles": {"p5": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
                                 "p75": float(q[3]), "p95": float(q[4])},
            "share_users_n_below_alpha0": float((nu < alpha0).mean()),
            "share_rows_history_below_alpha0": float((w * (n_before < alpha0)).sum() / w.sum()),
            "share_rows_history_zero": float((w * (n_before == 0)).sum() / w.sum()),
            "mean_rho_at_alpha0": mean_rho(alpha0),
            "mean_rho_at": {("%g" % k): mean_rho(k) for k in kappas}}


def effective_n(gamma, block_n_mean, T):
    """Effective item count of the DCR EWMA over T blocks of block_n_mean items:
    n_eff = (sum w)^2 / sum w^2 blocks, times the block size."""
    w = ewma_weights(gamma, T)
    return float(block_n_mean / (w ** 2).sum())


# ----------------------------------------------------------------------------
# T5: calibration cost
# ----------------------------------------------------------------------------
def kl_smoothed(p, q, alpha=0.01, eps=1e-12):
    """Steck's smoothed KL, as evaluate.py computes C_KL: sum_{p>eps} p log p / ((1-a)q + a p)."""
    qq = (1 - alpha) * q + alpha * p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > eps, p * np.log(np.where(p > eps, p, 1) / np.where(p > eps, qq, 1)), 0.0)
    return t.sum(-1)


def chi2_half(p, q, alpha=0.01, eps=1e-12):
    """0.5 * chi^2(p || q~) with the same smoothing: 0.5 sum (p - q~)^2 / q~ over p > eps."""
    qq = (1 - alpha) * q + alpha * p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > eps, (p - qq) ** 2 / np.where(qq > 0, qq, 1), 0.0)
    return 0.5 * t.sum(-1)


def user_ewma_mix(blocks, gamma):
    """(U,K) the DCR final-state personal mix d~_u at decay gamma (kappa = 0):
    acc <- gamma * acc + block mass, over all training blocks."""
    out = np.zeros((len(blocks.users), blocks.K))
    bu = blocks.block_user
    # backwards weights: block at position r of a user with last block L gets gamma^(L-r)
    last = np.repeat(blocks.offsets[1:] - 1, np.diff(blocks.offsets))
    w = gamma ** (last - np.arange(len(bu))) if gamma > 0 else (last == np.arange(len(bu))).astype(float)
    for k in range(blocks.K):                  # per column: no (B,K) temporary
        out[:, k] = np.bincount(bu, weights=blocks.mass[:, k] * w, minlength=len(blocks.users))
    return out / out.sum(1, keepdims=True)


def calibration_cost(blocks, gammas=(0.3, 0.5, 0.7, 0.9), alpha=0.01):
    """T5: per-user KL(d^1 || d^gamma) and its 0.5*chi^2 approximation."""
    d1 = user_ewma_mix(blocks, 1.0)
    out = {}
    for g in gammas:
        dg = user_ewma_mix(blocks, g)
        kl = kl_smoothed(d1, dg, alpha)
        ch = chi2_half(d1, dg, alpha)
        l1 = np.abs(d1 - dg).sum(1)
        out["%g" % g] = {"kl_mean": float(kl.mean()), "half_chi2_mean": float(ch.mean()),
                         "ratio_mean": float(kl.mean() / ch.mean()),
                         "corr_kl_chi2": float(np.corrcoef(kl, ch)[0, 1]),
                         "l1_mean": float(l1.mean()),
                         "kl_median": float(np.median(kl)), "half_chi2_median": float(np.median(ch))}
    return out


# ----------------------------------------------------------------------------
# T2 (mediator): Lipschitz constant from a checkpoint
# ----------------------------------------------------------------------------
def _load_model(checkpoint):
    import torch
    sys.path.insert(0, os.path.join(REPO, "code", "ml-1m"))
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def _feature_maps(dataset, repo=REPO):
    """Replicates data_utils.map_features: user features first, then item features."""
    p = dataset_paths(dataset, repo)
    uf = np.load(os.path.join(p["src"], "user_feature_file.npy"), allow_pickle=True).item()
    itf = np.load(os.path.join(p["src"], "item_feature_file.npy"), allow_pickle=True).item()
    fmap = {}
    for src in (uf, itf):
        for ID in src:
            for f in src[ID][0]:
                if f not in fmap:
                    fmap[f] = len(fmap)
    return uf, itf, fmap


def mediator_lipschitz(checkpoint, dataset, repo=REPO, shifts=None, base_prior=None,
                       n_score_users=300, topk=20, seed=0, chunk=256):
    """Lipschitz constant of the DecFM mediator M(d, u) in d, and what a
    population shift does to the ranking.

    M(d,u) = sum_{f<f'} x_f x_f' e_f*e_f' + sum_{f,g} x_f d_g e_f*e_g + sum_{g<g'} d_g d_g' e_g*e_g'
    dM/dd_g = (S_u + sum_{g' != g} d_g' e_g') * e_g,   S_u = sum_f x_f e_f.
    With ||.||_1 on d and ||.||_2 on M, L_u = sup_{d in simplex} max_g ||dM/dd_g||_2.
    Each column norm is convex in d, so the sup is at a vertex:
        L_u = max_g max( ||S_u * e_g||, max_{g' != g} ||(S_u + e_g') * e_g|| ).
    Then ||M(dbar_t,u) - M(dbar,u)||_2 <= L_u ||dbar_t - dbar||_1 (T2, mediator form).

    The score is LINEAR in M: yhat = c_u + c_i + sum_f s_f (S_u + I_i)_f M_f with s the
    eval-mode BatchNorm scale, so a shift dM changes item i's score by
    s*dM . I_i plus a per-user constant that cannot change the ranking.
    shifts: list of (K,) vectors dbar_t - dbar, applied to base_prior (default the
    checkpoint's own confounder_prior). Reports the exact rank-relevant score
    shift and the top-k overlap after the shift, on n_score_users users.
    """
    import torch
    model = _load_model(checkpoint)
    model.eval()
    sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
    Emb = sd["embeddings.weight"]
    E = sd["confounder_embed.weight"]
    K, F = E.shape
    nuf = int(model.num_user_features)
    if "FM_layers.0.weight" in sd:
        s_bn = sd["FM_layers.0.weight"] / np.sqrt(sd["FM_layers.0.running_var"] + 1e-5)
    else:
        s_bn = np.ones(F)
    uf, itf, fmap = _feature_maps(dataset, repo)
    uids = sorted(uf)
    S = np.stack([sum(float(x) * Emb[fmap[f]] for f, x in zip(uf[u][0][:nuf], uf[u][1][:nuf])) for u in uids])
    L = np.empty(len(uids))
    for a in range(0, len(uids), chunk):
        Sc = S[a:a + chunk]                                         # (c,F)
        A = (Sc[:, None, None, :] + E[None, None, :, :]) * E[None, :, None, :]   # (c,g,g',F)
        nA = np.sqrt((A ** 2).sum(-1))
        diag = np.sqrt(((Sc[:, None, :] * E[None]) ** 2).sum(-1))  # (c,g)
        idx = np.arange(K)
        nA[:, idx, idx] = diag
        L[a:a + chunk] = nA.reshape(len(Sc), -1).max(1)
    out = {"checkpoint": os.path.relpath(checkpoint, repo), "K": int(K), "factors": int(F),
           "L_u": {"mean": float(L.mean()), "median": float(np.median(L)),
                   "p90": float(np.percentile(L, 90)), "max": float(L.max())},
           "confounder_embed_norm": {"mean": float(np.linalg.norm(E, axis=1).mean()),
                                     "max": float(np.linalg.norm(E, axis=1).max())},
           "user_sum_norm_median": float(np.median(np.linalg.norm(S, axis=1)))}
    prior = np.asarray(model.confounder_prior.detach().cpu().numpy()).reshape(-1).astype(float) \
        if base_prior is None else np.asarray(base_prior, float)

    def M_of(d, Su):                       # (c,F) for one d and users Su
        V = np.concatenate([np.repeat(Su[:, None, :], 1, 1), (d[:, None] * E)[None].repeat(len(Su), 0)], 1)
        # user features enter via their sum: the user-user term does not depend on d,
        # so only S_u and the confounder rows matter for differences in d.
        return 0.5 * (V.sum(1) ** 2 - (V ** 2).sum(1))

    # typical size of M at the base prior, to put the bound in proportion
    M0 = M_of(prior, S)
    out["M_d_part_norm_median"] = float(np.median(np.linalg.norm(M0, axis=1)))
    if shifts:
        rng = np.random.default_rng(seed)
        sel = rng.choice(len(uids), min(n_score_users, len(uids)), replace=False)
        items = sorted(itf)
        # item feature sums I_i (all item features with their values)
        I = np.stack([sum(float(x) * Emb[fmap[f]] for f, x in zip(itf[i][0], itf[i][1])) for i in items])
        scores = _score_users(model, uf, itf, fmap, [uids[j] for j in sel], items)
        res = []
        for sh in shifts:
            sh = np.asarray(sh, float)
            d = np.clip(prior + sh, 0, None)
            d = d / d.sum()
            l1 = float(np.abs(d - prior).sum())
            dM = M_of(d, S[sel]) - M_of(prior, S[sel])            # (c,F)
            bound = L[sel] * l1
            dy = (dM * s_bn) @ I.T                                 # (c, items): rank-relevant part
            spread = scores.std(1)
            new = scores + dy
            ov = []
            for r in range(len(sel)):
                t0 = np.argpartition(-scores[r], topk)[:topk]
                t1 = np.argpartition(-new[r], topk)[:topk]
                ov.append(len(np.intersect1d(t0, t1)) / topk)
            res.append({"l1": l1,
                        "dM_over_bound_median": float(np.median(np.linalg.norm(dM, axis=1) / np.maximum(bound, 1e-15))),
                        "dM_norm_median": float(np.median(np.linalg.norm(dM, axis=1))),
                        "bound_median": float(np.median(bound)),
                        "rank_shift_over_score_sd_median": float(np.median(dy.std(1) / spread)),
                        "topk_overlap_mean": float(np.mean(ov))})
        out["shift_effects"] = res
        out["topk"] = topk
        out["n_score_users"] = int(len(sel))
    return out


def _score_users(model, uf, itf, fmap, users, items, batch=65536):
    """Model scores for users x all items (DecFM forward, static prior).
    Every item has the same feature list length (id + one slot per genre)."""
    import torch
    UF = np.array([[fmap[f] for f in uf[u][0]] for u in users])
    UV = np.array([uf[u][1] for u in users], dtype=np.float32)
    IF = np.array([[fmap[f] for f in itf[i][0]] for i in items])
    IV = np.array([itf[i][1] for i in items], dtype=np.float32)
    out = np.empty((len(users), len(items)))
    with torch.no_grad():
        for r in range(len(users)):
            F = torch.from_numpy(np.c_[np.repeat(UF[r:r + 1], len(items), 0), IF])
            V = torch.from_numpy(np.c_[np.repeat(UV[r:r + 1], len(items), 0), IV])
            out[r] = model(F, V).double().numpy()
    return out


# ----------------------------------------------------------------------------
# T6: everything, plus the predicted regime
# ----------------------------------------------------------------------------
def favourability(blocks, n_windows=8, edges=None, max_lag=20, fit_lags=None,
                  gammas_run=(0.3, 0.5, 0.7, 0.9, 1.0), kappas_run=(10,), lipschitz=None):
    """The full screen. `lipschitz` is an optional mediator_lipschitz(...) result."""
    pop = population_shift(blocks, n_windows=n_windows, edges=edges)
    vg = variogram(blocks, max_lag=max_lag)
    vgu = variogram(blocks, max_lag=max_lag, weighting="users")
    fit = fit_variogram(vg["lags"], vg["V"], weights=vg["npairs"], fit_lags=fit_lags)
    # diagnostic only (added after the freeze; never used for the regime call)
    vgd = variogram(blocks, max_lag=max_lag, detrend=20)
    fitd = fit_variogram(vgd["lags"], vgd["V"], weights=vgd["npairs"], fit_lags=fit_lags)
    T = int(np.median(np.diff(blocks.offsets)))
    rec = recency_gain(fit, T)
    C, nu = blocks.user_counts()
    dm_free = dirichlet_mle(C)
    dm_cat = dirichlet_mle(C, pi=blocks.prior) if blocks.prior is not None else None
    shr = shrinkage_summary(blocks, dm_cat["alpha0"] if dm_cat else dm_free["alpha0"], kappas_run)
    cal = calibration_cost(blocks, gammas=tuple(g for g in gammas_run if g < 1))
    pref = fit["preferred"]
    q = fit[pref]["q"]
    gain = rec[pref]["gain_vs_running_mean"]
    s_high = pop["max_tv"] >= S_HIGH_TV
    q_high = (q >= Q_HIGH) and (gain >= GAIN_HIGH)
    regime = predict_regime(s_high, q_high)
    out = {"blocks": int(len(blocks.ts)), "users": int(len(blocks.users)),
           "interactions": int(blocks.n.sum()), "K": int(blocks.K),
           "blocks_per_user_median": T, "items_per_block_mean": float(blocks.n.mean()),
           "population_shift": pop, "variogram": vg, "variogram_users": vgu,
           "variogram_fit": fit, "recency": rec,
           "variogram_fit_detrended_diagnostic": fitd,
           "dirichlet_free": {k: v for k, v in dm_free.items() if k != "alpha"},
           "dirichlet_catalogue_pi": None if dm_cat is None else {k: v for k, v in dm_cat.items() if k != "alpha"},
           "shrinkage": shr,
           "effective_n": {("%g" % g): effective_n(g, blocks.n.mean(), T) for g in gammas_run},
           "calibration": cal,
           "thresholds": {"S_HIGH_TV": S_HIGH_TV, "Q_HIGH": Q_HIGH, "GAIN_HIGH": GAIN_HIGH},
           "s_high": bool(s_high), "q_high": bool(q_high), "regime": regime}
    if lipschitz is not None:
        out["lipschitz"] = lipschitz
    return out


def predict_regime(s_high, q_high, beta_high=None):
    """T6 map. beta (exposure confounding) is not identifiable from logged mixes
    alone, so by default both beta rows are returned."""
    def one(b):
        if not b:
            return "R3" if q_high else "R0"
        if q_high:
            return "R4"
        return "R2" if s_high else "R1"
    if beta_high is None:
        return {"beta_low": one(False), "beta_high": one(True)}
    return one(beta_high)


PREDICTED_BEST = {
    "R0": "all ~ FM",
    "R1": "DecRS (static); TDecRS(kappa=0) ~ FMconf <= DecRS on the interventional test",
    "R2": "population time-indexed DecRS, gain <= TV * sup|E[Y|u,i,d]|",
    "R3": "recency models: FMconf ~ TDecRS ~ DecRS-R",
    "R4": "DecRS-R (adjust the static prior, recency as a covariate)",
}


def summary_line(s):
    fit = s["variogram_fit"]
    rw, ar = fit["random_walk"], fit["ar1"]
    pref = fit["preferred"]
    rec = s["recency"][pref]
    dm = s["dirichlet_catalogue_pi"] or s["dirichlet_free"]
    L = s.get("lipschitz", {}).get("L_u", {}).get("median")
    reg = s["regime"]
    return ("TV_max={:.4f} (s {}) | q_rw={:.3f} gamma*={:.2f} rho_x={:.2f} q_ar={:.3f} [{}] "
            "gamma_opt(T={})={:.2f} gain={:.1%} (q {}) | alpha0={:.1f} users n<alpha0={:.1%} | "
            "L_med={} | regime beta-low {} / beta-high {}").format(
        s["population_shift"]["max_tv"], "high" if s["s_high"] else "low",
        rw["q"], gamma_star(rw["q"]), ar["rho_x"], ar["q"], pref,
        s["recency"]["T"], rec["gamma_opt_finite_T"], rec["gain_vs_running_mean"],
        "high" if s["q_high"] else "low",
        dm["alpha0"], s["shrinkage"]["share_users_n_below_alpha0"],
        "n/a" if L is None else "{:.3f}".format(L), reg["beta_low"], reg["beta_high"])


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="reads data/<ds>/ and data/<ds>_temporal/per_user/")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--stored_windows", action="store_true",
                    help="use data/<ds>_temporal/global/window_bounds.npy (ml_1m: the windowed cell's windows)")
    ap.add_argument("--max_lag", type=int, default=20)
    ap.add_argument("--fit_lags", type=int, default=None, help="random-walk line fitted on h <= this")
    ap.add_argument("--checkpoint", default=None, help="DecFM .pth for the mediator Lipschitz constant")
    ap.add_argument("--score_users", type=int, default=300)
    ap.add_argument("--time_variogram", action="store_true", help="also V by calendar lag")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    blocks = load_blocks(args.dataset)
    edges = None
    if args.stored_windows:
        wb = np.load(os.path.join(dataset_paths(args.dataset)["temporal"], "global", "window_bounds.npy"))
        edges = np.r_[wb[:, 0], wb[-1, 1] + 1]
    lip = None
    if args.checkpoint:
        pop = population_shift(blocks, n_windows=args.windows, edges=edges)
        lip = mediator_lipschitz(args.checkpoint, args.dataset, shifts=pop["window_shift_vectors"],
                                 n_score_users=args.score_users)
    s = favourability(blocks, n_windows=args.windows, edges=edges, max_lag=args.max_lag,
                      fit_lags=args.fit_lags, lipschitz=lip)
    if args.time_variogram:
        s["variogram_by_time"] = variogram_by_time(blocks)
    s["dataset"] = args.dataset
    s["summary"] = summary_line(s)
    print(s["summary"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(_jsonable(s), f, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
