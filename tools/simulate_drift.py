"""Semi-synthetic MovieLens world with known drift and confounding.

WHY
---
Real MovieLens shows no time-aware gain, and the theory (T6)
predicts *where* time-aware deconfounding should help. This generator sets
the three regime knobs by hand, so the prediction can be checked against a
randomised-exposure (interventional) test set where the truth is known.

WHAT IS KEPT FROM ML-1M
-----------------------
The item catalogue (3,883 raw MovieID), its 18 genres, the item feature file,
the catalogue genre prior, item popularity (ML-1M positive counts), and the
item quality b_i and taste embeddings e_i of a trained FM checkpoint
(results/ml-1m/models/FM_*.pth). User taste embeddings e_u are drawn from a
Gaussian fitted to that checkpoint's user-ID embeddings.

THE GENERATIVE MODEL  (block t = 0..T-1, user u, item i, g(i) = 1/|genres| mass)
---------------------------------------------------------------------------------
 population   pi(t)    = softmax( log pi_0 + s * (t/(T-1) - 1/2) * v ),  v ~ N(0, I_K) fixed
 interest     theta_u(0) ~ Dir(alpha0 * pi(0))
              theta_u(t+1) = pi(t+1) + rho_x (theta_u(t) - pi(t))
                             + tau (xi - pi(t+1)),   xi ~ Dir(alpha0 * pi(t+1))
              clipped at 0 and renormalised. This is the
              Proj[rho_x theta + (1-rho_x) pi(t) + eta]. With the default
              tau = sqrt(1 - rho_x^2) the stationary cross-section keeps the
              Dir(alpha0 pi) mean and covariance at every t (alpha0 is the T4
              shrinkage constant by construction); rho_x = 1 with a given tau is a
              random walk around pi(t) with per-block theta-space drift
              sigma_eta^2 = tau^2 * sum_k pi_k (1 - pi_k) / (alpha0 + 1), which with
              the block size sets q (T3). The grid uses the random walk (see GRID).
 history mix  d_u^hist(t) = (sum of g(i) over u's clicks before t + c * pi(t)) / (n + c)
 exposure     E items per block, without replacement, never an item u already
              clicked:  P(expose i) proportional to pop_i^a * exp(beta * <g(i), d_u^hist(t)>)
 click        P(y=1 | exposed) = sigmoid( c0 + w * log(lift_ui(t) + eps)
                                          + sb * z(b_i) + se * z(e_u . e_i) ),
              lift_ui(t) = sum_k g_ik theta_uk(t) / e_k, with e_k the genre share
              of the beta=0 exposure distribution. With w = 1 and a small CTR the
              genre mix of a user's clicks would be ~ theta_u(t) when beta = 0 (up
              to the multi-genre blur). The default w = 2 sharpens it: at w = 1 the
              ~1.25 clicks per block carry too little genre signal for any world to
              reach the screen's q-high. The latent the estimators see is therefore
              x_u(t) = E[block click mix | state], saved as truth/x_expected_mix.npy.
              Beta > 0 tilts the exposed (hence clicked) mix toward the user's
              past genres: that is the confounding path D -> exposure -> Y.
 c0 = c0(t) is calibrated per world and block on a pilot (calibrate_c0) to the
 target clicks per user-block, so every world and every block has the same data
 volume; it is common to all users and items, so it carries no genre signal.

SPLIT (per user, by block)
--------------------------
 train  blocks 0 .. T-5      validation  T-4, T-3      observational test  T-2, T-1
 interventional test: at blocks T-2 and T-1, M items per block drawn uniformly (default M:
 every unseen item, for power: with M=200 interventional R@20 averaged ~0.6 hits per user)
 from the items the user has not clicked in train or validation, clicks drawn
 from the true click model (KuaiRand / Yahoo! R3 style). Users need >= 1 click in
 each of train, validation, observational test and interventional test; the
 first --users qualifying users are kept.

OUTPUTS  (world W; mirrors tools/build_ml10m.py)
-----------------------------------------------
 data/sim_W/                     DecRS files (same names as data/ml_10m), plus
     testing_dict_interventional.npy, manifest.json and truth/ (theta, pi(t),
     the oracle click probabilities over the whole catalogue at the test blocks)
 data/sim_W_temporal/per_user/   training_list.npy + *_timestamps.npy (block index)
 data/sim_W_temporal/global/     window_bounds.npy, prior_by_window_consumed.npy
 data/sim_W_dcr/                 g0.3_k0_static, g0.9_kinf_static (both by
     tools/build_dcr_priors.py --dataset sim_W) and g0.9_kinf_windowed (the
     population-only windowed prior, built here with tools/dcr_confounder.py's
     arithmetic because build_dcr_priors.py --pop windowed is ml_1m-only)

The FM score file the Dec* models blend against (code/ml-1m/inference/
sim_W_FM_user_item_scores.npy) needs a trained FM, so it is made on the GPU machine
by tools/gen_scores.py --data_path data/sim_W after the FM run.

USAGE
-----
    python tools/simulate_drift.py --grid                  # the 8-world grid
    python tools/simulate_drift.py --name sim_x --beta 8 --rho_x 0.7 --s 2.5
    python tools/simulate_drift.py --name sim_x --beta 8 --no_priors   # data only
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
ML1M = os.path.join(REPO, "data", "ml_1m")
FM_CKPT = os.path.join(REPO, "results", "ml-1m", "models",
                       "FM_ml_1m_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda.pth")
sys.path.insert(0, HERE)
import dcr_confounder  # noqa: E402

# The 2 x 2 x 2 grid, values chosen by the CPU validation (analysis/sim_validate.py,
# analysis/sim/README.md). beta_high: the observational/interventional gap is
# large. q: with ~1.25 clicks per block a mean-reverting AR(1) never reaches the
# screen's q-high (its recency gain stays < 1%), so qH is a random walk with
# innovation scale tau, and qL is nearly static. s_high: max TV(pi(t), mean) ~ 0.34.
BETA = {"b0": 0.0, "bH": 8.0}
Q = {"qL": (1.0, 0.05), "qH": (1.0, 0.35)}           # (rho_x, tau): near-static vs random walk
S = {"s0": 0.0, "sH": 2.5}
GRID = [(b, q, s) for b in BETA for q in Q for s in S]


def load_catalogue():
    item_feature = np.load(os.path.join(ML1M, "item_feature_file.npy"), allow_pickle=True).item()
    item_category = np.load(os.path.join(ML1M, "item_category.npy"), allow_pickle=True).item()
    cats = np.load(os.path.join(ML1M, "category_list.npy"), allow_pickle=True)
    prior = np.load(os.path.join(ML1M, "confounder_prior.npy")).astype(np.float64)
    items = np.array(list(item_feature.keys()), dtype=np.int64)
    K = len(cats)
    G = np.zeros((len(items), K))
    for j, i in enumerate(items):
        for c in item_category[int(i)]:
            G[j, c] = 1.0 / len(item_category[int(i)])
    # popularity = ML-1M positive counts (train + valid + test), +1 so every item can be exposed
    pop = np.ones(len(items))
    pos = {int(i): j for j, i in enumerate(items)}
    for u, i in np.load(os.path.join(ML1M, "training_list.npy"), allow_pickle=True):
        pop[pos[int(i)]] += 1
    for f in ("validation_dict.npy", "testing_dict.npy"):
        for xs in np.load(os.path.join(ML1M, f), allow_pickle=True).item().values():
            for i in xs:
                pop[pos[int(i)]] += 1
    return dict(item_feature=item_feature, item_category=item_category, cats=cats,
                prior=prior, items=items, G=G, pop=pop)


def load_fm_parts(items, n_users, rng):
    """Item quality b_i, item taste e_i (BN-scaled) from the FM checkpoint; e_u ~ N(fit to U-ID rows)."""
    import torch
    sys.path.insert(0, os.path.join(REPO, "code", "ml-1m"))         # the pickled FM class lives in model.py
    # the raw feature-name -> index map, built the way data_utils.map_features builds it
    feats = {}
    for path in ("user_feature_file.npy", "item_feature_file.npy"):
        for _id, (names, _v) in np.load(os.path.join(ML1M, path), allow_pickle=True).item().items():
            for f in names:
                if f not in feats:
                    feats[f] = len(feats)
    net = torch.load(FM_CKPT, map_location="cpu", weights_only=False)
    sd = net.state_dict()
    E = sd["embeddings.weight"].numpy().astype(np.float64)
    B = sd["biases.weight"].numpy()[:, 0].astype(np.float64)
    bn = sd["FM_layers.0.weight"].numpy() / np.sqrt(sd["FM_layers.0.running_var"].numpy() + 1e-5)
    idx_i = np.array([feats["I%d" % i] for i in items])
    idx_u = np.array([v for k, v in feats.items() if k.startswith("U") and k[1:].isdigit()])
    e_i = E[idx_i] * bn
    b_i = B[idx_i]
    Eu = E[idx_u]
    e_u = rng.multivariate_normal(Eu.mean(0), np.cov(Eu.T), size=n_users)
    return b_i, e_i, e_u


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def pi_path(prior, s, T, v):
    lp = np.log(prior)
    out = np.zeros((T, len(prior)))
    for t in range(T):
        z = lp + s * (t / (T - 1) - 0.5) * v
        z = np.exp(z - z.max())
        out[t] = z / z.sum()
    return out


def simulate(cfg, cat, fm, c0, n_users, rng, keep_truth=False):
    """Run the world for n_users. Returns per-user click lists (block, item-index) and truth."""
    G, pop = cat["G"], cat["pop"]
    I, K = G.shape
    T = cfg["T"]
    pis = cfg["pis"]
    b_z, e_i, e_u = fm
    e_u = e_u[:n_users]
    pref = cfg["sb"] * b_z[None, :] + cfg["se"] * zscore(e_u @ e_i.T)       # (N, I) static part
    log_pop = cfg["a"] * np.log(pop)
    e_share = (pop ** cfg["a"]) @ G / (pop ** cfg["a"]).sum()               # beta=0 exposure genre share
    Ge = G / e_share[None, :]                                               # lift weights

    alpha0, rho = cfg["alpha0"], cfg["rho_x"]
    tau = cfg["tau"] if cfg.get("tau") is not None else np.sqrt(1 - rho ** 2)
    theta = rng.dirichlet(alpha0 * pis[0], size=n_users)
    clicked = np.zeros((n_users, I), dtype=bool)
    hist_sum = np.zeros((n_users, K))
    hist_n = np.zeros(n_users)
    clicks = [[] for _ in range(n_users)]                                   # (t, item_index)
    thetas = np.zeros((n_users, T, K), dtype=np.float32) if keep_truth else None
    xs = np.zeros((n_users, T, K), dtype=np.float32) if keep_truth else None
    n_exposed_hist = np.zeros(n_users)
    rate = np.zeros(T)

    for t in range(T):
        if keep_truth:
            thetas[:, t] = theta
        # ---- exposure under the logging policy ------------------------------
        dh = (hist_sum + cfg["c_hist"] * pis[t]) / (hist_n + cfg["c_hist"])[:, None]
        logw = log_pop[None, :] + cfg["beta"] * (dh @ G.T)
        logw[clicked] = -np.inf
        gumbel = -np.log(-np.log(rng.random_sample((n_users, I))))
        ex = np.argpartition(-(logw + gumbel), cfg["E"], axis=1)[:, :cfg["E"]]
        if keep_truth:
            # x_u(t) = E[block click mix | state]: exposure share x click probability, over g(i).
            # This is the latent the variogram sees (theta seen through multi-genre items).
            q_ex = np.exp(logw - logw.max(1, keepdims=True))
            q_ex *= 1.0 / (1.0 + np.exp(-(c0[t] + cfg["w"] * np.log(theta @ Ge.T + cfg["eps"]) + pref)))
            xs[:, t] = (q_ex @ G) / q_ex.sum(1, keepdims=True)
        # ---- clicks ------------------------------------------------------------
        rows = np.arange(n_users)[:, None]
        lift = np.einsum("nk,nek->ne", theta, Ge[ex])                       # only the exposed items
        logit = c0[t] + cfg["w"] * np.log(lift + cfg["eps"]) + pref[rows, ex]
        y = rng.random_sample(logit.shape) < 1.0 / (1.0 + np.exp(-logit))
        rate[t] = y.sum() / n_users
        for u, j in zip(*np.nonzero(y)):
            it = ex[u, j]
            clicks[u].append((t, it))
            clicked[u, it] = True
            hist_sum[u] += G[it]
            hist_n[u] += 1
        n_exposed_hist += cfg["E"]
        # ---- interest dynamics -------------------------------------------------
        if t + 1 < T:
            nxt = pis[t + 1]
            xi = rng.dirichlet(alpha0 * nxt, size=n_users)
            theta = nxt + rho * (theta - pis[t]) + tau * (xi - nxt)
            theta = np.clip(theta, 1e-6, None)      # > 0, so raking never divides by an empty genre
            theta /= theta.sum(1, keepdims=True)
            # Clipping at 0 pushes users toward uniform, which would move the
            # population mean away from pi(t): spurious population drift in an
            # s = 0 world. Rake back (multiplicative, keeps theta >= 0): genre
            # means to pi(t+1), then every row to sum 1.
            for _ in range(5):
                theta *= nxt / theta.mean(0)
                theta /= theta.sum(1, keepdims=True)
            assert np.isfinite(theta).all(), "theta left the simplex at block {}".format(t + 1)
    return dict(clicks=clicks, thetas=thetas, xs=xs, pref=pref, Ge=Ge, e_share=e_share, rate=rate)


def true_click_prob(cfg, theta, pref, Ge, c0):
    logit = c0 + cfg["w"] * np.log(theta @ Ge.T + cfg["eps"]) + pref
    return 1.0 / (1.0 + np.exp(-logit))


def calibrate_c0(cfg, cat, fm, rng_seed, n_pilot=300, rounds=4):
    """Per-block intercept c0(t) so that clicks per user-block stay at cfg['target'].

    Without it the click rate decays over the 40 blocks (history-driven exposure
    goes stale under drift, and the best items get used up), leaving the test
    blocks thin. c0(t) is a population-level intercept: it rescales every item's
    click odds in block t alike, so it leaves the genre mix of clicks unchanged.
    Each round runs a pilot and moves c0(t) by log(target / rate(t)).
    """
    c0 = np.full(cfg["T"], -3.3)
    for _ in range(rounds):
        r = simulate(cfg, cat, fm, c0, n_pilot, np.random.RandomState(rng_seed))
        c0 = c0 + np.log(cfg["target"] / np.maximum(r["rate"], 1e-3))
    r = simulate(cfg, cat, fm, c0, n_pilot, np.random.RandomState(rng_seed))
    return c0, r["rate"]


def build_world(name, beta, rho_x, tau, s, a):
    cat = load_catalogue()
    T = a.T
    rng_v = np.random.RandomState(a.seed)              # v and e_u are shared by every world
    v = rng_v.randn(len(cat["cats"]))
    n_sim = int(a.users * a.oversample)
    b_i, e_i, e_u = load_fm_parts(cat["items"], max(n_sim, 400), rng_v)   # >= the c0 pilot size
    fm = (zscore(b_i), e_i, e_u)
    cfg = dict(T=T, beta=beta, rho_x=rho_x, tau=tau, s=s, alpha0=a.alpha0, a=a.pop_exp, E=a.E, w=a.w,
               eps=a.eps, sb=a.sb, se=a.se, c_hist=a.c_hist, target=a.target, M=a.M)
    cfg["pis"] = pi_path(cat["prior"], s, T, v)

    c0, rate = calibrate_c0(cfg, cat, fm, a.seed + 1)
    print("[{}] beta={:g} rho_x={:g} s={:g}  c0(t) {:.2f}..{:.2f}  pilot clicks/user-block {:.2f}..{:.2f}".format(
        name, beta, rho_x, s, c0.min(), c0.max(), rate.min(), rate.max()), flush=True)
    rng = np.random.RandomState(a.seed + 7)
    sim = simulate(cfg, cat, fm, c0, n_sim, rng, keep_truth=True)

    items = cat["items"]
    G = cat["G"]
    tr_end, va_end = T - 4, T - 2                       # train < T-4 <= valid < T-2 <= test
    I = len(items)
    kept, train, train_ts, valid, test, test_int = [], [], [], {}, {}, {}
    valid_ts, test_ts = {}, {}
    p_true, thetas_kept, xs_kept = [], [], []
    rng_int = np.random.RandomState(a.seed + 11)
    for u in range(n_sim):
        cl = sim["clicks"][u]
        tr = [(t, j) for t, j in cl if t < tr_end]
        va = [(t, j) for t, j in cl if tr_end <= t < va_end]
        te = [(t, j) for t, j in cl if t >= va_end]
        if not (tr and va and te):
            continue
        # interventional test: uniform exposure at the test blocks, true clicks
        seen = np.zeros(I, dtype=bool)
        seen[[j for _, j in tr + va]] = True
        cand = np.nonzero(~seen)[0]
        pu = [true_click_prob(cfg, sim["thetas"][u, t].astype(np.float64)[None], sim["pref"][u:u + 1],
                              sim["Ge"], c0[t])[0] for t in range(va_end, T)]
        pos_int = set()
        for k, t in enumerate(range(va_end, T)):
            ex = rng_int.choice(cand, size=min(a.M, len(cand)), replace=False)
            hit = ex[rng_int.random_sample(len(ex)) < pu[k][ex]]
            pos_int.update(hit.tolist())
        if not pos_int:
            continue
        uid = len(kept) + 1
        kept.append(u)
        train += [(uid, int(items[j])) for _, j in tr]
        train_ts += [t for t, _ in tr]
        valid[uid] = [int(items[j]) for _, j in va]
        valid_ts[uid] = [t for t, _ in va]
        test[uid] = [int(items[j]) for _, j in te]
        test_ts[uid] = [t for t, _ in te]
        test_int[uid] = sorted(int(items[j]) for j in pos_int)
        pt = np.mean(pu, 0)
        pt[seen] = 0.0                                   # train/valid items are never recommended
        p_true.append(pt.astype(np.float16))
        thetas_kept.append(sim["thetas"][u])
        xs_kept.append(sim["xs"][u])
        if len(kept) == a.users:
            break

    n_users = len(kept)
    user_feature = {uid: [["U%d" % uid], ["1"]] for uid in range(1, n_users + 1)}
    out = os.path.join(REPO, "data", name)
    tmp = os.path.join(REPO, "data", name + "_temporal", "per_user")
    glb = os.path.join(REPO, "data", name + "_temporal", "global")
    tru = os.path.join(out, "truth")
    for d in (out, tmp, glb, tru):
        os.makedirs(d, exist_ok=True)
    np.save(os.path.join(out, "training_list.npy"), np.array(train, dtype=np.int64))
    np.save(os.path.join(out, "validation_dict.npy"), valid)
    np.save(os.path.join(out, "testing_dict.npy"), test)
    np.save(os.path.join(out, "testing_dict_interventional.npy"), test_int)
    np.save(os.path.join(out, "user_feature_file.npy"), user_feature)
    np.save(os.path.join(out, "item_feature_file.npy"), cat["item_feature"])
    np.save(os.path.join(out, "item_category.npy"), cat["item_category"])
    np.save(os.path.join(out, "category_list.npy"), cat["cats"])
    np.save(os.path.join(out, "confounder_prior.npy"), cat["prior"])
    np.save(os.path.join(tmp, "training_list.npy"), np.array(train, dtype=np.int64))
    np.save(os.path.join(tmp, "training_timestamps.npy"), np.array(train_ts, dtype=np.int64))
    np.save(os.path.join(tmp, "validation_dict.npy"), valid)
    np.save(os.path.join(tmp, "testing_dict.npy"), test)
    np.save(os.path.join(tmp, "validation_timestamps.npy"), valid_ts)
    np.save(os.path.join(tmp, "testing_timestamps.npy"), test_ts)
    np.save(os.path.join(tru, "theta.npy"), np.stack(thetas_kept))          # (N, T, K), user uid = row+1
    np.save(os.path.join(tru, "x_expected_mix.npy"), np.stack(xs_kept))     # (N, T, K)
    np.save(os.path.join(tru, "sim_index.npy"), np.array(kept))
    np.save(os.path.join(tru, "pi_t.npy"), cfg["pis"])
    np.save(os.path.join(tru, "p_true_test.npy"), np.stack(p_true))         # (N, I) in item_feature order
    np.save(os.path.join(tru, "items.npy"), items)

    # windowed population prior: consumed genre mix of training clicks per window
    W = a.window
    bounds, wp = [], []
    gpos = {int(i): j for j, i in enumerate(items)}
    tr_items = np.array([i for _, i in train])
    tr_ts = np.array(train_ts)
    for st in range(0, tr_end, W):
        ed = min(st + W, tr_end) - 1
        m = (tr_ts >= st) & (tr_ts <= ed)
        mix = G[[gpos[int(i)] for i in tr_items[m]]].sum(0)
        wp.append(mix / mix.sum())
        bounds.append((st, ed))
    wp, bounds = np.array(wp), np.array(bounds, dtype=np.int64)
    np.save(os.path.join(glb, "prior_by_window_consumed.npy"), wp)
    np.save(os.path.join(glb, "window_bounds.npy"), bounds)

    tv_pi = 0.5 * np.abs(cfg["pis"] - cfg["pis"].mean(0)).sum(1)
    tv_win = 0.5 * np.abs(wp - wp.mean(0)).sum(1)
    n_int = np.array([len(x) for x in test_int.values()])
    man = {
        "world": name, "seed": a.seed, "users_simulated": n_sim, "users": n_users,
        "items": I, "num_groups": len(cat["cats"]), "num_user_features": 1,
        "params": {k: (float(v) if np.isscalar(v) else None) for k, v in cfg.items() if k != "pis"},
        "c0_t": c0.tolist(), "sim_clicks_per_user_block": sim["rate"].tolist(),
        "train": len(train), "valid": sum(map(len, valid.values())), "test_obs": sum(map(len, test.values())),
        "test_int": int(n_int.sum()), "test_int_per_user_median": float(np.median(n_int)),
        "positives_total": len(train) + sum(map(len, valid.values())) + sum(map(len, test.values())),
        "blocks": {"train": [0, tr_end - 1], "valid": [tr_end, va_end - 1], "test": [va_end, T - 1]},
        "pop_tv_from_mean": {"max_latent_pi": float(tv_pi.max()), "max_window_consumed": float(tv_win.max())},
        "generating": {
            "alpha0": a.alpha0, "rho_x": rho_x, "tau": tau,
            # theta-space innovation variance per block (total over genres)
            "sigma_eta2_theta": float((tau if tau is not None else np.sqrt(1 - rho_x ** 2)) ** 2
                                      * (cat["prior"] * (1 - cat["prior"])).sum() / (a.alpha0 + 1)),
        },
    }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print("[{}] users {}  train {}  valid {}  test_obs {}  test_int {}  (TV pi max {:.3f}, window {:.3f})".format(
        name, n_users, man["train"], man["valid"], man["test_obs"], man["test_int"],
        tv_pi.max(), tv_win.max()), flush=True)
    return man


def build_priors(name):
    py = sys.executable
    for g, k in (("0.3", "0"), ("0.9", "inf")):
        subprocess.run([py, os.path.join(HERE, "build_dcr_priors.py"), "--dataset", name,
                        "--gamma", g, "--kappa", k], check=True, stdout=subprocess.DEVNULL)
    # windowed population-only prior (kappa = inf), same arithmetic as build_dcr_priors
    src = os.path.join(REPO, "data", name)
    tmp = os.path.join(REPO, "data", name + "_temporal")
    out = os.path.join(REPO, "data", name + "_dcr")
    ic = np.load(os.path.join(src, "item_category.npy"), allow_pickle=True).item()
    NG = len(np.load(os.path.join(src, "confounder_prior.npy")))
    G = dcr_confounder.genre_matrix(ic, NG)
    wp = np.load(os.path.join(tmp, "global", "prior_by_window_consumed.npy"))
    wb = np.load(os.path.join(tmp, "global", "window_bounds.npy"))

    def pop_at(ts):
        k = int(np.searchsorted(wb[:, 1], ts))
        return wp[min(k, len(wp) - 1)]

    train = np.load(os.path.join(tmp, "per_user", "training_list.npy"))
    ts = np.load(os.path.join(tmp, "per_user", "training_timestamps.npy"))
    rows = np.zeros((len(train), NG), dtype=np.float32)
    users = {}
    order = np.lexsort((ts, train[:, 0]))               # stable: per user, by block
    starts = np.r_[0, np.nonzero(np.diff(train[order, 0]))[0] + 1, len(order)]
    for a_, b_ in zip(starts[:-1], starts[1:]):
        idx = order[a_:b_]
        r, fin = dcr_confounder.block_confounders(ts[idx].tolist(), train[idx, 1].tolist(), G,
                                                  pop_at, 0.9, float("inf"))
        rows[idx] = r
        users[int(train[idx[0], 0])] = fin
    np.save(os.path.join(out, "train_confounder_g0.9_kinf_windowed.npy"), rows)
    np.save(os.path.join(out, "user_confounder_g0.9_kinf_windowed.npy"), users, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="store_true", help="build the 8 worlds sim_{b0,bH}_{qL,qH}_{s0,sH}")
    ap.add_argument("--name", default=None)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--rho_x", type=float, default=0.995)
    ap.add_argument("--s", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=None,
                    help="innovation scale; default sqrt(1 - rho_x^2) (stationary Dir(alpha0 pi) moments)")
    ap.add_argument("--alpha0", type=float, default=2.0)
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--oversample", type=float, default=1.7,
                    help="simulate users*oversample, keep the first --users that qualify")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--E", type=int, default=20, help="exposures per user-block")
    ap.add_argument("--M", type=int, default=100000,
                    help="uniform exposures per interventional test block (default: every unseen item)")
    ap.add_argument("--target", type=float, default=1.25, help="clicks per user-block (c0 is calibrated to it)")
    ap.add_argument("--pop_exp", type=float, default=0.5, help="a in pop_i^a")
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--sb", type=float, default=0.5)
    ap.add_argument("--se", type=float, default=0.5)
    ap.add_argument("--c_hist", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=4, help="blocks per windowed-population window")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no_priors", action="store_true")
    a = ap.parse_args()

    worlds = ([("sim_{}_{}_{}".format(b, q, s), BETA[b], Q[q][0], Q[q][1], S[s]) for b, q, s in GRID]
              if a.grid else [(a.name, a.beta, a.rho_x, a.tau, a.s)])
    if not a.grid and not a.name:
        ap.error("--name or --grid")
    for name, beta, rho, tau, s in worlds:
        build_world(name, beta, rho, tau, s, a)
        if not a.no_priors:
            build_priors(name)
            print("[{}] confounders built".format(name), flush=True)


if __name__ == "__main__":
    main()
