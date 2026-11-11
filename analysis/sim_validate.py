"""CPU validation of the drift simulator before any GPU time.

Two checks, both on worlds made by tools/simulate_drift.py. The T3/T4
estimators (analysis/favourability.py) are imported, never re-implemented.

1. ESTIMATOR RECOVERY, at two levels.
   a. PURE (the premise of pre-registered P8): genre counts are drawn as a
      multinomial straight from the generator's theta_u(t), with the world's own
      block sizes, so the estimators see exactly the model they assume
      (y_t = theta_t + multinomial noise). Compared with the generating values:
        rho_x, q = sigma_eta^2 / sigma_eps^2 (theta-space lag-1 increment over the
        multinomial noise E[(1 - ||theta||^2) / n]), and alpha0: theta_u(0) ~ Dir(alpha0 pi)
        exactly, so alpha0 is fitted on multinomial counts from theta_u(0) with each
        user's history length (the pooled-history fit is reported too: drift and the
        simplex boundary make the time-averaged theta less dispersed than Dir(alpha0 pi)).
      P8: rho_x within +-0.05, q within a factor 1.5, alpha0 within +-25%.
   b. FULL PIPELINE: the estimators run on the simulated clicks. Clicks see theta
      only through multi-genre items, the logistic link and the exposure, so the
      truth is taken in OBSERVATION space: x_u(t) = E[block click mix | state]
      (truth/x_expected_mix.npy). sigma_eta^2 = latent lag-1 increment of x,
      sigma_eps^2 = mean ||y_block - x_u(t)||^2, alpha_eff = moment-matched
      concentration of the users' time-averaged x.

2. CONFOUNDING GAP. For worlds that differ only in beta (exposure confounding):
     tv_pop_obs_int  TV between the population genre mix of the observational test and of the
                     interventional test
     hist_bias_l1    mean ||observed history genre mix - mean latent theta over the train blocks||_1
     follow_gap      R@20 advantage of a "follow the history" ranker (pop^a * exp(beta_ref <g, d_hist>))
                     over popularity, observational test minus interventional test. A positive gap
                     means the observational test rewards echoing past genres more than true
                     preference does: the bias the interventional set is there to expose.

    python analysis/sim_validate.py --recovery sim_b0_qL_s0 ... --gap sim_b0_qL_s0 simval_b2_qL_s0 ... \
        --out analysis/sim/validation.json
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
import favourability as fav  # noqa: E402


def load_world(name):
    d = os.path.join(REPO, "data", name)
    L = lambda f: np.load(os.path.join(d, f), allow_pickle=True)
    w = {"man": json.load(open(os.path.join(d, "manifest.json"))),
         "train": L("training_list.npy"), "valid": L("validation_dict.npy").item(),
         "test": L("testing_dict.npy").item(), "test_int": L("testing_dict_interventional.npy").item(),
         "ic": L("item_category.npy").item(), "prior": L("confounder_prior.npy").astype(float),
         "theta": L("truth/theta.npy"), "x": L("truth/x_expected_mix.npy"), "items": L("truth/items.npy"),
         "ts": np.load(os.path.join(REPO, "data", name + "_temporal", "per_user", "training_timestamps.npy"))}
    w["G"] = fav.genre_matrix(w["ic"], len(w["prior"]))
    return w


def _fit(blocks):
    vg = fav.variogram(blocks, max_lag=20)
    fit = fav.fit_variogram(vg["lags"], vg["V"], weights=vg["npairs"])
    rec = fav.recency_gain(fit, int(np.median(np.diff(blocks.offsets))))
    dm = fav.dirichlet_mle(blocks.user_counts()[0], pi=blocks.prior)
    pref = fit["preferred"]
    return {"preferred": pref, "rho_x": fit["ar1"]["rho_x"], "q_ar1": fit["ar1"]["q"],
            "q_rw": fit["random_walk"]["q"], "q_pref": fit[pref]["q"],
            "sigma_eta2_rw": fit["random_walk"]["sigma_eta2"], "sigma_eta2_ar1": fit["ar1"]["sigma_eta2"],
            "sigma_eps2_ar1": fit["ar1"]["sigma_eps2"], "sigma_eps2_rw": fit["random_walk"]["sigma_eps2"],
            "gain": rec[pref]["gain_vs_running_mean"], "gamma_opt": rec[pref]["gamma_opt_finite_T"],
            "alpha0": dm["alpha0"], "sigma_eps2_multinomial": vg["sigma_eps2_multinomial"],
            "block_n_mean": vg["block_n_mean"], "V": [round(v, 4) for v in vg["V"][:8]]}


def _latent_rho(z):
    """Persistence the latent path actually has: AR(1) fit to its noise-free variogram.
    Clipping at the simplex boundary and raking lower it below the nominal rho_x."""
    lags = list(range(1, 21))
    V = [float(((z[:, h:] - z[:, :-h]) ** 2).sum(2).mean()) for h in lags]
    return fav.fit_variogram(lags, V)["ar1"]["rho_x"]


def _alpha_eff(z):
    m = z.mean(0)
    return float(np.median(m * (1 - m) / np.maximum(z.var(0), 1e-12) - 1))


def recovery(name, w, seed=0):
    man, gen = w["man"], w["man"]["generating"]
    tr_end = man["blocks"]["train"][1] + 1
    blocks = fav.load_blocks(name)
    bu = blocks.users[blocks.block_user] - 1                  # truth row of each block
    # ---- a. pure: multinomial genre draws from theta with the observed block sizes
    th = w["theta"].astype(float)
    rng = np.random.RandomState(seed)
    th_b = th[bu, blocks.ts]
    cnt = np.stack([rng.multinomial(n, p / p.sum()) for n, p in zip(blocks.n, th_b)]).astype(float)
    pb = fav.GenreBlocks(users=blocks.users, offsets=blocks.offsets, ts=blocks.ts, mass=cnt,
                         n=blocks.n, sq=cnt.sum(1), prior=blocks.prior)   # one genre per item: ||g||^2 = 1
    pure = _fit(pb)
    # alpha0: theta_u(0) is exactly Dir(alpha0 pi); counts with each user's history length
    nu = np.add.reduceat(blocks.n, blocks.offsets[:-1])
    C0 = np.stack([rng.multinomial(n, th[u - 1, 0] / th[u - 1, 0].sum())
                   for u, n in zip(blocks.users, nu)]).astype(float)
    pure["alpha0_pooled_history"] = pure["alpha0"]
    pure["alpha0"] = fav.dirichlet_mle(C0, pi=blocks.prior)["alpha0"]
    eta_th = float(((th[:, 1:tr_end] - th[:, :tr_end - 1]) ** 2).sum(2).mean())
    eps_th = float(((1 - (th_b ** 2).sum(1)) / blocks.n).mean())
    pure_truth = {"rho_x_nominal": gen["rho_x"], "rho_x": _latent_rho(th[:, :tr_end]), "sigma_eta2": eta_th, "sigma_eps2": eps_th, "q": eta_th / eps_th,
                  "alpha0": gen["alpha0"], "alpha_eff_time_avg": _alpha_eff(th[:, :tr_end].mean(1))}
    # ---- b. full pipeline, truth in observation space
    full = _fit(blocks)
    x = w["x"].astype(float)
    eta_x = float(((x[:, 1:tr_end] - x[:, :tr_end - 1]) ** 2).sum(2).mean())
    eps_x = float(((blocks.mix() - x[bu, blocks.ts]) ** 2).sum(1).mean())
    full_truth = {"rho_x_nominal": gen["rho_x"], "rho_x": _latent_rho(x[:, :tr_end]), "sigma_eta2": eta_x, "sigma_eps2": eps_x, "q": eta_x / eps_x,
                  "alpha_eff_time_avg": _alpha_eff(x[:, :tr_end].mean(1))}

    def p8(fitd, truth, use_alpha=True):
        q_fit = fitd["q_pref"]
        out = {"q_ratio": q_fit / truth["q"], "q_within_1.5x": bool(1 / 1.5 <= q_fit / truth["q"] <= 1.5)}
        if truth["rho_x_nominal"] < 0.999:   # a random walk has no rho to recover
            out["rho_err"] = fitd["rho_x"] - truth["rho_x"]
            out["rho_within_0.05"] = bool(abs(out["rho_err"]) <= 0.05)
        if use_alpha:
            out["alpha_rel_err"] = fitd["alpha0"] / truth["alpha0"] - 1
            out["alpha_within_25pct"] = bool(abs(out["alpha_rel_err"]) <= 0.25)
        return out
    return {"world": name, "users": man["users"], "train": man["train"],
            "pure": {"fit": pure, "truth": pure_truth, "p8": p8(pure, pure_truth)},
            "full": {"fit": full, "truth": full_truth, "p8": p8(full, full_truth, use_alpha=False)}}


def recall_at(rank_lists, gt, k=20):
    r = [len(set(rank_lists[u][:k]) & set(g)) / len(g) for u, g in gt.items() if g]
    return float(np.mean(r))


def gap(name, w, beta_ref=8.0, a=0.5):
    items = w["items"]
    pos = {int(i): j for j, i in enumerate(items)}
    Gi = w["G"][items]
    # ML-1M popularity, as the generator uses
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import simulate_drift as sd
    pop = sd.load_catalogue()["pop"]
    hist = {}
    for u, i in w["train"]:
        hist.setdefault(int(u), []).append(pos[int(i)])
    for u, xs in w["valid"].items():
        hist[u] += [pos[int(i)] for i in xs]
    pop_rank, fol_rank = {}, {}
    base = a * np.log(pop)
    pop_order = np.argsort(-base)
    mo_all, mi_all, bias = np.zeros(Gi.shape[1]), np.zeros(Gi.shape[1]), []
    th = w["theta"].astype(float)
    tr_end = w["man"]["blocks"]["train"][1] + 1
    for u, h in hist.items():
        seen = set(h)
        d = Gi[h].sum(0)
        d = d / d.sum()
        sc = base + beta_ref * (Gi @ d)
        sc[list(seen)] = -np.inf
        fol_rank[u] = [int(items[j]) for j in np.argsort(-sc)[:20]]
        pop_rank[u] = [int(items[j]) for j in pop_order if j not in seen][:20]
        mo_all += Gi[[pos[i] for i in w["test"][u]]].sum(0)
        mi_all += Gi[[pos[i] for i in w["test_int"][u]]].sum(0)
        bias.append(np.abs(d - th[u - 1, :tr_end].mean(0)).sum())
    out = {"world": name, "beta": w["man"]["params"]["beta"]}
    for nm, gt in (("obs", w["test"]), ("int", w["test_int"])):
        out["R20_pop_" + nm] = recall_at(pop_rank, gt)
        out["R20_follow_" + nm] = recall_at(fol_rank, gt)
        out["follow_adv_" + nm] = out["R20_follow_" + nm] / out["R20_pop_" + nm] - 1
    # oracle (true click probabilities) on both test sets
    pt = np.load(os.path.join(REPO, "data", name, "truth", "p_true_test.npy")).astype(float)
    orc = {u: [int(items[j]) for j in np.argsort(-pt[u - 1])[:20]] for u in hist}
    out["R20_oracle_obs"] = recall_at(orc, w["test"])
    out["R20_oracle_int"] = recall_at(orc, w["test_int"])
    out["follow_gap"] = out["follow_adv_obs"] - out["follow_adv_int"]
    out["tv_pop_obs_int"] = float(0.5 * np.abs(mo_all / mo_all.sum() - mi_all / mi_all.sum()).sum())
    out["hist_bias_l1"] = float(np.mean(bias))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery", nargs="*", default=[])
    ap.add_argument("--gap", nargs="*", default=[])
    ap.add_argument("--out", default=os.path.join(REPO, "analysis", "sim", "validation.json"))
    a = ap.parse_args()
    res = {"recovery": [], "gap": []}
    for n in a.recovery:
        r = recovery(n, load_world(n))
        res["recovery"].append(r)
        for lvl in ("pure", "full"):
            f, t, c = r[lvl]["fit"], r[lvl]["truth"], r[lvl]["p8"]
            print("{:22s} {:4s} pref={:11s} rho {:.3f}->{:.3f}  q {:.4f}->{:.4f} (x{:.2f})  qRW {:.4f}  gain {:.3f}  "
                  "alpha {}->{:.2f} (eff {:.2f})  P8 {}".format(
                      n, lvl, f["preferred"], t["rho_x"], f["rho_x"], t["q"], f["q_pref"], c["q_ratio"], f["q_rw"],
                      f["gain"], t.get("alpha0", "-"), f["alpha0"], t["alpha_eff_time_avg"],
                      {k: v for k, v in c.items() if isinstance(v, bool)}), flush=True)
    for n in a.gap:
        g = gap(n, load_world(n))
        res["gap"].append(g)
        print("{:28s} beta {:5.1f}  follow adv obs {:+.3f} int {:+.3f} gap {:+.3f}  TV(obs,int) {:.3f}  "
              "hist bias {:.3f}  oracle R20 obs {:.3f} int {:.3f}".format(
                  n, g["beta"], g["follow_adv_obs"], g["follow_adv_int"], g["follow_gap"],
                  g["tv_pop_obs_int"], g["hist_bias_l1"], g["R20_oracle_obs"], g["R20_oracle_int"]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
