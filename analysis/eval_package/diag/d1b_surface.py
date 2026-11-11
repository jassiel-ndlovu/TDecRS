"""D1b: is there a (gamma, kappa) where the TDecRS input helps? A grid over the confounder itself.

m_u(gamma, kappa) = (1 - rho_u) p + rho_u d_u^(gamma),   rho_u = n_u / (n_u + kappa),
p = the population static prior (data/<ds>/confounder_prior.npy), d_u^(gamma) the
gamma-discounted genre mix after the whole training history and n_u the user's
training-row count. This is exactly tools/dcr_confounder.block_confounders' final state:
d_u^(gamma) is taken from it at kappa = 0 and every kappa is derived from it, which is
checked against a direct block_confounders call. kappa = inf gives p for every user
(one column, constant in gamma).

Grid: gamma in {0.1, 0.2, 0.3, 0.5, 0.7, 0.82, 0.9, 0.94, 0.96, 1.0},
      kappa in {0, 1, 3, 10, alpha0/4, alpha0, 4 alpha0, inf}
(alpha0 from analysis/favourability_<ds>.json: 46.9 / 86.5 / 11.8).

The re-ranker and protocols are D1's (d1_headroom.py): seed-1 FM, s' = s + w log(0.01 + <m_u, g_i>),
  full     all test users; w chosen on validation R@20, and on test R@20 (upper bound)
  half-B   users with >= 2 test items, test items split A/B (same split as D1),
           A masked, scored on B; w from validation, and the B-best w (upper bound)
Per cell: R@20 rel. to FM with a paired per-user bootstrap 95% CI, and mean C_KL@20 of
the top-20 against the training mix and against the recent mix d_u^(0.3), kappa = 0
(the DCR input, as in E2c), at the validation-chosen w.

    python analysis/eval_package/diag/d1b_surface.py --ds ml_1m
Output: analysis/eval_package/diag/out/d1b_<ds>.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import ANALYSIS, DS_ALL, LABEL, Data, ckl, ld  # noqa: E402
from d1_headroom import EPS, boot_rel, fm_ckpt  # noqa: E402
from score_all import FastFM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402

OUT = os.path.join(HERE, "out")
GAMMAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.82, 0.9, 0.94, 0.96, 1.0)
GSTAR = {"ml_1m": 0.82, "ml_10m": 0.96, "goodreads": 0.94}
WS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
CB = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DS_ALL))
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds = args.ds
    c = DS_ALL[ds]
    D = Data(ds)
    U = D.users
    n = len(U)
    t0 = time.time()
    a0 = float(json.load(open(os.path.join(ANALYSIS, "favourability_%s.json" % ds)))["shrinkage"]["alpha0"])
    KAPPAS = (0.0, 1.0, 3.0, 10.0, a0 / 4, a0, 4 * a0, np.inf)
    KLAB = ["0", "1", "3", "10", "a0/4=%.1f" % (a0 / 4), "a0=%.1f" % a0, "4a0=%.0f" % (4 * a0), "inf"]

    # ---- the confounder grid
    pop = ld(os.path.join(c["data"], "confounder_prior.npy")).astype(np.float64)
    pop = pop / pop.sum()
    nu = np.array([len(D.train[u]) for u in U], float)[:, None]
    pers = {}
    for g in GAMMAS:
        r = D.recent_mix(g, 0.0)
        pers[g] = np.stack([np.asarray(r[u], float) for u in U])
    chk = D.recent_mix(0.5, a0)                  # derived kappa must equal block_confounders
    rho = nu / (nu + a0)
    der = (1 - rho) * pop + rho * pers[0.5]
    err = np.abs(der / der.sum(1, keepdims=True) - np.stack([np.asarray(chk[u], float) for u in U])).max()
    print("%s: derived kappa=alpha0 vs block_confounders max |diff| %.1e (%.0fs)" % (ds, err, time.time() - t0),
          flush=True)
    assert err < 1e-4
    cells, mix = [], []
    for g in GAMMAS:
        for k in KAPPAS:
            if np.isinf(k):
                m = np.repeat(pop[None], n, 0)
            else:
                rho = nu / (nu + k)
                m = (1 - rho) * pop + rho * pers[g]
                m = m / m.sum(1, keepdims=True)
            cells.append((g, k))
            mix.append(m)
    rec03 = pers[0.3]
    C = len(cells)

    # ---- FM seed 1, labels, the D1 split
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(c["data"], "user_feature_file.npy"), os.path.join(c["data"], "item_feature_file.npy"))
    urf, _, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, D.cat, c["G"], D.train, D.valid, D.test)
    item_idx = np.array(list(item_feature.keys()))
    pos = {i: p for p, i in enumerate(item_idx)}
    GI = torch.tensor(D.Gm[item_idx], dtype=torch.float32)
    with torch.no_grad():
        m = torch.load(fm_ckpt(ds, 1), map_location="cpu", weights_only=False)
        m.eval()
        fast = FastFM(m, aif, aifv)
    rng = np.random.RandomState(1)               # identical to d1_headroom.py's split
    split = {}
    for u in U:
        t = list(D.test[u])
        if len(t) >= 2:
            p = rng.permutation(len(t))
            split[u] = ([t[j] for j in p[:len(t) // 2]], [t[j] for j in p[len(t) // 2:]])
    in_split = np.array([u in split for u in U])

    W = len(WS)
    vR = np.zeros((C, W, n), np.float32)
    tR = np.zeros((C, W, n), np.float32)
    bR = np.zeros((C, W, n), np.float32)
    ckl_tr = np.zeros((C, W))
    ckl_re = np.zeros((C, W))
    Mt = [torch.tensor(x, dtype=torch.float32) for x in mix]
    with torch.no_grad():
        for b in range(0, n, CB):
            ub = U[b:b + CB]
            nb = len(ub)
            mask = torch.stack([urf[u][2] for u in ub])
            S = fast(urf, ub) + mask
            LV, LT, LB = (torch.zeros(nb, len(item_idx)) for _ in range(3))
            mA = mask.clone()
            for j, u in enumerate(ub):
                LV[j, [pos[i] for i in D.valid[u] if i in pos]] = 1.0
                LT[j, [pos[i] for i in D.test[u] if i in pos]] = 1.0
                if u in split:
                    mA[j, [pos[i] for i in split[u][0] if i in pos]] = -999.0
                    LB[j, [pos[i] for i in split[u][1] if i in pos]] = 1.0
            nV, nT, nB = LV.sum(1).clamp(min=1), LT.sum(1), LB.sum(1).clamp(min=1)
            SA = S - mask + mA
            his, rec = D.his_g[b:b + nb], rec03[b:b + nb]
            for ci in range(C):
                A = torch.log(EPS + Mt[ci][b:b + nb] @ GI.T)
                for wi, w in enumerate(WS):
                    if w == 0 and ci > 0:
                        vR[ci, 0, b:b + nb], tR[ci, 0, b:b + nb], bR[ci, 0, b:b + nb] = \
                            vR[0, 0, b:b + nb], tR[0, 0, b:b + nb], bR[0, 0, b:b + nb]
                        ckl_tr[ci, 0], ckl_re[ci, 0] = ckl_tr[0, 0], ckl_re[0, 0]
                        continue
                    top = torch.topk(S + w * A, 20).indices
                    vR[ci, wi, b:b + nb] = (LV.gather(1, top).sum(1) / nV).numpy()
                    tR[ci, wi, b:b + nb] = (LT.gather(1, top).sum(1) / nT).numpy()
                    Q = GI[top].mean(1).numpy()
                    ckl_tr[ci, wi] += ckl(his, Q).sum()
                    ckl_re[ci, wi] += ckl(rec, Q).sum()
                    bR[ci, wi, b:b + nb] = (LB.gather(1, torch.topk(SA + w * A, 20).indices).sum(1) / nB).numpy()
            print("  %d/%d users  %.0fs" % (b + nb, n, time.time() - t0), flush=True)
    ckl_tr /= n
    ckl_re /= n

    # ---- per-cell summary
    has_val = np.array([len(D.valid[u]) > 0 for u in U])
    fmT, fmB = tR[0, 0].astype(float), bR[0, 0].astype(float)
    res = []
    for ci, (g, k) in enumerate(cells):
        wv = int(np.argmax(vR[ci][:, has_val].mean(1)))
        wt = int(np.argmax(tR[ci].mean(1)))
        wb = int(np.argmax(bR[ci][:, in_split].mean(1)))
        row = {"gamma": g, "kappa": None if np.isinf(k) else k, "kappa_label": KLAB[KAPPAS.index(k)],
               "w_valid": WS[wv], "w_test": WS[wt], "w_B": WS[wb],
               "C_KL_train": float(ckl_tr[ci, wv]), "C_KL_recent": float(ckl_re[ci, wv])}
        for key, x, f, sel in (("full_valid_w", tR[ci, wv], fmT, None), ("full_test_w", tR[ci, wt], fmT, None),
                               ("halfB_valid_w", bR[ci, wv], fmB, in_split),
                               ("halfB_B_w", bR[ci, wb], fmB, in_split)):
            x = x.astype(float)
            if sel is not None:
                x, f = x[sel], f[sel]
            r, ci95 = boot_rel(x, f)
            row[key] = {"rel_pct": r, "ci95": ci95}
        res.append(row)
        print("g=%-4g k=%-10s w=%-4g full %+6.2f%% [%+.2f, %+.2f] | test-w %+6.2f%% | half-B %+6.2f%% [%+.2f, %+.2f] "
              "| B-w %+6.2f%% | C_KL tr %.4f re %.4f" % (
                  g, row["kappa_label"], WS[wv], row["full_valid_w"]["rel_pct"], *row["full_valid_w"]["ci95"],
                  row["full_test_w"]["rel_pct"], row["halfB_valid_w"]["rel_pct"], *row["halfB_valid_w"]["ci95"],
                  row["halfB_B_w"]["rel_pct"], row["C_KL_train"], row["C_KL_recent"]), flush=True)
    out = {"dataset": LABEL[ds], "alpha0": a0, "gamma_star": GSTAR[ds], "gammas": GAMMAS,
           "kappa_labels": KLAB, "w_grid": WS, "n_users": n, "n_halfB": int(in_split.sum()),
           "fm": {"R@20": float(fmT.mean()), "R@20_halfB": float(fmB[in_split].mean()),
                  "C_KL_train": float(ckl_tr[0, 0]), "C_KL_recent": float(ckl_re[0, 0])},
           "cells": res}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "d1b_%s.json" % ds), "w"), indent=1)


if __name__ == "__main__":
    main()
