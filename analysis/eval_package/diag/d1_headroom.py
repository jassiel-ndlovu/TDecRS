"""D1: headroom ceiling for genre-level time information.

Seed-1 FM is re-ranked with a genre-mix prior,
    s'(u, i) = s(u, i) + w * log(eps + <m_u, g_i>),      eps = 0.01,
s = FM's masked logit (training items at -999, as evaluate.Ranking), g_i = the item's
genre vector (sums to 1), and m_u one of
    static        d_u^(1), the user's training genre mix
    recent_g<x>   d_u^(gamma), kappa = 0, after the whole training history
                  (tools/dcr_confounder.py), gamma in {0.3, 0.5, 0.7, 0.9, gamma*}
    oracle        the genre mix of the user's TEST positives (not available at ranking time)
    placebo       the static mix of another user (one fixed permutation)
    oracle_split  the honest oracle: users with >= 2 test positives have them split at
                  random into halves A and B; m_u = the genre mix of A, A is masked like
                  training items, and the row (and its FM reference) is scored on B only.
The plain `oracle` contains the evaluated items' own genres. With a median of 1-12 test
items it nearly names them, so it bounds "knowing the test genres", not taste drift;
`oracle_split` is the ceiling for a taste-level genre mix. Every other mix is also
scored on the same half-B protocol (`split` block of the output; w from validation
as above, and the half-B-best w), so the (b)-to-(c) gap is like for like.
w is chosen on validation R@20 (valid positives, same mask). The oracle's w is chosen
on test R@20, so the oracle row is an upper bound: no genre-mix confounder, however
it is built, can move FM further by this route. Every row is also reported at its
test-best w (`test_best`), to show that the tuning is not what limits it.

Metrics: R@10, R@20, N@20 and C_KL@20 (training mix target), relative to FM (w = 0),
with a paired per-user bootstrap 95% CI of the ratio of means (2,000 draws).
Shift subgroup: users whose test genre mix is TV > 0.2 from their training mix. Small
test sets make TV large even without drift, so the share is set against a no-drift
null: n_test items drawn (with replacement) from the user's own training items.

For Goodreads it also builds the E2a control the eval package lacks for that dataset:
ENS_FM_s{2,3}+FM_s1 (FM seed s in DecFM's slot, FM seed 1 as the Eq. 11 partner,
exactly DecRS's blend), saved to diag/topk/goodreads/. Nothing as large as a score
matrix is written.

    python analysis/eval_package/diag/d1_headroom.py --ds ml_1m
Output: analysis/eval_package/diag/out/d1_<ds>.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import DS_ALL, LABEL, REPO, Data, ckl, eq11_lambda, ld  # noqa: E402
from score_all import HP, FastFM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402

RES = os.path.join(REPO, "results-azure")
OUT = os.path.join(HERE, "out")
EPS = 0.01
WS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0)
GAMMAS = {"ml_1m": (0.3, 0.5, 0.7, 0.9, 0.82), "ml_10m": (0.3, 0.5, 0.7, 0.9, 0.96),
          "goodreads": (0.3, 0.5, 0.7, 0.9, 0.94)}       # last = gamma* (favourability.py, T3)
TV_CUT = 0.2
CB = 512
N_BOOT = 2000


def fm_ckpt(ds, seed):
    if ds == "ml_1m":
        return os.path.join(RES, "results-fmseeds", "models", "FM_ml_1m_%s_seed%d.pth" % (HP, seed))
    job = {"ml_10m": {1: "ml10A"}, "goodreads": {1: "grA", 2: "grs2", 3: "grs3"}}[ds][seed]
    return os.path.join(RES, "results-" + job, "models", "FM_%s_%s_seed%d.pth" % (ds, HP, seed))


def boot_rel(a, b, seed=0):
    """Paired bootstrap of 100 * (mean a / mean b - 1) over users: point, 95% CI."""
    rng = np.random.RandomState(seed)
    n, bs = len(a), []
    for _ in range(N_BOOT // 200):
        idx = rng.randint(0, n, (200, n))
        bs.append(a[idx].mean(1) / b[idx].mean(1))
    bs = 100 * (np.concatenate(bs) - 1)
    return float(100 * (a.mean() / b.mean() - 1)), np.percentile(bs, [2.5, 97.5]).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DS_ALL))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds = args.ds
    c = DS_ALL[ds]
    D = Data(ds)
    U = D.users
    t0 = time.time()

    # ---- genre mixes, one row per evaluated user
    mixes = {"static": D.his_g}
    for g in GAMMAS[ds]:
        r = D.recent_mix(g, 0.0)
        mixes["recent_g%g" % g] = np.stack([np.asarray(r[u], float) for u in U])
    f03 = os.path.join(c["dcr"], "user_confounder_g0.3_k0_static.npy")
    if os.path.exists(f03):     # the in-memory rebuild must equal the DCR runs' confounder
        z = ld(f03)
        err = np.abs(mixes["recent_g0.3"] - np.stack([np.asarray(z[u], float) for u in U])).max()
        print("recent_g0.3 vs %s: max |diff| %.1e" % (os.path.basename(f03), err), flush=True)
        assert err < 1e-5
    mixes["oracle"] = D.test_g
    mixes["placebo"] = D.his_g[np.random.RandomState(0).permutation(len(U))]
    names = list(mixes)
    rng = np.random.RandomState(1)
    split = {}      # u -> (A, B) for users with >= 2 test positives
    for u in U:
        t = list(D.test[u])
        if len(t) >= 2:
            p = rng.permutation(len(t))
            split[u] = ([t[j] for j in p[:len(t) // 2]], [t[j] for j in p[len(t) // 2:]])
    in_split = np.array([u in split for u in U])
    mix_A = np.stack([D.gvec(split[u][0]) if u in split else D.his_g[j] for j, u in enumerate(U)])
    print("%s: %d users, mixes built %.0fs" % (ds, len(U), time.time() - t0), flush=True)

    # ---- FM seed 1 (and, for Goodreads, seeds 2-3 for the E2a control)
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(c["data"], "user_feature_file.npy"), os.path.join(c["data"], "item_feature_file.npy"))
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, D.cat, c["G"], D.train, D.valid, D.test)
    item_idx = np.array(list(item_feature.keys()))
    pos = {i: p for p, i in enumerate(item_idx)}
    GI = torch.tensor(D.Gm[item_idx], dtype=torch.float32)                   # (NI, G)
    seeds = [1, 2, 3] if ds == "goodreads" else [1]
    fast = {}
    with torch.no_grad():
        for s in seeds:
            m = torch.load(fm_ckpt(ds, s), map_location="cpu", weights_only=False)
            m.eval()
            fast[s] = FastFM(m, aif, aifv)
            # closed form vs forward() on 4 users
            ub = U[:4]
            NI = aif.size(0)
            feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
            vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
            err = (fast[s](urf, ub) - m(feat, vals).view(len(ub), NI)).abs().max().item()
            print("FM seed %d closed form vs forward: %.1e" % (s, err), flush=True)
            assert err < 1e-4
    all_users = list(D.test.keys())
    lam_all = dict(zip(all_users, eq11_lambda(ukl, all_users)))
    ens = {s: [] for s in seeds[1:]}

    V, W, n = len(names), len(WS), len(U)
    val = {"R@20": np.zeros((V, W, n))}
    tst = {k: np.zeros((V, W, n)) for k in ("R@10", "R@20", "N@20", "C_KL@20")}
    disc = torch.tensor(1.0 / np.log2(np.arange(2, 22)), dtype=torch.float32)
    Mt = {k: torch.tensor(v, dtype=torch.float32) for k, v in mixes.items()}
    MA = torch.tensor(mix_A, dtype=torch.float32)
    snames = [x for x in names if x != "oracle"] + ["oracle_split"]
    Mt["oracle_split"] = MA
    spl = {k: np.zeros((len(snames), W, n)) for k in ("R@20", "N@20")}    # half-B protocol (w=0: FM)
    with torch.no_grad():
        for b in range(0, n, CB):
            ub = U[b:b + CB]
            nb = len(ub)
            mask = torch.stack([urf[u][2] for u in ub])
            S = fast[1](urf, ub) + mask
            labs = {}
            for nm, d in (("valid", D.valid), ("test", D.test)):
                L = torch.zeros(nb, len(item_idx))
                for j, u in enumerate(ub):
                    ix = [pos[i] for i in d[u] if i in pos]
                    L[j, ix] = 1.0
                labs[nm] = L
            nval = labs["valid"].sum(1).clamp(min=1)
            ntst = labs["test"].sum(1)
            idcg = torch.stack([disc[:int(min(k, 20))].sum() for k in ntst.tolist()])
            his = torch.tensor(D.his_g[b:b + nb], dtype=torch.float32)
            for v, nm in enumerate(names):
                A = torch.log(EPS + Mt[nm][b:b + nb] @ GI.T)
                for wi, w in enumerate(WS):
                    if w == 0 and v > 0:      # w = 0 is FM for every mix
                        for k in tst:
                            tst[k][v, 0, b:b + nb] = tst[k][0, 0, b:b + nb]
                        val["R@20"][v, 0, b:b + nb] = val["R@20"][0, 0, b:b + nb]
                        continue
                    top = torch.topk(S + w * A, 20).indices
                    hv = labs["valid"].gather(1, top)
                    val["R@20"][v, wi, b:b + nb] = (hv.sum(1) / nval).numpy()
                    h = labs["test"].gather(1, top)
                    tst["R@10"][v, wi, b:b + nb] = (h[:, :10].sum(1) / ntst).numpy()
                    tst["R@20"][v, wi, b:b + nb] = (h.sum(1) / ntst).numpy()
                    tst["N@20"][v, wi, b:b + nb] = ((h * disc).sum(1) / idcg).numpy()
                    Q = GI[top].mean(1)
                    tst["C_KL@20"][v, wi, b:b + nb] = ckl(his.numpy(), Q.numpy())
            # split-half oracle: mask A, score B
            mA, LB = mask.clone(), torch.zeros(nb, len(item_idx))
            for j, u in enumerate(ub):
                if u in split:
                    mA[j, [pos[i] for i in split[u][0] if i in pos]] = -999.0
                    LB[j, [pos[i] for i in split[u][1] if i in pos]] = 1.0
            nB = LB.sum(1).clamp(min=1)
            idB = torch.stack([disc[:int(min(max(k, 1), 20))].sum() for k in LB.sum(1).tolist()])
            SA = S - mask + mA
            for v, nm in enumerate(snames):
                AA = torch.log(EPS + Mt[nm][b:b + nb] @ GI.T)
                for wi, w in enumerate(WS):
                    h = LB.gather(1, torch.topk(SA + w * AA, 20).indices)
                    spl["R@20"][v, wi, b:b + nb] = (h.sum(1) / nB).numpy()
                    spl["N@20"][v, wi, b:b + nb] = ((h * disc).sum(1) / idB).numpy()
            for s in ens:       # Goodreads E2a control: FM_s in the Dec* slot, FM_s1 partner
                lm = torch.tensor([lam_all[u] for u in ub], dtype=torch.float32).unsqueeze(1)
                e = lm * torch.sigmoid(fast[s](urf, ub) + mask) + (1 - lm) * torch.sigmoid(S)
                ens[s].append(item_idx[torch.topk(e, 100).indices.numpy()].astype(np.int32))
            print("  %d/%d users  %.0fs" % (b + nb, n, time.time() - t0), flush=True)

    if ens:
        od = os.path.join(HERE, "topk", ds)
        os.makedirs(od, exist_ok=True)
        for s, tops in ens.items():
            np.savez_compressed(os.path.join(od, "ENS_FM_s%d+FM_s1.npz" % s), users=np.array(U),
                                top=np.concatenate(tops))

    # ---- selection and reporting
    has_val = np.array([len(D.valid[u]) > 0 for u in U])
    tv = 0.5 * np.abs(D.test_g - D.his_g).sum(1)
    shift = tv > TV_CUT
    rng = np.random.RandomState(2)
    null = []
    for _ in range(5):          # no-drift null: test-sized draws from the user's training items
        g = np.stack([D.gvec(rng.choice(D.train[u], len(D.test[u]))) for u in U])
        null.append((0.5 * np.abs(g - D.his_g).sum(1) > TV_CUT).mean())
    fm = {k: tst[k][0, 0] for k in tst}
    out = {"dataset": LABEL[ds], "n_users": n, "eps": EPS, "w_grid": WS, "gammas": GAMMAS[ds],
           "fm_seed1": {k: float(v.mean()) for k, v in fm.items()},
           "tv_cut": TV_CUT, "share_tv_gt_cut": float(shift.mean()), "n_shift": int(shift.sum()),
           "share_tv_gt_cut_null": float(np.mean(null)),
           "tv_mean": float(tv.mean()), "tv_median": float(np.median(tv)),
           "n_test_median": float(np.median([len(D.test[u]) for u in U])), "rows": {}}
    for v, nm in enumerate(names):
        vbest = int(np.argmax(val["R@20"][v][:, has_val].mean(1)))
        tbest = int(np.argmax(tst["R@20"][v].mean(1)))
        wi = tbest if nm == "oracle" else vbest
        row = {"w": WS[wi], "w_rule": "test (upper bound)" if nm == "oracle" else "valid",
               "valid_R@20_curve": val["R@20"][v][:, has_val].mean(1).round(5).tolist(),
               "test_R@20_curve": tst["R@20"][v].mean(1).round(5).tolist()}
        for k in tst:
            x = tst[k][v, wi]
            r, ci = boot_rel(x, fm[k])
            row[k] = {"mean": float(x.mean()), "rel_pct": r, "ci95": ci}
        r, ci = boot_rel(tst["R@20"][v, tbest], fm["R@20"])
        row["test_best"] = {"w": WS[tbest], "R@20": float(tst["R@20"][v, tbest].mean()), "rel_pct": r, "ci95": ci}
        xs, fs = tst["R@20"][v, wi][shift], fm["R@20"][shift]
        r, ci = boot_rel(xs, fs)
        row["shift_R@20"] = {"fm": float(fs.mean()), "mean": float(xs.mean()), "rel_pct": r, "ci95": ci}
        xs, fs = tst["R@20"][v, wi][~shift], fm["R@20"][~shift]
        row["noshift_R@20"] = {"fm": float(fs.mean()), "mean": float(xs.mean()),
                               "rel_pct": float(100 * (xs.mean() / fs.mean() - 1))}
        out["rows"][nm] = row
        print("%-6s %-14s w=%-4g R@10 %+6.2f%%  R@20 %+6.2f%% [%+.2f, %+.2f]  N@20 %+6.2f%%  C_KL %+6.2f%%  "
              "| test-best w=%g R@20 %+6.2f%% | TV>%.1f R@20 %+6.2f%%" % (
                  LABEL[ds], nm, WS[wi], row["R@10"]["rel_pct"], row["R@20"]["rel_pct"], *row["R@20"]["ci95"],
                  row["N@20"]["rel_pct"], row["C_KL@20"]["rel_pct"], WS[tbest], row["test_best"]["rel_pct"],
                  TV_CUT, row["shift_R@20"]["rel_pct"]), flush=True)
    out["split"] = {"n_users": int(in_split.sum()), "rows": {}}
    sub = in_split & shift
    for v, nm in enumerate(snames):
        tb = int(np.argmax(spl["R@20"][v][:, in_split].mean(1)))
        wv = tb if nm == "oracle_split" else WS.index(out["rows"][nm]["w"])
        row = {"w": WS[wv], "w_rule": "half B (upper bound)" if nm == "oracle_split" else "valid",
               "test_R@20_curve": spl["R@20"][v][:, in_split].mean(1).round(5).tolist()}
        for k in spl:
            x, f = spl[k][v, wv, in_split], spl[k][v, 0, in_split]
            r, ci = boot_rel(x, f)
            row[k] = {"fm": float(f.mean()), "mean": float(x.mean()), "rel_pct": r, "ci95": ci}
        r, ci = boot_rel(spl["R@20"][v, tb, in_split], spl["R@20"][v, 0, in_split])
        row["test_best"] = {"w": WS[tb], "rel_pct": r, "ci95": ci}
        r, ci = boot_rel(spl["R@20"][v, wv, sub], spl["R@20"][v, 0, sub])
        row["shift_R@20"] = {"n": int(sub.sum()), "fm": float(spl["R@20"][v, 0, sub].mean()),
                             "mean": float(spl["R@20"][v, wv, sub].mean()), "rel_pct": r, "ci95": ci}
        out["split"]["rows"][nm] = row
        if nm == "oracle_split":
            out["rows"]["oracle_split"] = row
        print("%-6s half-B %-14s w=%-4g R@20 %.4f -> %.4f %+6.2f%% [%+.2f, %+.2f]  N@20 %+6.2f%% | B-best w=%g "
              "%+6.2f%% | TV>%.1f (n=%d) %+6.2f%%" % (
                  LABEL[ds], nm, WS[wv], row["R@20"]["fm"], row["R@20"]["mean"], row["R@20"]["rel_pct"],
                  *row["R@20"]["ci95"], row["N@20"]["rel_pct"], WS[tb], row["test_best"]["rel_pct"], TV_CUT,
                  sub.sum(), row["shift_R@20"]["rel_pct"]), flush=True)
    print("share TV>%.1f: %.3f (no-drift null %.3f)" % (TV_CUT, shift.mean(), np.mean(null)))
    print("share TV>%.1f: %.3f (n=%d); FM R@20 %.4f" % (TV_CUT, shift.mean(), shift.sum(), fm["R@20"].mean()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "d1_%s.json" % ds), "w"), indent=1)


if __name__ == "__main__":
    main()
