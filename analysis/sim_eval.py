"""Evaluate the simulator grid on both test sets and score the pre-registered T6 predictions.

For each world data/sim_<b>_<q>_<s>, each training seed and each of the six checkpoints in
results-sim/models/ (FM, DecFM, DecFM windowed, FMconf, TDecRS kappa=0, DecFMR),
every user is scored over the whole catalogue (training items masked, as in
code/ml-1m/evaluate.py), the Dec* models are blended with the world's FM scores
by Eq. 11 exactly as evaluate.Ranking does, and three metrics are computed:

  obs   R@K / N@K on the observational test (next blocks, biased exposure)
  int   R@K / N@K on the interventional test (uniform exposure, true clicks)
  util  true utility@K: sum of the oracle click probabilities of the top K,
        over the best achievable sum (truth/p_true_test.npy). Noise-free.

FMconf is reported raw (as trained and selected) and blended with FM by the same
Eq. 11 lambda_u ("FMconf_blend"), because P1 compares TDecRS with *blended* FMconf.

Then each prediction of the theory chapter (pre-registered on
1 Nov 2026) that the simulator can test is scored on interventional R@20:
P1 (TDecRS ~ blended FMconf, both tests), P2 (re-amplification), P3 (the
Table tab:map ordering for each world's regime, every relation scored
separately), P4 (windowed ~ DecRS when max TV < 0.05). The margin is the
pre-registered +-2% relative. Verdict rules, fixed before any result was seen:
  "~"  confirmed if the 90% paired-bootstrap CI of the relative difference is
       inside [-2%, +2%] (TOST); refuted if |diff| > 2% and the 95% CI excludes 0;
       otherwise inconclusive.
  ">"  confirmed if diff > +2% and the 95% CI excludes 0; refuted if diff < -2%
       and the 95% CI excludes 0; otherwise inconclusive.
  ">=" refuted if diff < -2% and the 95% CI excludes 0; confirmed otherwise.
With several seeds the difference is of the seed-pooled means, and the bootstrap
resamples seeds and users together (two-level, paired across models), so the CI
carries training-seed variance as well as user variance. Each row also lists the
per-seed point differences.

    python analysis/sim_eval.py                       # score everything found, seed 1
    python analysis/sim_eval.py --seeds 1 2 3 --reuse # pooled over seeds; reuse per-seed score files
    python analysis/sim_eval.py --plot                # figure from analysis/sim/sim_eval.json
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, os.pardir))
OUTD = os.path.join(REPO, "analysis", "sim")
RES = os.environ.get("SIM_RESULTS", os.path.join(REPO, "results-sim"))
WORLDS = ["sim_{}_{}_{}".format(b, q, s) for b in ("b0", "bH") for q in ("qL", "qH") for s in ("s0", "sH")]
# label -> (main.py model, dcr tag)
MODELS = {
    "FM": ("FM", ""),
    "DecRS": ("DecFM", ""),
    "windowed": ("DecFM", "g0.9_kinf_windowed"),
    "FMconf": ("FMconf", "g0.3_k0_static"),
    "TDecRS0": ("DecFM", "g0.3_k0_static"),
    "DecRS-R": ("DecFMR", "g0.3_k0_static"),
}
SUFFIX = "_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda"
ALPHA = 0.2
K = 20
MARGIN = 0.02
B_BOOT = 2000

# Table tab:map as groups, best first. Within a group every pair is "~"; every
# member of a group is ">" every member of a later group. R2's first relation
# is "windowed >= DecRS". R4 is scored as written: DecRS-R > every other model,
# TDecRS0 ~ FMconf_blend; DecRS vs FMconf is not predicted.
REGIME_OF = {"b0_qL_s0": "R0", "b0_qL_sH": "R0", "b0_qH_s0": "R3", "b0_qH_sH": "R3*",
             "bH_qL_s0": "R1", "bH_qL_sH": "R2", "bH_qH_s0": "R4", "bH_qH_sH": "R4"}


def regime_relations(reg):
    rel = []

    def chain(groups):
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    rel.append((g[i], "~", g[j]))
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                for x in groups[a]:
                    for y in groups[b]:
                        rel.append((x, ">", y))
    if reg == "R0":
        chain([["FM", "DecRS", "windowed", "FMconf", "TDecRS0", "DecRS-R"]])
    elif reg == "R1":
        chain([["DecRS", "windowed", "DecRS-R"], ["FM", "FMconf", "TDecRS0"]])
    elif reg == "R2":
        rel.append(("windowed", ">=", "DecRS"))
        chain([["DecRS", "DecRS-R"], ["FM", "FMconf", "TDecRS0"]])
        for y in ["FM", "FMconf", "TDecRS0"]:
            rel.append(("windowed", ">", y))
    elif reg in ("R3", "R3*"):
        chain([["FMconf", "TDecRS0", "DecRS-R"], ["FM", "DecRS", "windowed"]])
    elif reg == "R4":
        for y in ["FM", "DecRS", "windowed", "FMconf", "TDecRS0"]:
            rel.append(("DecRS-R", ">", y))
        rel.append(("TDecRS0", "~", "FMconf_blend"))
    return rel


# ----------------------------------------------------------------------------
def ckpt_path(world, label, seed=1):
    m, tag = MODELS[label]
    return os.path.join(RES, "models", "{}_{}{}{}_seed{}.pth".format(m, world, SUFFIX, ("_" + tag) if tag else "",
                                                                     seed))


def fm_scores_path(world, seed=1):
    # seed 1 keeps main.py's checkpoint name, other seeds carry _seed<S>
    return os.path.join(RES, "inference", "{}_FM_user_item_scores{}.npy".format(
        world, "" if seed == 1 else "_seed{}".format(seed)))


def per_user_metrics(top, gt, k=K):
    """Recall@k and NDCG@k per user, the formulas of evaluate.computeTopNAccuracy."""
    r, n = [], []
    for lst, g in zip(top, gt):
        g = set(g)
        hits = [1.0 if lst[j] in g else 0.0 for j in range(k)]
        dcg = sum(h / math.log2(j + 2) for j, h in enumerate(hits))
        idcg = sum(1.0 / math.log2(j + 2) for j in range(min(len(g), k)))
        r.append(sum(hits) / len(g))
        n.append(dcg / idcg)
    return np.array(r), np.array(n)


def score_world(world, seed=1):
    import torch
    sys.path.insert(0, os.path.join(REPO, "code", "ml-1m"))
    os.environ.setdefault("DECRS_DEVICE", "cpu")
    import data_utils
    import evaluate
    from device import get_device
    dev = get_device()
    d = os.path.join(REPO, "data", world)
    with contextlib.redirect_stdout(io.StringIO()):
        uf, itf, _ = data_utils.map_features(os.path.join(d, "user_feature_file.npy"),
                                             os.path.join(d, "item_feature_file.npy"))
        train = data_utils.FMData(os.path.join(d, "training_list.npy"), uf, itf, "log_loss")
    valid = data_utils.loadData(os.path.join(d, "validation_dict.npy"))
    obs = data_utils.loadData(os.path.join(d, "testing_dict.npy"))
    inter = data_utils.loadData(os.path.join(d, "testing_dict_interventional.npy"))
    ic = data_utils.loadData(os.path.join(d, "item_category.npy"))
    users = list(obs.keys())
    item_idx = np.array(list(itf.keys()))
    p_true = np.load(os.path.join(d, "truth", "p_true_test.npy")).astype(np.float64)
    assert (np.load(os.path.join(d, "truth", "items.npy")) == item_idx).all()
    best_util = -np.sort(-p_true, 1)[:, :K].sum(1)
    with contextlib.redirect_stdout(io.StringIO()):
        urf, ukl, aif, aifv = evaluate.pre_ranking(uf, itf, ic, 18, train.train_dict, valid, obs)
    kl_gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    lam = {u: ((ukl[u] - ukl["min_kl_score"]) / kl_gap) ** ALPHA for u in users}
    fm_path = fm_scores_path(world, seed)
    fm_scores = np.load(fm_path, allow_pickle=True).item() if os.path.exists(fm_path) else None

    def raw_scores(net, tag):
        uc = None
        if tag:
            uc = np.load(os.path.join(REPO, "data", world + "_dcr", "user_confounder_{}.npy".format(tag)),
                         allow_pickle=True).item()
        out = {}
        with torch.no_grad():
            for u in users:
                f, v, mask = urf[u]
                n = aif.size(0)
                bf = torch.cat([f.unsqueeze(0).expand(n, -1), aif], 1)
                bv = torch.cat([v.unsqueeze(0).expand(n, -1), aifv], 1)
                if uc is None:
                    p = net(bf, bv)
                else:
                    p = net(bf, bv, torch.tensor(np.asarray(uc[u], np.float32)).unsqueeze(0).expand(n, -1))
                out[u] = (p + mask).cpu().numpy().astype(np.float64)
        return out

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    res = {}
    for label in list(MODELS) + ["FMconf_blend"]:
        base = "FMconf" if label == "FMconf_blend" else label
        path = ckpt_path(world, base, seed)
        if not os.path.exists(path):
            continue
        blend = label == "FMconf_blend" or MODELS[base][0].startswith("Dec")
        if blend and fm_scores is None:
            continue
        if label != "FMconf_blend" or "FMconf" not in res:
            net = torch.load(path, map_location=dev, weights_only=False).to(dev).eval()
            raw = raw_scores(net, MODELS[base][1])
            if base == "FMconf":
                fmconf_raw = raw
        else:
            raw = fmconf_raw
        top = []
        util = []
        for k_, u in enumerate(users):
            s = raw[u]
            if blend:
                s = lam[u] * sig(s) + (1 - lam[u]) * sig(np.asarray(fm_scores[u], np.float64))
            t = np.argpartition(-s, K)[:K]
            t = t[np.argsort(-s[t])]
            top.append(item_idx[t].tolist())
            util.append(p_true[k_, t].sum() / best_util[k_])
        ro, no = per_user_metrics(top, [obs[u] for u in users])
        ri, ni = per_user_metrics(top, [inter[u] for u in users])
        res[label] = {"R_obs": ro, "N_obs": no, "R_int": ri, "N_int": ni, "util": np.array(util)}
        print("  {:14s} obs R@20 {:.4f}  int R@20 {:.4f}  util@20 {:.4f}".format(
            label, ro.mean(), ri.mean(), np.mean(util)), flush=True)
    return res


# ----------------------------------------------------------------------------
def rel_diff(a, b, rng, n=B_BOOT):
    """Point and bootstrap distribution of mean(a)/mean(b) - 1.

    a, b: per-user arrays, or (seeds, users) arrays. Seeds and users are resampled
    together and paired across a and b; with one seed this is the plain user bootstrap."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    S, U = a.shape
    pt = a.mean() / b.mean() - 1
    bs = np.empty(n)
    for i in range(n):
        si = rng.randint(0, S, size=S)
        ui = rng.randint(0, U, size=U)
        bs[i] = a[si][:, ui].mean() / b[si][:, ui].mean() - 1
    return float(pt), np.percentile(bs, [2.5, 5, 95, 97.5])


def verdict(op, pt, ci):
    lo95, lo90, hi90, hi95 = ci
    excl0 = lo95 > 0 or hi95 < 0
    if op == "~":
        if lo90 >= -MARGIN and hi90 <= MARGIN:
            return "confirmed"
        return "refuted" if (abs(pt) > MARGIN and excl0) else "inconclusive"
    if op == ">":
        if pt > MARGIN and lo95 > 0:
            return "confirmed"
        return "refuted" if (pt < -MARGIN and hi95 < 0) else "inconclusive"
    if op == ">=":
        return "refuted" if (pt < -MARGIN and hi95 < 0) else "confirmed"
    raise ValueError(op)


def score_predictions(all_res, tvs, seed=0):
    rng = np.random.RandomState(seed)
    rows = []

    def add(pred, world, a, op, b, metric="R_int"):
        R = all_res[world]
        if a not in R or b not in R:
            rows.append({"pred": pred, "world": world, "rel": "{} {} {}".format(a, op, b), "metric": metric,
                         "verdict": "missing"})
            return
        A, B = np.atleast_2d(R[a][metric]), np.atleast_2d(R[b][metric])
        pt, ci = rel_diff(A, B, rng)
        rows.append({"pred": pred, "world": world, "rel": "{} {} {}".format(a, op, b), "metric": metric,
                     "diff": pt, "ci95": [float(ci[0]), float(ci[3])], "ci90": [float(ci[1]), float(ci[2])],
                     "per_seed_diff": [float(x.mean() / y.mean() - 1) for x, y in zip(A, B)],
                     "verdict": verdict(op, pt, ci)})

    for w in all_res:
        key = w[len("sim_"):]
        reg = REGIME_OF.get(key)
        add("P1", w, "TDecRS0", "~", "FMconf_blend", "R_int")
        add("P1", w, "TDecRS0", "~", "FMconf_blend", "R_obs")
        if key.startswith("bH_qL"):
            add("P2", w, "DecRS", ">", "TDecRS0")
        for a, op, b in regime_relations(reg):
            add("P3:" + reg, w, a, op, b)
        if tvs.get(w, 1.0) < 0.05:
            add("P4", w, "windowed", "~", "DecRS")
    return rows


def plot(js, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = ["DecRS", "windowed", "FMconf", "FMconf_blend", "TDecRS0", "DecRS-R"]
    worlds = [w for w in WORLDS if w in js["summary"]]
    M = np.full((len(worlds), len(models)), np.nan)
    L = np.full_like(M, np.nan)
    H = np.full_like(M, np.nan)
    for i, w in enumerate(worlds):
        for j, m in enumerate(models):
            a = js["summary"][w].get(m)
            if a and "adv_int_vs_FM" in a:
                M[i, j] = 100 * a["adv_int_vs_FM"]
                L[i, j], H[i, j] = (100 * x for x in a["adv_int_vs_FM_ci95"])
    lim = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(worlds)):
        for j in range(len(models)):
            if np.isfinite(M[i, j]):
                sigf = L[i, j] > 0 or H[i, j] < 0
                ax.text(j, i, "{:+.1f}{}".format(M[i, j], "*" if sigf else ""), ha="center", va="center",
                        fontsize=9, color="black")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(["DecRS", "DecRS\nwindowed", "FMconf\n(raw)", "FMconf\n(blend)", "TDecRS\n$\\kappa$=0",
                        "DecRS-R"])
    ax.set_yticks(range(len(worlds)))
    ax.set_yticklabels(["{}  [{}]".format(w[4:].replace("_", " "), REGIME_OF[w[4:]]) for w in worlds])
    seeds = js.get("seeds", [1])
    ax.set_title("Interventional R@20, % change vs FM ({}; * = 95% paired bootstrap CI excludes 0)".format(
        "seed {}".format(seeds[0]) if len(seeds) == 1 else "mean of seeds " + ",".join(map(str, seeds))))
    fig.colorbar(im, ax=ax, label="% vs FM")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", nargs="*", default=WORLDS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--reuse", action="store_true",
                    help="load analysis/sim/sim_eval_per_user_seed<S>.npz when it exists instead of rescoring")
    ap.add_argument("--plot", action="store_true", help="only draw the figure from sim_eval.json")
    a = ap.parse_args()
    os.makedirs(OUTD, exist_ok=True)
    jpath = os.path.join(OUTD, "sim_eval.json")
    if a.plot:
        plot(json.load(open(jpath)), os.path.join(OUTD, "regime_map.png"))
        return
    sys.path.insert(0, HERE)
    import favourability as fav
    # per seed: {world: {model: {metric: per-user array}}}, cached one file per seed
    by_seed = {}
    for sd in a.seeds:
        npz = os.path.join(OUTD, "sim_eval_per_user_seed{}.npz".format(sd))
        cached = {}
        if a.reuse and os.path.exists(npz):
            for key, v in np.load(npz).items():
                w, m, k = key.split("|")
                cached.setdefault(w, {}).setdefault(m, {})[k] = v
        res = {}
        for w in a.worlds:
            if w in cached:
                res[w] = cached[w]
                continue
            if not os.path.exists(ckpt_path(w, "FM", sd)):
                print("skip {} seed {} (no FM checkpoint)".format(w, sd))
                continue
            print(w, "seed", sd, flush=True)
            res[w] = score_world(w, sd)
        by_seed[sd] = res
        np.savez_compressed(npz, **{"{}|{}|{}".format(w, m, k): v for w, R in res.items()
                                    for m, r in R.items() for k, v in r.items()})
    # stack seeds: (S, users) per metric, only worlds and models present at every seed
    all_res, tvs, screen = {}, {}, {}
    for w in a.worlds:
        if not all(w in by_seed[sd] for sd in a.seeds):
            continue
        ms = set.intersection(*(set(by_seed[sd][w]) for sd in a.seeds))
        all_res[w] = {m: {k: np.stack([by_seed[sd][w][m][k] for sd in a.seeds])
                          for k in by_seed[a.seeds[0]][w][m]} for m in ms}
        f = fav.favourability(fav.load_blocks(w))
        tvs[w] = f["population_shift"]["max_tv"]
        screen[w] = {"max_tv": tvs[w], "q_pref": f["variogram_fit"][f["variogram_fit"]["preferred"]]["q"],
                     "gain": f["recency"][f["variogram_fit"]["preferred"]]["gain_vs_running_mean"],
                     "s_high": f["s_high"], "q_high": f["q_high"], "regime": f["regime"]}
    rng = np.random.RandomState(1)
    summary = {}
    for w, R in all_res.items():
        summary[w] = {}
        for m, r in R.items():
            s = {k: float(v.mean()) for k, v in r.items()}
            s.update({k + "_per_seed": v.mean(1).tolist() for k, v in r.items()})
            if "FM" in R and m != "FM":
                for met, tag in (("R_int", "int"), ("R_obs", "obs"), ("util", "util")):
                    pt, ci = rel_diff(r[met], R["FM"][met], rng)
                    s["adv_{}_vs_FM".format(tag)] = pt
                    s["adv_{}_vs_FM_ci95".format(tag)] = [float(ci[0]), float(ci[3])]
            summary[w][m] = s
    preds = score_predictions(all_res, tvs)
    json.dump({"summary": summary, "screen": screen, "predictions": preds, "seeds": a.seeds,
               "regime_of": REGIME_OF, "margin": MARGIN, "K": K}, open(jpath, "w"), indent=1)
    tally = {}
    for p in preds:
        tally.setdefault(p["pred"].split(":")[0], {}).setdefault(p["verdict"], 0)
        tally[p["pred"].split(":")[0]][p["verdict"]] += 1
    print(json.dumps(tally, indent=1))
    print("wrote", jpath)


if __name__ == "__main__":
    main()
