"""E4 (KuaiRand-Pure, click): re-score, blend, and test on an observational and an interventional test.

The analysis/goodreads_eval.py method, extended (that file is left untouched):
Stage 1 re-scores the krA/krB checkpoints on CPU along code/ml-1m/evaluate.Ranking's path (training
items masked, Eq. 11 blend with the seed-1 FM scores, the confounder state after the whole history).
FMconf is ranked raw and also blended offline. For each model it keeps
  (a) the top-100 over the catalogue, for testing_dict.npy (standard log 5/05-5/08: observational);
  (b) its score on every RANDOM impression of 5/05-5/08 (data/kuairand_temporal/exposure/random.npy,
      the analysis/kuairand_probe.py protocol: unbiased, since those items were exposed uniformly).
lambda_u is normalised by the min/max KL over the test-(a) users, as in training (so test (a) R@10 must
match the training log); users only in test (b) get lambda clipped to [0, 1].
FM+FM control (D3): an FM of another seed in the DecFM slot, seed-1 FM partner, same lambda_u.

Stage 2:
- obs: per-user R@10/R@20/N@10/N@20, C_KL@20 against the training genre mix, coverage@20.
- int: per-user AUC of clicked vs unclicked random impressions (users with both classes, training and
  validation items excluded), plus R@20 of random clicks over the catalogue (low resolution, reported only).
- Paired per-user Wilcoxon + bootstrap CI + TOST +-2%, seed-level means/SD/paired t, and the FM+FM
  shares. For AUC, TOST and rel % are relative to the comparator's mean AUC; dAUC is also reported.
- out["prereg"]: the decision rules pre-registered on 2 Nov 2026, before the runs: test (b)
  absolute margin 0.003 AUC, test (a) relative 2%; Superior needs the margin, a per-user paired bootstrap
  95% CI above 0 (2,000 resamples over seed-averaged users) and a 3-seed paired t with p < alpha (0.025 for
  the two primary candidates, 0.05 elsewhere); Equivalent = 90% CI inside the margin; primary win = a
  candidate (DecRS-W, TDecRS) Superior on (b) to each of DecRS, DecRS-pool, FMconf_blend and FM+FM.
  FM+FM ("ENS") is seed-paired as frozen: s1 = FM2+FM1, s2 = FM3+FM1, s3 = FM2+FM1.

    python analysis/kuairand_eval.py        # stage 1 cached in analysis/topk_kuairand/
Output: analysis/kuairand_results.json
"""
import datetime as dt
import json
import math
import os
import sys
import time

os.environ.setdefault("DECRS_DEVICE", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
CODE = os.path.join(REPO, "code", "ml-1m")
sys.path.insert(0, CODE)

import numpy as np
import torch
from scipy import stats

import data_utils
import evaluate
import model as _model  # noqa: F401  (needed to unpickle the checkpoints)

DS = "kuairand"
DATA = os.path.join(REPO, "data", DS)
TMP = os.path.join(REPO, "data", DS + "_temporal")
DCR = os.path.join(REPO, "data", DS + "_dcr")
RES = os.environ.get("KR_SANDBOX") or os.path.join(REPO, "results-azure")
OUT = os.path.join(os.environ.get("KR_SANDBOX") or HERE, "topk_" + DS)
os.makedirs(OUT, exist_ok=True)
NAME = "{}_" + DS + "_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth"
JOBS = ["krA", "krB"]
SEEDS = [1, 2, 3]
G = 24
GSTAR = "g0.98"
TOPN = [10, 20, 50, 100]
UB = 16
TEST_START = int(dt.datetime(2022, 5, 5, tzinfo=dt.timezone(dt.timedelta(hours=8))).timestamp())
torch.set_num_threads(os.cpu_count() or 4)

user_feature, item_feature, _ = data_utils.map_features(
    os.path.join(DATA, "user_feature_file.npy"), os.path.join(DATA, "item_feature_file.npy"))
train_dict = {}
for u, i in np.load(os.path.join(DATA, "training_list.npy"), allow_pickle=True).tolist():
    train_dict.setdefault(u, []).append(i)
valid_dict = data_utils.loadData(os.path.join(DATA, "validation_dict.npy"))
test_dict = data_utils.loadData(os.path.join(DATA, "testing_dict.npy"))
test_rand = data_utils.loadData(os.path.join(DATA, "testing_dict_rand.npy"))
item_category = data_utils.loadData(os.path.join(DATA, "item_category.npy"))
item_idx = torch.tensor(list(item_feature.keys()))
item_pos = {int(i): k for k, i in enumerate(item_idx.tolist())}

# interventional impressions (pre-registered protocol = analysis/kuairand_probe.py auc_rows): every random
# impression of 5/05-5/08, minus items among the user's training positives; users need >= 1 clicked and
# >= 1 unclicked impression; ties get the mid-rank (Mann-Whitney U)
_r = np.load(os.path.join(TMP, "exposure", "random.npy"))
_r = _r[_r[:, 2] >= TEST_START]
rand_imp = {}
for u, i, _t, c, _l in _r.tolist():
    if u in train_dict and i not in set(train_dict[u]):
        rand_imp.setdefault(u, []).append((i, c))
rand_imp = {u: (np.array([i for i, _ in v]), np.array([c for _, c in v])) for u, v in rand_imp.items()}
rand_imp = {u: v for u, v in rand_imp.items() if 0 < v[1].sum() < len(v[1])}
users = list(test_dict.keys())                          # test (a), the users ranked in training
users_all = users + [u for u in sorted(set(rand_imp) | set(test_rand)) if u not in test_dict]


def ckpt(model, tag, seed):
    f = NAME.format(model, "_" + tag if tag else "", seed)
    for j in JOBS:
        p = os.path.join(RES, "results-" + j, "models", f)
        if os.path.exists(p):
            return p
    return os.path.join(RES, "results-" + JOBS[0], "models", f)


def sfx(seed):
    return "" if seed == 1 else "_s%d" % seed


def models_for(seed):
    t = GSTAR + "_k0_static"
    return [({"raw": "FM" + sfx(seed)}, ckpt("FM", None, seed), None),
            ({"blend": "DecRS" + sfx(seed)}, ckpt("DecFM", None, seed), None),
            ({"blend": "DecRS-W" + sfx(seed)}, ckpt("DecFM", "g0.9_kinf_windowed", seed), "g0.9_kinf_windowed"),
            ({"blend": "DecRS-pool" + sfx(seed)}, ckpt("DecFM", "g0.9_kinf_pooled", seed), "g0.9_kinf_pooled"),
            ({"raw": "FMconf_raw" + sfx(seed), "blend": "FMconf_blend" + sfx(seed)}, ckpt("FMconf", t, seed), t),
            ({"blend": "TDecRS" + sfx(seed)}, ckpt("DecFM", t, seed), t),
            ({"blend": "DecRS-R" + sfx(seed)}, ckpt("DecFMR", t, seed), t)]


MODELS = [m for s in SEEDS for m in models_for(s) if os.path.exists(m[1])]
ENS = [(a, b) for b in SEEDS for a in SEEDS if a != b]
# pre-registered: in seed s the Dec* slot holds FM seed (s mod 3) + 1, or the next one if that is the partner
ENS_SEED = {s: ((s % 3) + 1 if (s % 3) + 1 != 1 else 2, 1) for s in SEEDS}     # {1: (2,1), 2: (3,1), 3: (2,1)}
ENS_MATCHED = ["ENS_FM_s%d+FM_s%d" % ENS_SEED[s] for s in SEEDS]


def save(name, top, rsc):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), users=np.array(users_all),
                        top=np.array(top, dtype=np.int32),
                        rand_users=np.array(list(rsc)), rand_scores=np.concatenate([rsc[u] for u in rsc]))


def report(name, top, t0):
    t = evaluate.computeTopNAccuracy([test_dict[u] for u in users], top[:len(users)], TOPN)
    print("%-18s test(a) R@10=%.4f N@10=%.4f R@20=%.4f  (%.0fs)" % (name, t[1][0], t[2][0], t[1][1],
                                                                    time.time() - t0), flush=True)


def prep():
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G, train_dict,
                                               valid_dict, {u: None for u in users_all})
    ka = [ukl[u] for u in users]
    lo, hi = min(ka), max(ka)
    lam = {u: float(np.clip((ukl[u] - lo) / (hi - lo), 0, 1)) ** 0.2 for u in users_all}
    return urf, lam, aif, aifv


def batches():
    for b in range(0, len(users_all), UB):
        yield users_all[b:b + UB]


def collect(s_vec, u, top, rsc):
    top.append(item_idx[torch.topk(s_vec, 100).indices].tolist())
    if u in rand_imp:
        rsc[u] = s_vec[[item_pos[i] for i in rand_imp[u][0]]].numpy().astype(np.float32)


def stage1():
    fm1 = ckpt("FM", None, 1)
    if not os.path.exists(fm1):
        sys.exit("no seed-1 FM checkpoint at " + fm1)
    todo = [m for m in MODELS if not all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in m[0].values())]
    if not todo:
        return
    urf, lam, aif, aifv = prep()
    NI = aif.size(0)
    run = [({"raw": "FM"}, fm1, None)] + [m for m in todo if m[1] != fm1]
    fm_score = {}
    for outs, path, tag in run:
        t0 = time.time()
        uconf = None
        if tag:
            uconf = np.load(os.path.join(DCR, "user_confounder_%s.npy" % tag), allow_pickle=True).item()
        m = torch.load(path, map_location="cpu", weights_only=False)
        m.eval()
        if hasattr(m, "confounder_prior"):
            m.confounder_prior = m.confounder_prior.cpu()
        tops = {mode: [] for mode in outs}
        rsc = {mode: {} for mode in outs}
        with torch.no_grad():
            for ub in batches():
                feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
                vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
                if uconf is None:
                    pred = m(feat, vals)
                else:
                    conf = torch.cat([torch.tensor(np.asarray(uconf[u], np.float32)).unsqueeze(0).expand(NI, -1)
                                      for u in ub])
                    pred = m(feat, vals, conf)
                pred = pred.view(len(ub), NI)
                for j, u in enumerate(ub):
                    p = pred[j] + urf[u][2]
                    if path == fm1:
                        fm_score[u] = p.clone()
                    if "raw" in outs:
                        collect(p, u, tops["raw"], rsc["raw"])
                    if "blend" in outs:
                        collect(lam[u] * torch.sigmoid(p) + (1 - lam[u]) * torch.sigmoid(fm_score[u]),
                                u, tops["blend"], rsc["blend"])
        for mode, name in outs.items():
            if not os.path.exists(os.path.join(OUT, name + ".npz")):
                save(name, tops[mode], rsc[mode])
                report(name, tops[mode], t0)


def stage1_ens():
    names = ["ENS_FM_s%d+FM_s%d" % p for p in ENS]
    fms = {s: ckpt("FM", None, s) for s in SEEDS}
    if all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in names) or \
            not all(os.path.exists(f) for f in fms.values()):
        return
    urf, lam, aif, aifv = prep()
    NI = aif.size(0)
    nets = {}
    for s, f in fms.items():
        nets[s] = torch.load(f, map_location="cpu", weights_only=False)
        nets[s].eval()
    tops = {n: [] for n in names}
    rsc = {n: {} for n in names}
    t0 = time.time()
    with torch.no_grad():
        for ub in batches():
            feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
            vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
            sig = {s: torch.sigmoid(nets[s](feat, vals).view(len(ub), NI)
                                    + torch.stack([urf[u][2] for u in ub])) for s in SEEDS}
            for j, u in enumerate(ub):
                for (a, c), n in zip(ENS, names):
                    collect(lam[u] * sig[a][j] + (1 - lam[u]) * sig[c][j], u, tops[n], rsc[n])
    for n in names:
        save(n, tops[n], rsc[n])
        report(n, tops[n], t0)


# ---------------------------------------------------------------- stage 2 ----
def genres(i):
    g = item_category[i]
    return [g] if np.isscalar(g) else list(g)


def gvec(items):
    v = np.zeros(G)
    for i in items:
        g = genres(i)
        for x in g:
            v[x] += 1.0 / len(g)
    return v / v.sum() if v.sum() else v


def ckl(p, q, a=0.01):
    q = (1 - a) * q + a * p
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def per_user(top, gt, k):
    gt = set(gt)
    hits = [1 if i in gt else 0 for i in top[:k]]
    dcg = sum(h / math.log2(j + 2) for j, h in enumerate(hits))
    idcg = sum(1 / math.log2(j + 2) for j in range(min(len(gt), k)))
    return sum(hits) / len(gt), dcg / idcg


def auc(s, y):
    return stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic / (y.sum() * (len(y) - y.sum()))


def stage2():
    his_g = {u: gvec(h) for u, h in train_dict.items()}
    names = list(dict.fromkeys(["FM"] + [n for outs, _, _ in MODELS for n in outs.values()]))
    names += [n for n in ("ENS_FM_s%d+FM_s%d" % p for p in ENS) if os.path.exists(os.path.join(OUT, n + ".npz"))]
    us = [u for u in users if len(test_dict[u])]
    ur = sorted(rand_imp)
    utr = [u for u in users_all if u in test_rand]
    per, out = {}, {"dataset": DS, "gamma_star": GSTAR, "n_users_obs": len(us), "n_users_auc": len(ur),
                    "n_users_rand_recall": len(utr), "models": {}}
    for n in names:
        z = np.load(os.path.join(OUT, n + ".npz"))
        top = dict(zip(z["users"].tolist(), z["top"]))
        off = np.cumsum([0] + [len(rand_imp[u][0]) for u in z["rand_users"].tolist()])
        rsa = z["rand_scores"]                  # read once: each z[key] access re-reads the npz member
        rs = {u: rsa[off[k]:off[k + 1]] for k, u in enumerate(z["rand_users"].tolist())}
        r10, n10, r20, n20, c20, items20 = [], [], [], [], [], set()
        for u in us:
            t = top[u]
            a, b = per_user(t, test_dict[u], 10)
            c, d = per_user(t, test_dict[u], 20)
            r10.append(a); n10.append(b); r20.append(c); n20.append(d)
            c20.append(ckl(his_g[u], gvec(t[:20].tolist())))
            items20.update(t[:20].tolist())
        per[n] = {"R@10": np.array(r10), "N@10": np.array(n10), "R@20": np.array(r20), "N@20": np.array(n20),
                  "C_KL@20": np.array(c20),
                  "AUC_rand": np.array([auc(rs[u], rand_imp[u][1]) for u in ur]),
                  "R@20_rand": np.array([per_user(top[u], test_rand[u], 20)[0] for u in utr])}
        out["models"][n] = {k: float(v.mean()) for k, v in per[n].items()}
        out["models"][n]["coverage@20"] = len(items20) / len(item_feature)
        print(n, {k: round(v, 4) for k, v in out["models"][n].items()})

    for sd in SEEDS:                     # the seed-paired FM+FM model "ENS"
        n = "ENS_FM_s%d+FM_s%d" % ENS_SEED[sd]
        if n in per:
            per["ENS" + sfx(sd)], out["models"]["ENS" + sfx(sd)] = per[n], out["models"][n]

    def test(a, b, key):
        x, y = per[a][key], per[b][key]
        rel = (x.mean() / y.mean() - 1) * 100 if y.mean() else float("nan")
        p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(x != y) else 1.0
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(x), (1000, len(x)))
        bs = (x[idx].mean(1) / y[idx].mean(1) - 1) * 100
        d, m = x - y, 0.02 * abs(y.mean())
        se = d.std(ddof=1) / np.sqrt(len(d))
        p_tost = float(max(stats.t.sf((d.mean() + m) / se, len(d) - 1), stats.t.cdf((d.mean() - m) / se, len(d) - 1))) \
            if se > 0 else float("nan")
        ci90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        return {"a": float(x.mean()), "b": float(y.mean()), "diff": float(d.mean()), "rel_pct": float(rel),
                "p": float(p), "p_t": float(stats.ttest_rel(x, y).pvalue) if np.any(x != y) else 1.0,
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "ci90": ci90, "p_tost2": p_tost, "equiv2_boot": bool(-2 < ci90[0] and ci90[1] < 2)}

    pairs = [("DecRS", "FM"), ("DecRS-W", "DecRS-pool"), ("DecRS-W", "ENS"), ("TDecRS", "ENS"), ("DecRS", "ENS"),
             ("TDecRS", "DecRS-pool"), ("DecRS-R", "FMconf_blend"), ("ENS", "FM"), ("DecRS-W", "DecRS"), ("DecRS-pool", "DecRS"),
             ("TDecRS", "FMconf_blend"), ("TDecRS", "DecRS"), ("TDecRS", "DecRS-R"), ("DecRS-R", "DecRS"),
             ("FMconf_blend", "DecRS"), ("FMconf_raw", "FM"), ("DecRS-W", "FMconf_blend")]
    pairs = [p for p in pairs if p[0] in per and p[1] in per]
    keys = ("R@20", "N@20", "R@10", "C_KL@20", "AUC_rand", "R@20_rand")
    out["tests"] = {"%s_vs_%s" % p: {k: test(p[0], p[1], k) for k in keys} for p in pairs}

    seeds = [s for s in SEEDS if s == 1 or ("FM" + sfx(s)) in per]
    base = ["FM", "DecRS", "DecRS-W", "DecRS-pool", "FMconf_raw", "FMconf_blend", "TDecRS", "DecRS-R", "ENS"]
    full = [b for b in base if all((b + sfx(s)) in per for s in seeds)]
    if len(seeds) > 1:
        for b in full:
            per[b + "_mean"] = {k: np.mean([per[b + sfx(s)][k] for s in seeds], 0) for k in per[b]}
        out["seeds"] = {"seeds": seeds, "models": {}, "tests": {}}
        for b in full:
            out["seeds"]["models"][b] = {}
            for k in keys + ("N@10", "coverage@20"):
                v = [out["models"][b + sfx(s)][k] for s in seeds]
                out["seeds"]["models"][b][k] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        for a, b in [p for p in pairs if p[0] in full and p[1] in full]:
            row = {}
            for k in keys:
                t = test(a + "_mean", b + "_mean", k)
                x = np.array(out["seeds"]["models"][a][k]["per_seed"])
                y = np.array(out["seeds"]["models"][b][k]["per_seed"])
                t["p_seed_t"] = float(stats.ttest_rel(x, y).pvalue)
                t["per_seed_rel_pct"] = [float(v) for v in (x / y - 1) * 100]
                t["per_seed_diff"] = [float(v) for v in x - y]
                row[k] = t
            out["seeds"]["tests"]["%s_vs_%s" % (a, b)] = row
        if all(n in per for n in ENS_MATCHED):
            allp = ["ENS_FM_s%d+FM_s%d" % p for p in ENS]
            per["ENS_mean"] = {k: np.mean([per[n][k] for n in ENS_MATCHED], 0) for k in per["FM"]}
            e = {"matched": ENS_MATCHED, "models": {}, "share_explained": {}, "tests": {}}
            for g, ns in (("ENS (matched)", ENS_MATCHED), ("ENS (all pairs)", allp)):
                e["models"][g] = {k: {"per_member": [out["models"][n][k] for n in ns],
                                      "mean": float(np.mean([out["models"][n][k] for n in ns])),
                                      "sd": float(np.std([out["models"][n][k] for n in ns], ddof=1))}
                                  for k in keys + ("coverage@20",)}
            for b in [x for x in base[1:] if x in full and x not in ("FMconf_raw", "ENS")]:
                e["share_explained"][b] = {}
                for k in keys:
                    f = out["seeds"]["models"]["FM"][k]["mean"]
                    d = out["seeds"]["models"][b][k]["mean"]
                    en = e["models"]["ENS (matched)"][k]["mean"]
                    e["share_explained"][b][k] = {"gain_vs_FM_pct": 100 * (d / f - 1), "ENS_vs_FM_pct": 100 * (en / f - 1),
                                                  "vs_ENS_pct": 100 * (d / en - 1),
                                                  "share": (en - f) / (d - f) if d != f else float("nan")}
                e["tests"]["%s_vs_ENS" % b] = {k: test(b + "_mean", "ENS_mean", k) for k in keys}
            e["tests"]["ENS_vs_FM"] = {k: test("ENS_mean", "FM_mean", k) for k in keys}
            out["ensemble_control"] = e
        out["prereg"] = verdicts(per, out, seeds, full)
        for k, v in out["prereg"]["primary"].items():
            print("PRIMARY", k, v["win"], v["vs"])
        for k, v in out["prereg"]["comparisons"].items():
            print("VERDICT", k, {m: r["verdict"] for m, r in v.items()})
    json.dump(out, open(os.path.join(os.environ.get("KR_SANDBOX") or HERE, DS + "_results.json"), "w"), indent=1)


# ------------------------------------------- pre-registered decision rules ----------------------
DELTA_B = 0.003       # test (b), absolute AUC
DELTA_A = 2.0         # test (a), relative %
NBOOT = 2000


def verdict(per, out, seeds, a, b, key, alpha):
    """Superior / Equivalent / Inferior / Inconclusive, as frozen. Per-user paired bootstrap of the
    seed-averaged per-user values (pooled over the 3 seeds, 2,000 resamples); 3-seed paired t on seed means."""
    x, y = per[a + "_mean"][key], per[b + "_mean"][key]
    rng = np.random.RandomState(0)
    idx = rng.randint(0, len(x), (NBOOT, len(x)))
    xs = np.array([out["models"][a + sfx(s)][key] for s in seeds])
    ys = np.array([out["models"][b + sfx(s)][key] for s in seeds])
    p_seed = float(stats.ttest_rel(xs, ys).pvalue) if np.any(xs != ys) else 1.0
    if key == "AUC_rand":
        est, bs, m = float(x.mean() - y.mean()), x[idx].mean(1) - y[idx].mean(1), DELTA_B
    else:
        est, bs, m = float((x.mean() / y.mean() - 1) * 100), (x[idx].mean(1) / y[idx].mean(1) - 1) * 100, DELTA_A
    ci95 = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    ci90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    if est >= m and ci95[0] > 0 and p_seed < alpha:
        v = "Superior"
    elif -m < ci90[0] and ci90[1] < m:
        v = "Equivalent"
    elif est <= -m and ci95[1] < 0:
        v = "Inferior"
    else:
        v = "Inconclusive"
    return {"verdict": v, "estimate": est, "unit": "AUC abs" if key == "AUC_rand" else "% rel",
            "margin": m, "ci95": ci95, "ci90": ci90, "p_seed_t": p_seed, "alpha": alpha}


def verdicts(per, out, seeds, full):
    res = {"primary": {}, "comparisons": {}}
    comps = ["DecRS", "DecRS-pool", "FMconf_blend", "ENS"]
    for cand in ("DecRS-W", "TDecRS"):
        if cand not in full or not all(c in full for c in comps):
            continue
        rows = {c: verdict(per, out, seeds, cand, c, "AUC_rand", 0.025) for c in comps}
        res["primary"][cand] = {"win": all(r["verdict"] == "Superior" for r in rows.values()),
                                "vs": {c: r["verdict"] for c, r in rows.items()}, "detail": rows}
    for a, b in (("TDecRS", "FMconf_blend"), ("DecRS-pool", "DecRS"), ("DecRS", "ENS"), ("DecRS", "FMconf_blend"),
                 ("DecRS-W", "DecRS-pool"), ("DecRS-R", "FMconf_blend"), ("FMconf_raw", "FM"), ("DecRS", "FM"),
                 ("ENS", "FM")):
        if a in full and b in full:
            res["comparisons"]["%s_vs_%s" % (a, b)] = {k: verdict(per, out, seeds, a, b, k, 0.05)
                                                      for k in ("AUC_rand", "R@20", "N@20")}
    return res


if __name__ == "__main__":
    stage1()
    stage1_ens()
    stage2()
