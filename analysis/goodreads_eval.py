"""E3 (Goodreads, read_at time): re-score, blend, segment, and test.

The same method as analysis/ml10m_eval.py, which is left untouched:
Stage 1 re-scores the grA/grB checkpoints (seeds 2-3: JOBDIR) on CPU along the
ranking path of code/ml-1m/evaluate.Ranking (training items masked, Eq. 11 blend
with the seed-1 FM scores, the DCR user confounder after the whole training history).
FMconf is ranked raw (as trained) and blended offline. The test Recall@10 printed per
model must match its training log. Models whose checkpoint is missing are skipped, so this
runs on seed 1 alone.

Stage 2: per-user R/N@10/20, C_KL@20 calibration against the training genre mix,
coverage@20, per-user Wilcoxon + bootstrap CI per pair, seed-level tests when seeds
2-3 exist, and segments by training span and by the oracle train->test genre shift.

    python analysis/goodreads_eval.py           # stage 1 is cached in analysis/topk_goodreads/
Output: analysis/goodreads_results.json
"""
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

DS = "goodreads"
DATA = os.path.join(REPO, "data", DS)
DCR = os.path.join(REPO, "data", DS + "_dcr")
RES = os.path.join(REPO, "results-azure")
# GR_SANDBOX=<dir>: read checkpoints from <dir>/results-*/ and write everything under <dir>
SANDBOX = os.environ.get("GR_SANDBOX")
if SANDBOX:
    RES = SANDBOX
NAME = "{}_" + DS + "_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth"
OUT = os.path.join(SANDBOX or HERE, "topk_" + DS)
os.makedirs(OUT, exist_ok=True)
G = 10
GSTAR = "g0.94"          # T3-predicted gamma* (analysis/favourability_goodreads.json)
TOPN = [10, 20, 50, 100]
UB = 16
torch.set_num_threads(os.cpu_count() or 4)

user_feature, item_feature, _ = data_utils.map_features(
    os.path.join(DATA, "user_feature_file.npy"), os.path.join(DATA, "item_feature_file.npy"))
train_dict = {}
for u, i in np.load(os.path.join(DATA, "training_list.npy"), allow_pickle=True).tolist():
    train_dict.setdefault(u, []).append(i)
valid_dict = data_utils.loadData(os.path.join(DATA, "validation_dict.npy"))
test_dict = data_utils.loadData(os.path.join(DATA, "testing_dict.npy"))
item_category = data_utils.loadData(os.path.join(DATA, "item_category.npy"))
users = list(test_dict.keys())
item_idx = torch.tensor(list(item_feature.keys()))

# seed -> (job holding FM / DecRS / DecRS-R, job holding FMconf / DCR)
JOBDIR = {1: ("grA", "grB"), 2: ("grs2", "grs2"), 3: ("grs3", "grs3")}
SEEDS = [1, 2, 3]


def ckpt(job, model, tag, seed):
    return os.path.join(RES, "results-" + job, "models", NAME.format(model, "_" + tag if tag else "", seed))


def sfx(seed):
    return "" if seed == 1 else "_s%d" % seed


def models_for(seed):
    a, b = JOBDIR[seed]
    t = GSTAR + "_k0_static"
    m = [({"raw": "FM" + sfx(seed)}, ckpt(a, "FM", None, seed), None),
         ({"blend": "DecRS" + sfx(seed)}, ckpt(a, "DecFM", None, seed), None),
         ({"blend": "DecRS-R" + sfx(seed)}, ckpt(a, "DecFMR", t, seed), t),
         ({"raw": "FMconf_raw" + sfx(seed), "blend": "FMconf_blend" + sfx(seed)}, ckpt(b, "FMconf", t, seed), t),
         ({"blend": "DCR_" + GSTAR + sfx(seed)}, ckpt(b, "DecFM", t, seed), t)]
    if seed == 1:
        m.append(({"blend": "DCR_g0.3"}, ckpt(b, "DecFM", "g0.3_k0_static", 1), "g0.3_k0_static"))
    # Q1: windowed-population DecRS and its pooled consumed-mix control (runs grw1/grw2)
    for lab, tg in (("DecRS-W", "g0.9_kinf_windowed"), ("DecRS-pool", "g0.9_kinf_pooled")):
        p = [ckpt(j, "DecFM", tg, seed) for j in ("grw1", "grw2")]
        m.append(({"blend": lab + sfx(seed)}, next((x for x in p if os.path.exists(x)), p[0]), tg))
    return m


MODELS = [m for s in SEEDS for m in models_for(s) if os.path.exists(m[1])]


def save(name, top):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), users=np.array(users),
                        top=np.array(top, dtype=np.int32))


def report(name, top, t0):
    v = evaluate.computeTopNAccuracy([valid_dict[u] for u in users], top, TOPN)
    t = evaluate.computeTopNAccuracy([test_dict[u] for u in users], top, TOPN)
    print("%-16s valid R@10=%.4f  test R@10=%.4f N@10=%.4f R@20=%.4f  (%.0fs)" % (
        name, v[1][0], t[1][0], t[2][0], t[1][1], time.time() - t0), flush=True)


def stage1():
    fm1 = ckpt(JOBDIR[1][0], "FM", None, 1)
    if not os.path.exists(fm1):
        sys.exit("no seed-1 FM checkpoint at " + fm1)
    todo = [m for m in MODELS if not all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in m[0].values())]
    if not todo:
        return
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G,
                                               train_dict, valid_dict, test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    # the blend's FM scores always come from the seed-1 FM, as in training
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
        with torch.no_grad():
            for b in range(0, len(users), UB):
                ub = users[b:b + UB]
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
                        tops["raw"].append(item_idx[torch.topk(p, 100).indices].tolist())
                    if "blend" in outs:
                        lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                        s = lam * torch.sigmoid(p) + (1 - lam) * torch.sigmoid(fm_score[u])
                        tops["blend"].append(item_idx[torch.topk(s, 100).indices].tolist())
        for mode, name in outs.items():
            if not os.path.exists(os.path.join(OUT, name + ".npz")):
                save(name, tops[mode])
                report(name, tops[mode], t0)


# FM+FM control (D3, as analysis/eval_package/e2a_ensemble.py): an independently trained
# FM in the DecFM slot of Eq. 11, same lam_u (alpha 0.2). ENS_FM_sA+FM_sB = slot A, partner B.
# The matched analogues of DecRS seeds 2/3 are FM_s2/FM_s3 + FM_s1 (every Dec* uses the
# seed-1 FM partner); the other 4 ordered pairs give the all-pairs spread.
ENS = [(a, b) for b in SEEDS for a in SEEDS if a != b]
ENS_MATCHED = ["ENS_FM_s%d+FM_s1" % a for a in SEEDS if a != 1]


def stage1_ens():
    names = ["ENS_FM_s%d+FM_s%d" % p for p in ENS]
    fms = {s: ckpt(JOBDIR[s][0], "FM", None, s) for s in SEEDS}
    if all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in names) or             not all(os.path.exists(f) for f in fms.values()):
        return
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G,
                                               train_dict, valid_dict, test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    nets = {}
    for s, f in fms.items():
        nets[s] = torch.load(f, map_location="cpu", weights_only=False)
        nets[s].eval()
    tops = {n: [] for n in names}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(users), UB):
            ub = users[b:b + UB]
            feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
            vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
            sig = {s: torch.sigmoid(nets[s](feat, vals).view(len(ub), NI)
                                    + torch.stack([urf[u][2] for u in ub])) for s in SEEDS}
            for j, u in enumerate(ub):
                lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                for (a, c), n in zip(ENS, names):
                    sc = lam * sig[a][j] + (1 - lam) * sig[c][j]
                    tops[n].append(item_idx[torch.topk(sc, 100).indices].tolist())
    for n in names:
        save(n, tops[n])
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


def per_user(top, u, k):
    gt = set(test_dict[u])
    hits = [1 if i in gt else 0 for i in top[:k]]
    dcg = sum(h / math.log2(j + 2) for j, h in enumerate(hits))
    idcg = sum(1 / math.log2(j + 2) for j in range(min(len(gt), k)))
    return sum(hits) / len(gt), dcg / idcg


def stage2():
    tts = np.load(os.path.join(REPO, "data", DS + "_temporal", "per_user", "training_timestamps.npy"))
    tl = np.load(os.path.join(DATA, "training_list.npy"))
    first, last = {}, {}
    for (u, _), t in zip(tl.tolist(), tts.tolist()):
        first[u] = min(first.get(u, t), t)
        last[u] = max(last.get(u, t), t)
    span = {u: (last[u] - first[u]) / 86400.0 for u in first}
    his_g = {u: gvec(h) for u, h in train_dict.items()}
    shift = {u: float(np.abs(gvec(test_dict[u]) - his_g[u]).sum()) for u in users}

    names = [n for outs, _, _ in MODELS for n in outs.values()]
    names = list(dict.fromkeys(["FM"] + names))
    names += [n for n in ("ENS_FM_s%d+FM_s%d" % p for p in ENS) if os.path.exists(os.path.join(OUT, n + ".npz"))]
    topk = {}
    for n in names:
        z = np.load(os.path.join(OUT, n + ".npz"))
        topk[n] = dict(zip(z["users"].tolist(), z["top"]))
    us = [u for u in users if len(test_dict[u])]
    per = {}
    out = {"dataset": DS, "gamma_star": GSTAR, "n_users": len(us), "models": {}}
    for n in names:
        r10, n10, r20, n20, c20 = [], [], [], [], []
        items20 = set()
        for u in us:
            t = topk[n][u]
            a, b = per_user(t, u, 10)
            c, d = per_user(t, u, 20)
            r10.append(a); n10.append(b); r20.append(c); n20.append(d)
            c20.append(ckl(his_g[u], gvec(t[:20].tolist())))
            items20.update(t[:20].tolist())
        per[n] = {"R@10": np.array(r10), "N@10": np.array(n10), "R@20": np.array(r20),
                  "N@20": np.array(n20), "C_KL@20": np.array(c20)}
        out["models"][n] = {k: float(v.mean()) for k, v in per[n].items()}
        out["models"][n]["coverage@20"] = len(items20) / len(item_feature)
        print(n, {k: round(v, 4) for k, v in out["models"][n].items()})

    def test(a, b, key, mask=None):
        x, y = per[a][key], per[b][key]
        if mask is not None:
            x, y = x[mask], y[mask]
        rel = (x.mean() / y.mean() - 1) * 100 if y.mean() else float("nan")
        p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(x != y) else 1.0
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(x), (1000, len(x)))
        bs = (x[idx].mean(1) / y[idx].mean(1) - 1) * 100
        # TOST at +-2% of b's mean: paired one-sided t-tests on d = x - y, plus the bootstrap 90% CI
        d, m = x - y, 0.02 * abs(y.mean())
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        p_tost = float(max(stats.t.sf((d.mean() + m) / se, len(d) - 1), stats.t.cdf((d.mean() - m) / se, len(d) - 1)))             if se > 0 else float("nan")
        ci90 = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
        return {"a": float(x.mean()), "b": float(y.mean()), "rel_pct": float(rel), "p": float(p),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "ci90": ci90, "p_tost2": p_tost, "equiv2_boot": bool(-2 < ci90[0] and ci90[1] < 2)}

    D = "DCR_" + GSTAR
    pairs = [("DecRS", "FM"), (D, "DecRS"), (D, "FMconf_blend"), (D, "FMconf_raw"), (D, "DecRS-R"), ("DecRS-R", "DecRS"),
             ("FMconf_blend", "DecRS"), ("FMconf_raw", "FM"), ("DCR_g0.3", "DecRS"), (D, "DCR_g0.3"),
             ("DecRS-W", "DecRS-pool"), ("DecRS-W", "DecRS"), ("DecRS-pool", "DecRS"), ("DecRS-W", "FMconf_blend")]
    pairs = [p for p in pairs if p[0] in per and p[1] in per]
    keys4 = ("R@10", "R@20", "N@10", "C_KL@20")
    out["tests"] = {"%s_vs_%s" % p: {k: test(p[0], p[1], k) for k in keys4} for p in pairs}

    # seed level, for the models present at every seed
    seeds = [s for s in SEEDS if s == 1 or ("FM" + sfx(s)) in per]
    base = ["FM", "DecRS", "DecRS-R", "FMconf_raw", "FMconf_blend", D, "DecRS-W", "DecRS-pool"]
    full = [b for b in base if all((b + sfx(s)) in per for s in seeds)]
    if len(seeds) > 1:
        for b in full:
            per[b + "_mean"] = {k: np.mean([per[b + sfx(s)][k] for s in seeds], 0) for k in per[b]}
        out["seeds"] = {"seeds": seeds, "models": {}, "tests": {}}
        for b in full:
            out["seeds"]["models"][b] = {}
            for k in ("R@10", "R@20", "N@10", "N@20", "C_KL@20", "coverage@20"):
                v = [out["models"][b + sfx(s)][k] for s in seeds]
                out["seeds"]["models"][b][k] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        for a, b in [p for p in pairs if p[0] in full and p[1] in full]:
            row = {}
            for k in keys4:
                t = test(a + "_mean", b + "_mean", k)
                x = np.array(out["seeds"]["models"][a][k]["per_seed"]); y = np.array(out["seeds"]["models"][b][k]["per_seed"])
                t["p_seed_t"] = float(stats.ttest_rel(x, y).pvalue)
                t["per_seed_rel_pct"] = [float(v) for v in (x / y - 1) * 100]
                row[k] = t
            out["seeds"]["tests"]["%s_vs_%s" % (a, b)] = row

    # D3: share of each Dec* gain over FM that the FM+FM ensemble explains
    if len(seeds) > 1 and all(n in per for n in ENS_MATCHED):
        allp = ["ENS_FM_s%d+FM_s%d" % p for p in ENS]
        per["ENS_mean"] = {k: np.mean([per[n][k] for n in ENS_MATCHED], 0) for k in per["FM"]}
        e = {"matched": ENS_MATCHED, "models": {}, "share_explained": {}, "tests": {}}
        for g, ns in (("ENS (matched)", ENS_MATCHED), ("ENS (all pairs)", allp)):
            e["models"][g] = {k: {"per_member": [out["models"][n][k] for n in ns],
                                  "mean": float(np.mean([out["models"][n][k] for n in ns])),
                                  "sd": float(np.std([out["models"][n][k] for n in ns], ddof=1))}
                              for k in ("R@10", "R@20", "N@10", "C_KL@20", "coverage@20")}
        for b in [x for x in ("DecRS", D, "DecRS-R", "FMconf_blend", "DecRS-W", "DecRS-pool") if x in full]:
            e["share_explained"][b] = {}
            for k in keys4:
                f = out["seeds"]["models"]["FM"][k]["mean"]
                d = out["seeds"]["models"][b][k]["mean"]
                en = e["models"]["ENS (matched)"][k]["mean"]
                e["share_explained"][b][k] = {"gain_vs_FM_pct": 100 * (d / f - 1), "ENS_vs_FM_pct": 100 * (en / f - 1),
                                              "vs_ENS_pct": 100 * (d / en - 1),
                                              "share": (en - f) / (d - f) if d != f else float("nan")}
            e["tests"]["%s_vs_ENS" % b] = {k: test(b + "_mean", "ENS_mean", k) for k in keys4}
        e["tests"]["ENS_vs_FM"] = {k: test("ENS_mean", "FM_mean", k) for k in keys4}
        # slot-matched seeds: DecRS_s vs ENS_FM_s+FM_s1, s = 2, 3
        e["slot_matched"] = {b: {s: {k: 100 * (out["models"][b + sfx(s)][k] / out["models"]["ENS_FM_s%d+FM_s1" % s][k] - 1)
                                     for k in ("R@20", "C_KL@20")} for s in SEEDS if s != 1}
                             for b in ("DecRS", D) if b in full}
        out["ensemble_control"] = e
        for b, v in e["share_explained"].items():
            print("ENS share", b, {k: round(x["share"], 2) for k, x in v.items()})

    sp = np.array([span[u] for u in us])
    sh = np.array([shift[u] for u in us])
    q = np.percentile(sh, [25, 50, 75])
    segs = {"span": [("<30d", sp < 30), ("30-180d", (sp >= 30) & (sp < 180)), ("180d-1y", (sp >= 180) & (sp < 365)),
                     ("1-3y", (sp >= 365) & (sp < 1095)), (">3y", sp >= 1095)],
            "shift_oracle": [("Q1 (least)", sh <= q[0]), ("Q2", (sh > q[0]) & (sh <= q[1])),
                             ("Q3", (sh > q[1]) & (sh <= q[2])), ("Q4 (most)", sh > q[2])]}
    seg_pairs = [p for p in [(D, "DecRS"), (D, "FMconf_blend"), ("DecRS", "FM"), ("DecRS-R", "DecRS"),
                             ("DecRS-W", "DecRS-pool")]
                 if p[0] in per and p[1] in per]
    out["segments"] = {}
    for s, bins in segs.items():
        out["segments"][s] = []
        for lab, mask in bins:
            row = {"seg": lab, "n": int(mask.sum())}
            if mask.sum() < 20:
                out["segments"][s].append(row)
                continue
            for n in [b for b in base if b in per]:
                row[n] = float(per[n]["R@20"][mask].mean())
            for a, b in seg_pairs:
                t = test(a, b, "R@20", mask)
                row["%s_vs_%s" % (a, b)] = {"rel_pct": t["rel_pct"], "p": t["p"]}
            out["segments"][s].append(row)
            print(s, lab, row["n"], {k: (round(v["rel_pct"], 1), round(v["p"], 3))
                                     for k, v in row.items() if isinstance(v, dict)})
    out["span_days"] = {"median": float(np.median(sp)), "p25": float(np.percentile(sp, 25)),
                        "p75": float(np.percentile(sp, 75))}
    json.dump(out, open(os.path.join(SANDBOX or HERE, DS + "_results.json"), "w"), indent=1)


if __name__ == "__main__":
    stage1()
    stage1_ens()
    stage2()
