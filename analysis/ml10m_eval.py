"""Tier 2 (long-span ML-10M): re-score, blend, segment, and test across seeds.

Stage 1 re-scores the ml10A/ml10B checkpoints on CPU along the ranking path of
code/ml-1m/evaluate.Ranking (training items masked with -999, Eq. 11 blend,
DCR user confounder after the whole training history), as analysis/rescore.py
does for ML-1M. The FM scores used by the blend are recomputed from the ml10A
FM checkpoint, which is what tools/gen_scores.py does before training. FMconf is ranked
both raw (as trained) and blended offline, as in Tier 1.5. The test Recall@10
printed per model must match its training log.

Seeds 2 and 3 (ml10s2/ml10s3) and the slow-decay cells DCR g0.7/g0.9 (seed 1)
are re-scored the same way. Every blend uses the seed-1 FM scores, as in training.

Stage 2 computes per-user metrics, C_KL calibration and coverage, splits users
by training span, and runs per-user Wilcoxon tests at seed 1 (the pilot
tables). Across seeds it reports mean and sd over the 3 seeds, a paired t-test
on the 3 seed means, and a Wilcoxon test on per-user metrics averaged over
seeds; the gamma cells are tested against DecRS at seed 1.

    python analysis/ml10m_eval.py          # stage 1 is cached in analysis/topk_ml10m/
Output: analysis/ml10m_results.json
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

DATA = os.path.join(REPO, "data", "ml_10m")
DCR = os.path.join(REPO, "data", "ml_10m_dcr")
RES = os.path.join(REPO, "results-azure")
NAME = "{}_ml_10m_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth"
OUT = os.path.join(HERE, "topk_ml10m")
os.makedirs(OUT, exist_ok=True)
G = 19
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


def save(name, top):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), users=np.array(users),
                        top=np.array(top, dtype=np.int32))


def report(name, top, t0):
    v = evaluate.computeTopNAccuracy([valid_dict[u] for u in users], top, TOPN)
    t = evaluate.computeTopNAccuracy([test_dict[u] for u in users], top, TOPN)
    print("%-14s valid R@10=%.4f  test R@10=%.4f N@10=%.4f R@20=%.4f  (%.0fs)" % (
        name, v[1][0], t[1][0], t[2][0], t[1][1], time.time() - t0), flush=True)


# (output names, checkpoint, confounder tag). Every model except FM is blended;
# FM and FMconf are also kept raw. Seed-1 outputs keep their pilot names; other
# seeds get a "_s<seed>" suffix.
JOBDIR = {1: ("ml10A", "ml10B"), 2: ("ml10s2", "ml10s2"), 3: ("ml10s3", "ml10s3")}
SEEDS = [1, 2, 3]
GAMMAS = [("g0.7", "ml10s2"), ("g0.9", "ml10s3")]


def ckpt(job, model, tag, seed):
    return os.path.join(RES, "results-" + job, "models", NAME.format(model, "_" + tag if tag else "", seed))


def sfx(seed):
    return "" if seed == 1 else "_s%d" % seed


MODELS = [({"raw": "FM"}, ckpt("ml10A", "FM", None, 1), None)]
for _s in SEEDS:
    a, b = JOBDIR[_s]
    if _s > 1:
        MODELS.append(({"raw": "FM" + sfx(_s)}, ckpt(a, "FM", None, _s), None))
    MODELS += [({"blend": "DecRS" + sfx(_s)}, ckpt(a, "DecFM", None, _s), None),
               ({"blend": "DCR_g0.3" + sfx(_s)}, ckpt(b, "DecFM", "g0.3_k0_static", _s), "g0.3_k0_static"),
               ({"raw": "FMconf_raw" + sfx(_s), "blend": "FMconf_blend" + sfx(_s)},
                ckpt(b, "FMconf", "g0.3_k0_static", _s), "g0.3_k0_static")]
for _g, _j in GAMMAS:
    MODELS.append(({"blend": "DCR_" + _g}, ckpt(_j, "DecFM", _g + "_k0_static", 1), _g + "_k0_static"))


def stage1():
    if all(os.path.exists(os.path.join(OUT, n + ".npz")) for o, _, _ in MODELS for n in o.values()):
        return
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G,
                                               train_dict, valid_dict, test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    fm_score = {}
    for outs, path, tag in MODELS:
        done = all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in outs.values())
        if done and outs.get("raw") != "FM":
            continue
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
                    if outs.get("raw") == "FM":
                        fm_score[u] = p.clone()
                    if "raw" in outs:
                        tops["raw"].append(item_idx[torch.topk(p, 100).indices].tolist())
                    if "blend" in outs:
                        lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                        s = lam * torch.sigmoid(p) + (1 - lam) * torch.sigmoid(fm_score[u])
                        tops["blend"].append(item_idx[torch.topk(s, 100).indices].tolist())
        for mode, name in outs.items():
            if done:
                continue
            save(name, tops[mode])
            report(name, tops[mode], t0)


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
    tts = np.load(os.path.join(REPO, "data", "ml_10m_temporal", "per_user", "training_timestamps.npy"))
    tl = np.load(os.path.join(DATA, "training_list.npy"))
    first, last = {}, {}
    for (u, _), t in zip(tl.tolist(), tts.tolist()):
        first[u] = min(first.get(u, t), t)
        last[u] = max(last.get(u, t), t)
    span = {u: (last[u] - first[u]) / 86400.0 for u in first}
    his_g = {u: gvec(h) for u, h in train_dict.items()}
    # oracle: how far the test-period genre mix sits from the training mix
    shift = {u: float(np.abs(gvec(test_dict[u]) - his_g[u]).sum()) for u in users}

    base = ["FM", "DecRS", "DCR_g0.3", "FMconf_raw", "FMconf_blend"]
    names = [b + sfx(sd) for sd in SEEDS for b in base] + ["DCR_" + g for g, _ in GAMMAS]
    topk = {}
    for n in names:
        z = np.load(os.path.join(OUT, n + ".npz"))
        topk[n] = dict(zip(z["users"].tolist(), z["top"]))
    us = [u for u in users if len(test_dict[u])]
    per = {}
    out = {"n_users": len(us), "models": {}}
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
    for b in base:   # per-user metrics averaged over the seeds
        per[b + "_mean"] = {k: np.mean([per[b + sfx(sd)][k] for sd in SEEDS], 0) for k in per[b]}

    def test(a, b, key, mask=None):
        x, y = per[a][key], per[b][key]
        if mask is not None:
            x, y = x[mask], y[mask]
        rel = (x.mean() / y.mean() - 1) * 100 if y.mean() else float("nan")
        p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue if np.any(x != y) else 1.0
        rng = np.random.RandomState(0)
        idx = rng.randint(0, len(x), (1000, len(x)))
        bs = (x[idx].mean(1) / y[idx].mean(1) - 1) * 100
        return {"a": float(x.mean()), "b": float(y.mean()), "rel_pct": float(rel), "p": float(p),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    pairs = [("DecRS", "FM"), ("DCR_g0.3", "DecRS"), ("DCR_g0.3", "FMconf_blend"),
             ("FMconf_blend", "DecRS"), ("FMconf_raw", "FM")]
    out["tests"] = {"%s_vs_%s" % p: {k: test(p[0], p[1], k) for k in ("R@10", "R@20", "N@10", "C_KL@20")}
                    for p in pairs}

    # seed level: mean/sd over seeds, paired t on seed means, Wilcoxon on seed-averaged users
    keys = ("R@10", "R@20", "N@10", "N@20", "C_KL@20")
    out["seeds"] = {"seeds": SEEDS, "models": {}, "tests": {}}
    for b in base:
        out["seeds"]["models"][b] = {k: {"per_seed": [out["models"][b + sfx(sd)][k] for sd in SEEDS]}
                                     for k in keys + ("coverage@20",)}
        for k, v in out["seeds"]["models"][b].items():
            v["mean"] = float(np.mean(v["per_seed"])); v["sd"] = float(np.std(v["per_seed"], ddof=1))
    for a, b in pairs:
        row = {}
        for k in ("R@10", "R@20", "N@10", "C_KL@20"):
            t = test(a + "_mean", b + "_mean", k)
            x = np.array(out["seeds"]["models"][a][k]["per_seed"]); y = np.array(out["seeds"]["models"][b][k]["per_seed"])
            t["p_seed_t"] = float(stats.ttest_rel(x, y).pvalue)
            t["per_seed_rel_pct"] = [float(v) for v in (x / y - 1) * 100]
            row[k] = t
        out["seeds"]["tests"]["%s_vs_%s" % (a, b)] = row

    # gamma: slow-decay DCR cells against DecRS and DCR g0.3, seed 1
    out["gamma"] = {}
    for g in ["g0.3"] + [g for g, _ in GAMMAS]:
        out["gamma"][g] = {"vs_DecRS": {k: test("DCR_" + g, "DecRS", k) for k in ("R@10", "R@20", "N@10", "C_KL@20")},
                           "vs_g0.3": {k: test("DCR_" + g, "DCR_g0.3", k) for k in ("R@10", "R@20", "N@10", "C_KL@20")}}

    sp = np.array([span[u] for u in us])
    sh = np.array([shift[u] for u in us])
    q = np.percentile(sh, [25, 50, 75])
    segs = {"span": [("30-180d", (sp >= 30) & (sp < 180)), ("180d-1y", (sp >= 180) & (sp < 365)),
                     ("1-3y", (sp >= 365) & (sp < 1095)), (">3y", sp >= 1095)],
            "shift_oracle": [("Q1 (least)", sh <= q[0]), ("Q2", (sh > q[0]) & (sh <= q[1])),
                             ("Q3", (sh > q[1]) & (sh <= q[2])), ("Q4 (most)", sh > q[2])]}
    out["segments"] = {}
    out["segments_mean"] = {}
    for suffix, dest in (("", out["segments"]), ("_mean", out["segments_mean"])):
        for s, bins in segs.items():
            dest[s] = []
            for lab, mask in bins:
                row = {"seg": lab, "n": int(mask.sum())}
                for n in base:
                    row[n] = float(per[n + suffix]["R@20"][mask].mean())
                for a, b in (("DCR_g0.3", "DecRS"), ("DCR_g0.3", "FMconf_blend"), ("DecRS", "FM")):
                    t = test(a + suffix, b + suffix, "R@20", mask)
                    row["%s_vs_%s" % (a, b)] = {"rel_pct": t["rel_pct"], "p": t["p"]}
                dest[s].append(row)
                print(s + suffix, lab, row["n"], {k: (round(v["rel_pct"], 1), round(v["p"], 3))
                                                  for k, v in row.items() if isinstance(v, dict)})
    out["span_days"] = {"median": float(np.median(sp)), "p25": float(np.percentile(sp, 25)),
                        "p75": float(np.percentile(sp, 75))}
    json.dump(out, open(os.path.join(HERE, "ml10m_results.json"), "w"), indent=1)


if __name__ == "__main__":
    stage1()
    stage2()
