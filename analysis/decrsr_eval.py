"""DecRS-R (DecFMR) evaluation: re-score its checkpoints on CPU and test the E1 gate.

Stage 1 re-scores each DecFMR checkpoint the way code/ml-1m/evaluate.Ranking does
(training items masked with -999, recency confounder after the whole training
history, Eq. 11 blend with FM, alpha 0.2) and saves each user's top-100. Test
R@10 is printed per checkpoint so it can be checked against the training log.

Stage 1 also keeps DecRS-R's raw ranking (lambda = 1, no FM blend), because
the E2a check found that an FM+FM Eq. 11 ensemble explains most of DecRS's
gain over FM: the raw panel asks whether recency helps outside the ensemble.

Stage 2 scores DecRS-R against the eval-package top-K files in
analysis/eval_package/topk/<dataset>/, paired over the seeds they share:
  blended: DecRS-R vs DecRS, FMconf_blend, DCR_g0.3 (FM for reference)
  raw:     DecRS-R_raw vs DecFM_raw, FMconf_raw, DCR_raw
  R/N@10/20, C_KL@10/20 against the user's training genre mix (Steck, a=0.01),
  catalogue coverage and Gini of the top-K exposure.

E1 gate (fixed before the runs), on seed means:
  R@20(DecRS-R) >= 0.98 * R@20(FMconf blended)   AND   C_KL@20(DecRS-R) < C_KL@20(DCR g0.3)

    python analysis/decrsr_eval.py                  # ML-1M
    python analysis/decrsr_eval.py --dataset ml_10m # ML-10M long-span
Outputs: analysis/topk_decrsr[_ml10m]/DecRSR_s<seed>.npz, analysis/decrsr_results[_ml10m].json
New outputs go in their own folder so analysis/segments.py and ml10m_eval.py,
which read every file in topk/ and topk_ml10m/, are not affected.
"""
import argparse
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

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="ml_1m", choices=["ml_1m", "ml_10m"])
ap.add_argument("--seeds", default=None, help="e.g. 1,2,3 (default: every checkpoint found)")
ap.add_argument("--jobs", default=None,
                help="results-azure job folders holding the DecFMR checkpoints, comma-separated "
                     "(default: decrsrA,decrsrB or ml10rA,ml10rB)")
args = ap.parse_args()

ML10 = args.dataset == "ml_10m"
TAG = "g0.3_k0_static"
G = 19 if ML10 else 18
DATA = os.path.join(REPO, "data", args.dataset)
DCR = os.path.join(REPO, "data", args.dataset + "_dcr")
RES = os.path.join(REPO, "results-azure")
NAME = "{}_%s_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth" % args.dataset
BASE_TOPK = os.path.join(HERE, "eval_package", "topk", args.dataset)
OUT = os.path.join(HERE, "topk_decrsr_ml10m" if ML10 else "topk_decrsr")
JSON_OUT = os.path.join(HERE, "decrsr_results_ml10m.json" if ML10 else "decrsr_results.json")
JOBS = (args.jobs or ("ml10rA,ml10rB" if ML10 else "decrsrA,decrsrB")).split(",")
os.makedirs(OUT, exist_ok=True)
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


def find_ckpt(seed):
    for job in JOBS:
        p = os.path.join(RES, "results-" + job, "models", NAME.format("DecFMR", "_" + TAG, seed))
        if os.path.exists(p):
            return p
    return None


SEEDS = ([int(s) for s in args.seeds.split(",")] if args.seeds
         else [s for s in range(1, 6) if find_ckpt(s)])


def save(name, top):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), users=np.array(users),
                        top=np.array(top, dtype=np.int32))


def report(name, top, t0):
    v = evaluate.computeTopNAccuracy([valid_dict[u] for u in users], top, TOPN)
    t = evaluate.computeTopNAccuracy([test_dict[u] for u in users], top, TOPN)
    print("%-12s valid R@10=%.4f  test R@10=%.4f N@10=%.4f R@20=%.4f  (%.0fs)" % (
        name, v[1][0], t[1][0], t[2][0], t[1][1], time.time() - t0), flush=True)


# ---------------------------------------------------------------- stage 1 ----
def fm_scores(urf, aif, aifv):
    """The FM half of the Eq. 11 blend, as used in training. ML-1M: the authors'
    released score file. ML-10M: the seed-1 FM checkpoint (ml10A), whose scores
    gen_scores.py wrote to the file training blended with -- the same scores
    analysis/ml10m_eval.py rebuilds."""
    if not ML10:
        f = data_utils.loadData(os.path.join(CODE, "inference", "FM_user_item_scores.npy"))
        return {u: torch.tensor(np.asarray(f[u], dtype=np.float32)) for u in users}
    m = torch.load(os.path.join(RES, "results-ml10A", "models", NAME.format("FM", "", 1)),
                   map_location="cpu", weights_only=False)
    m.eval()
    NI, out = aif.size(0), {}
    with torch.no_grad():
        for b in range(0, len(users), UB):
            ub = users[b:b + UB]
            feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
            vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
            pred = m(feat, vals).view(len(ub), NI)
            for j, u in enumerate(ub):
                out[u] = pred[j] + urf[u][2]
    return out


def stage1():
    todo = [s for s in SEEDS if not all(os.path.exists(os.path.join(OUT, n % s + ".npz"))
                                        for n in ("DecRSR_s%d", "DecRSR_raw_s%d"))]
    if not todo:
        return
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G,
                                               train_dict, valid_dict, test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    fm = fm_scores(urf, aif, aifv)
    uconf = np.load(os.path.join(DCR, "user_confounder_%s.npy" % TAG), allow_pickle=True).item()
    for seed in todo:
        path = find_ckpt(seed)
        if path is None:
            print("no DecFMR checkpoint for seed %d in %s" % (seed, JOBS))
            continue
        t0 = time.time()
        m = torch.load(path, map_location="cpu", weights_only=False)
        m.eval()
        m.confounder_prior = m.confounder_prior.cpu()
        top, top_raw = [], []
        with torch.no_grad():
            for b in range(0, len(users), UB):
                ub = users[b:b + UB]
                feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
                vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
                conf = torch.cat([torch.tensor(np.asarray(uconf[u], np.float32)).unsqueeze(0).expand(NI, -1)
                                  for u in ub])
                pred = m(feat, vals, conf).view(len(ub), NI)
                for j, u in enumerate(ub):
                    p = pred[j] + urf[u][2]
                    top_raw.append(item_idx[torch.topk(p, 100).indices].tolist())
                    lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                    s = lam * torch.sigmoid(p) + (1 - lam) * torch.sigmoid(fm[u])
                    top.append(item_idx[torch.topk(s, 100).indices].tolist())
        save("DecRSR_s%d" % seed, top)
        save("DecRSR_raw_s%d" % seed, top_raw)
        report("DecRSR_s%d" % seed, top, t0)
        report("DecRSR_raw_s%d" % seed, top_raw, t0)


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


def base_file(model, seed):
    """Stream-B top-K names: <model>_s<seed>. ML-1M's FM is the authors' single
    file (FM_A), which is also the blend partner."""
    name = "FM_A" if (model == "FM" and not ML10) else "%s_s%d" % (model, seed)
    return os.path.join(BASE_TOPK, name + ".npz")


MODELS = ["DecRSR", "DecRS", "FMconf_blend", "DCR_g0.3", "FM",
          "DecRSR_raw", "DecFM_raw", "FMconf_raw", "DCR_raw"]
COMPARE = {"DecRSR": ["DecRS", "FMconf_blend", "DCR_g0.3"],
           "DecRSR_raw": ["DecFM_raw", "FMconf_raw", "DCR_raw", "FM"]}
KEYS = ["R@10", "R@20", "N@10", "N@20", "C_KL@10", "C_KL@20", "Cov@10", "Cov@20", "Gini@10", "Gini@20"]


def score(path, ev_users, his_g, n_items):
    z = np.load(path)
    top = dict(zip(z["users"].tolist(), z["top"]))
    out = {}
    for k in (10, 20):
        rec, ndcg, c = [], [], []
        cnt = {}
        for u in ev_users:
            t = top[u][:k].tolist()
            gt = set(test_dict[u])
            hits = [1 if i in gt else 0 for i in t]
            rec.append(sum(hits) / len(gt))
            ndcg.append(sum(h / math.log2(j + 2) for j, h in enumerate(hits))
                        / sum(1 / math.log2(j + 2) for j in range(min(len(gt), k))))
            c.append(ckl(his_g[u], gvec(t)))
            for i in t:
                cnt[i] = cnt.get(i, 0) + 1
        x = np.sort(np.array(list(cnt.values()) + [0] * (n_items - len(cnt)), float))
        n = len(x)
        out["R@%d" % k] = float(np.mean(rec))
        out["N@%d" % k] = float(np.mean(ndcg))
        out["C_KL@%d" % k] = float(np.mean(c))
        out["Cov@%d" % k] = len(cnt) / n_items
        out["Gini@%d" % k] = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))
    return out


def stage2():
    ev_users = [u for u in users if len(test_dict[u])]
    his_g = {u: gvec(train_dict[u]) for u in ev_users}
    n_items = len(item_category)
    res = {}
    for m in MODELS:
        for s in SEEDS:
            p = (os.path.join(OUT, "%s_s%d.npz" % (m, s)) if m.startswith("DecRSR")
                 else base_file(m, s))
            if os.path.exists(p):
                res.setdefault(m, {})[s] = score(p, ev_users, his_g, n_items)
    if "DecRSR" not in res:
        print("no DecRS-R top-K files yet")
        return

    def seed_mean(m, key, seeds):
        return float(np.mean([res[m][s][key] for s in seeds]))

    out = {"dataset": args.dataset, "seeds": SEEDS, "per_seed": res, "means": {}, "paired": {}}
    print("\n%s, seeds %s, %d test users" % (args.dataset, SEEDS, len(ev_users)))
    print("%-13s" % "model" + "".join("%9s" % k for k in KEYS) + "  seeds")
    for m in MODELS:
        if m not in res:
            continue
        ss = sorted(res[m])
        out["means"][m] = {k: seed_mean(m, k, ss) for k in KEYS}
        print("%-13s" % m + "".join("%9.4f" % out["means"][m][k] for k in KEYS) + "  %s" % ss)

    print("\npaired over shared seeds: DecRS-R minus baseline (mean, rel %, paired t p, wins)")
    for me, b in [(me, b) for me in COMPARE for b in COMPARE[me]]:
        if b not in res or me not in res:
            continue
        ss = sorted(set(res[me]) & set(res[b]))
        out["paired"]["%s_vs_%s" % (me, b)] = pr = {"seeds": ss}
        line = []
        for k in ("R@10", "R@20", "N@10", "N@20", "C_KL@20", "Cov@20", "Gini@20"):
            a = np.array([res[me][s][k] for s in ss])
            c = np.array([res[b][s][k] for s in ss])
            d = a - c
            p = float(stats.ttest_rel(a, c).pvalue) if len(ss) > 1 and d.std() > 0 else float("nan")
            wins = int((d < 0).sum()) if k.startswith(("C_KL", "Gini")) else int((d > 0).sum())
            pr[k] = {"diff": float(d.mean()), "rel": float(d.mean() / c.mean()),
                                   "sd": float(d.std(ddof=1)) if len(ss) > 1 else float("nan"),
                                   "p": p, "wins": wins, "n": len(ss)}
            line.append("%s %+.4f (%+.1f%%, p=%.3f, %d/%d)" % (k, d.mean(), 100 * d.mean() / c.mean(),
                                                                 p, wins, len(ss)))
        print("  %s vs %-12s seeds %s\n    " % (me, b, ss) + "\n    ".join(line))

    # E1 gate, on means over the seeds DecRS-R shares with each comparator
    gate = {}
    ss = sorted(set(res["DecRSR"]) & set(res.get("FMconf_blend", {})))
    if ss:
        r, f = seed_mean("DecRSR", "R@20", ss), seed_mean("FMconf_blend", "R@20", ss)
        gate["acc"] = {"R@20_DecRSR": r, "R@20_FMconf": f, "ratio": r / f, "pass": r >= 0.98 * f}
    ss = sorted(set(res["DecRSR"]) & set(res.get("DCR_g0.3", {})))
    if ss:
        r, d = seed_mean("DecRSR", "C_KL@20", ss), seed_mean("DCR_g0.3", "C_KL@20", ss)
        gate["cal"] = {"C_KL@20_DecRSR": r, "C_KL@20_DCR": d, "pass": r < d}
    gate["pass"] = bool(gate.get("acc", {}).get("pass") and gate.get("cal", {}).get("pass"))
    out["gate"] = gate
    print("\nGATE  R@20 ratio vs FMconf = %.4f (need >= 0.98): %s" % (
        gate["acc"]["ratio"], "PASS" if gate["acc"]["pass"] else "FAIL") if "acc" in gate else "")
    print("      C_KL@20 DecRS-R %.4f vs DCR %.4f (need lower): %s" % (
        gate["cal"]["C_KL@20_DecRSR"], gate["cal"]["C_KL@20_DCR"],
        "PASS" if gate["cal"]["pass"] else "FAIL") if "cal" in gate else "")
    print("      overall: %s" % ("PASS" if gate["pass"] else "FAIL"))
    with open(JSON_OUT, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", JSON_OUT)


if __name__ == "__main__":
    stage1()
    stage2()
