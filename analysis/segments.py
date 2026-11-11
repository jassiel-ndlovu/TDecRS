"""Per-user metrics, beyond-accuracy metrics and segment breakdowns from analysis/topk/.

Segments (all computed from training data only, except the labelled oracle):
  span     : days between a user's first and last training interaction
  hist     : training-history length quartile
  recency  : L1 between the user's gamma=0.3 confounder (recent genre mix) and
             their gamma=1 running mean (whole-history genre mix) -- how far the
             user's recent taste sits from their average taste
  kl       : DecRS's own drift score (symmetric KL, first vs second half of
             history), which sets the Eq. 11 blend weight
  shift    : ORACLE -- L1 between the test-period genre mix and the training mix

Writes analysis/tier1_segments.json.
"""
import json
import math
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
D = os.path.join(REPO, "data")
os.environ.setdefault("DECRS_DEVICE", "cpu")
sys.path.insert(0, os.path.join(REPO, "code", "ml-1m"))
import evaluate  # noqa: E402


def ld(p):
    return np.load(p, allow_pickle=True)


train = ld(os.path.join(D, "ml_1m", "training_list.npy")).tolist()
tts = ld(os.path.join(D, "ml_1m_temporal", "per_user", "training_timestamps.npy")).tolist()
valid = ld(os.path.join(D, "ml_1m", "validation_dict.npy")).item()
test = ld(os.path.join(D, "ml_1m", "testing_dict.npy")).item()
cat = ld(os.path.join(D, "ml_1m", "item_category.npy")).item()
c03 = ld(os.path.join(D, "ml_1m_dcr", "user_confounder_g0.3_k0_static.npy")).item()
c1 = ld(os.path.join(D, "ml_1m_dcr", "user_confounder_g1_k0_static.npy")).item()


def genres(i):
    g = cat[i]
    return [g] if np.isscalar(g) else list(g)


def gvec(items):
    v = np.zeros(18)
    for i in items:
        g = genres(i)
        for x in g:
            v[x] += 1.0 / len(g)
    return v / v.sum() if v.sum() else v


hist, first, last, pop = {}, {}, {}, {}
for (u, i), t in zip(train, tts):
    hist.setdefault(u, []).append(i)
    first[u] = min(first.get(u, t), t)
    last[u] = max(last.get(u, t), t)
    pop[i] = pop.get(i, 0) + 1
n_users = len(hist)
his_g = {u: gvec(h) for u, h in hist.items()}

topk = {}
for f in sorted(os.listdir(os.path.join(HERE, "topk"))):
    z = np.load(os.path.join(HERE, "topk", f))
    topk[f[:-4]] = dict(zip(z["users"].tolist(), z["top"]))
users = [u for u in sorted(next(iter(topk.values()))) if len(test[u])]
K = (10, 20)


def per_user(top, u, k):
    gt = set(test[u])
    hits = [1 if i in gt else 0 for i in top[:k]]
    dcg = sum(h / math.log2(j + 2) for j, h in enumerate(hits))
    idcg = sum(1 / math.log2(j + 2) for j in range(min(len(gt), k)))
    rr = next((1 / (j + 1) for j, h in enumerate(hits) if h), 0.0)
    p = his_g[u]
    q = 0.99 * gvec(top[:k]) + 0.01 * p  # Steck's smoothing, as in evaluate.calibration
    nz = p > 0
    ckl = float(np.sum(p[nz] * np.log(p[nz] / q[nz])))
    gl = [set(genres(i)) for i in top[:k]]
    ild = np.mean([1 - len(gl[a] & gl[b]) / len(gl[a] | gl[b]) for a in range(k) for b in range(a + 1, k)])
    nov = np.mean([-math.log2(pop.get(i, 1) / n_users) for i in top[:k]])
    return {"Recall": sum(hits) / len(gt), "NDCG": dcg / idcg, "Precision": sum(hits) / k, "MRR": rr,
            "HR": float(sum(hits) > 0), "C_KL": ckl, "ILD": ild, "Novelty": nov}


pu, beyond = {}, {}
for name, top in topk.items():
    pu[name], beyond[name] = {}, {}
    for k in K:
        rows = [per_user(top[u], u, k) for u in users]
        pu[name][k] = {m: np.array([r[m] for r in rows]) for m in rows[0]}
        cnt = {}
        for u in users:
            for i in top[u][:k]:
                cnt[i] = cnt.get(i, 0) + 1
        x = np.sort(np.array(list(cnt.values()) + [0] * (len(cat) - len(cnt)), float))
        n = len(x)
        gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))
        beyond[name][k] = {"Coverage": len(cnt) / len(cat), "Gini": gini}
    print("scored", name, flush=True)

span = np.array([(last[u] - first[u]) / 86400 for u in users])
hl = np.array([len(hist[u]) for u in users])
rec = np.array([np.abs(np.asarray(c03[u], float) - np.asarray(c1[u], float)).sum() for u in users])
shift = np.array([np.abs(gvec(test[u]) - his_g[u]).sum() for u in users])
kl = np.array([evaluate.kl_score_calc(hist[u], valid[u], cat, 18) for u in users])


def quart(x, fmt="%.2f"):
    q = np.quantile(x, [0.25, 0.5, 0.75])
    edges = [x.min()] + list(q) + [x.max()]
    return np.digitize(x, q), ["Q%d [%s, %s]" % (j + 1, fmt % edges[j], fmt % edges[j + 1]) for j in range(4)]


segs = {
    "span": (np.digitize(span, [1, 30]), ["< 1 day", "1-30 days", "> 30 days"]),
    "hist": quart(hl, "%d"),
    "recency": quart(rec),
    "kl": quart(kl),
    "shift_oracle": quart(shift),
}
cells = {"FM": ["FM_authors"], "DecRS": ["DecRS_s%d" % s for s in range(1, 6)],
         "DCR_g0.5": ["DCR_g0.5_s%d" % s for s in range(1, 6)],
         "DCR_g0.3": ["DCR_g0.3_s%d" % s for s in range(1, 6)],
         # Tier 1.5 control: FM + the same g0.3 d_u as plain side features
         "FMconf_raw": ["FMconf_raw_s%d" % s for s in range(1, 6)],
         "FMconf_blend": ["FMconf_blend_s%d" % s for s in range(1, 6)]}


def cellmean(cell, k, m, mask=None):
    mask = np.ones(len(users), bool) if mask is None else mask
    return [float(pu[n][k][m][mask].mean()) for n in cells[cell]]


out = {"n_users": len(users), "overall": {}, "beyond": {}, "segments": {}, "user_test": {}}
for cell in cells:
    out["overall"][cell] = {k: {m: cellmean(cell, k, m) for m in pu[cells[cell][0]][k]} for k in K}
    out["beyond"][cell] = {k: {m: [beyond[n][k][m] for n in cells[cell]] for m in ("Coverage", "Gini")} for k in K}

# user-level paired test on seed-averaged per-user scores, DCR g0.3 vs DecRS
for k in K:
    for m in ("Recall", "NDCG", "HR", "MRR", "C_KL", "ILD", "Novelty"):
        a = np.mean([pu[n][k][m] for n in cells["DCR_g0.3"]], 0)
        b = np.mean([pu[n][k][m] for n in cells["DecRS"]], 0)
        out["user_test"]["%s@%d" % (m, k)] = {"diff": float((a - b).mean()),
                                              "p": float(stats.wilcoxon(a, b).pvalue),
                                              "better": int((a > b).sum()), "worse": int((a < b).sum())}

# the same test for DCR g0.3 vs the blended Tier 1.5 control
out["user_test_ctrl"] = {}
for k in K:
    for m in ("Recall", "NDCG", "HR", "MRR", "C_KL", "ILD", "Novelty"):
        a = np.mean([pu[n][k][m] for n in cells["DCR_g0.3"]], 0)
        b = np.mean([pu[n][k][m] for n in cells["FMconf_blend"]], 0)
        out["user_test_ctrl"]["%s@%d" % (m, k)] = {"diff": float((a - b).mean()),
                                                   "p": float(stats.wilcoxon(a, b).pvalue),
                                                   "better": int((a > b).sum()), "worse": int((a < b).sum())}

for sname, (lab, names) in segs.items():
    out["segments"][sname] = []
    for g, gname in enumerate(names):
        mask = lab == g
        row = {"name": gname, "n": int(mask.sum())}
        for cell in cells:
            for m in ("Recall", "NDCG"):
                row["%s %s@20" % (cell, m)] = cellmean(cell, 20, m, mask)
        for m, key in (("Recall", "R20"), ("NDCG", "N20")):
            a = np.array(row["DCR_g0.3 %s@20" % m])
            b = np.array(row["DecRS %s@20" % m])
            row["gain_%s_pct" % key] = float(100 * (a.mean() - b.mean()) / b.mean())
            row["gain_%s_p" % key] = float(stats.ttest_rel(a, b).pvalue)
            c = np.array(row["FMconf_blend %s@20" % m])
            row["ctrl_%s_pct" % key] = float(100 * (a.mean() - c.mean()) / c.mean())
            row["ctrl_%s_p" % key] = float(stats.ttest_rel(a, c).pvalue)
        out["segments"][sname].append(row)

gain = (np.mean([pu[n][20]["Recall"] for n in cells["DCR_g0.3"]], 0)
        - np.mean([pu[n][20]["Recall"] for n in cells["DecRS"]], 0))
out["corr"] = {
    "spearman_recency_gain": float(stats.spearmanr(rec, gain).correlation),
    "spearman_shift_gain": float(stats.spearmanr(shift, gain).correlation),
    "spearman_span_gain": float(stats.spearmanr(span, gain).correlation),
    "spearman_recency_shift": float(stats.spearmanr(rec, shift).correlation),
    "span_share": {"<1d": float((span < 1).mean()), "1-30d": float(((span >= 1) & (span < 30)).mean()),
                   ">30d": float((span >= 30).mean())},
}
json.dump(out, open(os.path.join(HERE, "tier1_segments.json"), "w"), indent=1)

for s, rows in out["segments"].items():
    print(s)
    for r in rows:
        print("  %-24s n=%5d  DecRS R@20=%.4f  DCR=%.4f  gain %+.1f%% (p=%.2g)  N@20 %+.1f%%" % (
            r["name"], r["n"], np.mean(r["DecRS Recall@20"]), np.mean(r["DCR_g0.3 Recall@20"]),
            r["gain_R20_pct"], r["gain_R20_p"], r["gain_N20_pct"]))
print(json.dumps(out["user_test"], indent=0))
print(json.dumps(out["corr"]))
for cell in cells:
    print(cell, {k: {m: round(float(np.mean(v)), 4) for m, v in out["overall"][cell][k].items()} for k in K},
          {k: {m: round(float(np.mean(v)), 4) for m, v in out["beyond"][cell][k].items()} for k in K})
