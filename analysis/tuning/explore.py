"""Exploratory, outside the protocol's comparisons and Holm families (labelled as such wherever used).

1. Goodreads windowed - FM+FM on the tuned lists, three seeds: PROTOCOL lists windowed - pooled and
   windowed - FMconf (blend) only, but the report's earlier statement is that windowed ties both controls.
2. Goodreads, seed 1: was FMconf held back by its cell? The selection reads FMconf's raw validation value
   (protocol rule 2), while it is compared blended. Blend FMconf L+ and compare it with windowed L+ at seed 1.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AN, "eval_package")); sys.path.insert(0, os.path.join(AN, "audit")); sys.path.insert(0, HERE)
from common import Data, CODE, DS_ALL, REPO, ld, per_user, ckl
from score_all import FastFM
from stats import compare, verdict
import torch, data_utils, evaluate
ds = "goodreads"
D = Data(ds); TK = os.path.join(HERE, "topk", ds)
sel = json.load(open(os.path.join(HERE, "alpha_selected.json")))[ds]
def met(f):
    z = np.load(f); top = dict(zip(z["users"].tolist(), z["top"])); o = {}
    for k in (10, 20):
        R, N, Q = per_user(D, top, k); o["R@%d" % k], o["N@%d" % k] = R, N
        if k == 20: o["C_KL@20"] = ckl(D.his_g, Q)
    return o
METRICS = ["R@20", "R@10", "N@20", "C_KL@20"]
out = {"alpha_own": sel["own"]}
W = {s: met(os.path.join(TK, "windowed_a%g_s%d.npz" % (sel["chosen"]["windowed"], s))) for s in (1, 2, 3)}
E = {s: met(os.path.join(TK, "ENS_a%g_s%d.npz" % (sel["chosen"]["ENS"], s))) for s in (1, 2, 3)}
out["windowed_vs_ENS"] = {}
for k in METRICS:
    r = compare(np.stack([W[s][k] for s in W]), np.stack([E[s][k] for s in E]))
    out["windowed_vs_ENS"][k] = {"d": r["d"], "ci95": r["ci95"], "p": r["p"], "verdict_unadj": verdict(r), "per_seed": r["per_seed"]}
    print("windowed-ENS tuned", k, "%+.1f [%+.1f,%+.1f] p=%.3f %s" % (r["d"], *r["ci95"], r["p"], verdict(r)), r["per_seed"])
# FMconf L+ blended, seed 1
c = DS_ALL[ds]
torch.set_num_threads(6)
uf, itf, _ = data_utils.map_features(os.path.join(c["data"], "user_feature_file.npy"), os.path.join(c["data"], "item_feature_file.npy"))
tr = {}
for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist(): tr.setdefault(u, []).append(i)
va = ld(os.path.join(c["data"], "validation_dict.npy")); te = ld(os.path.join(c["data"], "testing_dict.npy")); cat = ld(os.path.join(c["data"], "item_category.npy"))
urf, ukl, aif, aifv = evaluate.pre_ranking(uf, itf, cat, c["G"], tr, va, te)
users = list(te.keys()); idx = torch.tensor(list(itf.keys()))
gap = ukl["max_kl_score"] - ukl["min_kl_score"]
lam = torch.tensor(np.array([((ukl[u]-ukl["min_kl_score"])/gap)**0.2 for u in users]), dtype=torch.float32).unsqueeze(1)
R = os.path.join(REPO, "results-azure")
l = lambda p: (lambda m: (m.eval(), m)[1])(torch.load(p, map_location="cpu", weights_only=False))
fm = l(os.path.join(R, "results-grq1", "models", "FM_goodreads_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda_seed1.pth"))
cfm = l(os.path.join(R, "results-hpB", "models", "FMconf_goodreads_0.1lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda_g0.94_k0_static_seed1.pth"))
cfm.confounder_prior = cfm.confounder_prior.cpu()
conf = ld(os.path.join(c["dcr"], "user_confounder_g0.94_k0_static.npy"))
ff, fc = FastFM(fm, aif, aifv), FastFM(cfm, aif, aifv)
top = []
with torch.no_grad():
    for b in range(0, len(users), 512):
        ub = users[b:b+512]; mask = torch.stack([urf[u][2] for u in ub])
        cf = torch.tensor(np.stack([np.asarray(conf[u], np.float32) for u in ub]))
        s = lam[b:b+512]*torch.sigmoid(fc(urf, ub, cf)+mask)+(1-lam[b:b+512])*torch.sigmoid(ff(urf, ub, None)+mask)
        top.append(idx[torch.topk(s, 100).indices].numpy().astype(np.int32))
top = np.concatenate(top)
np.savez_compressed(os.path.join(TK, "explore_FMconf_Lp_a0.2_s1.npz"), users=np.array(users), top=top)
vr = float(evaluate.computeTopNAccuracy([va[u] for u in users], top.tolist(), [10, 20, 50, 100])[1][0])
print("FMconf L+ blend valid R@10 %.4f; D blend %.4f" % (vr, json.load(open(os.path.join(HERE, "rerank_goodreads.json")))["valid_R@10"]["FMconf_a0.2_s1"]))
FL = met(os.path.join(TK, "explore_FMconf_Lp_a0.2_s1.npz"))
FD = met(os.path.join(TK, "FMconf_a0.2_s1.npz"))
out["seed1_FMconf_Lp"] = {"valid_R@10_blend": vr, "R@20": float(FL["R@20"].mean()), "R@20_D": float(FD["R@20"].mean()), "windowed_vs": {}}
for lab, F in (("FMconf L+", FL), ("FMconf D", FD)):
    for k in METRICS:
        r = compare(W[1][k], F[k]); out["seed1_FMconf_Lp"]["windowed_vs"].setdefault(lab, {})[k] = {"d": r["d"], "ci95": r["ci95"], "p": r["p"], "verdict_unadj": verdict(r)}
        print("seed 1 windowed L+ -", lab, k, "%+.1f [%+.1f,%+.1f] p=%.3f %s" % (r["d"], *r["ci95"], r["p"], verdict(r)))
json.dump(out, open(os.path.join(HERE, "explore.json"), "w"), indent=1)

# 3. Scoring record point 1 sets pooled's alpha to DecRS's (0.2), but pooled alone would choose 0.5 on validation.
#    Sensitivity: windowed (0.2) against pooled at its own alpha, three seeds.
P5 = {s: met(os.path.join(TK, "pooled_a0.5_s%d.npz" % s)) for s in (1, 2, 3)}
out["windowed_vs_pooled_own_alpha"] = {}
for k in METRICS:
    r = compare(np.stack([W[s][k] for s in W]), np.stack([P5[s][k] for s in P5]))
    out["windowed_vs_pooled_own_alpha"][k] = {"d": r["d"], "ci95": r["ci95"], "p": r["p"], "verdict_unadj": verdict(r), "per_seed": r["per_seed"]}
    print("windowed - pooled(a0.5)", k, "%+.1f [%+.1f,%+.1f] p=%.3f %s" % (r["d"], *r["ci95"], r["p"], verdict(r)), r["per_seed"])
json.dump(out, open(os.path.join(HERE, "explore.json"), "w"), indent=1)
