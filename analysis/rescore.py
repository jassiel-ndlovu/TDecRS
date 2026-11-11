"""Re-score the dcrseedsA checkpoints on CPU and save each user's blended top-100.

Follows the ranking path of code/ml-1m/evaluate.Ranking exactly (training
items masked with -999, Eq. 11 blend with the authors' FM scores, DCR user
confounder after the whole training history). The only change is that
several users go through one forward pass, which is valid because DecFM scores
rows independently in eval mode. The test Recall@10 printed per checkpoint must
match its training log.

    python analysis/rescore.py
Output: analysis/topk/<cell>_s<seed>.npz  (users, top100)
"""
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

import data_utils
import evaluate
import model as _model  # noqa: F401  (needed to unpickle the checkpoints)

DATA = os.path.join(REPO, "data", "ml_1m")
CKPT = os.path.join(REPO, "results-azure", "results-dcrseedsA", "models",
                    "DecFM_ml_1m_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth")
OUT = os.path.join(HERE, "topk")
os.makedirs(OUT, exist_ok=True)
TOPN = [10, 20, 50, 100]
UB = 16  # users per forward pass

user_feature, item_feature, _ = data_utils.map_features(
    os.path.join(DATA, "user_feature_file.npy"), os.path.join(DATA, "item_feature_file.npy"))
train_dict = {}
for u, i in np.load(os.path.join(DATA, "training_list.npy"), allow_pickle=True).tolist():
    train_dict.setdefault(u, []).append(i)
valid_dict = data_utils.loadData(os.path.join(DATA, "validation_dict.npy"))
test_dict = data_utils.loadData(os.path.join(DATA, "testing_dict.npy"))
item_category = data_utils.loadData(os.path.join(DATA, "item_category.npy"))
FM_score = data_utils.loadData(os.path.join(CODE, "inference", "FM_user_item_scores.npy"))
urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, 18,
                                           train_dict, valid_dict, test_dict)
users = list(test_dict.keys())
item_idx = torch.tensor(list(item_feature.keys()))
NI = aif.size(0)
gap = ukl["max_kl_score"] - ukl["min_kl_score"]


def save(name, top):
    np.savez_compressed(os.path.join(OUT, name + ".npz"), users=np.array(users),
                        top=np.array(top, dtype=np.int32))


def report(name, top, t0):
    v = evaluate.computeTopNAccuracy([valid_dict[u] for u in users], top, TOPN)
    t = evaluate.computeTopNAccuracy([test_dict[u] for u in users], top, TOPN)
    print("%-12s valid R@10=%.4f  test R@10=%.4f N@10=%.4f R@20=%.4f  (%.0fs)" % (
        name, v[1][0], t[1][0], t[2][0], t[1][1], time.time() - t0), flush=True)


if not os.path.exists(os.path.join(OUT, "FM_authors.npz")):
    t0 = time.time()
    top = []
    for u in users:
        s = torch.tensor(np.asarray(FM_score[u], dtype=np.float32)) + urf[u][2]
        top.append(item_idx[torch.topk(s, 100).indices].tolist())
    save("FM_authors", top)
    report("FM_authors", top, t0)

# (cell, checkpoint template, tag suffix, outputs). "blend" is the Eq. 11 ranking
# DecRS/DCR use; "raw" ranks the model's own score, as main.py does for the
# FM-style baselines. FMconf (Tier 1.5) gets both, so it can be compared with
# DCR like for like.
CKPT_CTRL = os.path.join(REPO, "results-azure", "results-dcrctrl", "models",
                         "FMconf_ml_1m_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth")
cells = [("DecRS", CKPT, "", {"blend": "DecRS"}),
         ("DCR_g0.3", CKPT, "_g0.3_k0_static", {"blend": "DCR_g0.3"}),
         ("DCR_g0.5", CKPT, "_g0.5_k0_static", {"blend": "DCR_g0.5"}),
         ("FMconf", CKPT_CTRL, "_g0.3_k0_static", {"blend": "FMconf_blend", "raw": "FMconf_raw"})]
for cell, ckpt, suffix, outs in cells:
    tag = suffix[1:]
    uconf = None
    if tag:
        uconf = np.load(os.path.join(REPO, "data", "ml_1m_dcr", "user_confounder_%s.npy" % tag),
                        allow_pickle=True).item()
    for seed in range(1, 6):
        names = {mode: "%s_s%d" % (o, seed) for mode, o in outs.items()}
        if all(os.path.exists(os.path.join(OUT, n + ".npz")) for n in names.values()):
            continue
        t0 = time.time()
        m = torch.load(ckpt.format(suffix, seed), map_location="cpu", weights_only=False)
        m.eval()
        m.confounder_prior = m.confounder_prior.cpu()
        top, top_raw = [], []
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
                    if "raw" in outs:
                        top_raw.append(item_idx[torch.topk(p, 100).indices].tolist())
                    lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                    s = lam * torch.sigmoid(p) + (1 - lam) * torch.sigmoid(
                        torch.tensor(np.asarray(FM_score[u], dtype=np.float32)))
                    top.append(item_idx[torch.topk(s, 100).indices].tolist())
        for mode, lst in (("blend", top), ("raw", top_raw)):
            if mode in names:
                save(names[mode], lst)
                report(names[mode], lst, t0)
