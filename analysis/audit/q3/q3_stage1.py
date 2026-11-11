"""Q3: re-rank the 300-epoch seed-1 checkpoints along goodreads_eval's stage-1 path.

GR_SANDBOX points at ./gr300 (copies of results-grA/grB/grw1/grs2 with the 300-epoch checkpoints), so every
blend uses the 300-epoch FM s1. Adds ENS_FM_s2+FM_s1 (slot FM s2, partner FM s1) as stage1_ens does.
"""
import os
import sys
import time

SB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gr300")
os.environ["GR_SANDBOX"] = SB
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
import torch
import goodreads_eval as E

print("models:", [list(m[0].values()) for m in E.MODELS], flush=True)
E.stage1()

name = "ENS_FM_s2+FM_s1"
if not os.path.exists(os.path.join(E.OUT, name + ".npz")):
    urf, ukl, aif, aifv = E.evaluate.pre_ranking(E.user_feature, E.item_feature, E.item_category, E.G,
                                                 E.train_dict, E.valid_dict, E.test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    nets = {s: torch.load(E.ckpt(E.JOBDIR[s][0], "FM", None, s), map_location="cpu", weights_only=False) for s in (1, 2)}
    for n in nets.values():
        n.eval()
    top, t0 = [], time.time()
    with torch.no_grad():
        for b in range(0, len(E.users), E.UB):
            ub = E.users[b:b + E.UB]
            feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
            vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
            sig = {s: torch.sigmoid(nets[s](feat, vals).view(len(ub), NI) + torch.stack([urf[u][2] for u in ub]))
                   for s in (1, 2)}
            for j, u in enumerate(ub):
                lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                top.append(E.item_idx[torch.topk(lam * sig[2][j] + (1 - lam) * sig[1][j], 100).indices].tolist())
    E.save(name, top)
    E.report(name, top, t0)
print("done", flush=True)
