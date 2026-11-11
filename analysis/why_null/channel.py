"""Why-null Prop. 4: how far can the genre-mix channel move R@20, at fixed parameters?

For the kappa=0 models (FMconf, DCR = DecFM with d_u^0.3 in the mediator), swap the
trained input d^0.3 for the running mean d^1 (and for the static prior), rescore, and
compare R@20. Also checks the band bound: with B_u the half-range of the item-varying
score change, |dR_u| <= #{test items within 2 B_u of the K-th score} / n_u.
For DCR also the Lipschitz version B_u <= max_i||a*(sI_i - c)|| * L_u * ||dd||_1.

    python channel.py ml_1m|ml_10m
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "analysis", "eval_package"))
sys.path.insert(0, os.path.join(os.getcwd(), "analysis"))
import score_all as S  # noqa: E402
from common import DS, ld  # noqa: E402
import favourability as fv  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402

torch.set_num_threads(4)
ds = sys.argv[1]
c = DS[ds]
src = S.sources(ds)
NAMES = ["FMconf_s1", "DCR_s1"]
K = 20

uf, itf, _ = data_utils.map_features(os.path.join(c["data"], "user_feature_file.npy"),
                                     os.path.join(c["data"], "item_feature_file.npy"))
train = {}
for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist():
    train.setdefault(u, []).append(i)
valid = ld(os.path.join(c["data"], "validation_dict.npy"))
test = ld(os.path.join(c["data"], "testing_dict.npy"))
cat = ld(os.path.join(c["data"], "item_category.npy"))
urf, ukl, aif, aifv = evaluate.pre_ranking(uf, itf, cat, c["G"], train, valid, test)
users = list(test.keys())
item_idx = np.array(list(itf.keys()))
pos = {it: p for p, it in enumerate(item_idx)}

c03 = ld(os.path.join(c["dcr"], "user_confounder_g0.3_k0_static.npy"))
blocks = fv.load_blocks(ds)
d1_arr = fv.user_ewma_mix(blocks, 1.0)
d1 = {int(u): d1_arr[k] for k, u in enumerate(blocks.users)}
prior = ld(os.path.join(c["data"], "confounder_prior.npy")).astype(np.float32)
chk = np.abs(fv.user_ewma_mix(blocks, 0.3)[:200] - np.stack([c03[int(u)] for u in blocks.users[:200]])).max()
print("d^0.3 rebuild vs stored: %.1e" % chk, flush=True)
INPUTS = {"g0.3": lambda u: c03[u], "g1": lambda u: d1[u], "prior": lambda u: prior}

models, fast = {}, {}
for n in NAMES:
    m = torch.load(src[n][0], map_location="cpu", weights_only=False)
    m.eval()
    if hasattr(m, "confounder_prior"):
        m.confounder_prior = m.confounder_prior.cpu()
    models[n] = m
    fast[n] = S.FastFM(m, aif, aifv)

res = {n: {k: {"R": [], "dl1": [], "band": [], "bandL": [], "absdR": [], "B": [], "BL": []}
           for k in INPUTS} for n in NAMES}
CB = 256
with torch.no_grad():
    for b in range(0, len(users), CB):
        ub = users[b:b + CB]
        mask = torch.stack([urf[u][2] for u in ub])
        conf = {k: torch.tensor(np.stack([np.asarray(f(u), np.float32) for u in ub])) for k, f in INPUTS.items()}
        for n in NAMES:
            F = fast[n]
            P = {k: (F(urf, ub, conf[k]) + mask).numpy() for k in INPUTS}
            if n == "DCR_s1":   # Lipschitz pieces: L_u (Prop. T2) and the item radius
                m = models[n]
                E = m.embeddings.weight[torch.stack([urf[u][0] for u in ub])] * \
                    torch.stack([urf[u][1] for u in ub]).unsqueeze(-1)
                Su = E.sum(1)                                            # (nu, F)
                e = m.confounder_embed.weight                            # (G, F)
                L0 = (Su[:, None, :] * e[None]).norm(dim=-1)             # (nu, G)
                L1 = ((Su[:, None, None, :] + e[None, None, :, :]) * e[None, :, None, :]).norm(dim=-1)
                G = e.shape[0]
                L1[:, torch.arange(G), torch.arange(G)] = 0
                Lu = torch.maximum(L0, L1.max(-1).values).max(-1).values.numpy()
                aI = F.a * F.sI
                rad = (aI - 0.5 * (aI.max(0).values + aI.min(0).values)).norm(dim=-1).max().item()
            for j, u in enumerate(ub):
                tpos = np.array([pos[i] for i in test[u] if i in pos])
                nt = len(tpos)
                if nt == 0:
                    continue
                keep = mask[j].numpy() == 0
                base = P["g0.3"][j]
                kth = np.partition(base[keep], -K)[-K]
                topb = set(np.argpartition(-base, K)[:K].tolist())
                Rb = len(topb & set(tpos.tolist())) / nt
                for k in INPUTS:
                    s = P[k][j]
                    top = set(np.argpartition(-s, K)[:K].tolist())
                    R = len(top & set(tpos.tolist())) / nt
                    d = s[keep] - base[keep]
                    B = 0.5 * (d.max() - d.min())
                    tb = base[tpos]
                    band = np.sum(np.abs(tb - kth) <= 2 * B + 1e-9) / nt
                    dl1 = float(np.abs(np.asarray(INPUTS[k](u), float) - np.asarray(c03[u], float)).sum())
                    A = res[n][k]
                    A["R"].append(R); A["dl1"].append(dl1); A["band"].append(band)
                    A["absdR"].append(abs(R - Rb)); A["B"].append(B)
                    if n == "DCR_s1":
                        # ||dM|| <= L_u ||dd||_1; item-varying score change <= ||dM|| * radius
                        BL = Lu[j] * dl1 * rad
                        A["BL"].append(BL)
                        A["bandL"].append(np.sum(np.abs(tb - kth) <= 2 * BL + 1e-9) / nt)
        print(b + len(ub), flush=True)

out = {}
for n in NAMES:
    out[n] = {}
    for k in INPUTS:
        A = res[n][k]
        o = {"R20": float(np.mean(A["R"])), "mean_l1_shift_from_g0.3": float(np.mean(A["dl1"])),
             "mean_abs_dR_user": float(np.mean(A["absdR"])), "band_bound_mean": float(np.mean(A["band"])),
             "half_range_B_median": float(np.median(A["B"]))}
        if A["BL"]:
            o["lipschitz_band_bound_mean"] = float(np.mean(A["bandL"]))
            o["lipschitz_B_median"] = float(np.median(A["BL"]))
        out[n][k] = o
    base = out[n]["g0.3"]["R20"]
    for k in INPUTS:
        out[n][k]["rel_vs_g0.3"] = out[n][k]["R20"] / base - 1
    print(n, json.dumps(out[n], indent=1), flush=True)
json.dump(out, open(os.path.join("analysis", "why_null_channel_%s.json" % ds), "w"), indent=1)
