"""T2 mediator bound on Goodreads, and the P4 check against the Q1 results.

Part 1 is the tab:t2 procedure (favourability.mediator_lipschitz) on the static
DecFM seed-1 checkpoint (results-grA). Two sets of shifts are applied to the
checkpoint's catalogue prior:
  time  : dbar_t - pooled consumed mix   (what windowed adds over its pooled control)
  full  : dbar_t - catalogue prior        (the whole input change windowed vs DecRS)

Part 2 puts the bound in R@20 terms. The same checkpoints are re-scored on a
random subset of test users along goodreads_eval's ranking path (training items
masked, Eq. 11 blend with the seed-1 FM), changing only the prior fed at score time:
  static s1  x {catalogue, pooled, per-user window}   score-time reweighting = the case T2 bounds
  pooled s1  x {pooled, per-user window}              score-time part of W - pool
  windowed s1 x {pooled, per-user window}             training part of W - pool
The per-user window is the one goodreads_eval feeds W at test time (user_confounder_
g0.9_kinf_windowed: the window of the user's last training interaction).

    venv/Scripts/python.exe analysis/why_null/t2_goodreads.py [--users 8000]
Output: analysis/why_null_t2_goodreads.json
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("DECRS_DEVICE", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
CODE = os.path.join(REPO, "code", "ml-1m")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(REPO, "analysis"))

import numpy as np
import torch

import data_utils
import evaluate
import model as _model  # noqa: F401  (needed to unpickle the checkpoints)
import favourability as fav

DS = "goodreads"
DATA = os.path.join(REPO, "data", DS)
DCR = os.path.join(REPO, "data", DS + "_dcr")
RES = os.path.join(REPO, "results-azure")
NAME = "{}_" + DS + "_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda{}_seed{}.pth"
G = 10
UB = 16


def ckpt(job, model, tag, seed):
    return os.path.join(RES, "results-" + job, "models", NAME.format(model, "_" + tag if tag else "", seed))


STATIC = ckpt("grA", "DecFM", None, 1)
POOLED = ckpt("grw1", "DecFM", "g0.9_kinf_pooled", 1)
WINDOWED = ckpt("grw1", "DecFM", "g0.9_kinf_windowed", 1)
FM1 = ckpt("grA", "FM", None, 1)


def recall_at(top, truth, k=20):
    return np.array([len(set(t[:k]) & set(g)) / len(g) for t, g in zip(top, truth)])


def boot_rel(a, b, n=2000, seed=0):
    """relative mean difference (a-b)/b with a per-user bootstrap 95% CI"""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = (a[idx].mean(1) - b[idx].mean(1)) / b[idx].mean(1)
    return {"rel": float((a.mean() - b.mean()) / b.mean()), "lo": float(np.percentile(r, 2.5)),
            "hi": float(np.percentile(r, 97.5))}


def part1():
    P = np.load(os.path.join(REPO, "data", DS + "_temporal", "global", "prior_by_window_consumed.npy"))
    pool = np.asarray(np.load(os.path.join(DCR, "train_confounder_g0.9_kinf_pooled.npy"), mmap_mode="r")[0], float)
    cat = np.load(os.path.join(DATA, "confounder_prior.npy")).reshape(-1).astype(float)
    out = {"tv_window_vs_pooled": (0.5 * np.abs(P - pool).sum(1)).tolist(),
           "tv_window_vs_catalogue": (0.5 * np.abs(P - cat).sum(1)).tolist(),
           "tv_pooled_vs_catalogue": float(0.5 * np.abs(pool - cat).sum())}
    for name, shifts in (("time", P - pool), ("full", P - cat), ("pooled_only", [pool - cat])):
        t0 = time.time()
        out[name] = fav.mediator_lipschitz(STATIC, DS, shifts=list(shifts), base_prior=cat, n_score_users=300)
        print("part1", name, "%.0fs" % (time.time() - t0), flush=True)
    return out


def part2(n_users, seed=0):
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(DATA, "user_feature_file.npy"), os.path.join(DATA, "item_feature_file.npy"))
    train_dict = {}
    for u, i in np.load(os.path.join(DATA, "training_list.npy"), allow_pickle=True).tolist():
        train_dict.setdefault(u, []).append(i)
    valid_dict = data_utils.loadData(os.path.join(DATA, "validation_dict.npy"))
    test_dict = data_utils.loadData(os.path.join(DATA, "testing_dict.npy"))
    item_category = data_utils.loadData(os.path.join(DATA, "item_category.npy"))
    all_users = list(test_dict.keys())
    rng = np.random.default_rng(seed)
    users = [all_users[j] for j in sorted(rng.choice(len(all_users), min(n_users, len(all_users)), replace=False))]
    item_idx = torch.tensor(list(item_feature.keys()))
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, item_category, G,
                                               train_dict, valid_dict, test_dict)
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    win = np.load(os.path.join(DCR, "user_confounder_g0.9_kinf_windowed.npy"), allow_pickle=True).item()
    pool = np.load(os.path.join(DCR, "user_confounder_g0.9_kinf_pooled.npy"), allow_pickle=True).item()
    cat = np.load(os.path.join(DATA, "confounder_prior.npy")).reshape(-1).astype(np.float32)
    confs = {"cat": lambda u: cat, "pool": lambda u: np.asarray(pool[u], np.float32),
             "win": lambda u: np.asarray(win[u], np.float32)}
    runs = [("FM", FM1, [None]), ("static", STATIC, ["cat", "pool", "win"]),
            ("pooled", POOLED, ["pool", "win"]), ("windowed", WINDOWED, ["pool", "win"])]
    fm_score, tops = {}, {}
    for lab, path, cl in runs:
        m = torch.load(path, map_location="cpu", weights_only=False)
        m.eval()
        if hasattr(m, "confounder_prior"):
            m.confounder_prior = m.confounder_prior.cpu()
        for c in cl:
            t0 = time.time()
            key = lab if c is None else lab + "|" + c
            raw, bl = [], []
            with torch.no_grad():
                for b in range(0, len(users), UB):
                    ub = users[b:b + UB]
                    feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
                    vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
                    if c is None:
                        pred = m(feat, vals)
                    else:
                        conf = torch.cat([torch.from_numpy(confs[c](u)).unsqueeze(0).expand(NI, -1) for u in ub])
                        pred = m(feat, vals, conf)
                    pred = pred.view(len(ub), NI)
                    for j, u in enumerate(ub):
                        p = pred[j] + urf[u][2]
                        if c is None:
                            fm_score[u] = p.clone()
                        raw.append(item_idx[torch.topk(p, 20).indices].tolist())
                        lam = ((ukl[u] - ukl["min_kl_score"]) / gap) ** 0.2
                        s = lam * torch.sigmoid(p) + (1 - lam) * torch.sigmoid(fm_score[u])
                        bl.append(item_idx[torch.topk(s, 20).indices].tolist())
            tops[key] = {"raw": raw, "blend": bl}
            print("part2 %-14s R@20 raw %.4f blend %.4f (%.0fs)" % (
                key, recall_at(raw, [test_dict[u] for u in users]).mean(),
                recall_at(bl, [test_dict[u] for u in users]).mean(), time.time() - t0), flush=True)
        del m
    truth = [test_dict[u] for u in users]
    out = {"n_users": len(users), "R@20": {}, "contrasts": {}}
    R = {k: {mode: recall_at(v[mode], truth) for mode in v} for k, v in tops.items()}
    for k in R:
        out["R@20"][k] = {mode: float(R[k][mode].mean()) for mode in R[k]}
    pairs = [("static|pool", "static|cat", "score-time: catalogue -> pooled (constant)"),
             ("static|win", "static|cat", "score-time: catalogue -> per-user window (windowed - DecRS, mediator only)"),
             ("static|win", "static|pool", "score-time: pooled -> per-user window, static model"),
             ("pooled|win", "pooled|pool", "score-time: pooled -> per-user window, pooled model (W - pool, mediator only)"),
             ("windowed|pool", "pooled|pool", "training only: W model vs pool model, both fed pooled"),
             ("windowed|win", "pooled|pool", "W vs pool as evaluated (training + score time)"),
             ("windowed|win", "windowed|pool", "score-time part inside the W model"),
             ("windowed|win", "static|cat", "W vs DecRS as evaluated (seed 1)")]
    for a, b, desc in pairs:
        row = {"desc": desc}
        for mode in ("raw", "blend"):
            row[mode] = boot_rel(R[a][mode], R[b][mode])
            # gross gain: test hits entering a's top-20 that were not in b's (an upper bound on the gain)
            ta, tb = tops[a][mode], tops[b][mode]
            gross = np.array([len((set(x) - set(y)) & set(g)) / len(g) for x, y, g in zip(ta, tb, truth)])
            row[mode]["gross_gain_rel"] = float(gross.mean() / R[b][mode].mean())
            row[mode]["top20_overlap"] = float(np.mean([len(set(x) & set(y)) / 20 for x, y in zip(ta, tb)]))
        out["contrasts"][a + " vs " + b] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=8000)
    ap.add_argument("--skip1", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    out = {"checkpoints": {k: os.path.relpath(v, REPO) for k, v in
                           (("static", STATIC), ("pooled", POOLED), ("windowed", WINDOWED), ("fm", FM1))}}
    if not args.skip1:
        out["mediator"] = part1()
    out["score_time"] = part2(args.users)
    p = os.path.join(REPO, "analysis", "why_null_t2_goodreads.json")
    with open(p, "w") as f:
        json.dump(fav._jsonable(out), f, indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
