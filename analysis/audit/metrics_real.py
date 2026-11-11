"""Recompute per-user metrics for every real-dataset model x seed from the stored top-100 lists.

Output: analysis/audit/cache/<ds>.npz with keys "<model>|<seed>|<metric>" (per-user arrays, users in
the dataset's evaluation order) and "<model>|<seed>|Cov@20" (scalar).

    python analysis/audit/metrics_real.py ml_1m ml_10m goodreads kuairand
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AN, "eval_package"))
sys.path.insert(0, AN)
CACHE = os.path.join(HERE, "cache")
os.makedirs(CACHE, exist_ok=True)

EP = os.path.join(AN, "eval_package", "topk")
# dataset -> model label -> {seed: file}
ENS_MAP = {1: (2, 1), 2: (3, 1), 3: (2, 1)}          # KuaiRand frozen mapping, partner FM seed 1


def sfx(s):
    return "" if s == 1 else "_s%d" % s


def files(ds):
    if ds == "ml_1m":
        S = [1, 2, 3, 4, 5]
        d = lambda n: os.path.join(EP, "ml_1m", n + ".npz")
        r = lambda n: os.path.join(AN, "topk_decrsr", n + ".npz")
        return {"FM_A": {s: d("FM_A") for s in S}, "FM": {s: d("FM_s%d" % s) for s in S},
                "DecRS": {s: d("DecRS_s%d" % s) for s in S}, "TDecRS": {s: d("DCR_g0.3_s%d" % s) for s in S},
                "FMconf_blend": {s: d("FMconf_blend_s%d" % s) for s in S},
                "FMconf_raw": {s: d("FMconf_raw_s%d" % s) for s in S},
                "DecFM_raw": {s: d("DecFM_raw_s%d" % s) for s in S},
                "TDecRS_raw": {s: d("DCR_raw_s%d" % s) for s in S},
                "DecRS-R": {s: r("DecRSR_s%d" % s) for s in S}, "DecRS-R_raw": {s: r("DecRSR_raw_s%d" % s) for s in S},
                "ENS": {s: d("ENS_FM_s%d+FM_A" % s) for s in S}}
    if ds == "ml_10m":
        S = [1, 2, 3]
        d = lambda n: os.path.join(EP, "ml_10m", n + ".npz")
        r = lambda n: os.path.join(AN, "topk_decrsr_ml10m", n + ".npz")
        return {"FM": {s: d("FM_s%d" % s) for s in S}, "DecRS": {s: d("DecRS_s%d" % s) for s in S},
                "TDecRS": {s: d("DCR_g0.3_s%d" % s) for s in S},
                "FMconf_blend": {s: d("FMconf_blend_s%d" % s) for s in S},
                "FMconf_raw": {s: d("FMconf_raw_s%d" % s) for s in S},
                "DecFM_raw": {s: d("DecFM_raw_s%d" % s) for s in S},
                "DecRS-R": {s: r("DecRSR_s%d" % s) for s in S},
                "ENS": {s: d("ENS_FM_s%d+FM_s%d" % ENS_MAP[s]) for s in S},
                # the Tier-2 scorer's copies, for the cross-check
                "x_DecRS_t2": {s: os.path.join(AN, "topk_ml10m", "DecRS%s.npz" % sfx(s)) for s in S},
                "x_TDecRS_t2": {s: os.path.join(AN, "topk_ml10m", "DCR_g0.3%s.npz" % sfx(s)) for s in S},
                "TDecRS_g0.7": {1: os.path.join(AN, "topk_ml10m", "DCR_g0.7.npz")},
                "TDecRS_g0.9": {1: os.path.join(AN, "topk_ml10m", "DCR_g0.9.npz")}}
    S = [1, 2, 3]
    base = os.path.join(AN, "topk_" + ds)
    d = lambda n: os.path.join(base, n + ".npz")
    td = "DCR_g0.94" if ds == "goodreads" else "TDecRS"
    m = {lab: {s: d(stem + sfx(s)) for s in S} for lab, stem in
         [("FM", "FM"), ("DecRS", "DecRS"), ("TDecRS", td), ("FMconf_blend", "FMconf_blend"),
          ("FMconf_raw", "FMconf_raw"), ("DecRS-R", "DecRS-R"), ("DecRS-W", "DecRS-W"), ("DecRS-pool", "DecRS-pool")]}
    m["ENS"] = {s: d("ENS_FM_s%d+FM_s%d" % ENS_MAP[s]) for s in S}
    m["ENS_alt"] = {2: d("ENS_FM_s2+FM_s1"), 3: d("ENS_FM_s3+FM_s1")}   # the report's 2-seed FM+FM
    if ds == "goodreads":
        m["TDecRS_g0.3"] = {1: d("DCR_g0.3")}
    return m


def real_metrics(ds):
    from common import Data, per_user, ckl
    D = Data(ds)
    out = {}
    for lab, by in files(ds).items():
        for s, f in by.items():
            if not os.path.exists(f):
                print("MISSING", ds, lab, s, f)
                continue
            z = np.load(f)
            top = dict(zip(z["users"].tolist(), z["top"]))
            miss = [u for u in D.users if u not in top]
            assert not miss, (f, len(miss))
            for k in (10, 20):
                R, N, Q = per_user(D, top, k)
                out["%s|%d|R@%d" % (lab, s, k)], out["%s|%d|N@%d" % (lab, s, k)] = R, N
                if k == 20:
                    out["%s|%d|C_KL@20" % (lab, s)] = ckl(D.his_g, Q)
                    items = np.unique(np.concatenate([np.asarray(top[u][:20]) for u in D.users]))
                    out["%s|%d|Cov@20" % (lab, s)] = np.array(len(items) / D.n_items)
            print(ds, lab, s, "R@20=%.4f" % out["%s|%d|R@20" % (lab, s)].mean(), flush=True)
    out["_n_users"] = np.array(len(D.users))
    return out


def kuairand_metrics():
    sys.argv = sys.argv[:1]
    import kuairand_eval as K                          # module level loads the data only
    us = [u for u in K.users if len(K.test_dict[u])]
    ur = sorted(K.rand_imp)
    his_g = {u: K.gvec(K.train_dict[u]) for u in us}
    out = {}
    for lab, by in files("kuairand").items():
        for s, f in by.items():
            if not os.path.exists(f):
                print("MISSING kuairand", lab, s, f)
                continue
            z = np.load(f)
            top = dict(zip(z["users"].tolist(), z["top"]))
            off = np.cumsum([0] + [len(K.rand_imp[u][0]) for u in z["rand_users"].tolist()])
            rsa = z["rand_scores"]
            rs = {u: rsa[off[k]:off[k + 1]] for k, u in enumerate(z["rand_users"].tolist())}
            for k in (10, 20):
                pu = [K.per_user(top[u], K.test_dict[u], k) for u in us]
                out["%s|%d|R@%d" % (lab, s, k)] = np.array([a for a, _ in pu])
                out["%s|%d|N@%d" % (lab, s, k)] = np.array([b for _, b in pu])
            out["%s|%d|C_KL@20" % (lab, s)] = np.array([K.ckl(his_g[u], K.gvec(top[u][:20].tolist())) for u in us])
            items = set()
            for u in us:
                items.update(top[u][:20].tolist())
            out["%s|%d|Cov@20" % (lab, s)] = np.array(len(items) / len(K.item_feature))
            out["%s|%d|AUC" % (lab, s)] = np.array([K.auc(rs[u], K.rand_imp[u][1]) for u in ur])
            print("kuairand", lab, s, "R@20=%.4f AUC=%.4f" % (out["%s|%d|R@20" % (lab, s)].mean(),
                                                               out["%s|%d|AUC" % (lab, s)].mean()), flush=True)
    out["_n_users"] = np.array(len(us))
    out["_n_users_auc"] = np.array(len(ur))
    return out


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        res = kuairand_metrics() if ds == "kuairand" else real_metrics(ds)
        np.savez_compressed(os.path.join(CACHE, ds + ".npz"), **res)
        print("saved", ds, flush=True)
