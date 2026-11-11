"""Score every checkpoint the E2 package needs and save each ranking variant's top-100.

One pass over the users (batches of 16, as analysis/rescore.py) runs every model
once and builds every variant from the same raw scores, so nothing as large as a
full user x item score matrix is ever written to disk. Ranking follows
code/ml-1m/evaluate.Ranking: training items masked with -999, DCR/FMconf get the
user's confounder after the whole training history, and Dec* variants use the
Eq. 11 blend  lam_u * sigmoid(x) + (1 - lam_u) * sigmoid(fm).

The blend partner `fm` is what the original runs used:
  ML-1M   the authors' released FM scores (code/ml-1m/inference/FM_user_item_scores.npy)
  ML-10M  FM seed 1 (masked raw scores), as tools/gen_scores.py writes them

Variants (top-100 per user; "score" kept where E2d needs it):
  raw          <model>_raw / FM_*                        lambda = 1, the model's own ranking
  Eq. 11       DecRS_s, DCR_g0.3_s, FMconf_blend_s       lambda_u from raw KL (Eq. 11)
  E2a control  ENS_<x>+<y>                               FM x in the Dec* slot, FM y as partner
  E2e          DecRSz_s, DCRz_s                          lambda_u from the drift z-score
  E2b          grid_<tag>                                ML-1M seed-1 grid checkpoints, Eq. 11

    python analysis/eval_package/score_all.py --ds ml_1m
    python analysis/eval_package/score_all.py --ds ml_10m
Output: analysis/eval_package/topk/<ds>/<variant>.npz  (users, top[, score])
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CODE, DS, OUT_DIR, REPO, TOPK_DIR, Data, eq11_lambda, ld  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402
import model as _model  # noqa: E402,F401  (needed to unpickle the checkpoints)

RES = os.path.join(REPO, "results-azure")
UB = 16      # users per forward() call in the closed-form check
CB = 512     # users per closed-form chunk


class FastFM:
    """Closed-form eval-mode scores of FM, FMconf and DecFM for a chunk of users x all items.

    All three end in  sum_k BN_k(bi_k) + biases,  where bi is the bi-interaction
    0.5 * ((sum of rows)^2 - sum of rows^2) over the embedding rows of one (user, item)
    row. Split the rows into a user part U and an item part I:
        bi = bi_U + bi_I + s_U * s_I            (s = sum of rows)
    BatchNorm in eval mode is a per-factor affine map a*x + b and dropout is the
    identity, so
        score(u, i) = c_u + c_i + (a * s_U) . s_I
    a rank-K bilinear form plus row and column offsets: one GEMM per chunk.
    User part: the user's feature rows, plus DecFM's mediator M(d, u) (it depends on
    the user and the confounder only -- every item carries the same G genre slots, so
    confounder_part is 0..G-1 for every row) or FMconf's G weighted confounder rows.
    Item part: the item's feature rows. Checked against forward() in main().
    """

    def __init__(self, m, aif, aifv):
        self.m = m
        K = m.num_factors
        a, b = torch.ones(K), torch.zeros(K)
        for mod in m.FM_layers:
            if isinstance(mod, torch.nn.BatchNorm1d):
                s = mod.weight / torch.sqrt(mod.running_var + mod.eps)
                a, b = a * s, b * s + mod.bias - mod.running_mean * s
            elif not isinstance(mod, torch.nn.Dropout):
                raise TypeError(type(mod))
        self.a, self.b = a, b
        E = m.embeddings.weight[aif] * aifv.unsqueeze(-1)              # (NI, n_item_feats, K)
        sI = E.sum(1)
        self.sI = sI
        self.cI = (0.5 * (sI.pow(2) - E.pow(2).sum(1)) * a).sum(1) + (m.biases.weight[aif].squeeze(-1) * aifv).sum(1)

    def __call__(self, urf, ub, conf=None):
        m = self.m
        uf = torch.stack([urf[u][0] for u in ub])
        uv = torch.stack([urf[u][1] for u in ub])
        E = m.embeddings.weight[uf] * uv.unsqueeze(-1)                  # (nu, n_user_feats, K)
        cU = (m.biases.weight[uf].squeeze(-1) * uv).sum(1)
        kind = type(m).__name__
        if kind in ("DecFM",):
            prior = m.confounder_prior.view(1, -1, 1) if conf is None else conf.unsqueeze(-1)
            ce = (m.confounder_embed.weight.unsqueeze(0) * prior).expand(len(ub), -1, -1)
            uce = torch.cat([E, ce], 1)
            M = 0.5 * (uce.sum(1).pow(2) - uce.pow(2).sum(1))
            E = torch.cat([E, M.unsqueeze(1)], 1)
        elif kind == "FMconf":
            cf = m.confounder_prior.unsqueeze(0).expand(len(ub), -1) if conf is None else conf
            E = torch.cat([E, m.confounder_embed.weight.unsqueeze(0) * cf.unsqueeze(-1)], 1)
            cU = cU + cf @ m.confounder_bias.weight.squeeze(-1)
        elif kind != "FM":
            raise TypeError(kind)
        sU = E.sum(1)
        cU = cU + (0.5 * (sU.pow(2) - E.pow(2).sum(1)) * self.a).sum(1)
        return (cU + self.b.sum() + m.bias_).unsqueeze(1) + self.cI.unsqueeze(0) + (sU * self.a) @ self.sI.T
HP = "0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda"


def sources(ds):
    """name -> (checkpoint path or None for a score file, confounder tag or None)."""
    if ds == "ml_1m":
        m = os.path.join(RES, "results-dcrseedsA", "models", "DecFM_ml_1m_" + HP + "{}_seed{}.pth")
        src = {
            # the only surviving trained FM checkpoints with the DecRS hyper-parameters
            # (results-baseseeds kept one file for 5 seeds; results/ml-1m is the 2026-09-15 run)
            "FM_b": (os.path.join(RES, "results-baseseeds", "models", "FM_ml_1m_%s.pth" % HP), None),
            "FM_r": (os.path.join(REPO, "results", "ml-1m", "models", "FM_ml_1m_%s.pth" % HP), None),
            # hyper-parameter sweep FMs (different lr / weight decay): extra ensemble members
            "FM_sw1": (os.path.join(REPO, "results-sweep", "models",
                                    "FM_ml_1m_0.01lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda.pth"), None),
            "FM_sw2": (os.path.join(REPO, "results-sweep", "models",
                                    "FM_ml_1m_0.01lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.0lamda.pth"), None),
            "FM_sw3": (os.path.join(REPO, "results-sweep", "models",
                                    "FM_ml_1m_0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.0lamda.pth"), None),
        }
        # FM seeds 1-4 retrained with seed-tagged names (fmseeds, logs identical to the lost
        # baseseeds seeds 1-4); seed 5 is the surviving baseseeds file (FM_b)
        for s in (1, 2, 3, 4):
            src["FM_s%d" % s] = (os.path.join(RES, "results-fmseeds", "models",
                                              "FM_ml_1m_%s_seed%d.pth" % (HP, s)), None)
        src["FM_s5"] = src["FM_b"]
        for s in DS[ds]["seeds"]:
            src["DecFM_s%d" % s] = (m.format("", s), None)
            src["DCR_s%d" % s] = (m.format("_g0.3_k0_static", s), "g0.3_k0_static")
            src["FMconf_s%d" % s] = (os.path.join(RES, "results-dcrctrl", "models", "FMconf_ml_1m_" + HP +
                                                  "_g0.3_k0_static_seed%d.pth" % s), "g0.3_k0_static")
        g2 = os.path.join(RES, "results-dcrgrid2", "models", "DecFM_ml_1m_" + HP + "_{}_seed1.pth")
        for tag in ("g0.3_k0_static", "g0.3_k10_static", "g0.5_k0_static", "g0.5_k10_static",
                    "g0.9_kinf_windowed"):
            src["grid_" + tag] = (g2.format(tag), tag)
        # dcrgrid saved every cell to one untagged file; the last job to finish was
        # g0.9_k0_windowed (started 14:37). With kappa=0 its final user confounder is
        # the personal g0.9 mix, i.e. g0.9_k0_static. Checked against the log below.
        src["grid_g0.9_k0_windowed"] = (os.path.join(RES, "results-dcrgrid", "models",
                                                     "DecFM_ml_1m_%s.pth" % HP), "g0.9_k0_static")
        return src
    src = {}
    for s in DS[ds]["seeds"]:
        job_a, job_b = {1: ("ml10A", "ml10B"), 2: ("ml10s2", "ml10s2"), 3: ("ml10s3", "ml10s3")}[s]
        p = os.path.join(RES, "results-{}", "models", "{}_ml_10m_" + HP + "{}_seed%d.pth" % s)
        src["FM_s%d" % s] = (p.format(job_a, "FM", ""), None)
        src["DecFM_s%d" % s] = (p.format(job_a, "DecFM", ""), None)
        src["DCR_s%d" % s] = (p.format(job_b, "DecFM", "_g0.3_k0_static"), "g0.3_k0_static")
        src["FMconf_s%d" % s] = (p.format(job_b, "FMconf", "_g0.3_k0_static"), "g0.3_k0_static")
    return src


def variants(ds):
    """name -> (x source, partner source or None for raw, lambda key, keep score?)."""
    seeds = DS[ds]["seeds"]
    v = {}
    if ds == "ml_1m":
        part = "FM_A"
        fms = ["FM_b", "FM_r", "FM_sw1", "FM_sw2", "FM_sw3"]
        v["FM_A"] = ("FM_A", None, None, True)
        for f in fms:
            v[f] = (f, None, None, f == "FM_b")
            v["ENS_%s+FM_A" % f] = (f, part, "kl", False)      # exactly DecRS's slot structure
        for s in seeds:
            v["FM_s%d" % s] = ("FM_s%d" % s, None, None, True)
            v["ENS_FM_s%d+FM_A" % s] = ("FM_s%d" % s, part, "kl", False)        # matched, seed by seed
            t = s % len(seeds) + 1                                                # 5 trained pairs s -> s+1
            v["ENS_FM_s%d+FM_s%d" % (s, t)] = ("FM_s%d" % s, "FM_s%d" % t, "kl", False)
        v["ENS_FM_b+FM_r"] = ("FM_b", "FM_r", "kl", False)
        v["ENS_FM_r+FM_b"] = ("FM_r", "FM_b", "kl", False)
        for tag in ("g0.3_k0_static", "g0.3_k10_static", "g0.5_k0_static", "g0.5_k10_static",
                    "g0.9_kinf_windowed", "g0.9_k0_windowed"):
            v["grid_" + tag] = ("grid_" + tag, part, "kl", False)
    else:
        part = "FM_s1"
        for s in seeds:
            v["FM_s%d" % s] = ("FM_s%d" % s, None, None, True)
        for a in seeds:
            for b in seeds:
                if a != b:
                    v["ENS_FM_s%d+FM_s%d" % (a, b)] = ("FM_s%d" % a, "FM_s%d" % b, "kl", False)
    for s in seeds:
        v["DecRS_s%d" % s] = ("DecFM_s%d" % s, part, "kl", True)
        v["DecFM_raw_s%d" % s] = ("DecFM_s%d" % s, None, None, False)
        v["DCR_g0.3_s%d" % s] = ("DCR_s%d" % s, part, "kl", False)
        v["DCR_raw_s%d" % s] = ("DCR_s%d" % s, None, None, False)
        v["FMconf_blend_s%d" % s] = ("FMconf_s%d" % s, part, "kl", False)
        v["FMconf_raw_s%d" % s] = ("FMconf_s%d" % s, None, None, False)
        v["DecRSz_s%d" % s] = ("DecFM_s%d" % s, part, "z", False)
        v["DCRz_s%d" % s] = ("DCR_s%d" % s, part, "z", False)
    return v


def _srcs(v):
    return {v[0]} | ({v[1]} if v[1] else set())


def _is(prefixes):
    return lambda v: all(s.startswith(prefixes) for s in _srcs(v))


# Forwards run in priority order, one stage at a time. Each stage builds the
# variants whose sources all belong to it (FM_A, a score file, costs nothing).
STAGES = {
    "A": _is(("FM_A", "FM_b", "FM_r", "FM_s", "DecFM_")),
    "B": lambda v: any(s.startswith(("grid_", "FM_sw")) for s in _srcs(v)) and _is(("FM_A", "grid_", "FM_sw"))(v),
    "C": lambda v: any(s.startswith(("DCR_", "FMconf_")) for s in _srcs(v)),
    "all": lambda v: True,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DS))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--stage", choices=list(STAGES), default="all",
                    help="A: FM and DecFM (E2a, E2d, E2e); B: grid and sweep FMs (E2b, extra E2a members); "
                         "C: DCR and FMconf forwards (only for variants import_existing.py could not supply)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ds = args.ds
    c = DS[ds]
    out = os.path.join(TOPK_DIR, ds)
    os.makedirs(out, exist_ok=True)
    V = variants(ds)
    todo = {n: v for n, v in V.items() if not os.path.exists(os.path.join(out, n + ".npz"))
            and STAGES[args.stage](v)}
    if not todo:
        print("all %d variants present" % len(V))
        return
    need = set()
    for x, y, _, _ in todo.values():
        need.update([x] + ([y] if y else []))
    S = {k: v for k, v in sources(ds).items() if k in need}

    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(c["data"], "user_feature_file.npy"), os.path.join(c["data"], "item_feature_file.npy"))
    train_dict = {}
    for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist():
        train_dict.setdefault(u, []).append(i)
    valid = ld(os.path.join(c["data"], "validation_dict.npy"))
    test = ld(os.path.join(c["data"], "testing_dict.npy"))
    cat = ld(os.path.join(c["data"], "item_category.npy"))
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, cat, c["G"], train_dict, valid, test)
    users = list(test.keys())
    item_idx = torch.tensor(list(item_feature.keys()))
    NI = aif.size(0)
    lam = {"kl": torch.tensor(eq11_lambda(ukl, users), dtype=torch.float32)}
    zf = os.path.join(OUT_DIR, "drift_z_%s.npz" % ds)
    if any(v[2] == "z" for v in todo.values()):
        z = np.load(zf)
        zd = dict(zip(z["users"].tolist(), z["z"]))
        lam["z"] = torch.tensor(eq11_lambda(zd, users), dtype=torch.float32)

    models, confs = {}, {}
    fm_file = None
    for name, (path, tag) in S.items():
        if name == "FM_A":
            continue
        m = torch.load(path, map_location="cpu", weights_only=False)
        m.eval()
        if hasattr(m, "confounder_prior"):
            m.confounder_prior = m.confounder_prior.cpu()
        models[name] = m
        if tag and tag not in confs:
            f = os.path.join(c["dcr"], "user_confounder_%s.npy" % tag)
            if os.path.exists(f):
                confs[tag] = ld(f)
            else:   # k10 grid cells: rebuilt in memory with tools/dcr_confounder.py
                g, k, pop = tag.split("_")
                assert pop == "static", tag
                confs[tag] = Data(ds).recent_mix(float(g[1:]), float(k[1:]))
    if "FM_A" in need:
        fm_file = ld(os.path.join(CODE, "inference", "FM_user_item_scores.npy"))
    print("%s: %d models, %d variants to build" % (ds, len(models) + (fm_file is not None), len(todo)),
          flush=True)

    def conf_of(name, ub):
        tag = S[name][1]
        return None if tag is None else torch.tensor(np.stack([np.asarray(confs[tag][u], np.float32) for u in ub]))

    def forward(name, ub):
        """The model's own forward pass, exactly as analysis/rescore.py runs it."""
        m, nb = models[name], len(ub)
        feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
        vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
        cf = conf_of(name, ub)
        pred = m(feat, vals) if cf is None else m(feat, vals, cf.repeat_interleave(NI, 0))
        return pred.view(nb, NI)

    fast = {name: FastFM(m, aif, aifv) for name, m in models.items()}
    with torch.no_grad():   # the closed form must reproduce forward() before it is used
        ub = users[:UB]
        for name in models:
            err = (fast[name](urf, ub, conf_of(name, ub)) - forward(name, ub)).abs().max().item()
            print("  closed form vs forward, %-26s max |diff| = %.2e" % (name, err), flush=True)
            assert err < 1e-4, name

    tops = {n: [] for n in todo}
    scs = {n: [] for n in todo if todo[n][3]}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(users), CB):
            ub = users[b:b + CB]
            nb = len(ub)
            mask = torch.stack([urf[u][2] for u in ub])
            P = {name: fast[name](urf, ub, conf_of(name, ub)) + mask for name in models}
            if fm_file is not None:
                P["FM_A"] = torch.stack([torch.tensor(np.asarray(fm_file[u], dtype=np.float32)) for u in ub])
            for n, (x, y, lk, keep) in todo.items():
                if y is None:
                    s = P[x] + mask if x == "FM_A" else P[x]       # rescore.py masks FM_authors again
                else:
                    lm = lam[lk][b:b + nb].unsqueeze(1)
                    s = lm * torch.sigmoid(P[x]) + (1 - lm) * torch.sigmoid(P[y])
                val, ind = torch.topk(s, 100)
                tops[n].append(item_idx[ind].numpy().astype(np.int32))
                if keep:
                    scs[n].append(val.numpy().astype(np.float32))
            print("  %d/%d users  %.0fs" % (b + nb, len(users), time.time() - t0), flush=True)
    for n in todo:
        kw = {"users": np.array(users), "top": np.concatenate(tops[n])}
        if n in scs:
            kw["score"] = np.concatenate(scs[n])
        np.savez_compressed(os.path.join(out, n + ".npz"), **kw)
        t = kw["top"].tolist()
        r = evaluate.computeTopNAccuracy([test[u] for u in users], t, [10, 20])
        v = evaluate.computeTopNAccuracy([valid[u] for u in users], t, [10, 20])
        print("%-26s valid R@10=%.4f  test R@10=%.4f R@20=%.4f N@10=%.4f" % (n, v[1][0], r[1][0], r[1][1], r[2][0]))


if __name__ == "__main__":
    main()
