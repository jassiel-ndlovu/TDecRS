"""Re-rank the tuning winners on CPU (PROTOCOL.md, "Scoring"), at inference alpha 0.2 and 0.5.

Ranking path: masked scores, Eq. 11 blend, partner as in training (ML-1M FM_A unmasked, as rescore.py;
Goodreads the 300-epoch FM s1, masked, as goodreads_eval.stage1). Scores come from the closed-form
scorer in eval_package/score_all.py, which is checked here against each model's forward() before use.

Checks printed for every model, none of which reads alpha 0.5 test data:
  * the alpha-0.2 (or raw, for FM and FMconf) test R@10 against the training log's best-epoch [Test] line;
  * top-20 agreement with the stored D-cell lists at alpha 0.2, where those exist.

    python analysis/tuning/rerank.py ml_1m|goodreads
Output: analysis/tuning/topk/<ds>/<class>_{raw,a0.2,a0.5}_s<seed>.npz, rerank_<ds>.json
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("DECRS_DEVICE", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
AN = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AN, "eval_package"))
from common import CODE, DS_ALL, REPO, ld  # noqa: E402
from score_all import FastFM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import data_utils  # noqa: E402
import evaluate  # noqa: E402

RES = os.path.join(REPO, "results-azure")
TOPN = [10, 20, 50, 100]
CB = 512
ALPHAS = (0.2, 0.5)
HP_D = "0.05lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda"
HP_L = "0.1lr_1024bs_[0.3,0.3]dropout_0.2alpha_0.1lamda"


def p(job, stem):
    return os.path.join(RES, "results-" + job, "models", stem + ".pth")


def log(job, name):
    return os.path.join(RES, "results-" + job, "log", name + ".txt")


def jobs(ds):
    """class -> {seed: (checkpoint, confounder tag, log file or None)}. Winners are in PROTOCOL's record."""
    S = (1, 2, 3)
    if ds == "ml_1m":
        t = "g0.3_k0_static"
        n = "{}_ml_1m_" + HP_D + "{}_seed{}"
        return {
            "FM": {s: (p("fmseeds", n.format("FM", "", s)), None, None) for s in S},
            "DecRS": {s: (p("hpA", "DecFM_ml_1m_%s_seed%d" % (HP_L, s)), None, log("hpA", "hp_ml1m_DecRS_Lp_s%d" % s))
                      for s in S},
            "TDecRS": {s: (p("dcrseedsA", n.format("DecFM", "_" + t, s)), t, None) for s in S},
            "FMconf": {s: (p("dcrctrl", n.format("FMconf", "_" + t, s)), t, None) for s in S}}
    t = "g0.94_k0_static"
    d = "{}_goodreads_" + HP_D + "{}_seed{}"
    li = "DecFM_goodreads_" + HP_L + "{}_seed{}"
    fmj = {1: "grq1", 2: "grq1", 3: "hpB"}
    cfj = {1: "grq2", 2: "hpB", 3: "hpB"}
    return {
        "FM": {s: (p(fmj[s], d.format("FM", "", s)), None,
                   log("hpB", "hp_gr_FM_D_s%d" % s) if s == 3 else None) for s in S},
        "DecRS": {s: (p("hpB", li.format("", s)), None, log("hpB", "hp_gr_DecRS_Lp_s%d" % s)) for s in S},
        "TDecRS": {s: (p("hpB", li.format("_" + t, s)), t, log("hpB", "hp_gr_TDecRS_Lp_s%d" % s)) for s in S},
        "FMconf": {s: (p(cfj[s], d.format("FMconf", "_" + t, s)), t,
                       log("hpB", "hp_gr_FMconf_D_s%d" % s) if s > 1 else None) for s in S},
        "windowed": {s: (p("hpA", li.format("_g0.9_kinf_windowed", s)), "g0.9_kinf_windowed",
                         log("hpA", "hp_gr_windowed_Lp_s%d" % s)) for s in S},
        "pooled": {s: (p("hpA", li.format("_g0.9_kinf_pooled", s)), "g0.9_kinf_pooled",
                       log("hpA", "hp_gr_pooled_Lp_s%d" % s)) for s in S}}


def logged_test_r10(f):
    if not f or not os.path.exists(f):
        return None
    txt = open(f, errors="ignore").read()
    m = re.search(r"End\. Best epoch \d+\s*\n\[Test\]:.*?Recall: ([\d.]+)", txt)
    return float(m.group(1)) if m else None


def main():
    ds = sys.argv[1]
    torch.set_num_threads(os.cpu_count() or 4)
    c = DS_ALL[ds]
    out = os.path.join(HERE, "topk", ds)
    os.makedirs(out, exist_ok=True)
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(c["data"], "user_feature_file.npy"), os.path.join(c["data"], "item_feature_file.npy"))
    train = {}
    for u, i in ld(os.path.join(c["data"], "training_list.npy")).tolist():
        train.setdefault(u, []).append(i)
    valid = ld(os.path.join(c["data"], "validation_dict.npy"))
    test = ld(os.path.join(c["data"], "testing_dict.npy"))
    cat = ld(os.path.join(c["data"], "item_category.npy"))
    urf, ukl, aif, aifv = evaluate.pre_ranking(user_feature, item_feature, cat, c["G"], train, valid, test)
    users = list(test.keys())
    item_idx = torch.tensor(list(item_feature.keys()))
    NI = aif.size(0)
    gap = ukl["max_kl_score"] - ukl["min_kl_score"]
    lam0 = np.array([(ukl[u] - ukl["min_kl_score"]) / gap for u in users], np.float64)
    lam = {a: torch.tensor(lam0 ** a, dtype=torch.float32).unsqueeze(1) for a in ALPHAS}

    J = jobs(ds)
    nets, confs, fast = {}, {}, {}
    for cl, by in J.items():
        for s, (path, tag, _) in by.items():
            assert os.path.exists(path), path
            m = torch.load(path, map_location="cpu", weights_only=False)
            m.eval()
            if hasattr(m, "confounder_prior"):
                m.confounder_prior = m.confounder_prior.cpu()
            nets[cl, s] = m
            fast[cl, s] = FastFM(m, aif, aifv)
            if tag and tag not in confs:
                confs[tag] = ld(os.path.join(c["dcr"], "user_confounder_%s.npy" % tag))

    def conf_of(cl, s, ub):
        tag = J[cl][s][1]
        return None if tag is None else torch.tensor(np.stack([np.asarray(confs[tag][u], np.float32) for u in ub]))

    # the closed form must reproduce forward() before it is used
    with torch.no_grad():
        ub = users[:16]
        feat = torch.cat([torch.cat([urf[u][0].unsqueeze(0).expand(NI, -1), aif], 1) for u in ub])
        vals = torch.cat([torch.cat([urf[u][1].unsqueeze(0).expand(NI, -1), aifv], 1) for u in ub])
        for (cl, s), m in nets.items():
            cf = conf_of(cl, s, ub)
            f = m(feat, vals) if cf is None else m(feat, vals, cf.repeat_interleave(NI, 0))
            err = (fast[cl, s](urf, ub, cf) - f.view(len(ub), NI)).abs().max().item()
            assert err < 1e-4, (cl, s, err)
            print("closed form vs forward %-9s s%d max|diff|=%.1e" % (cl, s, err), flush=True)
    del feat, vals

    fm_file = ld(os.path.join(CODE, "inference", "FM_user_item_scores.npy")) if ds == "ml_1m" else None
    # FM+FM (frozen): ML-1M FM s + FM_A; Goodreads slot s2 / s3 / s2 at rows 1 / 2 / 3, partner FM s1
    ens_slot = {1: 1, 2: 2, 3: 3} if ds == "ml_1m" else {1: 2, 2: 3, 3: 2}
    tops = {}
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(users), CB):
            ub = users[b:b + CB]
            mask = torch.stack([urf[u][2] for u in ub])
            if ds == "ml_1m":
                partner = torch.stack([torch.tensor(np.asarray(fm_file[u], dtype=np.float32)) for u in ub])
            else:
                partner = fast["FM", 1](urf, ub, None) + mask
            sp = torch.sigmoid(partner)
            P = {k: fast[k](urf, ub, conf_of(k[0], k[1], ub)) + mask for k in nets}

            def put(name, s):
                tops.setdefault(name, []).append(item_idx[torch.topk(s, 100).indices].numpy().astype(np.int32))

            for (cl, s), x in P.items():
                if cl in ("FM", "FMconf"):
                    put("%s_raw_s%d" % (cl, s), x)
                if cl != "FM":
                    sx = torch.sigmoid(x)
                    for a in ALPHAS:
                        lm = lam[a][b:b + len(ub)]
                        put("%s_a%g_s%d" % (cl, a, s), lm * sx + (1 - lm) * sp)
            for row, slot in ens_slot.items():
                sx = torch.sigmoid(P["FM", slot])
                for a in ALPHAS:
                    lm = lam[a][b:b + len(ub)]
                    put("ENS_a%g_s%d" % (a, row), lm * sx + (1 - lm) * sp)
            print("  %d/%d users  %.0fs" % (b + len(ub), len(users), time.time() - t0), flush=True)
    for n in tops:
        tops[n] = np.concatenate(tops[n])
        np.savez_compressed(os.path.join(out, n + ".npz"), users=np.array(users), top=tops[n])

    # validation R@10 of every list (alpha selection reads this only); test R@10 only for the checks
    vu = [valid[u] for u in users]
    res = {"valid_R@10": {n: float(evaluate.computeTopNAccuracy(vu, tops[n].tolist(), TOPN)[1][0]) for n in tops},
           "log_check": {}, "stored_check": {}}
    tu = [test[u] for u in users]
    for cl, by in J.items():
        for s, (_, _, lg) in by.items():
            n = "%s_raw_s%d" % (cl, s) if cl in ("FM", "FMconf") else "%s_a0.2_s%d" % (cl, s)
            got = float(evaluate.computeTopNAccuracy(tu, tops[n].tolist(), TOPN)[1][0])
            want = logged_test_r10(lg)
            res["log_check"][n] = {"rerank": got, "log": want}
            print("%-18s test R@10 rerank %.4f  log %s" % (n, got, "-" if want is None else "%.4f" % want), flush=True)

    # D-cell lists already stored, at alpha 0.2
    if ds == "ml_1m":
        ref = lambda n: os.path.join(AN, "eval_package", "topk", "ml_1m", n + ".npz")
        pairs = {"FM_raw_s%d": "FM_s%d", "TDecRS_a0.2_s%d": "DCR_g0.3_s%d", "FMconf_raw_s%d": "FMconf_raw_s%d",
                 "FMconf_a0.2_s%d": "FMconf_blend_s%d"}
        checks = {k % s: ref(v % s) for k, v in pairs.items() for s in (1, 2, 3)}
        checks.update({"ENS_a0.2_s%d" % s: ref("ENS_FM_s%d+FM_A" % s) for s in (1, 2, 3)})
    else:
        ref = lambda n: os.path.join(AN, "topk_goodreads_300ep", n + ".npz")
        checks = {"FM_raw_s1": ref("FM"), "FM_raw_s2": ref("FM_s2"), "FMconf_raw_s1": ref("FMconf_raw"),
                  "FMconf_a0.2_s1": ref("FMconf_blend"), "ENS_a0.2_s1": ref("ENS_FM_s2+FM_s1")}
    for n, f in checks.items():
        z = np.load(f)
        assert z["users"].tolist() == users, f
        same = float((tops[n][:, :20] == z["top"][:, :20]).all(1).mean())
        res["stored_check"][n] = same
        print("%-18s users with an identical top-20 vs stored list: %.4f" % (n, same), flush=True)
    json.dump(res, open(os.path.join(HERE, "rerank_%s.json" % ds), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
