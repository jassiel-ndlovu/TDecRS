"""E2d: Steck's calibrated re-ranking (RecSys 2018) of the FM and DecRS top-100 to a top-20.

Greedy maximisation of  (1 - lam) * sum_{i in I} s(i)  -  lam * C_KL(p_u || q(I)),
p_u = the user's training genre mix (the same target as the C_KL metric),
q(I) = the genre mix of the list built so far, with Steck's 0.01 smoothing,
s(i) = the model's ranking score as a probability (sigmoid of FM's logit;
DecRS's Eq. 11 score already is one). lam = 0 reproduces the model's own top-20.

Bases: ML-1M FM (authors' scores, the DecRS blend partner), FM seeds 1-5 (trained) and
DecRS seeds 1-5; ML-10M FM seeds 1-3 and DecRS seeds 1-3.

    python analysis/eval_package/e2d_steck.py
Output: analysis/eval_package/out/e2d_steck.json, e2d_steck.csv
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DS, LABEL, OUT_DIR, Data, load_top, metrics  # noqa: E402

import numpy as np  # noqa: E402

LAMS = (0.0, 0.3, 0.5, 0.9)
K = 20
A = 0.01


def steck(D, top, score, lam, prob):
    """Greedy calibrated re-rank of every user's top-100, vectorised over users."""
    U = len(D.users)
    cand = np.stack([np.asarray(top[u]) for u in D.users])               # (U, 100)
    s = np.stack([np.asarray(score[u], float) for u in D.users])
    s = s if prob else 1 / (1 + np.exp(-s))
    M = D.Gm[cand]                                                       # (U, 100, G)
    p = D.his_g[:, None, :]
    logp = np.log(np.where(p > 0, p, 1))
    taken = np.zeros(cand.shape, bool)
    qsum, ssum = np.zeros((U, D.G)), np.zeros(U)
    out = np.zeros((U, K), int)
    rows = np.arange(U)
    for t in range(K):
        q = (qsum[:, None, :] + M) / (t + 1)
        q = (1 - A) * q + A * p
        kl = np.where(p > 0, p * (logp - np.log(np.where(q > 0, q, 1))), 0).sum(-1)    # (U, 100)
        val = (1 - lam) * (ssum[:, None] + s) - lam * kl
        val[taken] = -np.inf
        j = val.argmax(1)
        out[:, t] = cand[rows, j]
        taken[rows, j] = True
        qsum += M[rows, j]
        ssum += s[rows, j]
    return dict(zip(D.users, out))


def main():
    out = {"lambdas": LAMS, "datasets": {}}
    rows = []
    for ds in DS:
        D = Data(ds)
        seeds = DS[ds]["seeds"]
        bases = {"FM": (["FM_s%d" % s for s in seeds], False),
                 "DecRS": (["DecRS_s%d" % s for s in seeds], True)}
        if ds == "ml_1m":
            bases = {"FM (authors)": (["FM_A"], False), **bases}
        res = {}
        for b, (names, prob) in bases.items():
            res[b] = {}
            for lam in LAMS:
                ms = []
                for n in names:
                    top, sc = load_top(ds, n, with_scores=True)
                    ms.append(metrics(D, steck(D, top, sc, lam, prob)))
                r = {k: float(np.mean([m[k].mean() for m in ms])) for k in ("R@10", "R@20", "N@10", "N@20",
                                                                            "C_KL@20")}
                r["sd_R@20"] = float(np.std([m["R@20"].mean() for m in ms], ddof=1)) if len(ms) > 1 else 0.0
                res[b][str(lam)] = r
                rows.append({"dataset": LABEL[ds], "base": b, "lambda": lam, "n_runs": len(names),
                             **{k: round(v, 4) for k, v in r.items()}})
                print("%-6s %-13s lam=%.1f  R@20 %.4f  N@20 %.4f  C_KL@20 %.4f" % (
                    LABEL[ds], b, lam, r["R@20"], r["N@20"], r["C_KL@20"]), flush=True)
        out["datasets"][ds] = res
    json.dump(out, open(os.path.join(OUT_DIR, "e2d_steck.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "e2d_steck.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
