"""E2e, part 1: null-calibrated drift z-score per user.

DecRS's drift score kl_u (evaluate.kl_score_calc) is the symmetric KL between the
genre mix of the first and second half of the user's train+valid sequence. Raw KL
is dominated by sampling noise for short histories. The null keeps the user's
own timestamp blocks (interactions sharing a timestamp move together) and
shuffles their ORDER, so it breaks temporal structure and keeps everything else:

    z_u = (kl_u - mean_null) / sd_null,   N_PERM permutations of block order

Users whose null has zero spread (a single block, or blocks that can only give
one split) get z = 0: no evidence of drift.

    python analysis/eval_package/drift_z.py [ml_1m] [ml_10m]
Output: analysis/eval_package/out/drift_z_<ds>.npz  (users, kl, null_mean, null_sd, z, n_blocks, n_items)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DS, OUT_DIR, Data, ld  # noqa: E402

N_PERM = 200
EPS = 0.01   # kl_div's smoothing


def sym_kl(a, b):
    """Vectorised evaluate.kl_score_calc on rows of genre-count vectors a, b (already /len)."""
    def kl(p, q):
        qq = (1 - EPS) * q + EPS * p
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(p >= 1e-5, p * np.log(np.where(p >= 1e-5, p, 1) / np.where(qq > 0, qq, 1)), 0.0)
        return t.sum(-1)
    return kl(a, b) + kl(b, a)


def main():
    only = sys.argv[1:]
    for ds in DS:
        if only and ds not in only:
            continue
        t0 = time.time()
        D = Data(ds)
        c = DS[ds]
        vts = ld(os.path.join(c["temporal"], "validation_timestamps.npy"))
        tts = ld(os.path.join(c["temporal"], "training_timestamps.npy"))
        tl = ld(os.path.join(c["data"], "training_list.npy")).tolist()
        tstamp = {}
        for (u, _), t in zip(tl, tts.tolist()):
            tstamp.setdefault(u, []).append(t)
        # kl_score_calc rounds 1/|genres| to 4 decimals; copy that for exactness
        Gr = np.zeros_like(D.Gm)
        for i, cs in D.cat.items():
            cs = [cs] if np.isscalar(cs) else list(cs)
            for x in cs:
                Gr[i, x] += round(1.0 / len(cs), 4)
        kl_ref = D.kl_scores()
        users = D.all_users
        out = {k: np.zeros(len(users)) for k in ("kl", "null_mean", "null_sd", "z", "n_blocks", "n_items")}
        for j, u in enumerate(users):
            items = np.array(D.train[u] + list(D.valid[u]), int)
            times = np.array(tstamp[u] + list(vts[u]))
            n, h = len(items), len(items) // 2
            X = Gr[items]
            obs = sym_kl(X[:h].sum(0)[None] / h, X[h:].sum(0)[None] / (n - h))[0]
            # blocks = maximal runs of equal timestamps in the observed order
            starts = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
            ends = np.r_[starts[1:], n]
            nb = len(starts)
            rng = np.random.RandomState(int(u) % (2 ** 31))
            if nb > 1:
                # block-level: whole blocks before the cut, plus a prefix of the block it falls in
                perms = np.argsort(rng.rand(N_PERM, nb), 1)
                C = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, 0)])
                Bs = C[ends] - C[starts]                      # (nb, G) block genre sums
                size = ends - starts
                cs = np.cumsum(size[perms], 1)                # (P, nb)
                k = (cs <= h).sum(1)                          # blocks wholly in the first half
                before = np.where(k > 0, cs[np.arange(N_PERM), np.maximum(k - 1, 0)], 0)
                whole = np.cumsum(Bs[perms], 1)               # (P, nb, G)
                head = np.where((k > 0)[:, None], whole[np.arange(N_PERM), np.maximum(k - 1, 0)], 0.0)
                kb = perms[np.arange(N_PERM), np.minimum(k, nb - 1)]
                r = h - before                                # items taken from block kb (0 if k == nb)
                part = C[starts[kb] + r] - C[starts[kb]]
                a = (head + part) / h
                b = (X.sum(0)[None] - head - part) / (n - h)
                null = sym_kl(a, b)
                mu, sd = null.mean(), null.std(ddof=1)
            else:
                mu, sd = obs, 0.0
            z = (obs - mu) / sd if sd > 1e-12 else 0.0
            for k, v in (("kl", obs), ("null_mean", mu), ("null_sd", sd), ("z", z), ("n_blocks", nb),
                         ("n_items", n)):
                out[k][j] = v
        ref = np.array([kl_ref[u] for u in users])
        err = np.abs(out["kl"] - ref).max()
        print("%s: %d users, max |kl - kl_score_calc| = %.2e, z>1.96 for %.1f%%, corr(kl, log n) = %.3f, "
              "corr(z, log n) = %.3f  (%.0fs)" % (
                  ds, len(users), err, 100 * (out["z"] > 1.96).mean(),
                  np.corrcoef(out["kl"], np.log(out["n_items"]))[0, 1],
                  np.corrcoef(out["z"], np.log(out["n_items"]))[0, 1], time.time() - t0), flush=True)
        assert err < 1e-6, "observed KL does not reproduce evaluate.kl_score_calc"
        np.savez_compressed(os.path.join(OUT_DIR, "drift_z_%s.npz" % ds), users=np.array(users), **out)


if __name__ == "__main__":
    main()
