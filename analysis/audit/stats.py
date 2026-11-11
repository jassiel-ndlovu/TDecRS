"""Unified paired statistics for the evidence audit (PROTOCOL.md, frozen 3 Nov 2026).

compare(A, B) takes (S, U) arrays of per-user metrics, paired by seed (rows) and user (columns),
and returns the relative (or absolute) effect, the two-component t interval
(user bootstrap + between-seed variance, Welch-Satterthwaite df), TOST and leave-one-seed-out signs.
"""
import numpy as np
from scipy import stats

B_BOOT = 2000


def _effect(a, b, absolute):
    """Grand-mean effect and per-seed effects. a, b: (S, U)."""
    ma, mb = a.mean(1), b.mean(1)
    if absolute:
        return float(ma.mean() - mb.mean()), ma - mb
    return float(100 * (ma.mean() / mb.mean() - 1)), 100 * (ma / mb - 1)


def compare(a, b, margin=2.0, absolute=False, seed=0, n_boot=B_BOOT):
    a, b = np.atleast_2d(np.asarray(a, float)), np.atleast_2d(np.asarray(b, float))
    assert a.shape == b.shape, (a.shape, b.shape)
    S, U = a.shape
    d, ds = _effect(a, b, absolute)
    rng = np.random.RandomState(seed)
    # user bootstrap: the same resampled users for every seed and both models; sums via counts
    sa, sb = a.T, b.T                                         # (U, S)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        w = np.bincount(rng.randint(0, U, U), minlength=U).astype(float)
        ma, mb = w @ sa / U, w @ sb / U
        bs[i] = (ma.mean() - mb.mean()) if absolute else 100 * (ma.mean() / mb.mean() - 1)
    se_u = float(bs.std(ddof=1))
    se_s = float(ds.std(ddof=1) / np.sqrt(S)) if S > 1 else 0.0
    se = float(np.hypot(se_u, se_s))
    df = float(se ** 4 * (S - 1) / se_s ** 4) if se_s > 0 else np.inf
    t95, t90 = stats.t.ppf(0.975, df), stats.t.ppf(0.95, df)
    p = float(2 * stats.t.sf(abs(d) / se, df)) if se > 0 else (0.0 if d else 1.0)
    p_tost = float(max(stats.t.sf((d + margin) / se, df), stats.t.cdf((d - margin) / se, df))) if se > 0 else 0.0
    loso = [] if S < 3 else [float(_effect(np.delete(a, k, 0), np.delete(b, k, 0), absolute)[0]) for k in range(S)]
    return {"d": d, "per_seed": ds.tolist(), "S": S, "U": U, "se_user": se_u, "se_seed": se_s, "se": se,
            "df": df, "ci95": [d - t95 * se, d + t95 * se], "ci90": [d - t90 * se, d + t90 * se],
            "p": p, "p_tost": p_tost, "margin": margin, "absolute": absolute,
            "a": float(a.mean()), "b": float(b.mean()),
            "loso_same_sign": bool(all(np.sign(x) == np.sign(d) for x in loso)) if loso else None,
            "seeds_same_sign": int((np.sign(ds) == np.sign(d)).sum())}


def holm(ps):
    """Holm-Bonferroni adjusted p-values (monotone)."""
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def verdict(r, p_adj=None, p_tost_adj=None):
    p = r["p"] if p_adj is None else p_adj
    pt = r["p_tost"] if p_tost_adj is None else p_tost_adj
    eq = pt < 0.05
    if eq:
        return "equivalent (significant, < margin)" if p < 0.05 else "equivalent"
    if p < 0.05:
        return "better" if r["d"] > 0 else "worse"
    return "inconclusive"


def family(rows):
    """rows: list of (label, result). Adds Holm-adjusted p, p_tost and the verdict in place."""
    pa = holm([r["p"] for _, r in rows])
    pt = holm([r["p_tost"] for _, r in rows])
    for (lab, r), x, y in zip(rows, pa, pt):
        r["p_holm"], r["p_tost_holm"] = x, y
        r["verdict"] = verdict(r, x, y)
        r["verdict_unadj"] = verdict(r)
    return rows


def fmt(r, nd=1):
    if r["absolute"]:
        s = "%+.4f [%+.4f, %+.4f]" % (r["d"], *r["ci95"])
    else:
        s = "%+.*f%% [%+.*f, %+.*f]" % (nd, r["d"], nd, r["ci95"][0], nd, r["ci95"][1])
    return "%s p=%.3g p_holm=%.3g df=%.1f  %s%s" % (
        s, r["p"], r.get("p_holm", np.nan), r["df"], r.get("verdict", verdict(r)) + (
            "" if r.get("verdict_unadj", r.get("verdict")) == r.get("verdict") else " (unadj: %s)" % r["verdict_unadj"]),
        "" if r["loso_same_sign"] in (None, True) else "  [LOSO sign flips]")
