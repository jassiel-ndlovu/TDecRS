"""Precompute DCR's time-dependent confounder d_u^(tau).

THE FORMULA
-----------
A user's history is ordered *timestamp blocks* B_u^(1..T), not individual
interactions -- 44.5% of ml-1m interactions share a timestamp with another by
the same user, so an index over individual interactions is partly arbitrary,
while block order is well defined (median user: 27 blocks, 48 interactions).

    personal   d~_u^(tau)  proportional to  sum_{s<tau} gamma^(tau-1-s) * sum_{i in B_s} g(i)
    shrinkage  rho_u^(tau) = n_u^(<tau) / (n_u^(<tau) + kappa)
    combined   d_u^(tau)   = (1-rho) * dbar_pop(tau) + rho * d~_u^(tau)

g(i) puts mass 1/|genres(i)| on each of item i's genres -- the same convention
the shipped confounder_prior uses.

Everything is strictly causal: block tau sees only blocks s < tau. The first
block of every user has no history, so rho = 0 and it falls back to the
population prior, which is how cold start is handled.

NESTING -- this is the point of the parameterisation:
    kappa = inf           -> d_u = dbar_pop           -> DecRS exactly
    kappa = 0, gamma = 1  -> user's running mean       -> personalisation, no recency
    kappa = 0, gamma < 1  -> DCR
    kappa > 0, gamma < 1  -> DCR with cold-start shrinkage

OUTPUTS (into data/ml_1m_dcr/)
    train_confounder_<tag>.npy  (N_train, 18) float32, row-aligned with
                                per_user/training_list.npy
    user_confounder_<tag>.npy   {userID: (18,) float32} -- state after the whole
                                training history, used when ranking at test time

USAGE
    python tools/build_dcr_priors.py --gamma 0.9 --kappa 10
    python tools/build_dcr_priors.py --kappa inf          # reproduces DecRS
"""
import argparse
import collections
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, os.pardir)
SRC = os.path.join(REPO, "data", "ml_1m")
TEMP = os.path.join(REPO, "data", "ml_1m_temporal", "per_user")
OUT = os.path.join(REPO, "data", "ml_1m_dcr")


def tag_for(gamma, kappa, pop):
    k = "inf" if np.isinf(kappa) else ("%g" % kappa)
    return "g{:g}_k{}_{}".format(gamma, k, pop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9,
                    help="recency decay per block; 1.0 = running mean, <1 = recency weighted")
    ap.add_argument("--kappa", default="0",
                    help="shrinkage strength toward the population prior; 'inf' reproduces DecRS")
    ap.add_argument("--pop", choices=["static", "windowed", "pooled"], default="static",
                    help="population prior: the shipped catalogue distribution, per time window, or "
                         "pooled (the consumed genre mix of all training rows: windowed's time-invariant control)")
    ap.add_argument("--num_groups", type=int, default=None,
                    help="default: the length of the dataset's confounder_prior (18 for ml-1m)")
    ap.add_argument("--dataset", default="ml_1m",
                    help="reads data/<dataset>/ and data/<dataset>_temporal/per_user/, "
                         "writes data/<dataset>_dcr/ (e.g. ml_10m from tools/build_ml10m.py)")
    args = ap.parse_args()
    global SRC, TEMP, OUT
    SRC = os.path.join(REPO, "data", args.dataset)
    TEMP = os.path.join(REPO, "data", args.dataset + "_temporal", "per_user")
    OUT = os.path.join(REPO, "data", args.dataset + "_dcr")
    WG = os.path.join(REPO, "data", args.dataset + "_temporal", "global")
    if args.pop == "windowed" and not os.path.exists(os.path.join(WG, "prior_by_window_consumed.npy")):
        sys.exit("--pop windowed needs data/%s_temporal/global/ (ml_1m: build_temporal_ml1m.py; "
                 "review datasets: tools/build_window_prior.py)" % args.dataset)
    kappa = float("inf") if str(args.kappa).lower() in ("inf", "infinity") else float(args.kappa)
    NG = args.num_groups or len(np.load(os.path.join(SRC, "confounder_prior.npy")))

    item_category = np.load(os.path.join(SRC, "item_category.npy"), allow_pickle=True).item()
    pop_static = np.load(os.path.join(SRC, "confounder_prior.npy")).astype(np.float64)
    train = np.load(os.path.join(TEMP, "training_list.npy"), allow_pickle=True)
    stamps = np.load(os.path.join(TEMP, "training_timestamps.npy"), allow_pickle=True)

    # g(i) lookup
    G = np.zeros((max(item_category) + 1, NG))
    for i, cats in item_category.items():
        for c in cats:
            G[i, c] = 1.0 / len(cats)

    if args.pop == "windowed":
        wp = np.load(os.path.join(WG, "prior_by_window_consumed.npy"))
        wb = np.load(os.path.join(WG, "window_bounds.npy"))

        def pop_at(ts):
            k = int(np.searchsorted(wb[:, 1], ts))
            return wp[min(k, len(wp) - 1)]
    elif args.pop == "pooled":
        pop_pooled = G[train[:, 1].astype(int)].sum(0)
        pop_pooled /= pop_pooled.sum()

        def pop_at(ts):
            return pop_pooled
    else:
        def pop_at(ts):
            return pop_static

    # rows grouped per user, then per timestamp block
    per_user = collections.defaultdict(list)
    for idx, ((u, i), t) in enumerate(zip(train, stamps)):
        per_user[int(u)].append((int(t), int(i), idx))

    train_conf = np.zeros((len(train), NG), dtype=np.float32)
    user_conf = {}
    moved, n_blocks_tot = [], []

    for u, rows in per_user.items():
        rows.sort(key=lambda r: r[0])
        blocks = []
        for t, grp in itertools.groupby(rows, key=lambda r: r[0]):
            blocks.append(list(grp))
        n_blocks_tot.append(len(blocks))

        acc = np.zeros(NG)      # sum_{s<tau} gamma^(tau-1-s) * blockvec(s)
        seen = 0                # n_u^(<tau)
        first_d = last_d = None

        for blk in blocks:
            # --- confounder for THIS block, from strictly earlier blocks only ---
            if seen == 0 or acc.sum() <= 0:
                personal = None
            else:
                personal = acc / acc.sum()
            rho = 0.0 if (seen == 0 or np.isinf(kappa)) else seen / (seen + kappa)

            ts = blk[0][0]
            base = pop_at(ts)
            d = base if personal is None else (1.0 - rho) * base + rho * personal
            s = d.sum()
            d = (d / s) if s > 0 else base

            for _t, _i, idx in blk:
                train_conf[idx] = d
            if first_d is None:
                first_d = d.copy()
            last_d = d

            # --- advance the accumulator past this block ---
            acc = args.gamma * acc + G[[i for _t, i, _x in blk]].sum(0)
            seen += len(blk)

        # State after the entire training history -> used when ranking.
        # With kappa=inf there is no personal component, so this must be the
        # SAME population prior the training rows saw: with --pop windowed that
        # is the window containing the user's last training interaction, not the
        # static catalogue prior. Getting this wrong silently turns the intended
        # population-only control into a personalised model at evaluation time.
        last_ts = blocks[-1][0][0] if blocks else None
        if np.isinf(kappa):
            d = pop_at(last_ts) if last_ts is not None else pop_static
        elif seen and acc.sum() > 0:
            personal = acc / acc.sum()
            rho = seen / (seen + kappa)
            d = (1.0 - rho) * pop_at(last_ts) + rho * personal
            d = d / d.sum()
        else:
            d = pop_at(last_ts) if last_ts is not None else pop_static
        user_conf[u] = d.astype(np.float32)
        if first_d is not None and last_d is not None:
            moved.append(np.abs(last_d - first_d).sum())

    os.makedirs(OUT, exist_ok=True)
    tag = tag_for(args.gamma, kappa, args.pop)
    np.save(os.path.join(OUT, "train_confounder_{}.npy".format(tag)), train_conf)
    np.save(os.path.join(OUT, "user_confounder_{}.npy".format(tag)), user_conf, allow_pickle=True)

    moved = np.array(moved)
    drift_vs_pop = np.abs(np.stack(list(user_conf.values())) - pop_static).sum(1)
    print("tag                      : {}".format(tag))
    print("gamma={:g}  kappa={}  pop={}".format(args.gamma, kappa, args.pop))
    print("train rows               : {}".format(len(train_conf)))
    print("users                    : {}".format(len(user_conf)))
    print("blocks per user (median) : {:.0f}".format(np.median(n_blocks_tot)))
    print("L1 movement within a user, first block -> last block:")
    print("   median {:.4f}   p75 {:.4f}   p90 {:.4f}   max {:.4f}".format(
        np.median(moved), np.percentile(moved, 75), np.percentile(moved, 90), moved.max()))
    print("L1 distance of the final user confounder from the static prior:")
    print("   median {:.4f}   p90 {:.4f}".format(
        np.median(drift_vs_pop), np.percentile(drift_vs_pop, 90)))
    if np.isinf(kappa):
        same = np.allclose(train_conf, pop_static.astype(np.float32), atol=1e-6) if args.pop == "static" else None
        print("kappa=inf sanity (static only): every row equals the static prior -> {}".format(same))
    return 0


if __name__ == "__main__":
    sys.exit(main())
