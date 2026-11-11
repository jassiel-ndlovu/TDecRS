"""Build MovieLens-10M in the DecRS file format, for the Tier 2 pilot.

Follows the recipe recovered for the DecRS ml-1m release
(tools/build_temporal_ml1m.py) wherever ML-10M allows it:

  ids      : raw MovieLens UserID / MovieID.
  filter   : implicit positive = rating >= 4 (ML-10M has half stars: 4, 4.5, 5).
  split    : per-user temporal. Each user's positives are stably sorted by
             timestamp; the last round(10%) (at least 1) is test, the round(10%)
             (at least 1) before it is validation, and the rest is training.
             Users with fewer than MIN_POS positives are dropped.
  genres   : movies.dat genres. ML-10M has 19 (IMAX is new), plus one movie
             with "(no genres listed)", which is dropped from the catalogue.
  prior    : the item-catalogue genre distribution, each item contributing
             round(1/|genres|, 4) per genre, normalised.
  features : user = [U<id>] only, because ML-10M has no demographics
             (num_user_features = 1). item = [I<id>, IC0..IC18] with the genre
             slots weighted round(1/|genres|, 2), as in the ml-1m item file.

Population: ML-10M is nearly as time-concentrated as ML-1M (74.7% of eligible
users' training histories span under one day; median 26 min), so the Tier 2
pilot keeps only users whose TRAINING history spans >= 30 days
(--min_span_days 30: 9,855 users, 1.32M training positives). That is the
population where drift and within-session recency can be told apart. The
full catalogue is kept. --user_frac (a seeded random user subset) is also
available but unused.

Outputs
  data/ml_10m/                       DecRS-format files (same names as data/ml_1m)
  data/ml_10m_temporal/per_user/     training_list.npy + *_timestamps.npy, which
                                     tools/build_dcr_priors.py --dataset ml_10m reads
  data/ml_10m/manifest.json          counts, span distribution, the choices above

    python -I tools/build_ml10m.py --raw data/raw/ml-10m/ml-10M100K --min_span_days 30
"""
import argparse
import collections
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
MIN_POS = 5
DAY = 86400.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder holding ratings.dat and movies.dat")
    ap.add_argument("--user_frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min_span_days", type=float, default=0.0,
                    help="keep only users whose TRAINING history spans at least this many days")
    ap.add_argument("--name", default="ml_10m")
    a = ap.parse_args()

    # ---- catalogue -----------------------------------------------------------
    genres = {}
    with open(os.path.join(a.raw, "movies.dat"), encoding="latin-1") as f:
        for line in f:
            mid, _title, g = line.rstrip("\n").split("::")
            if g != "(no genres listed)":
                genres[int(mid)] = g.split("|")
    cats = sorted({g for gs in genres.values() for g in gs})
    G = len(cats)
    cidx = {c: k for k, c in enumerate(cats)}
    item_category = {i: [cidx[g] for g in gs] for i, gs in genres.items()}

    prior = np.zeros(G)
    for cs in item_category.values():
        for c in cs:
            prior[c] += round(1.0 / len(cs), 4)
    prior /= prior.sum()

    item_feature = {}
    for i, cs in item_category.items():
        w = round(1.0 / len(cs), 2)
        item_feature[i] = [["I%d" % i] + ["IC%d" % k for k in range(G)],
                           ["1"] + [("%g" % w) if k in cs else "0" for k in range(G)]]

    # ---- interactions ----------------------------------------------------------
    pos = collections.defaultdict(list)
    n_raw = 0
    with open(os.path.join(a.raw, "ratings.dat")) as f:
        for k, line in enumerate(f):
            u, i, r, t = line.split("::")
            n_raw += 1
            if float(r) >= 4 and int(i) in genres:
                pos[int(u)].append((int(t), k, int(i)))   # k keeps file order for ties
    users = sorted(u for u, xs in pos.items() if len(xs) >= MIN_POS)
    n_eligible = len(users)
    if a.user_frac < 1:
        rng = np.random.RandomState(a.seed)
        users = sorted(rng.choice(users, int(round(a.user_frac * len(users))), replace=False).tolist())

    train, train_ts, valid, test, valid_ts, test_ts = [], [], {}, {}, {}, {}
    spans, lens = [], []
    kept = []
    for u in users:
        xs = sorted(pos[u])
        n = len(xs)
        nt = max(1, int(round(0.1 * n)))
        nv = max(1, int(round(0.1 * n)))
        tr, va, te = xs[:n - nt - nv], xs[n - nt - nv:n - nt], xs[n - nt:]
        # span of the TRAINING part only, so selecting on it never looks at
        # validation or test timestamps
        if (tr[-1][0] - tr[0][0]) / DAY < a.min_span_days:
            continue
        kept.append(u)
        train += [(u, i) for _, _, i in tr]
        train_ts += [t for t, _, _ in tr]
        valid[u] = [i for _, _, i in va]
        test[u] = [i for _, _, i in te]
        valid_ts[u] = [t for t, _, _ in va]
        test_ts[u] = [t for t, _, _ in te]
        spans.append((tr[-1][0] - tr[0][0]) / DAY)
        lens.append(len(tr))
    spans, lens = np.array(spans), np.array(lens)
    users = kept

    user_feature = {u: [["U%d" % u], ["1"]] for u in users}

    # ---- write -----------------------------------------------------------------
    out = os.path.join(REPO, "data", a.name)
    tmp = os.path.join(REPO, "data", a.name + "_temporal", "per_user")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    np.save(os.path.join(out, "training_list.npy"), np.array(train, dtype=np.int64))
    np.save(os.path.join(out, "validation_dict.npy"), valid)
    np.save(os.path.join(out, "testing_dict.npy"), test)
    np.save(os.path.join(out, "user_feature_file.npy"), user_feature)
    np.save(os.path.join(out, "item_feature_file.npy"), item_feature)
    np.save(os.path.join(out, "item_category.npy"), item_category)
    np.save(os.path.join(out, "category_list.npy"), np.array(cats))
    np.save(os.path.join(out, "confounder_prior.npy"), prior)
    np.save(os.path.join(tmp, "training_list.npy"), np.array(train, dtype=np.int64))
    np.save(os.path.join(tmp, "training_timestamps.npy"), np.array(train_ts, dtype=np.int64))
    np.save(os.path.join(tmp, "validation_dict.npy"), valid)
    np.save(os.path.join(tmp, "testing_dict.npy"), test)
    np.save(os.path.join(tmp, "validation_timestamps.npy"), valid_ts)
    np.save(os.path.join(tmp, "testing_timestamps.npy"), test_ts)

    q = lambda x, p: float(np.percentile(x, p))
    man = {
        "source": "MovieLens-10M (ml-10M100K)", "raw_ratings": n_raw,
        "users_with_min_pos": n_eligible, "user_frac": a.user_frac, "seed": a.seed,
        "min_span_days": a.min_span_days,
        "users": len(users), "items": len(item_feature), "genres": cats, "num_groups": G,
        "train": len(train), "valid": sum(map(len, valid.values())), "test": sum(map(len, test.values())),
        "train_span_days": {"<1d": float((spans < 1).mean()), "1-30d": float(((spans >= 1) & (spans < 30)).mean()),
                            ">30d": float((spans >= 30).mean()), ">365d": float((spans >= 365).mean()),
                            "median": q(spans, 50), "p75": q(spans, 75), "p90": q(spans, 90)},
        "train_len": {"median": q(lens, 50), "p90": q(lens, 90)},
        "recipe": {"filter": "rating >= 4", "min_pos": MIN_POS,
                   "split": "per-user temporal: last round(10%) test, preceding round(10%) valid (each >= 1)",
                   "prior": "item catalogue genre distribution, round(1/|genres|,4) mass",
                   "user_features": "user ID only (no demographics in ML-10M)"},
    }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: man[k] for k in ("users_with_min_pos", "users", "items", "num_groups", "train",
                                          "valid", "test", "train_span_days", "train_len")}, indent=1))


if __name__ == "__main__":
    main()
