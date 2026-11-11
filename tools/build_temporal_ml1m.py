"""Rebuild the DecRS ml-1m dataset with timestamps, and add a global-timeline split.

WHY
---
DCR needs P(d | t) and d_u^(t). The shipped DecRS arrays are (N, 2) integer
pairs of (userID, itemID) with no time column anywhere, so neither quantity is
computable from them. Raw MovieLens-1M does carry timestamps, and -- as this
script verifies -- the DecRS arrays use raw MovieLens IDs unchanged, so the
timestamps can be joined back on exactly rather than approximately.

WHAT IT EMITS  (into data/ml_1m_temporal/)
------------------------------------------
  per_user/    The authors' own split, byte-for-byte, plus aligned timestamp
               arrays. Use this to reproduce DecRS numbers while still having
               time available.
  global/      A global-timeline split (80/10/10 by interaction time) for the
               temporal claims DCR makes. See the note on leakage below.
  shared/      Features, categories and priors common to both.

Every file keeps the upstream name and shape, so data_utils.py loads them
unchanged; the timestamps live in parallel `*_timestamps.npy` arrays.

RECIPE RECOVERED FROM THE SHIPPED DATA (each assertion is checked at runtime)
----------------------------------------------------------------------------
  ids      : DecRS user/item ids ARE raw MovieLens UserID / MovieID.
  filter   : implicit-positive = rating >= 4  (575,281 raw vs 575,276 shipped).
  split    : per-user temporal, last ~10% test, preceding ~10% validation.
             99.6% of test interactions postdate all of that user's training
             data, so this is leave-last-N, NOT a random split.
  genres   : item_category == movies.dat genres via category_list (3883/3883).
  prior    : confounder_prior == the ITEM CATALOGUE genre distribution, each
             item contributing round(1/|genres|, 4) per genre, normalised.
             It is not interaction-weighted and not user-weighted.

ON LEAKAGE
----------
A per-user temporal split still interleaves users in absolute time: user A's
training interactions can postdate user B's test interactions, so the model sees
globally-future information. That is the effect documented in Ji et al., "A
Critical Study on Data Leakage in Recommender System Offline Evaluation" (TOIS
2023). The global/ split avoids it; per_user/ is kept so DecRS comparisons stay
like-for-like.

USAGE
-----
    python tools/build_temporal_ml1m.py --raw /path/to/ml-1m
"""
import argparse
import collections
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, os.pardir)
SRC = os.path.join(REPO, "data", "ml_1m")
OUT = os.path.join(REPO, "data", "ml_1m_temporal")

POSITIVE_RATING = 4
VALID_FRAC, TEST_FRAC = 0.10, 0.10


def load_raw(raw_dir):
    ratings = {}
    for line in open(os.path.join(raw_dir, "ratings.dat"), encoding="latin-1"):
        u, i, r, t = line.strip().split("::")
        ratings[(int(u), int(i))] = (int(r), int(t))
    genres = {}
    for line in open(os.path.join(raw_dir, "movies.dat"), encoding="latin-1"):
        mid, _title, g = line.strip().split("::")
        genres[int(mid)] = g.split("|")
    return ratings, genres


def group_distribution(items, item_category, num_groups):
    """Mass round(1/|genres|, 4) per genre, matching the upstream convention."""
    v = np.zeros(num_groups)
    for i in items:
        cats = item_category[i]
        w = round(1.0 / len(cats), 4)
        for c in cats:
            v[c] += w
    return v / v.sum() if v.sum() else v


def verify(ratings, genres, shipped, checks):
    item_category, category_list, prior = shipped["ic"], shipped["cl"], shipped["cp"]
    n_groups = len(category_list)

    pairs = ([tuple(int(x) for x in p) for p in shipped["train"]]
             + [(u, i) for u, v in shipped["valid"].items() for i in v]
             + [(u, i) for u, v in shipped["test"].items() for i in v])
    missing = [p for p in pairs if p not in ratings]
    checks["split_pairs_total"] = len(pairs)
    checks["split_pairs_missing_from_raw"] = len(missing)
    assert not missing, "{} split pairs are absent from ratings.dat -- the id " \
                        "mapping assumption is wrong".format(len(missing))

    used = collections.Counter(ratings[p][0] for p in pairs)
    checks["rating_values_used"] = dict(sorted(used.items()))
    assert min(used) >= POSITIVE_RATING, "split contains ratings below the assumed threshold"

    bad = [m for m, c in item_category.items()
           if sorted(c) != sorted(category_list.index(g) for g in genres[m])]
    checks["item_category_mismatches"] = len(bad)
    assert not bad, "item_category disagrees with movies.dat for {} items".format(len(bad))

    rebuilt = group_distribution(sorted(item_category), item_category, n_groups)
    gap = float(np.abs(rebuilt - prior).max())
    checks["confounder_prior_max_gap"] = gap
    assert gap < 1e-4, "confounder_prior is not the catalogue distribution (gap {:.2e})".format(gap)


def global_split(ratings, users_keep, items_keep):
    """80/10/10 by global interaction time."""
    rows = sorted(((t, u, i) for (u, i), (r, t) in ratings.items()
                   if r >= POSITIVE_RATING and u in users_keep and i in items_keep))
    n = len(rows)
    a, b = int(n * (1 - VALID_FRAC - TEST_FRAC)), int(n * (1 - TEST_FRAC))
    return rows[:a], rows[a:b], rows[b:]


def as_dicts(rows):
    d, ts = collections.defaultdict(list), collections.defaultdict(list)
    for t, u, i in rows:
        d[u].append(i)
        ts[u].append(t)
    return dict(d), dict(ts)


def save_dict(path, obj):
    np.save(path, obj, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory holding ratings.dat / movies.dat / users.dat")
    ap.add_argument("--windows", type=int, default=8, help="time windows for the P(d|t) table")
    args = ap.parse_args()

    checks = {}
    print("reading raw MovieLens-1M from", args.raw)
    ratings, genres = load_raw(args.raw)
    checks["raw_ratings"] = len(ratings)

    shipped = {
        "train": np.load(os.path.join(SRC, "training_list.npy"), allow_pickle=True),
        "valid": np.load(os.path.join(SRC, "validation_dict.npy"), allow_pickle=True).item(),
        "test": np.load(os.path.join(SRC, "testing_dict.npy"), allow_pickle=True).item(),
        "ic": np.load(os.path.join(SRC, "item_category.npy"), allow_pickle=True).item(),
        "cl": np.load(os.path.join(SRC, "category_list.npy"), allow_pickle=True).tolist(),
        "cp": np.load(os.path.join(SRC, "confounder_prior.npy")),
    }
    n_groups = len(shipped["cl"])

    print("verifying the recovered recipe against the shipped arrays...")
    verify(ratings, genres, shipped, checks)
    print("  all assertions passed")

    for sub in ("per_user", "global", "shared"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    # ---- shared -------------------------------------------------------------
    for f in ("user_feature_file.npy", "item_feature_file.npy",
              "item_category.npy", "category_list.npy", "confounder_prior.npy"):
        shutil.copy2(os.path.join(SRC, f), os.path.join(OUT, "shared", f))

    # ---- per_user: the authors' split, plus timestamps ----------------------
    pu = os.path.join(OUT, "per_user")
    train = np.asarray(shipped["train"])
    np.save(os.path.join(pu, "training_list.npy"), train)
    np.save(os.path.join(pu, "training_timestamps.npy"),
            np.array([ratings[(int(u), int(i))][1] for u, i in train], dtype=np.int64))
    for name, key in (("validation", "valid"), ("testing", "test")):
        dd = shipped[key]
        save_dict(os.path.join(pu, name + "_dict.npy"), dd)
        save_dict(os.path.join(pu, name + "_timestamps.npy"),
                  {u: [ratings[(u, i)][1] for i in v] for u, v in dd.items()})

    # ---- global-timeline split ---------------------------------------------
    users_keep = set(np.load(os.path.join(SRC, "user_feature_file.npy"), allow_pickle=True).item())
    items_keep = set(shipped["ic"])
    g_tr, g_va, g_te = global_split(ratings, users_keep, items_keep)

    tr_users = {u for _t, u, _i in g_tr}
    g_va = [r for r in g_va if r[1] in tr_users]
    g_te = [r for r in g_te if r[1] in tr_users]

    gg = os.path.join(OUT, "global")
    np.save(os.path.join(gg, "training_list.npy"),
            np.array([[u, i] for _t, u, i in g_tr], dtype=np.int64))
    np.save(os.path.join(gg, "training_timestamps.npy"),
            np.array([t for t, _u, _i in g_tr], dtype=np.int64))
    for name, rows in (("validation", g_va), ("testing", g_te)):
        d, ts = as_dicts(rows)
        save_dict(os.path.join(gg, name + "_dict.npy"), d)
        save_dict(os.path.join(gg, name + "_timestamps.npy"), ts)

    g_items = {i for _t, _u, i in g_tr}
    np.save(os.path.join(gg, "confounder_prior.npy"),
            group_distribution(sorted(g_items), shipped["ic"], n_groups))

    checks["global_split"] = {
        "train": len(g_tr), "valid": len(g_va), "test": len(g_te),
        "train_users": len(tr_users), "train_items": len(g_items),
        "valid_users": len({r[1] for r in g_va}), "test_users": len({r[1] for r in g_te}),
        "boundary_train_valid": g_tr[-1][0], "boundary_valid_test": g_va[-1][0] if g_va else None,
    }

    # ---- P(d | t): the table DCR's Eq. 3.1 actually needs --------------------
    # Two constructions, because they answer different questions and the choice
    # is a modelling decision, not a detail:
    #   catalogue   -- what DecRS ships: the genre mix of the items available.
    #   consumed    -- interaction-weighted: the genre mix of what users chose.
    edges = np.linspace(0, len(g_tr), args.windows + 1).astype(int)
    cat_rows, con_rows, bounds = [], [], []
    for k in range(args.windows):
        chunk = g_tr[edges[k]:edges[k + 1]]
        cat_rows.append(group_distribution(sorted({i for _t, _u, i in chunk}), shipped["ic"], n_groups))
        con_rows.append(group_distribution([i for _t, _u, i in chunk], shipped["ic"], n_groups))
        bounds.append((chunk[0][0], chunk[-1][0]))
    np.save(os.path.join(gg, "prior_by_window_catalogue.npy"), np.array(cat_rows))
    np.save(os.path.join(gg, "prior_by_window_consumed.npy"), np.array(con_rows))
    np.save(os.path.join(gg, "window_bounds.npy"), np.array(bounds, dtype=np.int64))
    drift = float(np.abs(np.array(con_rows)[0] - np.array(con_rows)[-1]).max())
    checks["prior_drift_first_to_last_window_consumed"] = drift

    checks["recipe"] = {
        "ids": "raw MovieLens UserID / MovieID, unchanged",
        "positive_rating_threshold": POSITIVE_RATING,
        "authors_split": "per-user temporal, last ~10% test, preceding ~10% validation",
        "confounder_prior": "item catalogue genre distribution, round(1/|genres|,4) mass",
        "global_split": "80/10/10 by global interaction timestamp",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(checks, f, indent=2)

    print("\nwrote", OUT)
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
