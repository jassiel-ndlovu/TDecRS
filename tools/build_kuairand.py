"""Build KuaiRand-Pure in the DecRS file format (E4), with an observational AND an interventional test.

Source: KuaiRand-Pure (Gao et al., CIKM'22; zenodo.org/records/10439422, CC BY 4.0), extracted to
data/raw/kuairand/KuaiRand-Pure/data/:
  log_standard_4_08_to_4_21_pure.csv, log_standard_4_22_to_5_08_pure.csv   policy exposures (is_rand=0)
  log_random_4_22_to_5_08_pure.csv                                         uniformly random exposures (is_rand=1)
  video_features_basic_pure.csv                                            `tag`: 1-3 tag ids per video

  positive : is_click (--label long_view is the alternative). Repeat (user, video) positives keep the first.
  groups   : the tags on at least --min_tag_videos videos, each its own group; every other tag goes
             to "other". Videos with no tag are dropped. Multi-label, like ML genres.
  split    : by GLOBAL date, not per user:
               train = standard log, dates <= --train_end
               valid = standard log, (--train_end, --valid_end]
               test  = standard log, (--valid_end, end]           -> testing_dict.npy (observational)
               test  = random log on the same test days          -> testing_dict_rand.npy (interventional)
             Users need >= MIN_POS training positives. A test user needs >= 1 positive in that test set.
             Every training user has a validation key (possibly empty), because evaluate.py reads
             valid_dict[u] for every test user. Test and validation items a user already clicked in
             training are dropped (the ranker masks them).
  catalogue: every tagged video, including videos with no training positive, so all random-log
             videos can be ranked.
  prior / features: as review_ds_common (catalogue group mass, user-ID feature only).

Extra outputs, for the screen and the confounding probe (data/kuairand_temporal/exposure/):
  standard.npy, random.npy   int64 (N, 5): user, item, unix seconds, is_click, long_view; every impression
                             of a kept user on a catalogue video, all dates

    python -I tools/build_kuairand.py --raw data/raw/kuairand/KuaiRand-Pure/data
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_ds_common as rc  # noqa: E402

STD = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]
RND = "log_random_4_22_to_5_08_pure.csv"


def read_log(path, label):
    """-> int64 (N, 6): raw user, raw video, date yyyymmdd, unix seconds, is_click, long_view; file order."""
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append((int(r["user_id"]), int(r["video_id"]), int(r["date"]), int(r["time_ms"]) // 1000,
                        int(r["is_click"]), int(r["long_view"])))
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--label", choices=["is_click", "long_view"], default="is_click")
    ap.add_argument("--train_end", type=int, default=20220501)
    ap.add_argument("--valid_end", type=int, default=20220504)
    ap.add_argument("--min_tag_videos", type=int, default=100)
    ap.add_argument("--name", default="kuairand")
    a = ap.parse_args()
    col = 4 if a.label == "is_click" else 5

    vtags = {}
    with open(os.path.join(a.raw, "video_features_basic_pure.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ts = [t.strip() for t in r["tag"].split(",") if t.strip()]
            if ts:
                vtags[int(r["video_id"])] = ts
    cnt = {}
    for ts in vtags.values():
        for t in ts:
            cnt[t] = cnt.get(t, 0) + 1
    kept_tags = sorted((t for t in cnt if cnt[t] >= a.min_tag_videos), key=lambda t: (-cnt[t], int(t)))
    cats = ["tag%s" % t for t in kept_tags] + ["other"]
    tix = {t: k for k, t in enumerate(kept_tags)}
    item_groups = {v: sorted({tix.get(t, len(kept_tags)) for t in ts}) for v, ts in vtags.items()}

    std = np.concatenate([read_log(os.path.join(a.raw, f), a.label) for f in STD])
    rnd = read_log(os.path.join(a.raw, RND), a.label)
    std = std[np.isin(std[:, 1], list(vtags))]
    rnd = rnd[np.isin(rnd[:, 1], list(vtags))]

    def first_pos(rows):
        """{user: [(t, file_order, video)]}, positives only, first occurrence of each (user, video)."""
        pos, seen = {}, set()
        for k in np.argsort(rows[:, 3], kind="stable"):
            u, v, _d, t, *_ = rows[k]
            if rows[k, col] and (u, v) not in seen:
                seen.add((u, v))
                pos.setdefault(int(u), []).append((int(t), int(k), int(v)))
        return pos

    tr_rows = std[std[:, 2] <= a.train_end]
    va_rows = std[(std[:, 2] > a.train_end) & (std[:, 2] <= a.valid_end)]
    te_rows = std[std[:, 2] > a.valid_end]
    tr_rand = rnd[rnd[:, 2] > a.valid_end]
    p_tr, p_va, p_te, p_rd = first_pos(tr_rows), first_pos(va_rows), first_pos(te_rows), first_pos(tr_rand)

    users_raw = sorted(u for u, xs in p_tr.items() if len(xs) >= rc.MIN_POS)
    umap = {r: k + 1 for k, r in enumerate(users_raw)}
    items_raw = sorted(vtags)
    imap = {r: k + 1 for k, r in enumerate(items_raw)}
    train, train_ts, train_items = [], [], {}
    for ur in users_raw:
        u = umap[ur]
        for t, _k, v in sorted(p_tr[ur]):
            train.append((u, imap[v]))
            train_ts.append(t)
            train_items.setdefault(u, set()).add(imap[v])

    def held_out(p):
        d, dts = {}, {}
        for ur in users_raw:
            u = umap[ur]
            xs = [(t, imap[v]) for t, _k, v in sorted(p.get(ur, [])) if imap[v] not in train_items[u]]
            if xs:
                d[u] = [i for _t, i in xs]
                dts[u] = [t for t, _i in xs]
        return d, dts

    valid, valid_ts = held_out(p_va)
    test, test_ts = held_out(p_te)
    test_r, test_r_ts = held_out(p_rd)
    for u in umap.values():           # evaluate.py reads valid_dict[u] for every test user
        valid.setdefault(u, [])
        valid_ts.setdefault(u, [])
    # an item can be both a validation and a test positive; the ranker masks validation items at test time
    test = {u: [i for i in xs if i not in set(valid[u])] for u, xs in test.items()}
    test = {u: xs for u, xs in test.items() if xs}
    test_r = {u: [i for i in xs if i not in set(valid[u])] for u, xs in test_r.items()}
    test_r = {u: xs for u, xs in test_r.items() if xs}

    G = len(cats)
    inv = {v: k for k, v in imap.items()}
    item_category = {i: item_groups[inv[i]] for i in imap.values()}
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
    user_feature = {u: [["U%d" % u], ["1"]] for u in umap.values()}

    out = os.path.join(rc.REPO, "data", a.name)
    tmp = os.path.join(rc.REPO, "data", a.name + "_temporal", "per_user")
    exd = os.path.join(rc.REPO, "data", a.name + "_temporal", "exposure")
    for d in (out, tmp, exd):
        os.makedirs(d, exist_ok=True)
    np.save(os.path.join(out, "training_list.npy"), np.array(train, dtype=np.int64))
    np.save(os.path.join(out, "validation_dict.npy"), valid)
    np.save(os.path.join(out, "testing_dict.npy"), test)
    np.save(os.path.join(out, "testing_dict_rand.npy"), test_r)
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
    np.save(os.path.join(tmp, "testing_dict_rand.npy"), test_r)
    np.save(os.path.join(tmp, "testing_timestamps_rand.npy"), test_r_ts)
    for nm, rows in (("standard", std), ("random", rnd)):
        rows = rows[np.isin(rows[:, 0], users_raw)]
        np.save(os.path.join(exd, nm + ".npy"), np.stack(
            [np.vectorize(umap.get)(rows[:, 0]), np.vectorize(imap.get)(rows[:, 1]),
             rows[:, 3], rows[:, 4], rows[:, 5]], 1).astype(np.int64))
    json.dump({"users": users_raw, "items": items_raw}, open(os.path.join(out, "id_map.json"), "w"))

    lens = np.array([len(train_items[u]) for u in umap.values()])
    rate = lambda r: float(r[:, col].mean()) if len(r) else float("nan")
    man = {
        "source": "KuaiRand-Pure (zenodo 10439422, CC BY 4.0)", "label": a.label,
        "users": len(umap), "items": len(item_feature), "genres": cats, "num_groups": G,
        "train": len(train), "valid": sum(map(len, valid.values())), "test": sum(map(len, test.values())),
        "test_rand": sum(map(len, test_r.values())), "test_users": len(test), "test_rand_users": len(test_r),
        "dates": {"train": "<= %d" % a.train_end, "valid": "(%d, %d]" % (a.train_end, a.valid_end),
                  "test": "> %d (to 20220508)" % a.valid_end},
        "positive_rate": {"train_std": rate(tr_rows), "test_std": rate(te_rows), "test_rand": rate(tr_rand),
                          "random_all": rate(rnd), "standard_all": rate(std)},
        "train_len": {"median": float(np.median(lens)), "p90": float(np.percentile(lens, 90))},
        "prior": dict(zip(cats, np.round(prior, 4).tolist())),
        "recipe": {"min_pos": rc.MIN_POS, "min_tag_videos": a.min_tag_videos,
                   "split": "global date; observational test = standard log, interventional test = random log, same days",
                   "prior": "item catalogue group distribution, round(1/|groups|,4) mass",
                   "user_features": "user ID only (num_user_features = 1)"},
    }
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: man[k] for k in ("users", "items", "num_groups", "train", "valid", "test", "test_rand",
                                          "test_users", "test_rand_users", "positive_rate", "train_len")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
