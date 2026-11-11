"""Build KuaiRec 2.0 in the DecRS file format (E4), with an observational and an interventional test.

Source: KuaiRec 2.0 (Gao et al., CIKM'22; zenodo.org/records/18164998, CC BY 4.0), extracted to
data/raw/kuairec/KuaiRec 2.0/data/:
  big_matrix.csv     policy exposures, 7,176 users x 10,728 videos, 2020-07-05..09-05
  small_matrix.csv   FULLY OBSERVED block: 1,411 users x 3,327 videos (every pair was shown)
  item_categories.csv  1-4 category ids per video

  positive : watch_ratio >= --pos_ratio (2.0, the usual KuaiRec convention; there is no click label).
  groups   : categories on >= --min_cat_items videos, each its own group; the rest go to "other".
  split    : by global date on the big matrix (UTC+8 days):
               train = big, date <= --train_end;  valid = (--train_end, --valid_end];  test = after
             testing_dict.npy      observational: big-matrix positives on the test days
             testing_dict_rand.npy interventional: small-matrix positives of kept small users (all days,
                                   since the block is fully observed), minus their training items
             (user, video) pairs that are in the small matrix are dropped from the big matrix first, so
             the interventional test never overlaps training.
  catalogue: every categorised video. Prior and features as review_ds_common.

Extra outputs (data/kuairec_temporal/exposure/): big.npy, small.npy  int64 (N, 4): user, item, unix s, positive.

    python -I tools/build_kuairec.py --raw "data/raw/kuairec/KuaiRec 2.0/data"
"""
import argparse
import ast
import csv
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_ds_common as rc  # noqa: E402

TZ = dt.timezone(dt.timedelta(hours=8))


def read(path, thr):
    """-> int64 (N, 5): raw user, raw video, yyyymmdd (UTC+8), unix s, positive; rows with no timestamp dropped."""
    out = []
    with open(path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for u, v, _pd, _vd, _tm, _d, ts, wr in r:
            if not ts or ts == "nan":
                continue
            t = int(float(ts))
            day = int(dt.datetime.fromtimestamp(t, TZ).strftime("%Y%m%d"))
            out.append((int(u), int(v), day, t, int(float(wr) >= thr)))
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--pos_ratio", type=float, default=2.0)
    ap.add_argument("--train_end", type=int, default=20200822)
    ap.add_argument("--valid_end", type=int, default=20200828)
    ap.add_argument("--min_cat_items", type=int, default=100)
    ap.add_argument("--name", default="kuairec")
    a = ap.parse_args()

    vc = {}
    with open(os.path.join(a.raw, "item_categories.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = ast.literal_eval(r["feat"])
            if c:
                vc[int(r["video_id"])] = c
    cnt = {}
    for cs in vc.values():
        for c in cs:
            cnt[c] = cnt.get(c, 0) + 1
    kept = sorted((c for c in cnt if cnt[c] >= a.min_cat_items), key=lambda c: (-cnt[c], c))
    cats = ["cat%d" % c for c in kept] + ["other"]
    cix = {c: k for k, c in enumerate(kept)}
    item_groups = {v: sorted({cix.get(c, len(kept)) for c in cs}) for v, cs in vc.items()}

    big = read(os.path.join(a.raw, "big_matrix.csv"), a.pos_ratio)
    small = read(os.path.join(a.raw, "small_matrix.csv"), a.pos_ratio)
    big = big[np.isin(big[:, 1], list(vc))]
    small = small[np.isin(small[:, 1], list(vc))]
    sp = set(zip(small[:, 0].tolist(), small[:, 1].tolist()))
    n_big = len(big)
    big = big[np.array([(u, v) not in sp for u, v in zip(big[:, 0].tolist(), big[:, 1].tolist())])]

    def first_pos(rows):
        pos, seen = {}, set()
        for k in np.argsort(rows[:, 3], kind="stable"):
            u, v, _d, t, p = rows[k].tolist()
            if p and (u, v) not in seen:
                seen.add((u, v))
                pos.setdefault(u, []).append((t, k, v))
        return pos

    p_tr = first_pos(big[big[:, 2] <= a.train_end])
    p_va = first_pos(big[(big[:, 2] > a.train_end) & (big[:, 2] <= a.valid_end)])
    p_te = first_pos(big[big[:, 2] > a.valid_end])
    p_rd = first_pos(small)
    users_raw = sorted(u for u, xs in p_tr.items() if len(xs) >= rc.MIN_POS)
    umap = {r: k + 1 for k, r in enumerate(users_raw)}
    imap = {r: k + 1 for k, r in enumerate(sorted(vc))}
    train, train_ts, titems = [], [], {}
    for ur in users_raw:
        u = umap[ur]
        for t, _k, v in sorted(p_tr[ur]):
            train.append((u, imap[v]))
            train_ts.append(t)
            titems.setdefault(u, set()).add(imap[v])

    def held_out(p):
        d, dts = {}, {}
        for ur in users_raw:
            u = umap[ur]
            xs = [(t, imap[v]) for t, _k, v in sorted(p.get(ur, [])) if imap[v] not in titems[u]]
            if xs:
                d[u], dts[u] = [i for _t, i in xs], [t for t, _i in xs]
        return d, dts

    valid, valid_ts = held_out(p_va)
    test, test_ts = held_out(p_te)
    test_r, test_r_ts = held_out(p_rd)
    for u in umap.values():
        valid.setdefault(u, [])
        valid_ts.setdefault(u, [])
    test = {u: [i for i in xs if i not in set(valid[u])] for u, xs in test.items()}
    test = {u: xs for u, xs in test.items() if xs}

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
    for d in (out, tmp):
        np.save(os.path.join(d, "training_list.npy"), np.array(train, dtype=np.int64))
        np.save(os.path.join(d, "validation_dict.npy"), valid)
        np.save(os.path.join(d, "testing_dict.npy"), test)
        np.save(os.path.join(d, "testing_dict_rand.npy"), test_r)
    np.save(os.path.join(out, "user_feature_file.npy"), user_feature)
    np.save(os.path.join(out, "item_feature_file.npy"), item_feature)
    np.save(os.path.join(out, "item_category.npy"), item_category)
    np.save(os.path.join(out, "category_list.npy"), np.array(cats))
    np.save(os.path.join(out, "confounder_prior.npy"), prior)
    np.save(os.path.join(tmp, "training_timestamps.npy"), np.array(train_ts, dtype=np.int64))
    np.save(os.path.join(tmp, "validation_timestamps.npy"), valid_ts)
    np.save(os.path.join(tmp, "testing_timestamps.npy"), test_ts)
    np.save(os.path.join(tmp, "testing_timestamps_rand.npy"), test_r_ts)
    for nm, rows in (("big", big), ("small", small)):
        rows = rows[np.isin(rows[:, 0], users_raw)]
        np.save(os.path.join(exd, nm + ".npy"), np.stack(
            [np.vectorize(umap.get)(rows[:, 0]), np.vectorize(imap.get)(rows[:, 1]), rows[:, 3], rows[:, 4]], 1))
    json.dump({"users": users_raw, "items": sorted(vc)}, open(os.path.join(out, "id_map.json"), "w"))
    man = {"source": "KuaiRec 2.0 (zenodo 18164998, CC BY 4.0)", "positive": "watch_ratio >= %g" % a.pos_ratio,
           "users": len(umap), "items": len(item_feature), "genres": cats, "num_groups": G,
           "train": len(train), "valid": sum(map(len, valid.values())), "test": sum(map(len, test.values())),
           "test_rand": sum(map(len, test_r.values())), "test_users": len(test), "test_rand_users": len(test_r),
           "big_rows_dropped_as_small_pairs": int(n_big - len(big)),
           "positive_rate": {"big": float(big[:, 4].mean()), "small": float(small[:, 4].mean())},
           "dates": {"train": "<= %d" % a.train_end, "valid": "(%d, %d]" % (a.train_end, a.valid_end),
                     "test": "> %d" % a.valid_end}}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in man.items() if k not in ("genres",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
