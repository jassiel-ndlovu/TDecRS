"""Shared split-and-write step for the E3 candidate datasets (tools/build_amazon_mtv.py,
tools/build_goodreads.py).

It is the build_ml10m.py recipe, factored out so that every candidate is built the same way:
  split    : per-user temporal. A user's positives are stably sorted by timestamp
             (ties keep file order); the last round(10%) (at least 1) is test, the
             round(10%) (at least 1) before it is validation, and the rest is training.
             Users with fewer than MIN_POS positives are dropped.
  prior    : the item-catalogue group distribution, round(1/|groups|, 4) mass per group.
  features : user = [U<id>] (num_user_features = 1); item = [I<id>, IC0..IC{G-1}] with the
             group slots weighted round(1/|groups|, 2).

Item and user IDs are remapped to 1-based integers; the raw IDs go to id_map.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir))
MIN_POS = 5
DAY = 86400.0


def cap_catalogue(pos, max_items, rounds=3):
    """pos: {user: [(t, k, item)]}. Keep the max_items items with the most positives,
    then users with >= MIN_POS positives on them; repeat (the user filter moves counts)."""
    for _ in range(rounds):
        cnt = {}
        for xs in pos.values():
            for _, _, i in xs:
                cnt[i] = cnt.get(i, 0) + 1
        keep = set(sorted(cnt, key=lambda i: (-cnt[i], i))[:max_items])
        pos = {u: [x for x in xs if x[2] in keep] for u, xs in pos.items()}
        pos = {u: xs for u, xs in pos.items() if len(xs) >= MIN_POS}
    return pos


def sample_users(pos, n_users, seed):
    """Seeded random subset of n_users eligible users (keeps the FM score file,
    users x items float32, near ML-10M's size)."""
    users = sorted(u for u, xs in pos.items() if len(xs) >= MIN_POS)
    if n_users and n_users < len(users):
        users = np.random.RandomState(seed).choice(users, n_users, replace=False).tolist()
    return {u: pos[u] for u in users}


def split_and_write(name, pos, item_groups, cats, source, recipe, min_span_days=0.0, extra=None):
    """pos: {raw_user: [(t_seconds, file_order, raw_item)]}; item_groups: {raw_item: [group idx]}."""
    users_raw = sorted(u for u, xs in pos.items() if len(xs) >= MIN_POS)
    n_eligible = len(users_raw)
    items_raw = sorted({i for u in users_raw for _, _, i in pos[u]})
    imap = {r: k + 1 for k, r in enumerate(items_raw)}
    G = len(cats)

    train, train_ts, valid, test, valid_ts, test_ts = [], [], {}, {}, {}, {}
    spans, lens, kept = [], [], []
    for u_raw in users_raw:
        xs = sorted(pos[u_raw])
        n = len(xs)
        nt = max(1, int(round(0.1 * n)))
        nv = max(1, int(round(0.1 * n)))
        tr, va, te = xs[:n - nt - nv], xs[n - nt - nv:n - nt], xs[n - nt:]
        if (tr[-1][0] - tr[0][0]) / DAY < min_span_days:   # training part only
            continue
        u = len(kept) + 1
        kept.append(u_raw)
        train += [(u, imap[i]) for _, _, i in tr]
        train_ts += [int(t) for t, _, _ in tr]
        valid[u] = [imap[i] for _, _, i in va]
        test[u] = [imap[i] for _, _, i in te]
        valid_ts[u] = [int(t) for t, _, _ in va]
        test_ts[u] = [int(t) for t, _, _ in te]
        spans.append((tr[-1][0] - tr[0][0]) / DAY)
        lens.append(len(tr))
    spans, lens = np.array(spans), np.array(lens)

    used = sorted({i for _, i in train} | {i for v in valid.values() for i in v} | {i for v in test.values() for i in v})
    inv = {v: k for k, v in imap.items()}
    item_category = {i: item_groups[inv[i]] for i in used}
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
    user_feature = {u: [["U%d" % u], ["1"]] for u in range(1, len(kept) + 1)}

    out = os.path.join(REPO, "data", name)
    tmp = os.path.join(REPO, "data", name + "_temporal", "per_user")
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
    json.dump({"users": kept, "items": [inv[i] for i in used]}, open(os.path.join(out, "id_map.json"), "w"))

    q = lambda x, p: float(np.percentile(x, p))
    man = {
        "source": source, "users_with_min_pos": n_eligible, "min_span_days": min_span_days,
        "users": len(kept), "items": len(item_feature), "genres": list(cats), "num_groups": G,
        "train": len(train), "valid": sum(map(len, valid.values())), "test": sum(map(len, test.values())),
        "train_span_days": {"<1d": float((spans < 1).mean()), "1-30d": float(((spans >= 1) & (spans < 30)).mean()),
                            ">30d": float((spans >= 30).mean()), ">365d": float((spans >= 365).mean()),
                            "median": q(spans, 50), "p75": q(spans, 75), "p90": q(spans, 90)},
        "train_len": {"median": q(lens, 50), "p90": q(lens, 90)},
        "prior": dict(zip(cats, np.round(prior, 4).tolist())),
        "recipe": dict(recipe, min_pos=MIN_POS,
                       split="per-user temporal: last round(10%) test, preceding round(10%) valid (each >= 1)",
                       prior="item catalogue group distribution, round(1/|groups|,4) mass",
                       user_features="user ID only (num_user_features = 1)"),
    }
    if extra:
        man.update(extra)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=2)
    print(json.dumps({k: man[k] for k in ("users_with_min_pos", "users", "items", "num_groups", "train",
                                          "valid", "test", "train_span_days", "train_len")}, indent=1))
    return man
