"""Per-user chronological sequences from timestamped ML-1M, with a causal
per-position DCR confounder.

SOURCE
    data/ml_1m_temporal/per_user/ -- the DecRS positives (rating >= 4) with
    timestamps, built by tools/build_temporal_ml1m.py. The authors' train/valid/
    test pieces are concatenated back into one history per user and re-split
    leave-one-out, so this is a different protocol from Tier 1 (numbers are not
    comparable across tiers, only across adaptors on the same backbone).

ORDER
    Stable sort by timestamp over (train rows in file order, then valid, then
    test). Items sharing a timestamp form a block whose internal order is
    arbitrary but deterministic.

SPLIT (leave-one-out, full-catalogue ranking)
    train : seq[:-2]    valid target : seq[-2]    test target : seq[-1]
    Users with fewer than 3 positives are dropped.

CONFOUNDER
    D[u] is (len(seq), 18): row r is d_u^(tau(r)), computed by
    tools/dcr_confounder.block_confounders from blocks strictly before r's
    block. Whenever the model predicts the item at position r it is given D[u][r],
    so the confounder never sees the genres of the target or of anything that
    shares its timestamp, even though the input sequence may contain those items.

ITEM IDS
    Raw MovieLens ids are remapped to 1..num_items in sorted order; 0 is padding.

USAGE
    python seqdata.py                     # stats + writes data/ml_1m_seq/
    python seqdata.py --gamma 0.9 --kappa 10
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(REPO, "tools"))
import dcr_confounder  # noqa: E402

TEMP = os.path.join(REPO, "data", "ml_1m_temporal")
OUT = os.path.join(REPO, "data", "ml_1m_seq")


def parse_kappa(k):
    return float("inf") if str(k).lower() in ("inf", "infinity") else float(k)


def conf_tag(gamma, kappa):
    k = "inf" if np.isinf(kappa) else ("%g" % kappa)
    return "g{:g}_k{}_static".format(gamma, k)


def load_raw_histories(temporal_dir=TEMP):
    """{raw user id: (times int64[n], raw items int64[n])}, chronological."""
    pu = os.path.join(temporal_dir, "per_user")
    train = np.load(os.path.join(pu, "training_list.npy"), allow_pickle=True)
    stamps = np.load(os.path.join(pu, "training_timestamps.npy"), allow_pickle=True)
    rows = {}
    for (u, i), t in zip(train, stamps):
        rows.setdefault(int(u), []).append((int(t), int(i)))
    for name in ("validation", "testing"):
        d = np.load(os.path.join(pu, name + "_dict.npy"), allow_pickle=True).item()
        ts = np.load(os.path.join(pu, name + "_timestamps.npy"), allow_pickle=True).item()
        for u, items in d.items():
            rows.setdefault(int(u), []).extend((int(t), int(i)) for t, i in zip(ts[u], items))
    out = {}
    for u, r in rows.items():
        r.sort(key=lambda x: x[0])     # stable: ties keep file order
        out[u] = (np.array([t for t, _ in r], dtype=np.int64),
                  np.array([i for _, i in r], dtype=np.int64))
    return out


class SeqData:
    """Everything the trainer needs, in remapped ids.

    users      : list of raw user ids (index = dense user index)
    seqs       : list of int64 arrays of item ids in 1..num_items
    times      : list of int64 timestamp arrays aligned with seqs
    conf       : list of float32 (len, NG) arrays, or None if not requested
    prior      : float32 (NG,) static catalogue prior (DecRS's dbar)
    num_items  : catalogue size (ids 1..num_items)
    """

    def __init__(self, gamma=0.9, kappa=10.0, with_conf=True, max_users=0,
                 subset_seed=0, temporal_dir=TEMP, min_len=3):
        shared = os.path.join(temporal_dir, "shared")
        item_category = np.load(os.path.join(shared, "item_category.npy"), allow_pickle=True).item()
        self.num_groups = len(np.load(os.path.join(shared, "category_list.npy"), allow_pickle=True))
        pop = np.load(os.path.join(shared, "confounder_prior.npy")).astype(np.float64)
        self.prior = pop.astype(np.float32)
        self.gamma, self.kappa = gamma, kappa

        hist = load_raw_histories(temporal_dir)
        users = sorted(u for u, (_t, it) in hist.items() if len(it) >= min_len)
        # The item vocabulary is fixed from ALL users so a smoke subset ranks the
        # same catalogue as a full run.
        raw_items = sorted({int(i) for u in users for i in hist[u][1]})
        if max_users and max_users < len(users):
            rng = np.random.RandomState(subset_seed)
            users = sorted(rng.choice(users, max_users, replace=False).tolist())
        self.item_raw = np.array([0] + raw_items, dtype=np.int64)   # dense -> raw
        remap = np.zeros(max(raw_items) + 1, dtype=np.int64)
        remap[raw_items] = np.arange(1, len(raw_items) + 1)
        self.num_items = len(raw_items)

        G = dcr_confounder.genre_matrix(item_category, self.num_groups)
        pop_at = dcr_confounder.static_pop(pop)
        self.users, self.seqs, self.times, self.conf = users, [], [], [] if with_conf else None
        for u in users:
            t, it = hist[u]
            self.seqs.append(remap[it])
            self.times.append(t)
            if with_conf:
                rows, _final = dcr_confounder.block_confounders(t, it, G, pop_at, gamma, kappa)
                self.conf.append(rows)

    def __len__(self):
        return len(self.users)

    def stats(self):
        lens = np.array([len(s) for s in self.seqs])
        blocks = np.array([len(np.unique(t)) for t in self.times])
        tie_last = np.mean([t[-1] == t[-2] for t in self.times])
        return {
            "users": len(self), "items": self.num_items, "interactions": int(lens.sum()),
            "seq_len_median": float(np.median(lens)), "seq_len_p90": float(np.percentile(lens, 90)),
            "seq_len_max": int(lens.max()), "blocks_median": float(np.median(blocks)),
            "test_target_shares_timestamp_with_valid_target": float(tie_last),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--kappa", default="10")
    args = ap.parse_args()
    kappa = parse_kappa(args.kappa)
    ds = SeqData(args.gamma, kappa)
    st = ds.stats()
    st["confounder_tag"] = conf_tag(args.gamma, kappa)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "sequences.npy"),
            {u: (ds.item_raw[s], t) for u, s, t in zip(ds.users, ds.seqs, ds.times)}, allow_pickle=True)
    np.save(os.path.join(OUT, "confounder_{}.npy".format(st["confounder_tag"])),
            dict(zip(ds.users, ds.conf)), allow_pickle=True)
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(st, f, indent=2)
    print(json.dumps(st, indent=2))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
