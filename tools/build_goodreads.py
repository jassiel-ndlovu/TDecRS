"""Build Goodreads (UCSD Book Graph) in the DecRS file format (E3 candidate).

Source (McAuley lab, academic use only; cite Wan & McAuley RecSys'18, Wan et al. ACL'19):
  goodreads_reviews_dedup.json.gz, streamed through a filter that keeps
      user_id, book_id, rating, read_at, started_at, date_added  -> reviews_dedup_slim.tsv.gz
  goodreads_book_genres_initial.json.gz   10 coarse genres with shelf-vote counts per book

  filter   : implicit positive = rating >= 4 (rating 0 means "not rated" and is dropped).
  time     : read_at, the date the user finished the book (consumption time). date_added can
             be bulk shelving, so it is not used; rows without read_at, or with a read_at
             outside [2000-01-01, 2017-12-31], are dropped.
  groups   : a book's genres are those with at least --genre_share of its top genre's votes
             (multi-label, like ML). Books with no genre are dropped.
  catalogue: capped at --max_items books by number of positives (review_ds_common.cap_catalogue),
             then an optional seeded --n_users user subset, so the FM score file fits.
  then the build_ml10m.py recipe (review_ds_common.split_and_write).

    python -I tools/build_goodreads.py --raw data/raw/goodreads --max_items 15000 --name goodreads_full   # screen
    python -I tools/build_goodreads.py --raw data/raw/goodreads --max_items 10000 --n_users 20000       # training set
"""
import argparse
import datetime as dt
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_ds_common as rc  # noqa: E402

T_MIN = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc).timestamp()
T_MAX = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc).timestamp()
MONTHS = {m: k + 1 for k, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}


def parse_date(s):
    """'Sat Oct 07 00:00:00 -0700 2017' -> unix seconds (strptime is ~10x slower)."""
    p = s.split()
    if len(p) != 6:
        return None
    hh, mm, ss = p[3].split(":")
    z = p[4]
    off = (1 if z[0] == "+" else -1) * (int(z[1:3]) * 3600 + int(z[3:5]) * 60)
    try:
        t = dt.datetime(int(p[5]), MONTHS[p[1]], int(p[2]), int(hh), int(mm), int(ss), tzinfo=dt.timezone.utc)
    except (ValueError, KeyError):
        return None
    return t.timestamp() - off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--max_items", type=int, default=15000)
    ap.add_argument("--genre_share", type=float, default=0.2)
    ap.add_argument("--min_span_days", type=float, default=0.0)
    ap.add_argument("--n_users", type=int, default=0, help="seeded random user subset after the cap (0 = all)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--name", default="goodreads")
    a = ap.parse_args()

    votes = {}
    with gzip.open(os.path.join(a.raw, "goodreads_book_genres_initial.json.gz"), "rt", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            if d["genres"]:
                votes[d["book_id"]] = d["genres"]
    cats = sorted({g for v in votes.values() for g in v})
    cidx = {c: k for k, c in enumerate(cats)}
    item_groups = {}
    for b, v in votes.items():
        top = max(v.values())
        item_groups[b] = sorted(cidx[g] for g, c in v.items() if c >= a.genre_share * top)

    pos, n_raw, n_pos, n_no_read = {}, 0, 0, 0
    with gzip.open(os.path.join(a.raw, "reviews_dedup_slim.tsv.gz"), "rt", encoding="utf8") as f:
        next(f)
        for k, line in enumerate(f):
            u, b, r, read_at, _started, _added = line.rstrip("\n").split("\t")
            n_raw += 1
            if not r or int(r) < 4:
                continue
            n_pos += 1
            t = parse_date(read_at) if read_at else None
            if t is None or not (T_MIN <= t < T_MAX):
                n_no_read += 1
                continue
            if b in item_groups:
                pos.setdefault(u, []).append((t, k, b))
    n_pos_kept = sum(map(len, pos.values()))
    pos = rc.cap_catalogue(pos, a.max_items)
    n_capped = len(pos)
    pos = rc.sample_users(pos, a.n_users, a.seed)
    rc.split_and_write(
        a.name, pos, item_groups, cats,
        source="UCSD Book Graph: goodreads_reviews_dedup (read_at) + goodreads_book_genres_initial",
        recipe={"filter": "rating >= 4", "time": "read_at in [2000, 2018)",
                "groups": "genres with >= %g x top genre votes" % a.genre_share, "max_items": a.max_items},
        min_span_days=a.min_span_days,
        extra={"raw_reviews": n_raw, "raw_positives": n_pos, "positives_without_valid_read_at": n_no_read,
               "positives_kept_before_cap": n_pos_kept, "users_after_cap": n_capped,
               "n_users": a.n_users, "seed": a.seed})


if __name__ == "__main__":
    main()
