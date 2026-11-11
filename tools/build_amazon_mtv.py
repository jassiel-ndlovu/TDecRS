"""Build Amazon Reviews 2023, Movies & TV, in the DecRS file format (E3 candidate).

Source (McAuley lab, academic use; cite Hou et al. 2024, arXiv:2403.03952):
  benchmark/5core/rating_only/Movies_and_TV.csv.gz   user_id, parent_asin, rating, timestamp (ms)
  raw/meta_categories/meta_Movies_and_TV.jsonl.gz    item metadata

  filter   : implicit positive = rating >= 4; the timestamp is the review time.
  groups   : the metadata has no clean genre field. `categories` mixes genres with formats,
             studios and moods, and `details["Genre"]` is a free-text string. Both are
             mapped to GENRES below by keyword rules; anything else is ignored. Items with
             no genre are dropped from the catalogue.
  catalogue: capped at --max_items items by number of positives (review_ds_common.cap_catalogue),
             so DecRS's full-ranking evaluation fits a T4.
  then the build_ml10m.py recipe (review_ds_common.split_and_write).

    python -I tools/build_amazon_mtv.py --raw data/raw/amazon_mtv --max_items 15000
"""
import argparse
import gzip
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import review_ds_common as rc  # noqa: E402

# (group, regex on a lower-cased category entry or Genre token)
GENRES = [
    ("Action & Adventure", r"\baction\b|\badventure\b"),
    ("Anime", r"\banime\b"),
    ("Comedy", r"\bcomedy\b"),
    ("Documentary", r"\bdocumentar"),
    ("Drama", r"\bdrama\b"),
    ("Faith & Spirituality", r"\bfaith\b|spiritual|religio"),
    ("Fitness", r"fitness|\bexercise\b|\byoga\b"),
    ("Holiday", r"christmas|\bholiday"),
    ("Horror", r"\bhorror\b"),
    ("International", r"international|foreign|art house"),
    ("Kids & Family", r"\bkids\b|\bfamily\b|\banimation\b"),
    ("Military & War", r"\bmilitary\b|\bwar\b"),
    ("Music & Performing Arts", r"\bmusic|\bconcert|\bclassical\b|performing arts|\bopera\b|\bballet\b"),
    ("Mystery & Suspense", r"\bmystery\b|\bsuspense\b|\bthriller"),
    ("Romance", r"\bromance\b"),
    ("Sci-Fi & Fantasy", r"science fiction|\bsci-fi\b|\bfantasy\b"),
    ("Sports", r"\bsports?\b"),
    ("Westerns", r"\bwesterns?\b"),
]
RX = [(g, re.compile(p)) for g, p in GENRES]
# category entries that match a rule but are not genres
NOT_GENRE = {"genre for featured categories", "kids & family deals", "sony pictures classics"}


def groups_of(meta):
    toks = [c for c in (meta.get("categories") or [])]
    g = (meta.get("details") or {}).get("Genre")
    if isinstance(g, str):
        toks += re.split(r",|/", g)
    found = set()
    for t in toks:
        t = t.strip().lower()
        if not t or t in NOT_GENRE:
            continue
        for k, (_, rx) in enumerate(RX):
            if rx.search(t):
                found.add(k)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--max_items", type=int, default=15000)
    ap.add_argument("--min_span_days", type=float, default=0.0)
    ap.add_argument("--name", default="amazon_mtv")
    a = ap.parse_args()

    item_groups = {}
    n_meta = 0
    with gzip.open(os.path.join(a.raw, "meta_Movies_and_TV.jsonl.gz"), "rt", encoding="utf8") as f:
        for line in f:
            d = json.loads(line)
            n_meta += 1
            gs = groups_of(d)
            if gs:
                item_groups[d["parent_asin"]] = gs

    pos, n_raw, n_pos_all = {}, 0, 0
    with gzip.open(os.path.join(a.raw, "Movies_and_TV_5core.csv.gz"), "rt") as f:
        next(f)
        for k, line in enumerate(f):
            u, i, r, t = line.rstrip("\n").split(",")
            n_raw += 1
            if float(r) >= 4:
                n_pos_all += 1
                if i in item_groups:
                    pos.setdefault(u, []).append((int(t) / 1000.0, k, i))
    n_pos_genre = sum(map(len, pos.values()))
    pos = rc.cap_catalogue(pos, a.max_items)
    rc.split_and_write(
        a.name, pos, item_groups, [g for g, _ in GENRES],
        source="Amazon Reviews 2023 Movies_and_TV, 5-core rating_only + raw metadata",
        recipe={"filter": "rating >= 4", "groups": "keyword rules on categories + details.Genre",
                "max_items": a.max_items},
        min_span_days=a.min_span_days,
        extra={"raw_ratings": n_raw, "raw_positives": n_pos_all, "positives_on_genred_items": n_pos_genre,
               "meta_items": n_meta, "meta_items_with_genre": len(item_groups)})


if __name__ == "__main__":
    main()
