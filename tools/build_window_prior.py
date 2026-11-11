"""Per-time-window population prior P(d|t) for a review-style dataset.

The table that `tools/build_dcr_priors.py --pop windowed` reads, built for datasets made by
tools/review_ds_common.py (goodreads, amazon_mtv, ...). ML-1M has its own table, from
tools/build_temporal_ml1m.py's global split; that one is not touched here.

Same construction as build_temporal_ml1m.py's "consumed" table:
- The training interactions (data/<ds>_temporal/per_user/training_list.npy and their timestamps)
  are sorted by time and cut into --windows equal-count windows.
- Each window's row is the consumed genre mix: item i puts mass 1/|genres(i)| on each of its genres.
- The rows are normalised.
- Equal-count windows match analysis/favourability.py's population_shift
  (its 8-window TV is the screen's s).

OUTPUTS (data/<ds>_temporal/global/)
    prior_by_window_consumed.npy  (W, G) float64, rows sum to 1
    window_bounds.npy             (W, 2) int64, (first, last) training timestamp in each window

    python tools/build_window_prior.py --dataset goodreads --windows 8
    python tools/build_window_prior.py --dataset kuairand --days 7 --tz_hours 8
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, os.pardir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--days", type=int, default=0,
                    help="calendar windows of this many days instead of equal-count windows (0 = equal-count)")
    ap.add_argument("--min_share", type=float, default=0.01, help="with --days: merge smaller windows")
    ap.add_argument("--tz_hours", type=float, default=0.0, help="day boundaries in UTC+this (KuaiRand: 8)")
    args = ap.parse_args()
    if args.dataset == "ml_1m":
        sys.exit("ml_1m's window table comes from tools/build_temporal_ml1m.py; not rebuilt here")
    src = os.path.join(REPO, "data", args.dataset)
    tmp = os.path.join(REPO, "data", args.dataset + "_temporal")
    ic = np.load(os.path.join(src, "item_category.npy"), allow_pickle=True).item()
    NG = len(np.load(os.path.join(src, "confounder_prior.npy")))
    train = np.load(os.path.join(tmp, "per_user", "training_list.npy"), allow_pickle=True)
    ts = np.load(os.path.join(tmp, "per_user", "training_timestamps.npy"), allow_pickle=True).astype(np.int64)

    Gm = np.zeros((max(ic) + 1, NG))
    for i, cats in ic.items():
        for c in cats:
            Gm[i, c] = 1.0 / len(cats)

    order = np.argsort(ts, kind="stable")
    items = train[order, 1].astype(int)
    ts = ts[order]
    if args.days:
        off = int(args.tz_hours * 3600)
        day0 = (ts[0] + off) // 86400 * 86400 - off
        cuts = np.arange(day0, ts[-1] + 1, args.days * 86400)[1:]
        edges = np.concatenate([[0], np.searchsorted(ts, cuts), [len(ts)]]).astype(int)
        edges = edges[np.r_[True, np.diff(edges) > 0]]
        # merge windows holding < --min_share of the rows into a neighbour (stray timestamps, data gaps)
        while len(edges) > 2:
            sizes = np.diff(edges)
            k = int(np.argmin(sizes))
            if sizes[k] >= args.min_share * len(ts):
                break
            edges = np.delete(edges, k if k == len(sizes) - 1 else k + 1)
    else:
        edges = np.linspace(0, len(ts), args.windows + 1).astype(int)
    rows, bounds = [], []
    for k in range(len(edges) - 1):
        m = Gm[items[edges[k]:edges[k + 1]]].sum(0)
        rows.append(m / m.sum())
        bounds.append((ts[edges[k]], ts[edges[k + 1] - 1]))
    rows = np.array(rows)
    out = os.path.join(tmp, "global")
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "prior_by_window_consumed.npy"), rows)
    np.save(os.path.join(out, "window_bounds.npy"), np.array(bounds, dtype=np.int64))

    static = np.load(os.path.join(src, "confounder_prior.npy"))
    pooled = Gm[items].sum(0)
    pooled /= pooled.sum()
    import datetime as _dt
    for k, (r, (a, b)) in enumerate(zip(rows, bounds)):
        print("window %d  %s .. %s  TV vs pooled consumed %.3f" % (
            k, _dt.datetime.fromtimestamp(int(a), _dt.timezone.utc).date(), _dt.datetime.fromtimestamp(int(b), _dt.timezone.utc).date(),
            0.5 * np.abs(r - pooled).sum()))
    print("TV max (windows vs pooled consumed): %.3f" % (0.5 * np.abs(rows - pooled).sum(1)).max())
    print("TV pooled consumed vs static catalogue prior: %.3f" % (0.5 * np.abs(pooled - static).sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
