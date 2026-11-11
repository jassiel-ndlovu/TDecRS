"""Frozen selection rule of analysis/tuning/PROTOCOL.md: read [Valid] Recall@10 only, never [Test].

    python select_cell.py D=<log or value> L+=<log> L-=<log> R=<log>
prints the winning cell name on stdout. A cell's argument is a training log, or a logged value for a
D cell that was not re-run. A non-D cell must beat D by >= 1% (relative); the highest such cell wins,
and ties go to the higher learning rate (L+ before R before L-).
"""
import re
import sys

MIN_GAIN = 0.01
TIE_ORDER = ["L+", "R", "L-"]


def best_valid(path):
    best = -1.0
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith("[Valid]"):
                best = max(best, float(re.search(r"Recall: ([\d.]+)", line).group(1)))
    if best < 0:
        raise SystemExit("no [Valid] line in " + path)
    return best


def value(arg):
    try:
        return float(arg)
    except ValueError:
        return best_valid(arg)


def select(vals):
    d = vals["D"]
    ok = [c for c in TIE_ORDER if c in vals and vals[c] >= d * (1 + MIN_GAIN)]
    if not ok:
        return "D"
    return max(ok, key=lambda c: (vals[c], -TIE_ORDER.index(c)))


if __name__ == "__main__":
    vals = {k: value(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    print({k: round(v, 4) for k, v in vals.items()}, file=sys.stderr)
    print(select(vals))
