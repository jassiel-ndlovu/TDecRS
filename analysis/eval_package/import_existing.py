"""Copy the DCR and FMconf top-100 lists the earlier scorers already produced into the package.

analysis/rescore.py (ML-1M) and analysis/ml10m_eval.py (ML-10M) re-scored these
checkpoints along the same ranking path score_all.py uses, so re-running their
forwards would only repeat work. This copies them under the package's names
(seed 1 of ML-10M gets an explicit _s1). Lists still missing afterwards are built by
score_all.py --stage C. score_all.py rebuilds DecRS itself (it needs the scores for
E2d); compare_existing() checks those rebuilt lists against the originals.

    python analysis/eval_package/import_existing.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS, DS, TOPK_DIR  # noqa: E402

import numpy as np  # noqa: E402

SRC = {"ml_1m": os.path.join(ANALYSIS, "topk"), "ml_10m": os.path.join(ANALYSIS, "topk_ml10m")}
CELLS = ("DCR_g0.3", "FMconf_blend", "FMconf_raw")


def src_name(ds, cell, s):
    return "%s.npz" % cell if (ds == "ml_10m" and s == 1) else "%s_s%d.npz" % (cell, s)


def main():
    for ds in DS:
        dst = os.path.join(TOPK_DIR, ds)
        os.makedirs(dst, exist_ok=True)
        for cell in CELLS:
            for s in DS[ds]["seeds"]:
                f = os.path.join(SRC[ds], src_name(ds, cell, s))
                g = os.path.join(dst, "%s_s%d.npz" % (cell, s))
                if os.path.exists(g):
                    continue
                if not os.path.exists(f):
                    print("missing (score_all.py --stage C builds it):", ds, cell, s)
                    continue
                z = np.load(f)
                np.savez_compressed(g, users=z["users"], top=z["top"])
                print("imported", ds, os.path.basename(g))
    compare_existing()


def compare_existing():
    """Rebuilt lists vs the earlier scorers' lists: share of users with identical top-20."""
    for ds in DS:
        for cell in ("DecRS", "FM"):
            for s in DS[ds]["seeds"]:
                if ds == "ml_1m" and cell == "FM":
                    continue
                a = os.path.join(SRC[ds], src_name(ds, cell, s))
                b = os.path.join(TOPK_DIR, ds, "%s_s%d.npz" % (cell, s))
                if os.path.exists(a) and os.path.exists(b):
                    x, y = np.load(a), np.load(b)
                    assert (x["users"] == y["users"]).all()
                    same = (x["top"][:, :20] == y["top"][:, :20]).all(1).mean()
                    print("check %s %s_s%d: identical top-20 for %.2f%% of users" % (ds, cell, s, 100 * same))


if __name__ == "__main__":
    main()
