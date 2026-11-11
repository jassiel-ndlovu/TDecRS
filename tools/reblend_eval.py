"""Re-evaluate trained Dec* checkpoints under a different FM score file.

The Eq. 11 blend is applied at INFERENCE only -- the training loss never touches
FM_score -- so the trained weights are identical regardless of which FM scores
are used. Only the reported metric changes. That means the "self-contained"
DecRS numbers can be recomputed from existing checkpoints without retraining.

One caveat this cannot fix: the checkpoint itself was *selected* by validation
Recall@10 computed under the released-FM blend, so the epoch chosen may differ
slightly from what self-blend selection would have picked. Retraining closes
that gap; expect the effect to be small, since validation recall is flat across
the last ~100 epochs.

    python tools/reblend_eval.py --checkpoint <pth> --scores <npy> [--scores <npy> ...]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, os.pardir)
ML1M = os.path.join(REPO, "code", "ml-1m")
sys.path.insert(0, ML1M)

import data_utils          # noqa: E402
import evaluate            # noqa: E402
from device import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scores", action="append", required=True,
                    help="FM score file to blend against; repeat to compare several")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--data_path", default=os.path.join(REPO, "data", "ml_1m"))
    args = ap.parse_args()

    dev = get_device()
    d = args.data_path
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(d, "user_feature_file.npy"), os.path.join(d, "item_feature_file.npy"))
    train_dataset = data_utils.FMData(
        os.path.join(d, "training_list.npy"), user_feature, item_feature, "log_loss")
    valid_dict = data_utils.loadData(os.path.join(d, "validation_dict.npy"))
    test_dict = data_utils.loadData(os.path.join(d, "testing_dict.npy"))
    item_category = data_utils.loadData(os.path.join(d, "item_category.npy"))

    net = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    net.to(dev).eval()

    urf, uks, aif, aifv = evaluate.pre_ranking(
        user_feature, item_feature, item_category, 18,
        train_dataset.train_dict, valid_dict, test_dict)

    print("\ncheckpoint: {}".format(os.path.basename(args.checkpoint)))
    print("model     : {}   alpha={}".format(type(net).__name__, args.alpha))
    print("{:<48} {:>8} {:>8} {:>8} {:>8}".format("blended against", "R@10", "R@20", "N@10", "N@20"))
    for path in args.scores:
        fm = data_utils.loadData(path)
        t0 = time.time()
        with torch.no_grad():
            _v, t, _c = evaluate.Ranking(
                net, type(net).__name__, valid_dict, test_dict, train_dataset.train_dict,
                item_feature, urf, uks, aif, aifv, fm, 4096, [10, 20, 50, 100], 18, args.alpha)
        print("{:<48} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}   ({:.0f}s)".format(
            os.path.basename(path), t[1][0], t[1][1], t[2][0], t[2][1], time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
