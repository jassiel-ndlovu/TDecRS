"""Regenerate a *_user_item_scores.npy file from a locally trained checkpoint.

WHY
---
`main.py` loads `./inference/FM_user_item_scores.npy` and Eq. 11 blends every
Dec* prediction against it. Shipped, that file holds the AUTHORS' pretrained FM
scores, so a "trained DecFM" number is really the trained model mixed with
someone else's artefact -- measured at ~34% of the final score for DecFM
(alpha=0.2) and ~55% for DecNFM (alpha=0.4).

Regenerating the file from your own FM checkpoint removes that dependency and
makes the comparison self-contained.

FORMAT
------
Matches the released file exactly: a pickled dict {userID: np.float32 array of
scores over item_feature.keys() order}, with training items masked by adding
-999.0 (verified against the released file, where training-item scores average
-996 against -5 elsewhere).

USAGE
-----
    python tools/gen_scores.py \
        --checkpoint results/ml-1m/models/FM_....pth \
        --out code/ml-1m/inference/FM_user_item_scores_selftrained.npy
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, os.pardir)
ML1M = os.path.join(REPO, "code", "ml-1m")
sys.path.insert(0, ML1M)

import data_utils          # noqa: E402
from device import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_path", default=os.path.join(REPO, "data", "ml_1m"))
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--compare", default=None,
                    help="optional released score file to sanity-check the output against")
    args = ap.parse_args()

    dev = get_device()
    d = args.data_path
    user_feature, item_feature, _ = data_utils.map_features(
        os.path.join(d, "user_feature_file.npy"), os.path.join(d, "item_feature_file.npy"))
    train_dataset = data_utils.FMData(
        os.path.join(d, "training_list.npy"), user_feature, item_feature, "log_loss")
    test_dict = data_utils.loadData(os.path.join(d, "testing_dict.npy"))
    train_dict = train_dataset.train_dict

    net = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    net.to(dev).eval()
    print("loaded {} from {}".format(type(net).__name__, os.path.basename(args.checkpoint)))

    item_idx = list(item_feature.keys())
    item_pos = {it: i for i, it in enumerate(item_idx)}
    n_items = len(item_idx)
    all_item_features = torch.tensor(
        np.array([item_feature[i][0] for i in item_idx]), dtype=torch.long, device=dev)
    all_item_feature_values = torch.tensor(
        np.array([item_feature[i][1] for i in item_idx], dtype=np.float32), device=dev)

    scores = {}
    with torch.no_grad():
        for n, userID in enumerate(test_dict):
            uf = torch.tensor(np.array(user_feature[userID][0]), dtype=torch.long, device=dev)
            uv = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32), device=dev)
            chunks, st = [], 0
            while st < n_items:
                ed = min(st + args.batch_size, n_items)
                k = ed - st
                chunks.append(net(
                    torch.cat([uf.unsqueeze(0).expand(k, -1), all_item_features[st:ed]], 1),
                    torch.cat([uv.unsqueeze(0).expand(k, -1), all_item_feature_values[st:ed]], 1)))
                st = ed
            pred = (chunks[0] if len(chunks) == 1 else torch.cat(chunks, 0)).cpu().numpy()

            # same masking convention as the released file
            mask = np.zeros(n_items, dtype=np.float32)
            for itemID in train_dict.get(userID, []):
                if itemID in item_pos:
                    mask[item_pos[itemID]] = -999.0
            scores[userID] = (pred + mask).astype(np.float32)
            if (n + 1) % 1000 == 0:
                print("  {} / {} users".format(n + 1, len(test_dict)), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, scores, allow_pickle=True)
    print("wrote {} ({} users x {} items)".format(args.out, len(scores), n_items))

    sample = np.concatenate([scores[u] for u in list(scores)[:200]])
    kept = sample[sample > -500]
    print("unmasked score range: {:.2f} .. {:.2f}  mean {:.2f}".format(
        kept.min(), kept.max(), kept.mean()))
    print("masked fraction: {:.2%}".format(np.mean(sample <= -500)))

    if args.compare:
        rel = np.load(args.compare, allow_pickle=True).item()
        common = [u for u in scores if u in rel][:200]
        agree = np.mean([
            len(set(np.argsort(-scores[u])[:100]) & set(np.argsort(-np.asarray(rel[u]))[:100])) / 100.0
            for u in common])
        print("top-100 overlap with {}: {:.1%} (low is expected -- different training run)".format(
            os.path.basename(args.compare), agree))
    return 0


if __name__ == "__main__":
    sys.exit(main())
