"""Tier 3: train/evaluate SASRec or BERT4Rec with an optional DCR adaptor.

    python main.py --backbone sasrec --adaptor none
    python main.py --backbone sasrec --adaptor static
    python main.py --backbone sasrec --adaptor dcr --gamma 0.9 --kappa 10
    python main.py --backbone bert4rec --adaptor dcr --kappa inf   # == static, see smoke_test.py

Protocol: leave-one-out, full-catalogue ranking (no sampled negatives at eval),
already-seen items excluded from the ranking. Model selection on validation
Recall@10, as in code/ml-1m/main.py. With one held-out item per user Recall@K
equals HR@K; both are printed so the log carries the name readers expect.

Logs go to log/<backbone>_<adaptor>[_<tag>]_seed<seed>.txt in the ml-1m format
(Precision/Recall/NDCG/MRR lines, "Runing Epoch", "End. Best epoch"), so
analysis/parse_logs.py-style parsers work; a .json with the best-epoch metrics
is written next to it.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import device
from device import get_device
import models
import seqdata

TOPN = [10, 20]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--backbone", choices=["sasrec", "bert4rec"], default="sasrec")
    p.add_argument("--adaptor", choices=["none", "static", "dcr"], default="none",
                   help="none: plain backbone; static: DecRS prior (kappa=inf); dcr: d_u^(tau)")
    p.add_argument("--gamma", type=float, default=0.9, help="DCR recency decay per block")
    p.add_argument("--kappa", default="10", help="DCR shrinkage; 'inf' reproduces --adaptor static")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many evaluations without a better valid Recall@10; 0 = never")
    p.add_argument("--max_len", type=int, default=200)
    p.add_argument("--dim", type=int, default=None, help="default 50 (SASRec) / 64 (BERT4Rec)")
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=None, help="default 1 (SASRec) / 2 (BERT4Rec)")
    p.add_argument("--dropout", type=float, default=None, help="default 0.2 (SASRec) / 0.1 (BERT4Rec)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=None, help="default 0 (SASRec) / 0.01 (BERT4Rec)")
    p.add_argument("--batch_size", type=int, default=None, help="default 128 (SASRec) / 256 (BERT4Rec)")
    p.add_argument("--loss", choices=["bce", "ce"], default=None,
                   help="SASRec only: bce = paper (1 sampled negative per position), "
                        "ce = full softmax. BERT4Rec always uses full softmax.")
    p.add_argument("--mask_prob", type=float, default=0.2, help="BERT4Rec cloze rate (paper: 0.2 on ML-1M)")
    p.add_argument("--eval_batch", type=int, default=512)
    p.add_argument("--max_users", type=int, default=0, help="random user subset for smoke tests; 0 = all")
    p.add_argument("--log_dir", default="./log/", help="'' disables the log file")
    p.add_argument("--save_model", type=int, default=0)
    p.add_argument("--model_path", default="./models/")
    p.add_argument("--job", default="local", help="free-text job name for the log header")
    a = p.parse_args(argv)
    a.kappa = seqdata.parse_kappa(a.kappa)
    bert = a.backbone == "bert4rec"
    if a.weight_decay is None:
        a.weight_decay = 0.01 if bert else 0.0
    if a.batch_size is None:
        a.batch_size = 256 if bert else 128
    if a.loss is None:
        a.loss = "ce" if bert else "bce"
    return a


def run_tag(a):
    if a.adaptor == "dcr":
        return seqdata.conf_tag(a.gamma, a.kappa)
    return a.adaptor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Tee:
    def __init__(self, path):
        self.f, self.stdout = open(path, "w", encoding="utf-8"), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()

    def close(self):
        self.f.close()


#############################  DATA -> TENSORS  #############################

def left_pad(rows, L, fill, width=None):
    """rows: list of arrays (len <= L [, width]); returns (n, L[, width])."""
    shape = (len(rows), L) if width is None else (len(rows), L, width)
    out = np.empty(shape, dtype=np.asarray(fill).dtype if width is not None else np.int64)
    out[...] = fill
    for r, x in enumerate(rows):
        if len(x):
            out[r, L - len(x):] = x
    return out


def prepare(ds, a, dev):
    """Training inputs, eval inputs and exclusion masks, all on `dev`.

    conf tensors hold, at every position, the confounder of the item being
    PREDICTED there (the target), never of the input item. Padding gets the
    static prior so 'static' and 'dcr kappa=inf' see identical tensors.
    """
    L, N, bert = a.max_len, ds.num_items, a.backbone == "bert4rec"
    prior = ds.prior
    if a.adaptor == "dcr":
        conf = ds.conf
    else:   # static, and a placeholder for none (never read)
        conf = [np.broadcast_to(prior, (len(s), len(prior))) for s in ds.seqs]

    tr_in, tr_tgt, tr_cf = [], [], []
    ev = {"valid": ([], [], []), "test": ([], [], [])}
    for s, D in zip(ds.seqs, conf):
        n = len(s)
        tr = s[:n - 2]
        if bert:
            tr_in.append(tr[-L:])
            tr_cf.append(D[:n - 2][-L:])
        else:
            tr_in.append(tr[:-1][-L:])
            tr_tgt.append(tr[1:][-L:])
            tr_cf.append(D[1:n - 2][-L:])
        for split, cut in (("valid", n - 2), ("test", n - 1)):
            hist = s[:cut]
            inp = np.append(hist[-(L - 1):], ds.num_items + 1) if bert else hist[-L:]
            ev[split][0].append(inp)
            ev[split][1].append(s[cut])
            ev[split][2].append(D[cut])

    T = lambda x, dt=torch.long: torch.as_tensor(x, dtype=dt, device=dev)
    G = len(prior)
    data = {"train_in": T(left_pad(tr_in, L, 0)),
            "train_cf": T(left_pad(tr_cf, L, prior, G), torch.float32)}
    if not bert:
        data["train_tgt"] = T(left_pad(tr_tgt, L, 0))
    for split, (inp, tgt, cf) in ev.items():
        data[split + "_in"] = T(left_pad(inp, L, 0))
        data[split + "_tgt"] = T(np.array(tgt))
        data[split + "_cf"] = T(np.stack(cf), torch.float32)

    # exclusion masks: everything before the target (valid / test), and the
    # user's whole sequence (negative sampling must not draw a true positive)
    U = len(ds.seqs)
    for name, cut in (("valid_seen", 2), ("test_seen", 1), ("all_seen", 0)):
        m = torch.zeros(U, N + 1, dtype=torch.bool)
        for u, s in enumerate(ds.seqs):
            m[u, torch.as_tensor(s[:len(s) - cut])] = True
        m[:, 0] = True
        data[name] = m.to(dev)
    return data


#############################  TRAIN / EVAL  #############################

def sample_negatives(seen, users, shape, N):
    neg = torch.randint(1, N + 1, shape, device=seen.device)
    for _ in range(100):
        bad = seen[users.view(-1, *[1] * (len(shape) - 1)).expand(shape), neg]
        if not bad.any():
            break
        neg[bad] = torch.randint(1, N + 1, (int(bad.sum()),), device=seen.device)
    return neg


def train_epoch(model, data, a, opt, N):
    model.train()
    bert = a.backbone == "bert4rec"
    U = data["train_in"].size(0)
    perm = torch.randperm(U).to(data["train_in"].device)
    total, steps = 0.0, 0
    for b in range(0, U, a.batch_size):
        users = perm[b:b + a.batch_size]
        ids, cf = data["train_in"][users], data["train_cf"][users]
        if bert:
            real = ids > 0
            mask = (torch.rand(ids.shape, device=ids.device) < a.mask_prob) & real
            # every non-empty row gets at least one cloze target: its last item
            empty = ~mask.any(1) & real.any(1)
            mask[empty, -1] = True
            if not mask.any():
                continue
            labels = ids[mask]
            ids = ids.masked_fill(mask, model.backbone.mask_id)
            h = model.represent(ids, cf)[mask]
            loss = F.cross_entropy(model.backbone.logits(h), labels)
        else:
            tgt = data["train_tgt"][users]
            valid = tgt > 0
            if not valid.any():
                continue
            h = model.represent(ids, cf)
            if a.loss == "ce":
                loss = F.cross_entropy(model.backbone.logits(h[valid]), tgt[valid])
            else:
                neg = sample_negatives(data["all_seen"], users, tgt.shape, N)
                pos_l = model.backbone.logits(h, tgt.unsqueeze(-1)).squeeze(-1)
                neg_l = model.backbone.logits(h, neg.unsqueeze(-1)).squeeze(-1)
                loss = (F.binary_cross_entropy_with_logits(pos_l[valid], torch.ones_like(pos_l[valid]))
                        + F.binary_cross_entropy_with_logits(neg_l[valid], torch.zeros_like(neg_l[valid])))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, data, split, a):
    """Full-catalogue rank of the single held-out item. Returns
    (precision, recall, ndcg, mrr, hr), each a list over TOPN, rounded like ml-1m."""
    model.eval()
    inp, tgt, cf = data[split + "_in"], data[split + "_tgt"], data[split + "_cf"]
    seen = data[split + "_seen"]
    ranks = []
    for b in range(0, inp.size(0), a.eval_batch):
        sl = slice(b, b + a.eval_batch)
        h = model.backbone.encode(inp[sl])[:, -1]
        h = model.adaptor(h, cf[sl])
        scores = model.backbone.logits(h)
        t = tgt[sl]
        ts = scores.gather(1, t.unsqueeze(1))
        scores = scores.masked_fill(seen[sl], float("-inf"))
        ranks.append((scores > ts).sum(1))
    r = torch.cat(ranks).double()
    out = {"precision": [], "recall": [], "ndcg": [], "mrr": [], "hr": []}
    for k in TOPN:
        hit = (r < k).double()
        out["precision"].append(round((hit / k).mean().item(), 4))
        out["recall"].append(round(hit.mean().item(), 4))
        out["hr"].append(round(hit.mean().item(), 4))
        out["ndcg"].append(round((hit / torch.log2(r + 2)).mean().item(), 4))
        out["mrr"].append(round((hit / (r + 1)).mean().item(), 4))
    return out


def fmt(res):
    j = lambda xs: "-".join(str(x) for x in xs)
    return "Precision: {} Recall: {} NDCG: {} MRR: {} HR: {}".format(
        j(res["precision"]), j(res["recall"]), j(res["ndcg"]), j(res["mrr"]), j(res["hr"]))


#############################  ENTRY  #############################

def run(a, ds=None):
    """Train + evaluate. Returns a dict with the best-epoch results and the
    final model. `ds` may be passed in to reuse a loaded SeqData (smoke tests);
    it must have been built with this run's gamma/kappa when adaptor == 'dcr'."""
    tag = run_tag(a)
    tee = None
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)
        stem = os.path.join(a.log_dir, "{}_{}_seed{}".format(a.backbone, tag, a.seed))
        tee = Tee(stem + ".txt")
        sys.stdout = tee
    try:
        return _run(a, ds, tag, stem if a.log_dir else None)
    finally:
        if tee is not None:
            sys.stdout = tee.stdout
            tee.close()


def _run(a, ds, tag, stem):
    print("args:", a)
    print("# job: {}  dcr_tag: {}  seed: {}".format(a.job, tag, a.seed))
    print(device.describe(), flush=True)
    set_seed(a.seed)
    dev = get_device()

    t0 = time.time()
    if ds is None:
        ds = seqdata.SeqData(a.gamma, a.kappa, with_conf=(a.adaptor == "dcr"), max_users=a.max_users)
    elif a.adaptor == "dcr":
        assert (ds.gamma, ds.kappa) == (a.gamma, a.kappa) and ds.conf is not None
    print("number of users: {}".format(len(ds)))
    print("number of items: {}".format(ds.num_items))
    data = prepare(ds, a, dev)
    print("data ready. costs " + time.strftime("%H: %M: %S", time.gmtime(time.time() - t0)), flush=True)

    set_seed(a.seed)    # model init independent of how much RNG data loading used
    model = models.build(a.backbone, ds.num_items, ds.num_groups, a.adaptor, a.max_len,
                         a.dim, a.num_blocks, a.num_heads, a.dropout).to(dev)
    n_adapt = sum(p.numel() for p in model.adaptor.parameters())
    print("parameters: backbone {}  adaptor {}".format(
        sum(p.numel() for p in model.backbone.parameters()), n_adapt), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=a.weight_decay)

    best = {"epoch": -1, "valid": None, "test": None}
    best_recall, bad = -1.0, 0
    for epoch in range(a.epochs):
        t0 = time.time()
        loss = train_epoch(model, data, a, opt, ds.num_items)
        if epoch % a.eval_every == 0 or epoch == a.epochs - 1:
            valid, test = evaluate(model, data, "valid", a), evaluate(model, data, "test", a)
            print("---" * 18)
            print("Runing Epoch {:03d} ".format(epoch) + "costs " +
                  time.strftime("%H: %M: %S", time.gmtime(time.time() - t0)))
            print("[Train]: loss: {:.4f}".format(loss))
            print("[Valid]: " + fmt(valid))
            print("[Test]: " + fmt(test), flush=True)
            if valid["recall"][0] > best_recall:
                best_recall, bad = valid["recall"][0], 0
                best = {"epoch": epoch, "valid": valid, "test": test}
                print("------------Best model------------")
                if a.save_model:
                    os.makedirs(a.model_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(
                        a.model_path, "{}_{}_seed{}.pth".format(a.backbone, tag, a.seed)))
            else:
                bad += 1
                if a.patience and bad >= a.patience:
                    print("early stop: {} evaluations without improvement".format(bad))
                    break

    print("End. Best epoch {:03d}".format(best["epoch"]))
    print("[Valid]: " + fmt(best["valid"]))
    print("[Test]: " + fmt(best["test"]), flush=True)
    if stem:
        rec = {k: (v if not isinstance(v, float) or np.isfinite(v) else str(v)) for k, v in vars(a).items()}
        with open(stem + ".json", "w") as f:
            json.dump({"args": rec, "tag": tag, "adaptor_params": n_adapt, **best}, f, indent=2)
    best["model"] = model
    return best


if __name__ == "__main__":
    run(parse_args())
