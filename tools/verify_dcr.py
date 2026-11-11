"""Prove the DCR plumbing does not change DecRS.

Two checks:

  1. STATIC EQUIVALENCE -- scoring with an explicit per-row confounder, where
     every row carries the shipped static prior, must be bit-identical to the
     legacy path that uses self.confounder_prior. If this holds, every DecRS
     number already produced stays valid and DecRS is genuinely the kappa=inf
     corner of the DCR family rather than a separately-coded baseline.

  2. KAPPA=INF FILE -- the confounder file built with --kappa inf must equal the
     static prior on every row.

  3. DecRS-R NESTING -- DecFMR with its recency table zeroed and frozen must give
     DecFM's scores bit for bit, while being fed a real per-user DCR confounder
     (so its mediator provably ignores it). Checked on fresh weights and again
     after a few identical training steps, so the gradients nest too.

    DECRS_DEVICE=cpu python tools/verify_dcr.py --num_users 20
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
import model as model_lib  # noqa: E402
from device import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_users", type=int, default=20)
    ap.add_argument("--data_path", default=os.path.join(REPO, "data", "ml_1m"))
    args = ap.parse_args()

    dev = get_device()
    d = args.data_path
    user_feature, item_feature, num_features = data_utils.map_features(
        os.path.join(d, "user_feature_file.npy"), os.path.join(d, "item_feature_file.npy"))
    test_dict = data_utils.loadData(os.path.join(d, "testing_dict.npy"))
    prior = np.load(os.path.join(d, "confounder_prior.npy"))
    NG = len(prior)

    item_idx = list(item_feature.keys())
    all_if = torch.tensor(np.array([item_feature[i][0] for i in item_idx]), dtype=torch.long, device=dev)
    all_iv = torch.tensor(np.array([item_feature[i][1] for i in item_idx], dtype=np.float32), device=dev)
    n_items = all_if.size(0)

    print("check 1: static per-row confounder == legacy path")
    worst = 0.0
    for name, build in (("DecFM", lambda: model_lib.DecFM(num_features, NG, 64, 1, [0.3, 0.3], 5, prior)),
                        ("DecNFM", lambda: model_lib.DecNFM(num_features, NG, 64, "relu", [64], 1,
                                                            [0.3, 0.3], 5, prior)),
                        ("FMconf", lambda: model_lib.FMconf(num_features, NG, 64, 1, [0.3, 0.3], prior))):
        torch.manual_seed(1)
        net = build().to(dev)
        net.eval()
        static_row = torch.tensor(np.tile(prior.astype(np.float32), (n_items, 1)), device=dev)
        gap = 0.0
        with torch.no_grad():
            for userID in list(test_dict)[: args.num_users]:
                uf = torch.tensor(np.array(user_feature[userID][0]), dtype=torch.long, device=dev)
                uv = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32), device=dev)
                feats = torch.cat([uf.unsqueeze(0).expand(n_items, -1), all_if], 1)
                vals = torch.cat([uv.unsqueeze(0).expand(n_items, -1), all_iv], 1)
                legacy = net(feats, vals)
                perrow = net(feats, vals, static_row)
                gap = max(gap, float((legacy - perrow).abs().max()))
        print("  {:<7} max |legacy - per-row| = {:.3e}".format(name, gap))
        worst = max(worst, gap)

    print("\ncheck 2: the kappa=inf confounder file equals the static prior")
    f = os.path.join(REPO, "data", "ml_1m_dcr", "train_confounder_g0.9_kinf_static.npy")
    if os.path.exists(f):
        rows = np.load(f)
        gap2 = float(np.abs(rows - prior.astype(np.float32)).max())
        print("  rows {}  max |row - prior| = {:.3e}".format(len(rows), gap2))
    else:
        gap2 = 0.0
        print("  (not built; run: python tools/build_dcr_priors.py --kappa inf)")

    print("\ncheck 3: DecFMR with the recency table zeroed and frozen == DecFM")
    gap3 = check_decfmr_nesting(args, dev, num_features, prior, user_feature, test_dict, all_if, all_iv)

    ok = worst == 0.0 and gap2 < 1e-6 and gap3 == 0.0
    print("\n{}: DecRS is unchanged by the DCR plumbing".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def check_decfmr_nesting(args, dev, num_features, prior, user_feature, test_dict, all_if, all_iv,
                         tag="g0.3_k0_static", steps=5, batch=256):
    NG = len(prior)
    uc = os.path.join(REPO, "data", "ml_1m_dcr", "user_confounder_{}.npy".format(tag))
    users = list(test_dict)[: args.num_users]
    if os.path.exists(uc):
        user_conf = np.load(uc, allow_pickle=True).item()
    else:
        print("  ({} not built; using random per-user confounders)".format(os.path.basename(uc)))
        rng = np.random.RandomState(0)
        user_conf = {u: rng.dirichlet(np.ones(NG)).astype(np.float32) for u in users}

    torch.manual_seed(1)
    base = model_lib.DecFM(num_features, NG, 64, 1, [0.3, 0.3], 5, prior).to(dev)
    torch.manual_seed(1)
    net = model_lib.DecFMR(num_features, NG, 64, 1, [0.3, 0.3], 5, prior).to(dev)
    # DecFMR's constructor builds DecFM's parameters first, so the seeded
    # draws must already agree; this is asserted, not assumed
    for k, v in base.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), "init differs at " + k
    with torch.no_grad():
        net.recency_embed.weight.zero_()
        net.recency_bias.weight.zero_()
    net.recency_embed.weight.requires_grad_(False)
    net.recency_bias.weight.requires_grad_(False)

    n_items = all_if.size(0)

    def score_gap():
        base.eval(); net.eval()
        gap = 0.0
        with torch.no_grad():
            for userID in users:
                uf = torch.tensor(np.array(user_feature[userID][0]), dtype=torch.long, device=dev)
                uv = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32), device=dev)
                feats = torch.cat([uf.unsqueeze(0).expand(n_items, -1), all_if], 1)
                vals = torch.cat([uv.unsqueeze(0).expand(n_items, -1), all_iv], 1)
                conf = torch.tensor(np.asarray(user_conf[userID], dtype=np.float32),
                                    device=dev).unsqueeze(0).expand(n_items, -1)
                gap = max(gap, float((base(feats, vals) - net(feats, vals, conf)).abs().max()))
        return gap

    g_init = score_gap()
    print("  fresh weights:          max |DecFM - DecFMR| = {:.3e}".format(g_init))

    # identical training: same batches, same dropout draws, Adagrad as in main.py
    opt_b = torch.optim.Adagrad(base.parameters(), lr=0.05, initial_accumulator_value=1e-8)
    opt_n = torch.optim.Adagrad([p for p in net.parameters() if p.requires_grad],
                                lr=0.05, initial_accumulator_value=1e-8)
    crit = torch.nn.BCEWithLogitsLoss(reduction="sum")
    rng = np.random.RandomState(1)
    for _ in range(steps):
        rows_u = rng.choice(users, batch)
        rows_i = rng.randint(0, n_items, batch)
        uf = torch.tensor(np.array([user_feature[u][0] for u in rows_u]), dtype=torch.long, device=dev)
        uv = torch.tensor(np.array([user_feature[u][1] for u in rows_u], dtype=np.float32), device=dev)
        feats = torch.cat([uf, all_if[rows_i]], 1)
        vals = torch.cat([uv, all_iv[rows_i]], 1)
        conf = torch.tensor(np.array([user_conf[u] for u in rows_u], dtype=np.float32), device=dev)
        label = torch.tensor(rng.randint(0, 2, batch).astype(np.float32), device=dev)
        for m, opt, extra in ((base, opt_b, ()), (net, opt_n, (conf,))):
            m.train()
            torch.manual_seed(100 + _)     # same dropout mask for both
            m.zero_grad()
            loss = crit(m(feats, vals, *extra), label) + 0.1 * m.embeddings.weight.norm()
            loss.backward()
            opt.step()
    assert float(net.recency_embed.weight.abs().max()) == 0.0
    g_train = score_gap()
    print("  after {} train steps:    max |DecFM - DecFMR| = {:.3e}".format(steps, g_train))
    return max(g_init, g_train)


if __name__ == "__main__":
    sys.exit(main())
