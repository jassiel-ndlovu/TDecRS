"""CPU smoke test for Tier 3. Run from code/seqrec/:

    python smoke_test.py

Checks, in order (any failure exits non-zero):
  1. tools/dcr_confounder.py reproduces build_dcr_priors.py's saved arrays exactly
  2. the confounder is block-causal: same-timestamp rows never see each other
  3. kappa=inf gives the static prior, bit for bit, at every position
  4. the adaptor's mediator equals DecFM's FM bi-interaction over {h, d_g e_g}
  5. every backbone x adaptor trains 2 epochs on a user subset with finite metrics
  6. nesting: DCR with kappa=inf == static prior, identical metrics AND weights
"""
import os
import sys
import time

import numpy as np
import torch

import main
import models
import seqdata
from seqdata import dcr_confounder

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = seqdata.REPO
USERS, EPOCHS = 300, 2
LOG = os.path.join(HERE, "log", "smoke")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name, ("  -- " + detail) if detail else ""), flush=True)


def t1_matches_build_dcr_priors():
    """Recompute the Tier 1 per_user training confounder with the shared function."""
    src = os.path.join(REPO, "data", "ml_1m_temporal")
    ic = np.load(os.path.join(src, "shared", "item_category.npy"), allow_pickle=True).item()
    pop = np.load(os.path.join(src, "shared", "confounder_prior.npy")).astype(np.float64)
    train = np.load(os.path.join(src, "per_user", "training_list.npy"), allow_pickle=True)
    stamps = np.load(os.path.join(src, "per_user", "training_timestamps.npy"), allow_pickle=True)
    G = dcr_confounder.genre_matrix(ic, len(pop))
    per_user = {}
    for idx, ((u, i), t) in enumerate(zip(train, stamps)):
        per_user.setdefault(int(u), []).append((int(t), int(i), idx))
    for tag, gamma, kappa in (("g0.9_k10_static", 0.9, 10.0), ("g0.9_kinf_static", 0.9, float("inf")),
                              ("g1_k0_static", 1.0, 0.0)):
        ref_path = os.path.join(REPO, "data", "ml_1m_dcr", "train_confounder_{}.npy".format(tag))
        if not os.path.exists(ref_path):
            check("confounder == build_dcr_priors [{}]".format(tag), True, "reference missing, skipped")
            continue
        ref = np.load(ref_path)
        ref_user = np.load(ref_path.replace("train_", "user_"), allow_pickle=True).item()
        out = np.zeros_like(ref)
        user_ok = True
        for u, rows in per_user.items():
            rows.sort(key=lambda r: r[0])
            d, final = dcr_confounder.block_confounders([r[0] for r in rows], [r[1] for r in rows],
                                                        G, dcr_confounder.static_pop(pop), gamma, kappa)
            out[[r[2] for r in rows]] = d
            user_ok &= np.array_equal(final, ref_user[u])
        check("confounder == build_dcr_priors [{}]".format(tag),
              np.array_equal(out, ref) and user_ok,
              "max |diff| {:.2e}, per-user final states equal: {}".format(np.abs(out - ref).max(), user_ok))


def t2_block_causal():
    G = np.eye(4)[[0, 0, 1, 2, 3, 0]]            # item k -> genre
    pop = np.full(4, 0.25)
    times = [10, 10, 20, 20, 20, 30]
    items = [0, 2, 3, 4, 1, 5]
    d, _ = dcr_confounder.block_confounders(times, items, G, dcr_confounder.static_pop(pop), 0.9, 0.0)
    within = np.array_equal(d[2], d[3]) and np.array_equal(d[3], d[4]) and np.array_equal(d[0], d[1])
    first = np.array_equal(d[0], pop)                          # no history -> prior
    G2 = G.copy()
    G2[3] = np.eye(4)[1]                                       # change genre of an item in block 2
    d2, _ = dcr_confounder.block_confounders(times, items, G2, dcr_confounder.static_pop(pop), 0.9, 0.0)
    same_block_blind = np.array_equal(d[:5], d2[:5])
    later_sees = not np.array_equal(d[5], d2[5])
    check("block-causal confounder", within and first and same_block_blind and later_sees,
          "constant within block {}, first block = prior {}, blind to own block {}, next block sees it {}"
          .format(within, first, same_block_blind, later_sees))


def t3_kinf_is_prior(ds_inf):
    ok = all(np.array_equal(D, np.broadcast_to(ds_inf.prior, D.shape)) for D in ds_inf.conf)
    check("kappa=inf confounder == static prior at every position", ok)


def t4_adaptor_is_decfm():
    torch.manual_seed(0)
    G, d = 18, 16
    ad = models.DCRAdaptor("dcr", G, d, init_std=0.5)
    h = torch.randn(5, 7, d)
    conf = torch.softmax(torch.randn(5, 7, G), -1)
    # DecFM: stack {h, d_g e_g}, bi-interaction 0.5 * ((sum v)^2 - sum v^2)
    v = torch.cat([h.unsqueeze(-2), conf.unsqueeze(-1) * ad.confounder_embed.weight], -2)
    M_ref = 0.5 * (v.sum(-2).pow(2) - v.pow(2).sum(-2))
    M = ad.mediator(h, conf)
    params = {m: sum(p.numel() for p in models.DCRAdaptor(m, G, d).parameters())
              for m in ("none", "static", "dcr")}
    check("adaptor mediator == DecFM bi-interaction", torch.allclose(M, M_ref, atol=1e-5),
          "max |diff| {:.2e}".format((M - M_ref).abs().max().item()))
    check("adaptor parameters: none 0, static/dcr = G*d only",
          params == {"none": 0, "static": G * d, "dcr": G * d}, str(params))


def args_for(backbone, adaptor, kappa="10", extra=()):
    return main.parse_args(["--backbone", backbone, "--adaptor", adaptor, "--kappa", kappa,
                            "--epochs", str(EPOCHS), "--eval_every", "1", "--max_len", "50",
                            "--max_users", str(USERS), "--log_dir", LOG, "--patience", "0",
                            "--job", "smoke", *extra])


def finite(res):
    return all(np.isfinite(x) for split in ("valid", "test") for xs in res[split].values() for x in xs)


def main_():
    t0 = time.time()
    t1_matches_build_dcr_priors()
    t2_block_causal()
    ds10 = seqdata.SeqData(0.9, 10.0, max_users=USERS)
    dsinf = seqdata.SeqData(0.9, float("inf"), max_users=USERS)
    t3_kinf_is_prior(dsinf)
    t4_adaptor_is_decfm()

    runs = {}
    for bb in ("sasrec", "bert4rec"):
        for ad in ("none", "static", "dcr"):
            r = main.run(args_for(bb, ad), ds10)
            runs[bb, ad] = r
            check("train+eval {} / {}".format(bb, ad), finite(r) and r["epoch"] >= 0,
                  "best epoch {}, test R@10 {}".format(r["epoch"], r["test"]["recall"][0]))
    r = main.run(args_for("sasrec", "dcr", extra=("--loss", "ce")), ds10)
    check("train+eval sasrec / dcr / full-softmax loss", finite(r))

    for bb in ("sasrec", "bert4rec"):
        a = args_for(bb, "dcr", kappa="inf")
        r_inf, r_st = main.run(a, dsinf), runs[bb, "static"]
        same_metrics = r_inf["valid"] == r_st["valid"] and r_inf["test"] == r_st["test"]
        sd1, sd2 = r_inf["model"].state_dict(), r_st["model"].state_dict()
        same_w = sd1.keys() == sd2.keys() and all(torch.equal(sd1[k], sd2[k]) for k in sd1)
        check("nesting {}: dcr(kappa=inf) == static".format(bb), same_metrics and same_w,
              "metrics equal {}, weights equal {}".format(same_metrics, same_w))
        # and the confounder does matter: dcr(kappa=10) must differ from static
        sd3 = runs[bb, "dcr"]["model"].state_dict()
        check("non-degeneracy {}: dcr(kappa=10) != static".format(bb),
              not all(torch.equal(sd3[k], sd2[k]) for k in sd2))

    n_fail = sum(not ok for _n, ok in results)
    print("\n{} checks, {} failed, {:.0f}s".format(len(results), n_fail, time.time() - t0))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main_())
