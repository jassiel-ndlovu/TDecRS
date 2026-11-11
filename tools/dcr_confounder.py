"""The block-indexed DCR confounder d_u^(tau) for ONE user, as a reusable function.

This is the per-user loop of tools/build_dcr_priors.py, lifted out unchanged so
that other consumers (code/seqrec/ for Tier 3) compute the confounder with the
same arithmetic instead of a re-implementation that can drift. Equivalence is
checked, not assumed: code/seqrec/smoke_test.py rebuilds the per_user training
confounder with this function and compares it with the arrays build_dcr_priors.py
wrote to data/ml_1m_dcr/ (exact float32 equality).

    personal   d~_u^(tau)  proportional to  sum_{s<tau} gamma^(tau-1-s) * sum_{i in B_s} g(i)
    shrinkage  rho_u^(tau) = n_u^(<tau) / (n_u^(<tau) + kappa)
    combined   d_u^(tau)   = (1-rho) * dbar_pop(tau) + rho * d~_u^(tau)

Every row of block tau gets the same d, computed from blocks s < tau only, so
interactions that share a timestamp never see each other's genres.
"""
import itertools

import numpy as np


def genre_matrix(item_category, num_groups):
    """g(i): mass 1/|genres(i)| on each genre, indexed by raw item id."""
    G = np.zeros((max(item_category) + 1, num_groups))
    for i, cats in item_category.items():
        for c in cats:
            G[i, c] = 1.0 / len(cats)
    return G


def static_pop(pop):
    return lambda ts: pop


def block_confounders(times, items, G, pop_at, gamma, kappa):
    """One user's confounder per row.

    times, items : sequences of equal length, already in chronological order
                   (ties = same timestamp block; rows of a block must be adjacent)
    G            : genre_matrix(...)
    pop_at       : timestamp -> population prior (np.float64 vector)
    Returns
        rows  (n, NG) float32, row r = d_u^(tau(r)) from blocks strictly before r's block
        final (NG,)   float32, the state after the whole history (what
                      build_dcr_priors.py stores as user_confounder)
    """
    NG = G.shape[1]
    out = np.zeros((len(items), NG), dtype=np.float32)
    acc = np.zeros(NG)      # sum_{s<tau} gamma^(tau-1-s) * blockvec(s)
    seen = 0                # n_u^(<tau)
    blocks = []
    pos = 0
    for t, grp in itertools.groupby(zip(times, items), key=lambda r: r[0]):
        blk = [i for _t, i in grp]
        blocks.append(t)
        # --- confounder for THIS block, from strictly earlier blocks only ---
        if seen == 0 or acc.sum() <= 0:
            personal = None
        else:
            personal = acc / acc.sum()
        rho = 0.0 if (seen == 0 or np.isinf(kappa)) else seen / (seen + kappa)

        base = pop_at(t)
        d = base if personal is None else (1.0 - rho) * base + rho * personal
        s = d.sum()
        d = (d / s) if s > 0 else base
        out[pos:pos + len(blk)] = d
        pos += len(blk)

        # --- advance the accumulator past this block ---
        acc = gamma * acc + G[blk].sum(0)
        seen += len(blk)

    last_ts = blocks[-1] if blocks else None
    if np.isinf(kappa):
        d = pop_at(last_ts)
    elif seen and acc.sum() > 0:
        personal = acc / acc.sum()
        rho = seen / (seen + kappa)
        d = (1.0 - rho) * pop_at(last_ts) + rho * personal
        d = d / d.sum()
    else:
        d = pop_at(last_ts)
    return out, d.astype(np.float32)
