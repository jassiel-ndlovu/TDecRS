# Tier 3: the confounder as an adaptor on sequential recommenders

SASRec and BERT4Rec on timestamped ML-1M, each run with one of three adaptors:

| `--adaptor` | confounder `d` fed to the mediator | corresponds to |
|---|---|---|
| `none` | (no mediator) | plain backbone |
| `static` | the catalogue prior `dbar` at every position | DecRS, kappa = inf |
| `dcr` | `d_u^(tau)` per position (`--gamma`, `--kappa`) | TDecRS |

It asks whether the time-aware confounder still helps on a backbone that already models sequence order.

SASRec was run on 5 seeds with `none`, `static` and `dcr` at gamma 0.3, kappa 0 (`--loss ce`). BERT4Rec is implemented and passes the smoke test, but it was not run on a GPU.

## Run

From `code/seqrec/`:

```bash
python smoke_test.py
```

```bash
python main.py --backbone sasrec --adaptor dcr --gamma 0.3 --kappa 0 --loss ce --seed 1
```

```bash
python seqdata.py --gamma 0.9 --kappa 10
```

`seqdata.py` is optional. `main.py` builds the sequences in memory (about 10 s); the CLI only writes `data/ml_1m_seq/` so you can inspect them.

Main flags:

- `--seed`, `--backbone {sasrec,bert4rec}`, `--adaptor {none,static,dcr}`
- `--gamma`, `--kappa` (a number or `inf`)
- `--epochs`, `--eval_every`, `--patience`

The backbone defaults depend on `--backbone` (see below). `DECRS_DEVICE=cpu` forces CPU.

Logs go to `log/<backbone>_<tag>_seed<n>.txt`, plus a `.json` with the best-epoch metrics. They use the same format as `code/ml-1m`. `<tag>` is `none`, `static`, or a `build_dcr_priors.py` tag such as `g0.3_k0_static`.

## Files

| file | what |
|---|---|
| `seqdata.py` | per-user chronological sequences, leave-one-out split, per-position confounder |
| `../../tools/dcr_confounder.py` | the block-indexed `d_u^(tau)` for one user, the same arithmetic as `build_dcr_priors.py` |
| `models.py` | `SASRec`, `BERT4Rec`, `DCRAdaptor` |
| `main.py` | training, full-ranking evaluation, logging |
| `smoke_test.py` | the CPU checks below |
| `device.py` | copy of `code/ml-1m/device.py` |

## Protocol

- **Data.** `data/ml_1m_temporal/per_user/` holds the DecRS positives (rating ≥ 4). The authors' train/valid/test pieces are merged back into one history per user.
  - That gives 6,035 users (those with at least 3 positives), 3,533 items and 575,276 interactions.
  - The median sequence has 58 items and 33 timestamp blocks.
  - These are positives only. The usual SASRec ML-1M setup uses all ratings, so absolute numbers won't match the literature.
- **Order.** Interactions are stably sorted by timestamp. Order inside a timestamp block is file order.
- **Split.** Leave-one-out: train `seq[:-2]`, valid target `seq[-2]`, test target `seq[-1]`.
  - Ranking is over the full catalogue with no sampled negatives, and items already in the history are excluded.
  - Model selection uses validation Recall@10. With one target, Recall@K = HR@K.
- **Confounder and leakage.** `D[u][r]` is `d_u^(tau)` for the block holding position r, built only from strictly earlier blocks.
  - Each prediction gets the confounder of its target's position, so it never sees the genres of the target or of anything sharing the target's timestamp. This matters because 31.5% of test targets share a timestamp with the valid target.
  - The input sequence can still contain items from the target's own block, as in standard leave-one-out. That is the same for every adaptor, so it doesn't bias the comparison, but it inflates absolute numbers.
- **Population prior.** Only the static catalogue prior is used.

## The adaptor (`models.DCRAdaptor`)

DecFM computes `M(d,u)` as the FM bi-interaction over the user feature embeddings and `{d_g e_g}`, then adds `M` as one more feature of the outer FM. Here the user is the sequence vector `h` at the prediction position:

```
M = sum_{j<l} v_j * v_l  over v = {h, d_1 e_1, ..., d_G e_G}
  = h * S + 0.5 * (S^2 - Q),     S = sum_g d_g e_g,  Q = sum_g d_g^2 e_g^2
score(i) = <h + M, e_i>    (+ item bias for BERT4Rec)
```

**Parameters.** The only new parameters are the 18 × d confounder embedding: 900 for SASRec and 1,152 for BERT4Rec. The embedding is initialised N(0, 0.02²), so `M` starts near zero.

**Placement.** For SASRec the mediator goes after the final LayerNorm. For BERT4Rec it goes after the output head, just before the tied-embedding scoring.

## Smoke test (`smoke_test.py`, about 50 s on CPU)

1. `tools/dcr_confounder.py` reproduces the confounders that `build_dcr_priors.py` saved in `data/ml_1m_dcr/`, exactly (float32 equality).
2. The confounder is block-causal on a synthetic user. It is constant inside a block, and the first block gets the prior. Changing an item in block τ leaves block τ unchanged but changes block τ+1.
3. With kappa = inf, every position's confounder is the static prior.
4. The mediator equals DecFM's explicit bi-interaction to within 1.4e-6.
5. Every backbone × adaptor pair trains for 2 epochs on 300 users with finite metrics.
6. Nesting: `dcr` with kappa = inf matches `static` exactly in metrics and weights, and kappa = 10 does not. `static` and `dcr` share one code path, so this tests the plumbing; it is not an independent derivation.

## Backbone defaults

| | SASRec (Kang & McAuley 2018) | BERT4Rec (Sun et al. 2019) |
|---|---|---|
| d / blocks / heads | 50 / 2 / 1 | 64 / 2 / 2 |
| max_len / dropout | 200 / 0.2 | 200 / 0.1 |
| loss | BCE, 1 sampled negative per position; `--loss ce` = full softmax | cloze, mask 0.2, full softmax |
| optimiser | Adam lr 1e-3, β = (0.9, 0.98), batch 128 | AdamW lr 1e-3, wd 0.01, batch 256 |

Both are compact re-implementations, and no code is vendored. The SASRec block layout follows [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch).

BERT4Rec deviates from the paper in three ways:

- It uses lr 1e-3 with no decay, because the original 1e-4 needs a very long schedule (Petrov & Macdonald, RecSys 2022).
- There is no sliding-window augmentation.
- Masked tokens are always replaced by [MASK].

## Not done

- **Same-timestamp items in the input.** No attention mask stops a prediction from attending to items in the target's own timestamp block. Only the confounder is block-causal.
- **Not wired in:**
  - the windowed population prior;
  - DecRS's Eq. 11 inference blend (ranking uses `<h+M, e_i>` only);
  - calibration metrics;
  - an FMconf-style control (`d_u` as a plain input feature).
