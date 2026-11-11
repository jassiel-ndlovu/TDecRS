# Evidence audit protocol: frozen 3 Nov 2026, before any audit number was computed

Purpose: re-derive every empirical claim in the findings report (including the why-null analysis) from the raw per-user outputs. Every comparison uses one statistical method. Each claim is then marked as confirmed, corrected or withdrawn. The audit adds no new models and runs no new training. The 300-epoch Goodreads check (Q3) is scored separately, under its own frozen rule (`q3/q3_score.py`).

## Inputs (raw per-user outputs only; no earlier summaries are reused)

| Dataset | Top-100 lists | Ground truth |
|---|---|---|
| ML-1M (5 seeds) | `eval_package/topk/ml_1m/` | `common.Data("ml_1m")` |
| ML-10M long-span (3 seeds, 9,855 users) | `eval_package/topk/ml_10m/`, cross-checked against `topk_ml10m/` | `common.Data("ml_10m")` |
| Goodreads (3 seeds, 20k users) | `topk_goodreads/` | `common.Data("goodreads")` |
| KuaiRand-Pure (3 seeds) | `topk_kuairand/` (top-100 plus random-impression scores) | the loaders in `kuairand_eval.py` |
| Simulator (8 worlds × 6 models × 3 seeds, 2,000 users) | `sim/sim_eval_per_user_seed{1,2,3}.npz` | (per-user metrics already computed) |

Metrics are recomputed with `eval_package/common.py` (`per_user`, `ckl`, `beyond`) and the AUC of `kuairand_eval.py`. The simulator arrays are used as stored.

## Pairing
- Users are paired within each seed.
- Seeds are paired by training seed: seed s of model A is compared with seed s of model B.
- **FM⊕FM is matched to the Dec\* blend partner:**
  - ML-1M: seed s is `ENS_FM_s{s}+FM_A`, with the authors' FM as partner, as every ML-1M Dec\* blend uses.
  - ML-10M, Goodreads and KuaiRand: the partner is FM seed 1. Seed s uses the KuaiRand frozen mapping {1: s2+s1, 2: s3+s1, 3: s2+s1}.
- The ML-1M "FM" row is the authors' released scores (`FM_A`), as in the report. Wherever the report says "DecRS vs FM", ML-1M is also given against the 5 trained FM seeds, and the two are labelled.

## Effect size
- Δ = 100·(Ā/B̄ − 1) on the grand means. Ā is the mean over seeds of the per-seed user means.
- For KuaiRand test (b), AUC is compared as an absolute difference ΔAUC, as pre-registered.

## Uncertainty: the two-component t interval (primary)
- Per-seed effects: d_s = 100·(Ā_s/B̄_s − 1).
- Seed component: se_seed² = var(d_s, ddof=1)/S. Users are the same in every seed, so this variance is training noise only.
- User component: se_user is the standard deviation of Δ over B = 2000 user bootstraps. The same resampled user indices are applied to every seed and to both models.
- se = √(se_user² + se_seed²). Degrees of freedom by Welch–Satterthwaite, df = se⁴·(S−1)/se_seed⁴. The user term has infinite df.
- 95% and 90% CIs are Δ ± t·se. The p-value is two-sided.
- With S = 1 the interval is user-only and is flagged "single seed".

## Verdicts (margin δ = 2% relative; 0.003 AUC on KuaiRand (b))

| Verdict | Rule |
|---|---|
| equivalent | the 90% CI lies inside (−δ, +δ) (TOST, α = .05) |
| better / worse | Holm-adjusted p < .05 and not equivalent |
| inconclusive | anything else |

Significant-but-equivalent differences are reported as "equivalent (significant, < δ)".

## Multiplicity (Holm–Bonferroni, α = .05)
- A family is one dataset × one report section's comparisons on the primary metric (R@20; AUC for KuaiRand (b)).
- Both the difference p-values and the TOST p-values are adjusted within the family.
- Secondary metrics (R@10, N@k, C_KL, coverage) are reported, but they are not confirmatory.

## Pre-registered tests
- The simulator P1–P4 (frozen 1 Nov) and KuaiRand K1–K8 with its primary rule (frozen 2 Nov 07:56) are first re-scored under **their own frozen rules**.
- The unified method is then applied as a sensitivity analysis.
- Any disagreement is reported. It never changes a registered verdict.

## Robustness flags, reported for every primary comparison
- Leave-one-seed-out: the sign of Δ with each seed dropped.
- Metric agreement: whether R@10 and N@20 have the same sign as R@20.

## Number check
- Every number printed in the report is matched to its recomputed value.
- A match means agreement within the printed precision (±1 in the last printed digit).
- Each mismatch is listed with its cause (a different pairing, seed set or partner, an arithmetic slip, or a stale value) and corrected in the report.
