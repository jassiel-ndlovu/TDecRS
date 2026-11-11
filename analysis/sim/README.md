# Simulator: validation and grid

| step | script | output |
|---|---|---|
| Generator | `tools/simulate_drift.py` | the simulated worlds |
| Validation | `analysis/sim_validate.py` | `validation.json`, `validation.txt` |
| Grid evaluation | `analysis/sim_eval.py --seeds 1 2 3 --reuse` | `sim_eval.json`, `regime_map.png` |

The grid has 8 worlds. They cross exposure confounding (beta 0 or 8), individual drift (q low or high) and population drift (s 0 or high). Six models run in each world: FM, DecRS, windowed DecRS, FMconf, TDecRS with kappa = 0, and DecRS-R. Each model is trained for 300 epochs on 3 seeds, with 2,000 users per world.

## Design choices forced by validation (made before any GPU result)

| Choice | Why |
|---|---|
| Click link w = 2, alpha0 = 2 | At w = 1, the ~1.25 clicks per block carry too little genre signal for any world to reach the screen's q-high |
| qH = random walk (rho_x = 1, tau = 0.35); qL = (1, 0.05) | A mean-reverting AR(1) gives q ~ 0.06 but a recency gain under 1%, so the screen classes it q-low |
| theta raked back to pi(t) every block | Clipping at the simplex boundary moves the population mean, which creates spurious population drift in an s = 0 world |
| Per-block intercept c0(t) | Without it the click rate decays from 1.7 to 0.65 per block. The intercept is common to all items, so it adds no genre signal |
| Interventional test = uniform exposure over every unseen item | With 200 items per block, interventional R@20 averaged ~0.6 hits per user, too noisy for the ±2% margin |
| beta_high = 8 | The confounding gap is large here (below), and the data is not degenerate |

## Validation

**The confounding gap grows with beta.** These are the qL, s0 worlds. "follow" is a ranker that scores pop × exp(8 ⟨g, d_hist⟩).

| beta | follow-vs-pop R@20 adv., obs | adv., int | gap | TV(pop obs mix, pop int mix) | history bias L1 |
|---|---|---|---|---|---|
| 0 | +0.67 | +1.54 | -0.88 | 0.075 | 0.40 |
| 2 | +2.09 | +1.85 | +0.24 | 0.109 | 0.44 |
| 4 | +4.82 | +2.09 | +2.73 | 0.157 | 0.59 |
| 8 | +16.3 | +2.00 | +14.3 | 0.185 | 0.77 |
| 16 | +44.8 | +1.21 | +43.6 | 0.219 | 0.88 |

**Estimator recovery** (`favourability.py`, against the P8 thresholds):

- **q** is within 1.5x of the truth wherever it is non-negligible.
- **alpha0** is within ±25% in all 5 worlds.
- **rho_x** narrowly fails P8. The fitted value is 0.60–0.63 against a realised 0.69, outside the ±0.05 margin.

Two points matter for the theory:

1. **Population drift inflates the variogram's q about 9x** (true 0.002, fitted 0.02). Common drift reads as individual drift.
2. **Exposure feedback alone creates observed population shift.** At beta = 8, the s = 0 worlds show a TV of 0.12–0.20, above the 0.05 threshold. So when beta is high, the screen's s cannot be read as exogenous drift.

## Results (3 seeds)

Interventional R@20, % change vs FM, pooled over seeds. Every cell has a 95% CI that excludes 0, except FMconf raw in b0 qL s0 and in bH qL s0.

| world [designed regime] | DecRS | windowed | FMconf | FMconf blend | TDecRS0 | DecRS-R |
|---|---|---|---|---|---|---|
| b0 qL s0 [R0] | +6.1 | +5.4 | +1.0 | +5.9 | +5.1 | +5.1 |
| b0 qL sH [R0] | +3.1 | **+19.9** | +9.1 | +11.4 | +11.5 | +12.5 |
| b0 qH s0 [R3] | +6.2 | +5.1 | +15.5 | +15.8 | +17.4 | **+17.8** |
| b0 qH sH [R3*] | +5.4 | +31.6 | +32.9 | +30.4 | +26.9 | **+35.3** |
| bH qL s0 [R1] | +2.1 | +1.9 | -0.9 | **+2.2** | +1.8 | +1.8 |
| bH qL sH [R2] | +3.4 | +8.3 | **+18.8** | +13.3 | +7.2 | +12.8 |
| bH qH s0 [R4] | +2.7 | +3.2 | **+7.3** | +7.0 | +5.5 | +7.0 |
| bH qH sH [R4] | +2.0 | +12.8 | **+35.7** | +27.6 | +12.4 | +24.2 |

**Pre-registered T6 predictions** (confirmed / refuted / inconclusive). The tests use a ±2% relative TOST with a 90% CI; the seed bootstrap resamples seeds and users together.

| Prediction | 3 seeds | Notes |
|---|---|---|
| P1: TDecRS0 ~ FMconf blend | 2 / 3 / 11 | Refuted in b0 qH sH, bH qL sH and bH qH sH, with the same sign at every seed |
| P2: DecRS > TDecRS0 (beta-high, q-low) | 0 / 1 / 1 | In R2, TDecRS0 is 3.6% above DecRS |
| P3: regime ordering | 40 / 39 / 22 | R0 and R2 fail. All refutations have the same sign at every seed |
| P4: windowed ~ DecRS when max TV < 0.05 | 0 / 0 / 2 | Both inconclusive |

What the grid shows:

1. **A genre prior helps even with no confounding.** At beta = 0, every model with one gains about 5–6% over FM. The gain comes from the extra signal, not from deconfounding.
2. **Exposure confounding gives backdoor adjustment little.** DecRS gains +2.1% over FM in R1, right at the ±2% margin.
3. **Population drift separates the models.** Windowed is +19.9% in b0 qL sH. FMconf is best in both beta-high sH worlds.
4. **Under individual drift (qH),** the static-confounder feature models (FMconf, TDecRS0, DecRS-R) beat DecRS and windowed.
5. **The favourability screen misreads every beta-high world.** Exposure feedback pushes TV and q over the thresholds. All four beta = 0 worlds are classified correctly.

The seed-1-only outputs are kept as `sim_eval_seed1.json`, `regime_map_seed1.png` and `sim_eval_per_user_seed1.npz`. Per-seed scores are cached in `sim_eval_per_user_seed<S>.npz`.

A harmless overflow warning in `sigmoid` comes from `np.exp` of large negative logits. The result is 0, as intended.
