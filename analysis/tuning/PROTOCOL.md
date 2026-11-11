# Hyperparameter sensitivity: frozen protocol

Frozen on Wed 4 Nov 2026, before any tuning run was launched. Requested by the student on 4 Nov. Nothing below changes after launch.

## Question
Every earlier run used one configuration on all datasets: lr 0.05, batch 1024, dropout [0.3,0.3], lamda 0.1. These are the authors' published ML-1M settings. Do the report's conclusions survive when each model's hyperparameters are selected on validation data?

This is a **sensitivity analysis**. It does not replace the existing tables, the pre-registered tests, or their verdicts. It is a small validation-selected grid, so it cannot show that any model is at its optimum.

## Datasets
- **ML-1M:** the authors' tuned dataset, and our primary replication.
- **Goodreads:** the dataset our default settings were never fitted to, and where Q3 showed under-training.
- **KuaiRand and ML-10M are not tuned.** This is a stated limitation.

## Grid (one factor group at a time, 4 cells)
| cell | lr | dropout | lamda | origin |
|---|---|---|---|---|
| D | 0.05 | [0.3,0.3] | 0.1 | the authors' ML-1M settings, already used everywhere |
| L+ | 0.1 | [0.3,0.3] | 0.1 | faster learning, because Q3 showed under-training |
| L- | 0.02 | [0.3,0.3] | 0.1 | slower learning |
| R | 0.05 | [0.5,0.2] | 0 | the authors' published Amazon-Book regularisation |

Fixed: batch 1024, hidden factor 64, Adagrad, training alpha 0.2, the data, and each dataset's confounder tags. Epochs are 300 on both datasets.

## Model classes
Each class is tuned separately, on seed 1.

| class | ML-1M | Goodreads |
|---|---|---|
| FM | FM | FM |
| DecRS | DecFM (no tag) | DecFM (no tag) |
| FMconf | FMconf g0.3_k0_static | FMconf g0.94_k0_static |
| TDecRS | DecFM g0.3_k0_static | DecFM g0.94_k0_static |

Stretch, on Goodreads only, if time remains: windowed (g0.9_kinf_windowed) and pooled (g0.9_kinf_pooled) DecRS. They inherit the tuned DecRS cell, so this pair is matched but not separately tuned.

## Blend partner
- **ML-1M:** the authors' released FM scores (FM_A), as in every ML-1M table. It is already a tuned FM, and keeping it lets the D cells be reused.
- **Goodreads:** the seed-1 FM from the winning FM cell, at 300 epochs. FM is tuned first and its scores are generated before any Dec* cell starts. If FM's D cell wins, the existing `gr300/goodreads_FM_user_item_scores.npy` is used and the Q3 D cells are reused. If another cell wins, the three D cells (DecRS, FMconf, TDecRS) are re-run with the new partner.

## Selection rule
1. Use validation data only: the best logged `[Valid]` Recall@10 of each cell. This is the criterion `main.py` already uses to pick the best epoch. Test lines are never read for selection. The rule is implemented in `select_cell.py`.
2. FMconf is selected on its logged (raw) validation value.
3. A non-D cell replaces D only if its value exceeds D's by **at least 1% (relative)**. This avoids chasing seed-1 noise. If several qualify, the highest wins. Ties go to the higher learning rate.
4. Inference alpha is chosen offline per class, from {0.2, 0.5} (the authors' two published values), on validation R@10 at seed 1, and then applied to all seeds. FM⊕FM uses FM's winning cell and chooses its own alpha the same way.

D-cell validation baselines already logged:
- ML-1M: FM .0872, DecRS .0876, TDecRS .1107, FMconf .1175.
- Goodreads at 300 epochs: FM .0634, DecRS .0695, TDecRS .0713, FMconf .0641.

## Final runs
Each class's winning cell is trained on seeds 2 and 3. Seed 1 comes from the tuning run. A class whose D cell won reuses its existing checkpoints where the partner and epochs match. This holds on ML-1M, and on Goodreads if FM's D cell won; Goodreads seeds 2–3 at 300 epochs do not exist, so they still have to run. FM seeds run first, because FM⊕FM needs them.

FM⊕FM mapping (frozen):
- Goodreads: FM s2 slot + s1 partner at seed 1, s3 + s1 at seed 2, s2 + s1 at seed 3.
- ML-1M: FM s + FM_A, as now.

## Scoring (after all selections are fixed)
- Re-rank offline along the existing stage-1 path, and verify that test R@10 matches the training log.
- Use `analysis/audit/stats.py` `compare` (seed-aware CI, TOST at ±2%, Holm across the pairs below within dataset × metric).
- Primary metric R@20. R@10, N@20 and C_KL@20 are reported alongside.

Pre-stated comparisons:
- DecRS − FM
- TDecRS − FMconf (blend)
- DecRS − FM⊕FM
- TDecRS − DecRS
- FMconf (blend) − DecRS
- If the stretch runs complete: windowed − pooled, windowed − FMconf (blend)

**Robust** means the sign and the verdict class (better / equivalent / inconclusive / worse) match the same comparison at D on matched seeds (seeds 1–3). On Goodreads, D is reported at 200 epochs (the primary tables), and at 300 epochs on seed 1 from Q3. Any change is reported with both numbers. No conclusion is re-worded to hide it, and no threshold changes.

## Predictions (made before any result)
- HP1: TDecRS − FMconf (blend) stays non-significant on both datasets. This is the main negative result.
- HP2: DecRS − FM⊕FM is not significantly positive on either dataset.
- HP3: the ML-1M recency gain (TDecRS − DecRS, FMconf (blend) − DecRS) stays significant.
- HP4: ML-1M changes less than Goodreads, because the D settings came from ML-1M.

## Compute
Azure: 2 × NC4as_T4_v3 (one per dataset).

| stage | hpA (ML-1M) | hpB (Goodreads) |
|---|---|---|
| 1. FM cells | 3 runs | 3 runs |
| 2. DecRS, FMconf, TDecRS cells | 9 runs | 9, or 12 if FM's D cell loses |
| 3. seeds 2 and 3 of winners; all seed 2 before any seed 3 | ≤ 8 runs | 8 runs, FM first |
| stretch | Goodreads windowed/pooled, in the order W s1, pool s1, W s2, pool s2, W s3, pool s3 | none |

- Measured run times: about 56 min FM, about 70 min others, at 300 epochs.
- Total: at most about 48 VM-hours plus stretch, about USD 27–32.
- **Hard stop: Thu 5 Nov 19:30 SAST.** Any run unfinished then is discarded, the VMs are deleted, and the results are scored on the seeds that completed. That number of seeds is reported.

## Launch record (added after the freeze; the rules above are unchanged)
Launched on Wed 4 Nov 2026: hpA at 13:38 SAST, hpB at 13:43 SAST.

Implementation details:
1. `select_cell.py` is delivered to the VMs through blob `tools/select_cell.py`.
2. Run labels are `hp_<ds>_<class>_<cell>_s<seed>`: ds is ml1m or gr; class is FM, DecRS, FMconf, TDecRS, windowed or pooled; cell is D, Lp (L+), Lm (L-) or R.
3. Reuse:
   - If FM's D cell wins on Goodreads, FM s2 is `gr300_FM_s2` from results-grq1.
   - If FM and DecRS both win at D, windowed s1 and pooled s1 are the Q3 runs from results-grq1 and results-grq2.
4. VMs kill unfinished runs at 19:20 SAST, so that VMs and disks are deleted by 19:30. **Scoring uses only runs marked `done rc=0` in `status/hpA.txt` or `status/hpB.txt`.** A killed run's uploaded checkpoint is discarded.
5. Winners are published as blobs `hp/goodreads_winner_FM.txt` and `hp/goodreads_winner_DecRS.txt`. The partner is `hp/goodreads_FM_user_item_scores.npy`, only if FM's D cell loses.
6. If FM's D cell loses on Goodreads, the FMconf D cell is re-run as written, although `main.py` uses no partner for FMconf. That run is redundant. The selection uses the re-run value, as the job script already does, and the difference between the two D runs is reported as a measure of run-to-run noise.

Goodreads run times measure 72–82 min, not the 70 min estimated above, so the last stretch runs may be cut.

## Scoring record (added Thu 5 Nov, after selection and before any tuned comparison was computed; the rules above are unchanged)
Three points the protocol left open, fixed here before scoring:
1. **Windowed and pooled inference alpha.** They "inherit the tuned DecRS cell" and are "not separately tuned", so they also inherit DecRS's alpha. What each would choose alone is printed alongside.
2. **HP4 measure.** "Changes less" means two things, both reported: the mean absolute change in the effect (percentage points) over the five comparisons common to both datasets on R@20, and the number of those five whose verdict class changes. Goodreads is measured against its 200-epoch tables and, separately, against 300 epochs at seed 1. The 200-epoch reference also contains the change from 200 to 300 epochs.
3. **Scorer.** Lists come from the closed-form scorer of `eval_package/score_all.py`, asserted equal to `forward()` on 16 users per checkpoint (max difference under 1e-4), and checked against the training logs and the stored D-cell lists. See `rerank.py`.

The 300-epoch seed-1 reference is scored per user and without Holm, as in Q3. The 200-epoch and ML-1M D references use the same Holm family as the tuned results, on seeds 1-3.
