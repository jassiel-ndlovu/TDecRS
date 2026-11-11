# TDecRS: time-aware deconfounded recommendation

This is a fork of [DecRS](https://github.com/WenjieWWJ/DecRS) (Wang et al., KDD 2021), extended for an Honours research project at the University of the Witwatersrand (2026).

DecRS removes bias amplification with a backdoor adjustment over a user's genre mix `d`. It uses one static prior `d̄` for everyone. **TDecRS** replaces `d̄` with a per-user, time-dependent confounder `d_u^(τ)`. This is a recency-weighted genre mix of the user's earlier interactions (decay `γ`), shrunk towards a population prior (strength `κ`). With `κ = ∞` the model is DecRS exactly. In the code and file names TDecRS is still called **DCR**, which was its working name.

The repo also has the controls and datasets used to test whether any gain comes from the deconfounding or simply from the recency information:

- **FMconf**: a plain FM with the same `d_u` as extra features and no backdoor adjustment.
- **DecRS-R** (`DecFMR`): DecRS's adjustment with recency added as a covariate.
- **FM⊕FM**: two FMs combined with DecRS's inference blend (Eq. 11).
- **Windowed DecRS**: a population prior that changes over time.
- **Datasets**: ML-1M, long-span ML-10M, Goodreads, KuaiRand-Pure (which has a random-exposure test set), KuaiRec, and a semi-synthetic simulator where the true drift and confounding are known.

### Findings in short

The thesis has the full numbers, confidence intervals and caveats.

- **DecRS reproduces on ML-1M.** It beats FM, but an FM⊕FM ensemble with the same blend accounts for most of that gain.
- **TDecRS's gain is recency, not deconfounding.** TDecRS beats DecRS on ML-1M, but FMconf matches it.
- **No time-aware gain on real data.** There is none on long-span ML-10M or on KuaiRand's random-exposure test. The pre-registered KuaiRand test was not met.
- **The simulator shows where time helps.** A windowed population prior helps under population drift, while backdoor adjustment adds little under exposure confounding.

## Layout

| path | what |
|---|---|
| `code/ml-1m/` | Upstream DecRS training code, extended with `--dcr_tag` (time-aware confounder), `FMconf` and `DecFMR`, seeding, and CPU support. Every dataset except Amazon-Book trains from here. |
| `code/amazon-book/` | Upstream Amazon-Book code. It is only ported to run on CPU or any GPU. |
| `code/seqrec/` | SASRec and BERT4Rec with the confounder as an adaptor (Tier 3). See its README. |
| `tools/` | Dataset builders, the confounder builder (`build_dcr_priors.py`), the simulator (`simulate_drift.py`), FM score generation, and equivalence checks. |
| `analysis/` | CPU re-scoring, statistics and table scripts, one per experiment (see [Experiment labels](#experiment-labels)). |
| `data/` | Only upstream's `ml_1m` and `amazon_book` are in git (see [Data and results](#data-and-results)). |

## Setup

The code was tested with Python 3.14, PyTorch 2.10 (CPU) and NumPy 2.4 on Windows. The GPU runs used Linux with NVIDIA T4 cards. Any recent PyTorch should work.

```bash
python -m venv venv
pip install -r requirements.txt
```

`matplotlib` is only needed for the two plotting scripts.

- **Device.** It is picked automatically. Set `DECRS_DEVICE=cpu`, `cuda` or `cuda:0` to force one. `--gpu` sets `CUDA_VISIBLE_DEVICES`, and its upstream default is `5`, so pass `--gpu 0` on a single-GPU machine.
- **Windows.** Use `--num_workers 0`.
- **Seed.** Set it with `DECRS_SEED`, which defaults to 1, the upstream seed.

## Data and results

Files too large for git are shared in the [TDecRS Google Drive folder](https://drive.google.com/drive/folders/1MpdNveqyNb24fi8mgl-DDPD6FQ29ZqT6?usp=sharing). It holds a selection of:

- `data/`: the built datasets.
- `results-*/`: training logs and checkpoints.
- `analysis/` outputs: top-K lists and `*_results.json`.
- `code/ml-1m/inference/*_FM_user_item_scores.npy`: FM score files.

Put them in the repo root, keeping the folder names. Not every file is on the Drive. Anything missing can be rebuilt:

- a dataset, with the builders below;
- an FM score file, with `tools/gen_scores.py`;
- results, by training with `main.py` and re-scoring with the `analysis/` scripts.

You also need the authors' released files, from the [upstream Google Drive](https://drive.google.com/drive/folders/1xww1VA-6Rc911vsAhucA7hi-2Np9zyWX?usp=sharing):

- `FM_user_item_scores.npy` for ML-1M, placed in `code/ml-1m/inference/`.
- Amazon-Book's `item_feature_file.npy`.

To rebuild a dataset from raw files instead, run the builder. Each script's docstring gives the raw-file layout.

```bash
python tools/build_temporal_ml1m.py --raw data/raw/ml-1m
python tools/build_dcr_priors.py --dataset ml_1m --gamma 0.3 --kappa 0
python tools/build_ml10m.py --raw data/raw/ml-10m/ml-10M100K --min_span_days 30
python tools/build_goodreads.py --raw data/raw/goodreads --max_items 10000 --n_users 20000
python tools/build_kuairand.py --raw data/raw/kuairand/KuaiRand-Pure/data
python tools/simulate_drift.py --grid
```

`build_dcr_priors.py` writes `data/<ds>_dcr/{train,user}_confounder_<tag>.npy`. The tag is, for example, `g0.3_k0_static` (`γ = 0.3`, `κ = 0`, static population prior). Pass the same tag to `main.py --dcr_tag`.

## Training

Run from `code/ml-1m/`. The examples below use ML-1M with the upstream hyperparameters.

```bash
python main.py --model FM     --dataset ml_1m --lr 0.05 --batch_size 1024 --dropout [0.3,0.3] --lamda 0.1 --gpu 0
python main.py --model DecFM  --dataset ml_1m --lr 0.05 --batch_size 1024 --dropout [0.3,0.3] --lamda 0.1 --alpha 0.2 --gpu 0
python main.py --model DecFM  --dataset ml_1m --dcr_tag g0.3_k0_static --lr 0.05 --batch_size 1024 --dropout [0.3,0.3] --lamda 0.1 --alpha 0.2 --gpu 0
python main.py --model FMconf --dataset ml_1m --dcr_tag g0.3_k0_static --lr 0.05 --batch_size 1024 --dropout [0.3,0.3] --lamda 0.1 --gpu 0
```

These commands train, in order:

1. FM.
2. DecRS.
3. TDecRS: DecFM with a time-aware confounder.
4. The FMconf control.

`--model DecFMR` with a tag gives DecRS-R.

For any other dataset, add `--dataset <name> --num_groups <G> --num_user_features 1`. G is listed in each dataset's `manifest.json`: 19 for ML-10M, 10 for Goodreads, 24 for KuaiRand and 18 for the simulator. On a new dataset, train FM first, then write its score file with `tools/gen_scores.py`, because the Dec* models blend with it.

Logs go to `log/`, and the best checkpoint goes to `models/`, named with the tag and seed. Validation Recall@10 picks the best epoch.

The upstream inference scripts (`code/*/inference/DecFM.py`, `DecNFM.py`) still work as described in the [upstream README](https://github.com/WenjieWWJ/DecRS#usage).

## Evaluation

Every number in the thesis is recomputed on CPU from the saved checkpoints by the `analysis/` scripts. They do not rely on the training logs. Each script's docstring states its inputs and outputs.

| script | what it produces |
|---|---|
| `analysis/rescore.py`, `aggregate.py`, `segments.py` | ML-1M Tier 1 / 1.5: per-user top-100 lists, seed means and paired tests |
| `analysis/eval_package/` | ML-1M and ML-10M: `score_all.py` scores every checkpoint, then `e2a`–`e2f` run the experiments listed below; `diag/` holds D1, D2 and the (γ, κ) surface |
| `analysis/ml10m_eval.py` | ML-10M (Tier 2) |
| `analysis/goodreads_eval.py` | Goodreads (E3 and Q1) |
| `analysis/kuairand_eval.py`, `kuairand_probe.py`, `kuairec_probe.py` | KuaiRand / KuaiRec (E4), with the pre-registered decision rules |
| `analysis/decrsr_eval.py` | DecRS-R (E1) |
| `analysis/favourability.py` | the dataset screen: population drift, individual drift `q`, `γ*`, `κ*` |
| `analysis/sim_validate.py`, `sim_eval.py` | simulator validation and grid (see `analysis/sim/README.md`) |
| `analysis/why_null/` | checks on why the time-aware gain is null |
| `analysis/audit/` | the evidence audit: every claim re-derived with one method (seed-aware CIs, TOST at ±2%, Holm). See `PROTOCOL.md` |
| `analysis/tuning/` | the hyperparameter sensitivity study. See `PROTOCOL.md` |

The scripts in `tools/` check that the extensions don't change the original model:

- `verify_dcr.py` checks that `κ = ∞` reproduces DecRS bit for bit, and that DecRS-R nests DecFM.
- `code/seqrec/smoke_test.py` runs the same nesting checks for the sequential models.

## Experiment labels

The scripts and logs use these labels:

| label | meaning |
|---|---|
| Tier 1 | ML-1M: DecRS vs TDecRS, a (γ, κ) grid and 5 seeds |
| Tier 1.5 | the FMconf recency control |
| Tier 2 | ML-10M, restricted to users whose training history spans at least 30 days |
| Tier 3 | SASRec / BERT4Rec with the confounder adaptor (`code/seqrec`) |
| E1 | DecRS-R gate |
| E2a–f | CPU evaluation package. a: FM⊕FM ensemble control. b: κ/γ accuracy–calibration frontier. c: calibration reference. d: Steck re-ranking. e: drift z-score (H2). f: equivalence tests |
| E3, Q1, Q3 | Goodreads. Q1 is the windowed population prior; Q3 is the 300-epoch check |
| E4 | KuaiRand-Pure (pre-registered, random-exposure test) and KuaiRec |
| D1–D3 | diagnostics. D1: genre-mix headroom ceiling. D2: power. D3: FM⊕FM share of each gain |
| T1–T6, P1–P8 | results of the theory chapter and the simulator predictions pre-registered from them |
| HP1–HP4 | hyperparameter-sensitivity predictions |

## Changes to the upstream code

- **Device.** The hard-coded `.cuda()` calls go through `device.py`, so the code runs on CPU or any GPU.
- **Memory.** Evaluation builds the item feature table once instead of per user. The original used about 7 GiB of GPU memory on ML-1M.
- **Seeds.** The seed comes from `DECRS_SEED`, and the checkpoint name carries the tag and seed so that runs don't overwrite each other.
- **Blending.** FM, NFM and FMconf rank on their own scores with no Eq. 11 blend. Datasets other than ML-1M blend with FM scores from our own FM checkpoint (`tools/gen_scores.py`).
- **New options.** `--dcr_tag`, `--eval_every` and `--num_workers`, plus the `FMconf` and `DecFMR` models.

With no `--dcr_tag`, the DecRS path gives the same numbers as before (`tools/verify_dcr.py`).

## Citation

If you use the DecRS code or data, please cite the original paper:

```
@inproceedings{wang2021deconfounding,
  title={Deconfounded Recommendation for Alleviating Bias Amplification},
  author={Wenjie Wang, Fuli Feng, Xiangnan He, Xiang Wang, and Tat-Seng Chua},
  booktitle={KDD},
  year={2021},
  publisher={ACM}
}
```

## Acknowledgment

- DecRS by Wenjie Wang et al. (NExT++, NUS). The FM/NFM code comes from [NFM-torch](https://github.com/guoyang9/NFM-pyorch/) (Yangyang Guo) and [NFM-tensorflow](https://github.com/hexiangnan/neural_factorization_machine) (Xiangnan He).
- The SASRec block layout follows [pmixer/SASRec.pytorch](https://github.com/pmixer/SASRec.pytorch) (Apache-2.0); no code is vendored.
