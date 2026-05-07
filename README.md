# Demand Forecasting & Procurement Optimization

SKU-level demand forecasting and automated procurement recommendations for a Ukrainian toy distributor (Djeco, CubicFun, Infantino). Built end-to-end as a zero-cost Proof of Concept — trains entirely on laptop CPU + free GPU (Kaggle / Google Colab).

**TL;DR**

* 65 models trained, objectively ranked on a held-out test window (Jul 2025 → Jan 2026, 7 months never seen during training).
* Production model (`V12.2_champion`) hits **annual aggregate accuracy 99.4 %**, **monthly accuracy 92 %**, **per-pair (Partner × SKU × Month) accuracy 63 %** — close to the empirical ceiling for this data type (M5/Rossmann/Favorita benchmarks plateau at 62-67 %).
* Beats expert manual planning by ~5 percentage points on M-WAPE. First production model that consumes free open-data signals (UA macro, war intensity, blackouts, Wikipedia attention, Orthodox calendar) end-to-end.
* All results reproducible from CLI; full audit trail in `docs/`.

---

## Production model

**`V12.2_champion`** = `0.925 · V11_final + 0.075 · V12_external`

* `V11_final` — bias-aware drift-adaptive LAD ensemble with hyper-recent bases + streaming-EMA calibrator + post-LAD λ-blend with `V11_g93`.
* `V12_external` — V11 hyper-recent two-stage retrained on `abt_v12_external` (V11 features + 32 columns from 9 priority-1 free open-data loaders).
* Recipe picked by a 459-candidate joint multi-helper bias-laddered OOF search; the 0.075 admixture lets EXT signals contribute on the margin without breaking V11_final's well-calibrated bias trajectory.

### Headline numbers (test: Jul 2025 – Jan 2026, 18.3 k active SKU-month pairs)

| Metric | V12.2_champion | V12.1 | V11_final | V10 |
|---|---:|---:|---:|---:|
| Test SIMSCORE ↓ | **0.4435** | 0.4453 | 0.4489 | 0.4690 |
| Test WAPE ↓ | **0.3931** | 0.3937 | 0.3950 | 0.4013 |
| Test aggregate bias % | **+2.13** | +2.36 | +2.80 | +5.09 |
| Test Monthly-WAPE ↓ | **0.0794** | 0.0796 | 0.0799 | 0.0827 |
| Cumulative annual accuracy | **~99.4 %** (0.6 % error) | ~99 % | ~99 % | ~95 % |
| Cumulative monthly accuracy | **~92 %** | ~92 % | ~92 % | ~91 % |

### Business numbers (per brand)

| Brand | Actual | AI Forecast | Error |
|---|---:|---:|---:|
| Infantino | 11.46 M UAH | 11.81 M UAH | +3.0 % |
| Cubic Fun | 8.43 M UAH | 8.30 M UAH | −1.6 % |
| Djeco | 11.59 M UAH | 11.57 M UAH | −0.2 % |
| **TOTAL** | **31.49 M UAH** | **31.69 M UAH** | **+0.6 %** |

A management-facing presentation package (`output/business_demo_pkg/`) with 3 panels + Excel + 1-page exec summary is available in both Russian and English. Rebuild via `python -m scripts.build_business_demo_pkg --lang {ru,en}`.

---

## Visualizations

**Cross-version progression — V10 → V12.2**

![Full progression](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_full_progression.png)

**V11 quality showcase — production model deep-dive (full 19-month timeline + per-channel violins + calibration scatter + rolling MAE/WAPE)**

![V11 quality showcase](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_v11_quality_showcase.png)

**Why we are at the DATA ceiling, not the algorithm ceiling — four independent lines of evidence**

![Data ceiling proof](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_data_ceiling_proof.png)

**Objective comparison of all model versions**

![All models compared](https://raw.githubusercontent.com/Smikalo/business-process-modeling-demo/main/output/plot_all_models_comparison.png)

---

## Quick start

```bash
# 1. Clone (Git LFS required for cached parquets + joblibs)
git lfs install
git clone https://github.com/Smikalo/business-process-modeling-demo.git
cd business-process-modeling-demo

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Score the production model on the held-out test window
PYTHONPATH=. python -c "
from scripts.score_similarity import score_frame
import pandas as pd
print(score_frame(pd.read_csv('output/preds_v122_champion_test.csv')))
"

# 4. Rebuild the production model from scratch (~15 min)
PYTHONPATH=. python -m scripts.build_v12_external_abt        # build EXT-augmented ABT
PYTHONPATH=. python -m scripts.train_v12_external_base       # train V12_external (5 seeds)
PYTHONPATH=. python -m scripts.v122_multihelper              # OOF joint search → champion

# 5. Generate management presentation package (Russian or English)
PYTHONPATH=. python -m scripts.build_business_demo_pkg --lang ru
# Outputs: output/business_demo_pkg/{panel1,panel2,panel3}_*.png + Excel + README.md

# 6. Cross-version audit
PYTHONPATH=. python -m scripts.audit_full
```

For GPU experiments (V13 Chronos fine-tune, V14 GlobalNN), see `notebooks/v13_chronos_finetune_colab.py` and `notebooks/v14_globalnn_colab.py` (paste-and-run on Colab T4) or `output/v14_kaggle_kernel/v14_globalnn.ipynb` (Kaggle).

---

## Repository layout

```
business-process-modeling-demo/
├── README.md                          ← this file
├── STRUCTURE.md                       ← detailed layout map
├── requirements.txt
│
├── data/                              ← raw client data (gitignored)
├── src/                               ← core importable library
├── scripts/                           ← runnable CLI scripts (build/train/audit/viz)
├── pipelines/                         ← top-level entry points
├── notebooks/                         ← Colab/Kaggle GPU notebooks
│
├── docs/                              ← see docs/README.md for index
│   ├── adr/                           ← architecture decision records
│   ├── reports/                       ← per-version final reports (V6 → V11)
│   ├── retrospectives/                ← campaign retros (V12 → V14)
│   └── guides/                        ← operational + planning docs
│
├── output/                            ← all generated artifacts
│   ├── abt_v*_cached.parquet          ← cached feature-engineered ABTs
│   ├── model_*.joblib                 ← saved LightGBM models
│   ├── preds_<tag>_{val,test}.csv     ← predictions
│   ├── plot_*.png                     ← diagnostic figures
│   ├── v<NN>/                         ← per-version ensemble search outputs
│   ├── v14_*/                         ← V14 GlobalNN training + Kaggle pipeline
│   ├── external/                      ← cached external loader outputs
│   └── business_demo_pkg/             ← management presentation package
│
└── tests/
```

See [`STRUCTURE.md`](STRUCTURE.md) for the full layout map and [`docs/README.md`](docs/README.md) for the documentation index.

---

## Version progression

Honest comparison of all production-eligible model variants on the same held-out test window (Jul 2025 – Jan 2026, 18.3 k active SKU-month pairs). Lower SIMSCORE / WAPE is better; bias should be near zero.

| Version | Test SIMSCORE | Test WAPE | Bias % | Notes |
|---|---:|---:|---:|---|
| Seasonal Naive (lag-12) | — | 0.759 | — | trivial baseline |
| V1 LightGBM | — | 0.886 | −0.52 | first model |
| V4 Convex Ensemble | — | 0.490 | −0.51 | first ensemble |
| V6 (V5 + imputation + pinball) | — | 0.449 | +0.41 | rolling-origin CV |
| V7 (margins + price/cohort + stacker) | — | 0.421 | −0.46 | per-SKU economics |
| V7.1 (recency γ=0.95 + per-channel) | — | 0.412 | −0.56 | first cost-optimised |
| V8 (within-month features) | 0.4800 | 0.4113 | −2.57 | first creative leap |
| V9 (sales-leading + weekly Tweedie) | 0.4557 | 0.4150 | +0.25 | second creative leap |
| V10 (receipts/stock + MinT) | 0.4690 | 0.4013 | +5.09 | row-level WAPE all-time low |
| V11_final (bias-aware drift-adaptive) | 0.4489 | 0.3950 | +2.80 | bias halved vs V10 |
| V12 (failed) | 0.4607 | 0.3983 | +4.48 | val→test bias reversal |
| V12.1_champion | 0.4453 | 0.3937 | +2.36 | first to consume EXT signals |
| **V12.2_champion** | **0.4435** | **0.3931** | **+2.13** | **★ current production** |
| V13_chronos_ft (zero-shot + LoRA fine-tune) | 0.8473 | 0.6172 | −23.88 | earned 0 LAD weight |
| V14_globalnn (Kaggle GPU, leakage-corrected) | 0.5213 | 0.4745 | +3.42 | doesn't beat V12.2 |
| V13.2_relaxed (parallel sensitivity) | 0.4329 | 0.3913 | −0.05 | judgment-call, not OOF-defensible |

Detailed per-version writeups: [`docs/reports/`](docs/reports/) (V6 → V11) and [`docs/retrospectives/`](docs/retrospectives/) (V12 → V14).

---

## Key design decisions

| Decision | ADR | Rationale |
|---|---|---|
| LightGBM on CPU only | [adr-001](docs/adr/adr-001-training-architecture.md) | Zero-cost — no GPU, no cloud. Trains 2.8 M rows in ~40 s. |
| Active-pair filtering (V2+) | — | Keep (Partner, SKU) pairs with ≥3 nonzero months in trailing 12. Removes 82 % of rows; raises nonzero rate 8 % → 59 %. |
| Two-stage forecasting (V2+) | — | Classifier `P(demand>0)` × Tweedie regressor `E[qty\|demand>0]`. Right-sized for zero-inflated count data. |
| Convex-blend ensemble | [adr-002](docs/adr/adr-002-ensemble-architecture.md) | 3-4 weighted models beat learned GBDT meta-learner under limited val data. |
| External signal decision gate | [adr-003](docs/adr/adr-003-external-signals.md) | Add-one-source + leave-one-out ablation per loader before promotion. |
| Censored-demand imputation + pinball loss | [adr-004](docs/adr/adr-004-v6.md) | V6: stockout-aware label correction + α=0.6 pinball. |
| Per-SKU realised margins + ridge stacker + conformal | [adr-005](docs/adr/adr-005-v7.md) | V7: economics-calibrated. Distributor margins are ~10 %, not the 28 % textbook assumption. |
| Recency-weighted training + per-channel specialists | [adr-006](docs/adr/adr-006-v71.md) | V7.1: γ=0.95 recency + w=0.6 specialist-blend. |

Plus four cross-version mechanisms developed at V11 → V12:

* **Bias-constrained LAD search** — hard filter `gap ≤ 0.05 ∧ \|OOF bias %\| ≤ 1.0` on the per-channel LAD candidate grid.
* **Streaming-EMA bias recalibrator** — time-causal multiplicative correction using only past data.
* **Bias-direction-symmetry filter** — V12.1 LAD addition that rejects pools whose bias direction reverses across CV folds.
* **Joint multi-helper bias-laddered search** — V12.2's 459-candidate grid with bias ceilings 1.0/1.25/1.5/1.75/2.0 %.

---

## Limitations

* **Information ceiling, not algorithmic.** 65 model classes — LightGBM variants, two-stage, ensembles, hierarchical reconciliation, foundation models (Chronos, TimesFM), Transformer-encoder with embeddings, intermittent specialists — all converge to the same ~63 % annual / ~92 % monthly accuracy band. Open M5 / Rossmann / Favorita benchmarks plateau at 62-67 % on similar structures.
* **7 months of held-out test data**; 1-2 p.p. WAPE deltas are within natural between-month variance, so all selection is rolling-origin CV with the OOF SIMSCORE rule (gap ceiling 0.02-0.05).
* **Three brands** in scope (Djeco, Cubic Fun, Infantino); other 15+ brands in client's ERP not yet included.
* **Stock-out periods** are flagged and partially imputed (V6+), but the underlying signal — "demand we couldn't satisfy" — is fundamentally not in the data.
* **January overforecast (~24 %)** is the single systematic weakness — out of 6 historical Januaries in training, 5 were distorted by COVID lockdowns and war stocking. Adding pre-2020 history would fix this.

What would unlock 80-90 % accuracy (out of scope for current zero-budget work; 12-18 month bizdev project):

* **POS-level transaction data** from partners (timestamped baskets) — biggest single lever, +8-12 p.p.
* **Real partner inventory** (distinguish "no demand" from "out of stock at partner") — +3-5 p.p.
* **Full promo plans with budgets** (we have a binary flag, not depth or reach) — +3-5 p.p.
* **2-3 additional pre-pandemic years** of history for a clean seasonal anchor — +1-2 p.p.

See [`docs/guides/limitations-and-next-steps.md`](docs/guides/limitations-and-next-steps.md) for the full analysis.

---

## Compute & cost

| Stage | Time | Cost |
|---|---|---|
| Data ingest + ABT build (V12 era, ~316 k × 223 cols) | ~3 min (cached) | $0 |
| V12_external base training (5 seeds) | ~5 min CPU | $0 |
| V12.2 multi-helper joint OOF search (459 candidates) | ~5 min CPU | $0 |
| Business demo package generation | ~5 sec | $0 |
| V13 Chronos fine-tune | ~2 hr Colab T4 | $0 |
| V14 GlobalNN training | ~3 hr Kaggle T4/P100 | $0 |
| **Total — full pipeline reproduction** | **~3 hr wall-clock** | **$0** |

Free Kaggle CLI tooling (`scripts/v14_kaggle_check.sh`, `scripts/build_v14_kaggle_notebook.py`, `scripts/push_to_kaggle.sh`) reads `KAGGLE_USERNAME` + `KAGGLE_API_TOKEN` from `.env`. No browser clicks for GPU runs.
