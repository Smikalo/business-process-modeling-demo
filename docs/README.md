# Documentation index

## Quick links

* **Production model status:** [`retrospectives/v122_retrospective.md`](retrospectives/v122_retrospective.md)
* **All-versions comparison:** [`objective_model_comparison.md`](objective_model_comparison.md)
* **Data ceiling + roadmap:** [`guides/limitations-and-next-steps.md`](guides/limitations-and-next-steps.md)
* **External data sources catalog:** [`external-data-sources.md`](external-data-sources.md)

## Subdirectories

### `adr/` — Architecture Decision Records

The architectural choices that shaped the project, in chronological order.

| File | Decision |
|---|---|
| [`adr-001-training-architecture.md`](adr/adr-001-training-architecture.md) | Zero-cost CPU training (no GPU, no cloud) |
| [`adr-002-ensemble-architecture.md`](adr/adr-002-ensemble-architecture.md) | Convex-blend ensemble over learned meta-learner |
| [`adr-003-external-signals.md`](adr/adr-003-external-signals.md) | External data loader framework + decision gate |
| [`adr-004-v6.md`](adr/adr-004-v6.md) | V6 censored-demand imputation + promo-lifecycle + pinball loss |
| [`adr-005-v7.md`](adr/adr-005-v7.md) | V7 per-SKU realised margins + price/cohort features + stacker + conformal |
| [`adr-006-v71.md`](adr/adr-006-v71.md) | V7.1 recency weights + per-channel specialists |

### `reports/` — per-version final reports

Each report is a self-contained write-up of one model version's design, results, and ablations.

| Version | What changed |
|---|---|
| [V6](reports/v6_final_report.md) | Censored-demand imputation + promo-lifecycle + ?=0.6 pinball |
| [V7](reports/v7_final_report.md) | Realised margins + price/cohort features + ridge stacker |
| [V7.1](reports/v71_final_report.md) | Recency-weighted training + per-channel specialists |
| [V7.2](reports/v72_final_report.md) | Optuna re-tuned on UAH cost objective |
| [V7.3](reports/v73_final_report.md) | NNLS stack of V4/V5/V6/V7.1 (similarity-first) |
| [V7.4](reports/v74_final_report.md) | Per-channel NNLS stack |
| [V7.5](reports/v75_final_report.md) | LAD per-channel + hierarchical reconcile |
| [V7.7](reports/v77_final_report.md) | Recency-weighted V7 retrain + multi-axis reconcile + tilted LAD |
| [V7.8](reports/v78_final_report.md) | Extended LAD pool + ?=0.55 |
| [V8](reports/v8_final_report.md) | Within-month features from raw daily shipments |
| [V9](reports/v9_final_report.md) | Sales-leading-indicator features + weekly Tweedie |
| [V10](reports/v10_final_report.md) | Receipts/stock leading features + hierarchical MinT |
| [V11](reports/v11_final_report.md) | Bias-aware drift-adaptive ensemble (production was V11_final until V12.x) |

### `retrospectives/` — campaign retrospectives

Honest write-ups of what was tried, what worked, what didn't.

| File | Summary |
|---|---|
| [`v12_retrospective.md`](retrospectives/v12_retrospective.md) | V12 candidate failed acceptance gate (val?test bias-direction reversal) |
| [`v121_retrospective.md`](retrospectives/v121_retrospective.md) | V12.1 fix shipped — re-train on EXT ABT + bias-direction-symmetry filter |
| [`v122_retrospective.md`](retrospectives/v122_retrospective.md) | **V12.2 = current production champion** (multi-helper joint OOF search) |
| [`v131_retrospective.md`](retrospectives/v131_retrospective.md) | V13 Chronos zero-shot + relaxed variant (parallel sensitivity, not production) |
| [`v14_retrospective.md`](retrospectives/v14_retrospective.md) | V14 GlobalNN trained on Kaggle GPU; data-leakage discovered + fixed; doesn't beat V12.2 |

### `guides/` — operational + planning docs

| File | Content |
|---|---|
| [`gpu-workflow.md`](guides/gpu-workflow.md) | Free Kaggle / Colab GPU workflow |
| [`limitations-and-next-steps.md`](guides/limitations-and-next-steps.md) | Data ceiling analysis + business-side ask for next 1-2 % accuracy |
| [`external-data-plan.md`](guides/external-data-plan.md) | Original V5-era plan for external signal integration |
| [`v11_plan.md`](guides/v11_plan.md) | V11 plan (drift-aware bias-constrained ensemble) |
| [`v11_chronos_colab_guide.md`](guides/v11_chronos_colab_guide.md) | Step-by-step Colab guide for V11 Chronos run |

### Top-level (cross-version)

| File | Content |
|---|---|
| [`objective_model_comparison.md`](objective_model_comparison.md) | Every released version + variant ranked by held-out test SIMSCORE |
| [`external-data-sources.md`](external-data-sources.md) | Survey of all external data sources used + considered |
| [`v4-creative-approaches.md`](v4-creative-approaches.md) | V4 creative experiments (per-channel, log-target, hierarchical, isotonic, GBDT meta) |
| [`v71_optuna_comparison.md`](v71_optuna_comparison.md) | V7.1 hyperparameter sweep results |
