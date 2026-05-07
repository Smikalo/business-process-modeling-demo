# Repository structure

Map of where things live, what they're for, and what depends on what.

## Top-level layout

```
business-process-modeling-demo/
??? README.md                     ? project overview, results, version progression
??? STRUCTURE.md                  ? this file
??? requirements.txt
??? .env                          ? KAGGLE_USERNAME, KAGGLE_API_TOKEN (gitignored)
??? .gitignore
?
??? data/                         ? raw client data (xlsx, csv, txt Ч gitignored)
??? src/                          ? core importable library
??? scripts/                      ? runnable CLI scripts (build/train/audit/viz/Kaggle)
??? pipelines/                    ? top-level entry points (one-shot pipelines)
??? notebooks/                    ? Colab/Kaggle GPU notebooks
??? docs/                         ? architecture, reports, retros, guides
??? output/                       ? all generated artifacts (parquets, models, preds, figures)
??? tests/                        ? unit tests
```

## `src/` Ч core library (importable)

Used by everything via `from src.X import Y`. **Never edit without re-running tests.**

```
src/
??? config.py                     ? file paths, period boundaries, split dates
??? ingestion.py                  ? raw .txt/.xlsx loaders (cp1251 + utf-8 + calamine)
??? aggregation.py                ? monthly aggregation (clip negatives)
??? master.py                     ? dense (Period, Partner, SKU) skeleton
??? enrichment.py                 ? nomenclature + partners + prices + promo joins
??? enrichment_external.py        ? external loader join layer
??? external_data.py              ? BaseSignalLoader ABC, parquet cache
??? leakage_guard.py              ? publication-lag enforcement
??? features*.py                  ? feature engineering modules (per-version)
??? evaluation.py                 ? WAPE/MAPE/RMSE/Bias metrics + temporal split
??? losses.py                     ? pinball + asymmetric LightGBM objectives
??? model.py, model_v2.py, ...    ? model classes (V1 ? V4)
??? streaming_calibrator.py       ? V11 streaming-EMA bias recalibrator
??? adversarial_validation.py     ? V11 drift audit utility
??? intermittent.py               ? V12 Croston/SBA/TSB intermittent specialist
??? margin_table.py               ? V7 per-SKU realised margin
??? demand_imputation.py          ? V6 censored-demand imputation
??? procurement.py                ? order recommendations (q50/q90 safety stock)
??? loaders/                      ? external data loaders (one .py per source)
??? models/                       ? model architectures (e.g. V14 GlobalNN)
```

## `scripts/` Ч runnable CLI scripts

Run via `PYTHONPATH=. python -m scripts.<name>`. Organized by purpose (in spirit; files are flat for import compatibility).

| Pattern | Purpose | Examples |
|---|---|---|
| `build_v*_abt.py` | Build cached ABT for a version | `build_v6_abt.py`, `build_v12_external_abt.py` |
| `train_v*.py` | Train a model | `train_v5.py`, `train_v12_external_base.py` |
| `v*_lad_stack.py`, `v*_multihelper.py` | Per-version LAD ensemble search | `v8_lad_stack.py`, `v126_multihelper.py` |
| `v*_final_blend.py`, `v*_champion_blend.py` | Final ?-blend search | `v11_final_blend.py`, `v121_champion_blend.py` |
| `audit_*.py`, `score_similarity.py` | Quality audits | `audit_full.py`, `audit_v121.py` |
| `viz_*.py` | Visualisation | `viz_v122_progression.py`, `viz_data_ceiling_proof.py` |
| `push_*.sh`, `pull_*.sh`, `kaggle_env.sh` | Kaggle CLI tooling | `v14_kaggle_check.sh`, `push_to_kaggle.sh` |
| `build_business_demo_pkg.py` | Generate management presentation package | (RU + EN) |

Total: ~120 scripts spanning V1 ? V12.2 production + V13/V14 experimental work.

## `pipelines/` Ч top-level entry points

One-shot scripts that run a full version's pipeline end-to-end. Useful for reproduction.

```
pipelines/
??? run_pipeline.py               ? V1 (ingest ? V1 baseline)
??? run_v4_experiments.py         ? V4 (all base models comparison)
??? run_v4_round2.py              ? V4 round-2 calibration
??? run_v4_final.py               ? V4 ensemble production run
```

## `notebooks/` Ч GPU notebooks

```
notebooks/
??? kaggle_export.py              ? Kaggle dataset export utility
??? results_visualization.py      ? cross-version viz helper
??? v11_chronos_colab.ipynb       ? V11 Chronos zero-shot run (T4)
??? v13_chronos_finetune_colab.py ? V13 Chronos LoRA fine-tune (paste-and-run)
??? v14_globalnn_colab.py         ? V14 GlobalNN training (paste-and-run)
```

## `docs/` Ч documentation

```
docs/
??? adr/                          ? architecture decision records (6 docs)
?   ??? adr-001-training-architecture.md
?   ??? adr-002-ensemble-architecture.md
?   ??? adr-003-external-signals.md
?   ??? adr-004-v6.md
?   ??? adr-005-v7.md
?   ??? adr-006-v71.md
?
??? reports/                      ? per-version final reports (V6 ? V11, 13 docs)
?   ??? v6_final_report.md ... v11_final_report.md
?   ??? (each report = a complete write-up of that version's design + results)
?
??? retrospectives/               ? per-campaign retros (5 docs)
?   ??? v12_retrospective.md      ? V12 attempt that regressed
?   ??? v121_retrospective.md     ? V12.1 fix that shipped
?   ??? v122_retrospective.md     ? V12.2 production champion
?   ??? v131_retrospective.md     ? V13.1 Chronos relaxed variant
?   ??? v14_retrospective.md      ? V14 GlobalNN + leakage discovery
?
??? guides/                       ? operational + planning docs
?   ??? gpu-workflow.md
?   ??? limitations-and-next-steps.md
?   ??? external-data-plan.md
?   ??? v11_chronos_colab_guide.md
?   ??? v11_plan.md
?
??? external-data-sources.md      ? cross-version: data source catalog
??? objective_model_comparison.md ? cross-version: every model ranked
??? v4-creative-approaches.md     ? V4 creative experiments writeup
??? v71_optuna_comparison.md      ? V7.1 hyperparameter sweep summary
```

## `output/` Ч generated artifacts

Flat-by-default, with versioned subdirs for ensemble search results.

| Pattern | Content |
|---|---|
| `abt_v*_cached.parquet` | Cached feature-engineered ABTs (input to training) |
| `model_*.joblib` | Saved LightGBM models (per version) |
| `preds_<tag>_{val,test}.csv` | Predictions on val/test splits Ч keyed on `(ѕериод, ѕартнер, јртикул)` |
| `feature_importance_*.csv` | LightGBM gain importance |
| `*_metrics.csv` | Headline metrics per version |
| `*_rolling_cv.{json,md}` | Rolling-origin CV results |
| `cost_scorecard*.{json,md}` | UAH cost scorecard |
| `plot_*.png` | Diagnostic figures |
| `full_audit.{csv,md}` + `plot_full_progression.png` | Cross-version audit |
| `v<NN>/` | Per-version ensemble search outputs (LAD champion JSON, OOF CSV) |
| `v14_globalnn/` | V14 export tensors for GPU training |
| `v14_kaggle_{dataset,kernel,output}/` | V14 Kaggle pipeline state |
| `v13_fm/` | V13 foundation-model export data |
| `external/` | Cached parquet outputs of external data loaders |
| `business_demo_pkg/` | **Management presentation package** (RU + EN) |

## `tests/` Ч unit tests

Standard pytest layout.

## What's gitignored

- `data/` content (confidential customer data Ч patterns in `.gitignore`)
- `.env` (API tokens)
- `.venv/`, `__pycache__/`
- `output/gpu/` (transient Kaggle outputs)
- `logs/` (transient log files)
- `.cursor/`, `.beads/` (local tooling state)
- `*_pkg.zip` (regenerable packages)
- `.DS_Store`

## Quick orientation

| Want to... | Look at |
|---|---|
| Understand the project | `README.md` |
| Reproduce V12.2 production | `scripts/build_v12_external_abt.py` ? `scripts/train_v12_external_base.py` ? `scripts/v122_multihelper.py` |
| Score the production model | `scripts/audit_full.py` (writes `output/full_audit.{csv,md}`) |
| Generate the business presentation | `python -m scripts.build_business_demo_pkg --lang ru` (or `--lang en`) |
| Run V13 Chronos fine-tune | `notebooks/v13_chronos_finetune_colab.py` (paste-and-run on Colab T4) |
| Run V14 GlobalNN on Kaggle | `./scripts/v14_kaggle_check.sh status` (after `KAGGLE_USERNAME` set in `.env`) |
| Read what went wrong with V12 | `docs/retrospectives/v12_retrospective.md` |
| Read what fixed it | `docs/retrospectives/v122_retrospective.md` |
| Understand current limitations | `docs/guides/limitations-and-next-steps.md` |
