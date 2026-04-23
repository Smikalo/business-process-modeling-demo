# V7.1 — Optuna follow-up (negative result, kept as sibling)

## TL;DR

Kaggle Optuna kernel finished after 120 trials (`output/gpu/v7_optuna_best_params.json`).
Best pinball-val score **0.4352** at:

```json
{
  "num_leaves": 118,
  "learning_rate": 0.0191,
  "feature_fraction": 0.522,
  "bagging_fraction": 0.820,
  "bagging_freq": 8,
  "min_data_in_leaf": 22,
  "reg_lambda": 0.411
}
```

Re-running the full V7.1 pipeline (`rec95` + per-channel specialist blend)
with these params:

| Variant | Test WAPE | Test Bias | Annual UAH cost | Δ vs champion |
|---|---:|---:|---:|---:|
| **V7.1 champion** (defaults, `w=0.6`) | 0.4117 | −0.47 | **1,316,197** | — |
| V7.1 Optuna-tuned (`rec95_tuned` + `w=0.5`) | 0.4072 | −0.48 | 1,328,762 | **+12,565 UAH** |

## Why Optuna didn't help on business cost

Optuna optimised **val-set pinball loss** (α=0.45, single scalar). Pinball
loss and UAH cost are correlated but not identical objectives:

- Tuned params trade a small val WAPE win (−0.45pp) for a slightly higher
  holding-cost curve on the test set.
- The un-tuned defaults (`num_leaves=255`, `learning_rate=0.05`) happen to
  sit in a broad flat basin for UAH cost, so tuning for pinball shifted us
  into a slightly worse cost region.
- Optuna had no visibility into the per-SKU margin table or the asymmetric
  over/under cost ratio.

## Decision

- **Keep the pre-Optuna V7.1 champion** (`output/v71_champion.json`,
  `preds_v71_channels_{test,val}.csv`) as the official model.
- Preserve the tuned variant as a sibling for reproducibility:
  - `output/model_v7_rec95_tuned.joblib`
  - `output/preds_v71_channels_tuned_{val,test}.csv`
  - `output/v71_champion_tuned.json`
  - `output/cost_scorecard_v71_channels_tuned.{json,md}`
- Optuna best params cached at `output/v7_optuna_best.json`.

## V7.2 follow-up

Re-run Optuna with **UAH cost** as the objective (not pinball loss), using
the per-SKU margin table and the official scorecard. That is the only way
to recover a tuning win on the business metric.

See also: `docs/v71_final_report.md`, `docs/adr-006-v71.md`.
