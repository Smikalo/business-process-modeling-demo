# ADR-001: Zero-Cost Training Architecture

**Status:** Accepted  
**Date:** 2026-04-22  
**Context:** The project has zero budget for compute. All training must run for free.

## Decision

Use **LightGBM (CPU-only gradient-boosted trees)** as the primary model family. No GPU, no cloud, no cost.

## Rationale

### Why LightGBM on CPU is sufficient

| Factor | Value (as measured on V4) |
|--------|---------------------------|
| Dataset size (pre-filter) | ~2.8M rows × ~60 features after engineering |
| Dataset size (active-pair filter, V2+) | ~480k rows × ~75 features |
| Raw data on disk | ~50 MB |
| In-memory (feature-expanded, filtered) | ~120 MB |
| ABT build (ingest + features, first run) | ~4 min (cached thereafter) |
| V1 single training run | ~40 s |
| V2 / V3 single training run | ~70-90 s |
| **V4 ensemble training (3 models)** | **~3 min** |
| Optuna hyperparameter search (30 trials) | ~12 min |
| Model artifact size (V4 ensemble) | ~8 MB (joblib) |

LightGBM uses all CPU cores via `n_jobs=-1` and OpenMP (`libomp`). No GPU dependency anywhere in the codebase.

### Why not deep learning?

1. **Data is tabular and monthly**: 74 time steps, ~1 500 active SKUs. This is small by DL standards.
2. **Gradient-boosted trees dominate tabular benchmarks**: consistently win Kaggle competitions over neural nets on structured data of this size.
3. **The previous model (Modus, ~6 years ago) failed at ~15–20 % error.** The cause was almost certainly bad feature engineering (no stock-out awareness, no partner-type handling), not insufficient model complexity.
4. **DL would add cost and complexity with no expected accuracy gain** for this data profile.

### Fallback options (if ever needed)

| Platform | Free allocation | Use case |
|----------|----------------|----------|
| Kaggle Notebooks | 30 h/week GPU (T4), 20 h/week TPU | Neural experiments (N-BEATS, TFT) |
| Google Colab Free | Variable T4 GPU, ~12 h sessions | Quick one-off experiments |
| AWS SageMaker Studio Lab | 4 h GPU / 24 h period | Short demos |

None of these are expected to be needed for the PoC.

### Compute budget estimate

| Activity | Time | Cost |
|----------|------|------|
| ABT build (ingest → features → cache), first run | ~4 min | $0 |
| ABT build, cached runs | ~2 s (parquet load) | $0 |
| V4 ensemble training (3 base models + SLSQP weights) | ~3 min | $0 |
| V4 inference on test set (34k rows) | <1 s | $0 |
| Optuna search (30 trials) | ~12 min | $0 |
| Multi-horizon models (h=1,3,6 × point + quantile) | ~40 min | $0 |
| **Full end-to-end pipeline (first run)** | **~25 min** | **$0** |
| **Subsequent runs (cache hit)** | **~5 min** | **$0** |

## Consequences

- No `torch`, `tensorflow`, or GPU-related imports anywhere.
- `requirements.txt` contains only CPU-compatible packages.
- Pipeline is fully reproducible on any laptop with Python 3.11+ and 8 GB RAM.
- The cached ABT (`output/abt_v4_cached.parquet`, ~10 MB) makes iteration fast — retraining a base model with a new feature does not re-run ingestion.
- If the client later wants to scale to all 20 brands (~10× data), LightGBM on CPU still handles it in ~30–50 min per training run. GPU becomes relevant only beyond ~100 M rows.

## Scope update (V4)

The original decision (V1-V3) assumed a single LightGBM model family. V4 extends this to an **ensemble of LightGBM models** (V3 Tweedie + LogTarget MAE + PerChannel specialists). This does not change the zero-cost / CPU-only premise — all three components are LightGBM boosters; they train serially on the same hardware with total runtime still well under 5 min. See `docs/adr/adr-002-ensemble-architecture.md` for the ensemble rationale.
