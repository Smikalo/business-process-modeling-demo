# ADR-001: Zero-Cost Training Architecture

**Status:** Accepted  
**Date:** 2026-04-22  
**Context:** The project has zero budget for compute. All training must run for free.

## Decision

Use **LightGBM (CPU-only gradient-boosted trees)** as the primary model family. No GPU, no cloud, no cost.

## Rationale

### Why LightGBM on CPU is sufficient

| Factor | Value |
|--------|-------|
| Dataset size | ~2.8M rows × ~60 features after engineering |
| Raw data on disk | ~50 MB |
| In-memory (feature-expanded) | ~500 MB–1 GB |
| Single training run | 3–5 min on laptop (M1/M2 or modern Intel, 16 GB RAM) |
| 3-fold time-series CV | ~15 min |
| Optuna hyperparameter search (100 trials) | 4–8 hours (run overnight) |
| Model artifact size | < 10 MB (joblib) |

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
| Full pipeline run (ingest → features → train → eval) | ~30 min | $0 |
| Optuna search (100 trials) | 4–8 h | $0 |
| Multi-horizon models (h=1,3,6 × point + quantile) | ~40 min | $0 |
| **Total** | **< 10 h** | **$0** |

## Consequences

- No `torch`, `tensorflow`, or GPU-related imports anywhere.
- `requirements.txt` contains only CPU-compatible packages.
- Pipeline is fully reproducible on any laptop with Python 3.11+ and 8 GB RAM.
- If the client later wants to scale to all 20 brands (~10× data), LightGBM on CPU still handles it in ~30–50 min per training run. GPU becomes relevant only beyond ~100 M rows.
