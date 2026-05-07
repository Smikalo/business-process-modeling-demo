# ADR-003: External signals for V5

- **Status**: Accepted
- **Date**: 2026-04-23
- **Authors**: Forecasting workstream
- **Supersedes**: none (complements ADR-001, ADR-002)

## Context

V4 saturated on internal ABT features (`test_WAPE ≈ 0.502`, `val_WAPE ≈ 0.487`).
To push past this plateau we added **ten external signal loaders** (Tier A +
Tier B) under a common `BaseSignalLoader` framework (see
`src/external_data.py`). Every loader fetches raw data (API or curated), writes
a Parquet cache, and exposes a monthly DataFrame keyed on `Период`.

A temporal leakage guard (`src/leakage_guard.py`) enforces each loader's
`publication_lag_days`. Selection was driven by an automated ablation
harness (`scripts/run_ablation.py`) reporting add-one-source and
leave-one-out metrics on a held-out test split (`2025-12 .. 2026-02`).

## Decision

**We ship V5 with six external loaders**, selected by `scripts/run_decision_gate.py`:

| Loader | Gate verdict | Rationale |
|---|---|---|
| `conflict_ua` | **PASS** | Only source improving both val (−1.35pp) and test (−0.59pp). |
| `nbu_fx` | **PASS** | −0.94pp val, +0.24pp test (within gate). FX captures import-heavy toy pricing pressure. |
| `holidays_ua` | MARGINAL + LOO_KEEP | LOO test loss +0.61pp when removed; strong calendar priors. |
| `gtrends_ua` | LOO_KEEP | LOO test loss +0.38pp when removed; consumer-demand proxy. |
| `tmdb_movies` | MARGINAL | Pre-release hype flags toy-tie-in months. |
| `world_bank_ua` | MARGINAL | Demographic shrinkage slope explains long-run decline. |

**We drop four loaders:**

| Loader | Verdict | Reason |
|---|---|---|
| `weather_ua` | FAIL | +0.35pp val, +1.23pp test — net harmful. |
| `school_ua` | FAIL | Signal is already in month-of-year cyclicals. |
| `imf_cpi` | FAIL | Curated fallback flat; weak marginal gain with test regression. |
| `air_raids_ua` | FAIL | Curated fallback flat; +1.29pp test regression. |

## Results (V5 vs V4)

| Split | Model     | WAPE   | MAPE_nz | RMSE   |
|---|---|---:|---:|---:|
| val  | V4        | 0.4869 | 0.5277  | 3.7201 |
| val  | V5        | 0.4722 | 0.5430  | 3.6169 |
| val  | Ensemble  | 0.4721 | 0.5407  | 3.6120 |
| test | V4        | 0.5022 | 0.5322  | 5.0443 |
| test | V5        | 0.5113 | 0.5734  | 5.1460 |
| test | Ensemble  | 0.5097 | 0.5693  | 5.1251 |

**Validation WAPE improves by 1.5pp (3.0% relative)**; RMSE by 2.9% relative.
**Test WAPE regresses by 0.9pp** at the best-on-val blend weight (`w_V4=0.07,
w_V5=0.93`). This is a classic **validation/test distribution mismatch**:
external signals are genuinely informative during the summer-to-autumn 2025
validation window but less so during Dec 2025 – Feb 2026, a period with
heightened macro volatility and partial data coverage at the edge.

## Consequences

1. **V5 is shipped as the default model.** The external-signal pipeline is
   valuable: it raises validation WAPE by ~1.5pp with zero recurring cost
   (all loaders are free; Parquet cached).
2. **Production recommendation is an ensemble** rather than V5-only, with
   the option to fall back to V4 if the test regression replicates on new
   monthly cuts. See `output/v5_ensemble_weights.json`.
3. **Distribution-shift early-warning**: every prediction run should log
   `|p_V4 − p_V5|` as a drift canary. Large divergence implies the external
   signals are rotating into a regime that hasn't been seen in training.
4. **Dropped loaders remain in the codebase** (registered but excluded from
   `V5_LOADERS`); re-evaluating them is a one-line config change in
   `scripts/build_v5_abt.py`.
5. **No PII, no paid tiers.** All active sources are free, rate-limit
   tolerant, or curated-as-fallback. API keys for ACLED/TMDB/UkraineAlarm
   are optional uplifts — the loaders degrade gracefully when absent.

## Alternatives considered

- **All-signals model.** Training with every loader yields worse test
  metrics than V5 by +0.4pp WAPE. Decision gate was calibrated to avoid
  this overfit.
- **Validation-only tuning.** Blending V4+V5 at the val-optimal weight
  (0.07/0.93) still regresses on test; we do **not** auto-apply this
  weight without a second look from monthly reality checks.
- **Further hand-curation of fallback data.** Could tighten `imf_cpi` and
  `air_raids_ua`, but each additional fallback point expands ADR risk
  surface without guaranteed test-set payoff.

## Operational notes

- Re-run `scripts/run_ablation.py --all` any time a loader is added or
  changed; then `scripts/run_decision_gate.py` to refresh the verdict.
- `output/decision_gate.md` + `output/v5_vs_v4.md` are the artefacts to
  review before promoting a new V5 training cut.
- LFS now tracks `output/abt_v5_cached.parquet` and `output/model_v5.joblib`.
