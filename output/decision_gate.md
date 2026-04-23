# External signals — decision gate report

## Gate definitions
- **PASS**: `val_WAPE_delta ≤ -0.005` AND `test_WAPE_delta ≤ +0.005`
- **MARGINAL**: `val_WAPE_delta < 0` AND `test_WAPE_delta ≤ +0.010`
- **LOO_KEEP**: LOO shows `test_WAPE_loss ≥ +0.003` when signal removed
- **FAIL**: otherwise

## Per-source verdict

| source | val Δ WAPE | test Δ WAPE | LOO test loss | verdict |
|---|---:|---:|---:|---|
| school_ua | -0.0035 | +0.0110 | -0.0014 | **FAIL** |
| imf_cpi | -0.0043 | +0.0110 | +0.0000 | **FAIL** |
| weather_ua | +0.0035 | +0.0123 | -0.0020 | **FAIL** |
| air_raids_ua | -0.0086 | +0.0129 | +0.0000 | **FAIL** |
| gtrends_ua | -0.0078 | +0.0117 | +0.0038 | **LOO_KEEP** |
| holidays_ua | -0.0074 | +0.0056 | +0.0061 | **MARGINAL** |
| tmdb_movies | -0.0002 | +0.0091 | +0.0000 | **MARGINAL** |
| world_bank_ua | -0.0032 | +0.0099 | +0.0000 | **MARGINAL** |
| conflict_ua | -0.0135 | -0.0059 | +0.0000 | **PASS** |
| nbu_fx | -0.0094 | +0.0024 | -0.0024 | **PASS** |

## V5 candidate set

**Keep** (6): `conflict_ua`, `gtrends_ua`, `holidays_ua`, `nbu_fx`, `tmdb_movies`, `world_bank_ua`

**Drop** (4): `air_raids_ua`, `imf_cpi`, `school_ua`, `weather_ua`

## Notes
- Validation = 2025-06 .. 2025-11; test = 2025-12 .. 2026-02.
- Deltas use the freshest baseline per run_utc (so baseline swings from stochasticity don't contaminate the verdict).
- LOO is computed from the 'all signals in' model by dropping one loader's columns at a time.