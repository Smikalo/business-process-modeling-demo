# V12.1 Audit Report

Comparison of all production model candidates on the standard validation (Jul 2024 → Jun 2025) and held-out test (Jul 2025 → Mar 2026) windows.

## Test scores

| Model | n_rows | SIMSCORE ↓ | WAPE ↓ | Bias % | RMSE | M-WAPE |
|---|---:|---:|---:|---:|---:|---:|
| **V11_LAD** | 18298 | 0.4662 | 0.4006 | +4.86 | 4.812 | 0.0827 |
| **V11_final** | 18298 | 0.4489 | 0.3950 | +2.80 | 4.847 | 0.0799 |
| **V12_LAD** | 18298 | 0.4607 | 0.3983 | +4.48 | 4.832 | 0.0801 |
| **V12_final** | 18298 | 0.4607 | 0.3983 | +4.48 | 4.832 | 0.0801 |
| **V12.1_LAD** | 18298 | 0.4568 | 0.3969 | +3.86 | 4.843 | 0.0812 |
| **V12.1_final** | 18298 | 0.4556 | 0.3973 | +3.38 | 4.871 | 0.0829 |
| **V12.1_meta** | 18298 | 0.4523 | 0.3955 | +3.33 | 4.840 | 0.0803 |
| **V12.1_champion** | 18298 | 0.4453 | 0.3937 | +2.36 | 4.855 | 0.0796 |

## V12.1_champion vs V11_final (held-out test)

- **Δ SIMSCORE**: -0.0036 (-0.80% relative — lower SIMSCORE is better)
- **Δ WAPE**: -0.0013
- **Δ Bias%**: -0.44 pp (V12.1 bias is closer to zero)
- **Δ Monthly-WAPE**: -0.0003
- **Δ RMSE**: +0.008

**Recommendation:** ship V12.1_champion as the new production model. The improvement is small (~0.8% relative on SIMSCORE) but is supported by an honest 3-fold OOF lambda search, and the bias moves in the right direction. The V12_external base brings the real EXT signals (open-data sources) into the production stack for the first time.
