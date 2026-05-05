# Full Model Audit (cross-version)

Generated: 2026-05-05

All models scored on the same held-out test window (Jul 2025 – Mar 2026).
Lower is better for SIMSCORE / WAPE / Monthly-WAPE / RMSE; bias should be near zero.

## Test scores

| Model | n_rows | SIMSCORE ↓ | WAPE ↓ | Bias % | M-WAPE ↓ | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| V11_LAD | 18298 | 0.4662 | 0.4006 | +4.86 | 0.0827 | 4.812 |
| V11_final | 18298 | 0.4489 | 0.3950 | +2.80 | 0.0799 | 4.847 |
| V12_LAD | 18298 | 0.4607 | 0.3983 | +4.48 | 0.0801 | 4.832 |
| V12_final | 18298 | 0.4607 | 0.3983 | +4.48 | 0.0801 | 4.832 |
| V12.1_LAD | 18298 | 0.4568 | 0.3969 | +3.86 | 0.0812 | 4.843 |
| V12.1_champion | 18298 | 0.4453 | 0.3937 | +2.36 | 0.0796 | 4.855 |
| **V12.2_champion** | 18298 | 0.4435 | 0.3931 | +2.13 | 0.0794 | 4.859 |
| V13_chronos (zs) | 20792 | 0.8666 | 0.6304 | -26.05 | 0.2119 | 8.232 |
| V13.1_relaxed | 17507 | 0.4322 | 0.3907 | +0.05 | 0.0825 | 5.061 |

## Notes

* **V12.2_champion** is the new production model (`0.925·V11_final + 0.075·V12_external`, OOF-honest). Test SIM 0.4435, bias +2.13 %, WAPE 0.3931 (new all-time low).
* **V13.1_relaxed** is a parallel sensitivity artifact (`0.925·V12.1_champion + 0.075·Chronos`, judgment-call). Test SIM 0.4322 on aligned subset, bias +0.05 %. **Not OOF-defensible** — see `docs/v131_retrospective.md`.
* **V13_chronos** is the zero-shot Chronos-T5-Small run (LoRA fine-tune silently no-op'd; predictions are stock pretrained model). Earned 0 LAD weight under honest OOF.
