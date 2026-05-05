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
| V13_chronos_ft | 20792 | 0.8473 | 0.6172 | -23.88 | 0.2215 | 7.917 |
| V13.1_relaxed | 17507 | 0.4322 | 0.3907 | +0.05 | 0.0825 | 5.061 |
| V13.2_relaxed | 17507 | 0.4329 | 0.3913 | -0.05 | 0.0828 | 5.057 |

## Notes

* **V12.2_champion** is the production model (`0.925·V11_final + 0.075·V12_external`, OOF-honest). Test SIM 0.4435, bias +2.13 %, WAPE 0.3931.
* **V13.2_relaxed** is the latest parallel sensitivity artifact (`0.925·V12.2_champion + 0.075·V13_chronos_ft`, judgment-call). Test SIM 0.4329 on aligned subset (95.7 % coverage of V12.2), bias −0.05 %. Supersedes V13.1_relaxed.
* **V13_chronos_ft** is the LoRA fine-tuned Chronos-T5-Small (2 epochs, ~600K trainable params, ran on Colab T4). Standalone test WAPE 0.617 (vs zero-shot 0.630, −2.1 % lift). Earned 0 LAD weight in V12.3 multi-helper joint OOF search (same val→test bias-direction reversal that affected zero-shot).
* **V13_chronos (zs)** is the original zero-shot Chronos run (LoRA fine-tune in Cell 5 silently no-op'd due to context_len vs prediction_length mismatch — kept for historical reference).
* **V12.3 multi-helper joint search** (with FT Chronos in the pool) produced the same champion as V12.2 — 0 weight on Chronos in any OOF-defensible variant.
