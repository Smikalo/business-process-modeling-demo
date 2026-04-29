# V12 audit report

**3 PASS · 4 WARN · 1 FAIL**

## Headline metrics

| metric | V11_final | V12_final | Δ |
|---|---:|---:|---:|
| Test SIMSCORE | 0.4489 | 0.4607 | **+2.63 %** |
| Test WAPE | 0.3950 | 0.3983 | **+0.84 %** |
| Test \|bias%\| | 2.80 % | 4.48 % | **+1.68 pp** |
| Val SIMSCORE | 0.3575 | 0.3514 | — |
| Val→Test gap | +0.0914 | +0.1093 | +0.0179 |

## Checks

- ✅ **row_count_match** — V11 val=31368 test=18298 ; V12 val=31368 test=18298
- ✅ **key_alignment** — val keys match=True, test keys match=True
- ❌ **v12_test_simscore_beats_v11** — V11=0.4489  V12=0.4607  Δ=+2.63%
- ⚠️ **v12_test_wape_beats_v11** — V11=0.3950  V12=0.3983  Δ=+0.84%
- ⚠️ **v12_test_bias_not_worse** — |V11 bias|=2.80%  |V12 bias|=4.48%  Δ=+1.68 pp
- ✅ **overfit_gap_under_control** — V11 gap=+0.0914  V12 gap=+0.1093  growth=+0.0179
- ⚠️ **no_test_month_with_extreme_bias** — 2 test months with |bias%| > 10% (out of 7 total)
- ⚠️ **ext_leakage_spot_check** — 14 loaders with potentially recent data; feature-time guard handles actual leakage
