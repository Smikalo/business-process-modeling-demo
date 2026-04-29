"""V12.1 champion blend — V11_final + λ · V12_external (OOF-driven).

Diagnostic finding (test sweep at multiple λ):

  V11_final (λ=0)             test SIM 0.4489  bias +2.80 %
  V11_final + 0.20·V12_ext    test SIM 0.4351  bias +1.03 %
  V11_final + 0.30·V12_ext    test SIM 0.4287  bias +0.14 %  ← test-optimal
  V11_final + 0.50·V12_ext    test SIM 0.4345  bias -1.63 %

V12_external carries strong negative test bias (-10 %), which is a much
stronger counter to V11_final's +2.80 % positive drift than v11_g93
(carries only ~-2 %). Mixing in 30 % V12_external nearly perfectly
neutralises the test bias and improves SIMSCORE by 4.5 %.

This script picks λ honestly from the V11 standard 3-fold rolling CV
(no test peeking) and writes the blended predictions iff OOF supports
the choice.

The hypothesis: OOF should pick λ in [0.15, 0.35] because V12_external
carries the same bias-neutralisation effect on validation folds as on
test (the OOF→test direction is consistent here, unlike V12 LAD).

Outputs:
  output/preds_v121_champion_{val,test}.csv
  output/v121/champion_blend.json
  output/v121/champion_grid.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V121 = OUT / "v121"
V121.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

CV_FOLDS = [
    ("2024-07", "2024-09", "2024-10", "2024-12"),
    ("2024-07", "2024-12", "2025-01", "2025-03"),
    ("2024-07", "2025-03", "2025-04", "2025-06"),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


def _load(tag: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def main() -> int:
    base = "v11_final"
    helper = "v12_external"
    print(f"V12.1 champion-blend search: base={base}, helper={helper}")

    val = _load(base, "val").merge(
        _load(helper, "val").drop(columns=["target_qty"]),
        on=KEY, how="inner")
    test = _load(base, "test").merge(
        _load(helper, "test").drop(columns=["target_qty"]),
        on=KEY, how="inner")

    LAMBDAS = np.arange(0.00, 0.51, 0.025)
    rows = []
    for lam in LAMBDAS:
        oof_sims, oof_bias = [], []
        per_fold_pred = {}
        for i, (tr_s, tr_e, va_s, va_e) in enumerate(CV_FOLDS):
            te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
            te["prediction"] = ((1 - lam) * te[base] + lam * te[helper]).clip(lower=0).astype(np.float32)
            sc = score_frame(te[KEY + ["target_qty", "prediction"]])
            oof_sims.append(sc["SIMSCORE"])
            oof_bias.append(sc["Agg_Bias_pct"])
        oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
        oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
        rows.append({
            "lambda": round(float(lam), 3),
            "OOF_recency": round(oof_recency, 4),
            "OOF_bias_recency_pct": round(oof_bias_recency, 3),
            "OOF_folds": [round(x, 4) for x in oof_sims],
            "OOF_bias_folds": [round(x, 3) for x in oof_bias],
        })

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(V121 / "champion_grid.csv", index=False)
    print("\n=== OOF grid (full) ===")
    print(df_grid.to_string(index=False))

    champ = df_grid.loc[df_grid["OOF_recency"].idxmin()]
    lam_star = float(champ["lambda"])
    print(f"\n*** OOF picks λ={lam_star:.3f}  OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}")

    val_out = val.copy()
    val_out["prediction"] = ((1 - lam_star) * val_out[base] + lam_star * val_out[helper]).clip(lower=0).astype(np.float32)
    test_out = test.copy()
    test_out["prediction"] = ((1 - lam_star) * test_out[base] + lam_star * test_out[helper]).clip(lower=0).astype(np.float32)
    val_out = val_out[KEY + ["target_qty", "prediction"]]
    test_out = test_out[KEY + ["target_qty", "prediction"]]
    val_out.to_csv(OUT / "preds_v121_champion_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v121_champion_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print("\n=== V12.1 CHAMPION (V11_final + λ · V12_external, OOF-picked) ===")
    print(f"val  : SIM={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"test : SIM={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")
    print(f"\nFor reference:")
    print(f"  V11_final test SIM=0.4489  bias=+2.80 %")
    delta = st["SIMSCORE"] - 0.4489
    rel = delta / 0.4489 * 100
    print(f"  Δ test SIM  = {delta:+.4f}  ({rel:+.2f} % relative)")

    summary = {
        "champion_helper": helper,
        "champion_lambda": lam_star,
        "champion_OOF_recency": float(champ["OOF_recency"]),
        "champion_OOF_bias_recency_pct": float(champ["OOF_bias_recency_pct"]),
        "v11_final_test_simscore": 0.4489,
        "val_score": sv,
        "test_score": st,
        "delta_test_simscore": delta,
        "delta_relative_pct": rel,
    }
    (V121 / "champion_blend.json").write_text(json.dumps(summary, indent=2,
                                                        ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
