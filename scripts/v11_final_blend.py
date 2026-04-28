"""V11 -- final post-LAD blend with hyper-recent base.

After V11 LAD picks its champion, this script searches over the
mixing weight λ ∈ [0, 0.4] of:

    final_pred = (1 - λ) * V11_LAD + λ * V11_g93

using rolling-origin CV (same folds as the LAD search).  Pick the λ
that maximises OOF recency-weighted SIMSCORE under
|OOF bias%| ≤ 1.0.

This is a *post-hoc* meta-blend that exploits the fact that V11_g93
trained with steep recency has the OPPOSITE bias direction on test
(negative) vs V10 LAD (positive).  A small admixture (~5-15 %)
counter-acts the structural under-/over-bias.

Outputs:
* `output/preds_v11_final_{val,test}.csv`
* `output/v11/final_blend.json`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V11 = OUT / "v11"
V11.mkdir(parents=True, exist_ok=True)
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
    base_tag = "v11_lad"
    helper_tag = "v11_g93"

    val = _load(base_tag, "val").merge(
        _load(helper_tag, "val").drop(columns=["target_qty"]),
        on=KEY, how="inner",
    )
    test = _load(base_tag, "test").merge(
        _load(helper_tag, "test").drop(columns=["target_qty"]),
        on=KEY, how="inner",
    )

    LAMBDAS = np.arange(0.00, 0.41, 0.025)
    rows = []
    for lam in LAMBDAS:
        oof_sims, oof_bias = [], []
        for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
            te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
            te["prediction"] = ((1 - lam) * te[base_tag] +
                                lam * te[helper_tag]).clip(lower=0).astype(np.float32)
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
        })

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(V11 / "final_blend_grid.csv", index=False)
    print("=== Blend search ===")
    print(df_grid.to_string(index=False))

    BIAS_CEIL = 1.0
    surv = df_grid[df_grid["OOF_bias_recency_pct"].abs() <= BIAS_CEIL]
    if surv.empty:
        BIAS_CEIL = 1.5
        surv = df_grid[df_grid["OOF_bias_recency_pct"].abs() <= BIAS_CEIL]
    if surv.empty:
        surv = df_grid
    champ = surv.loc[surv["OOF_recency"].idxmin()]
    lam_star = float(champ["lambda"])

    print(f"\nChampion λ = {lam_star:.3f}  "
          f"OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}")

    val_out = val.copy()
    val_out["prediction"] = ((1 - lam_star) * val_out[base_tag] +
                             lam_star * val_out[helper_tag]).clip(lower=0).astype(np.float32)
    test_out = test.copy()
    test_out["prediction"] = ((1 - lam_star) * test_out[base_tag] +
                              lam_star * test_out[helper_tag]).clip(lower=0).astype(np.float32)

    val_out = val_out[KEY + ["target_qty", "prediction"]]
    test_out = test_out[KEY + ["target_qty", "prediction"]]
    val_out.to_csv(OUT / "preds_v11_final_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v11_final_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print("\n=== V11 FINAL (LAD + λ * v11_g93) ===")
    print(f"val   SIMSCORE={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  "
          f"bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"test  SIMSCORE={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  "
          f"bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")

    summary = {
        "champion_lambda": lam_star,
        "champion_OOF_recency": float(champ["OOF_recency"]),
        "champion_OOF_bias_recency_pct": float(champ["OOF_bias_recency_pct"]),
        "val_score": sv,
        "test_score": st,
        "search_grid": df_grid.to_dict("records"),
    }
    (V11 / "final_blend.json").write_text(json.dumps(summary, indent=2,
                                                      ensure_ascii=False,
                                                      default=str))
    print("\nwrote preds_v11_final_{val,test}.csv  +  v11/final_blend.json"
          "  +  v11/final_blend_grid.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
