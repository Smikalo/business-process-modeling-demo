"""V11 -- triple-blend with Chronos (V11_LAD + a·V11_g93 + b·V11_chronos).

Extends `v11_final_blend.py` to a two-helper search.  Same rolling-origin
folds + recency weighting + |OOF_bias%| ≤ 1.0 (relaxed to ≤ 1.5 if no
candidate fits, mirroring the V11 LAD logic).

Outputs:
* `output/preds_v11_chronos_blend_{val,test}.csv`  (only if it improves
  V11_final on OOF_recency)
* `output/v11/chronos_blend_grid.csv`
* `output/v11/chronos_blend_summary.json`
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

BASE_TAG = "v11_lad"
HELP_A_TAG = "v11_g93"
HELP_B_TAG = "v11_chronos"


def _load(tag: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def main() -> int:
    val = (_load(BASE_TAG, "val")
           .merge(_load(HELP_A_TAG, "val").drop(columns=["target_qty"]),
                  on=KEY, how="inner")
           .merge(_load(HELP_B_TAG, "val").drop(columns=["target_qty"]),
                  on=KEY, how="inner"))
    test = (_load(BASE_TAG, "test")
            .merge(_load(HELP_A_TAG, "test").drop(columns=["target_qty"]),
                   on=KEY, how="inner")
            .merge(_load(HELP_B_TAG, "test").drop(columns=["target_qty"]),
                   on=KEY, how="inner"))
    print(f"loaded val={len(val)}  test={len(test)}")

    A_GRID = np.arange(0.00, 0.41, 0.025)
    B_GRID = np.arange(0.00, 0.21, 0.025)

    rows = []
    for a in A_GRID:
        for b in B_GRID:
            if a + b >= 1.0:
                continue
            oof_sims, oof_bias = [], []
            for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
                te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
                te["prediction"] = ((1 - a - b) * te[BASE_TAG] +
                                    a * te[HELP_A_TAG] +
                                    b * te[HELP_B_TAG]).clip(lower=0).astype(np.float32)
                sc = score_frame(te[KEY + ["target_qty", "prediction"]])
                oof_sims.append(sc["SIMSCORE"])
                oof_bias.append(sc["Agg_Bias_pct"])
            oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
            oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
            rows.append({
                "a": round(float(a), 3), "b": round(float(b), 3),
                "OOF_recency": round(oof_recency, 4),
                "OOF_bias_recency_pct": round(oof_bias_recency, 3),
                "OOF_folds": [round(x, 4) for x in oof_sims],
            })

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(V11 / "chronos_blend_grid.csv", index=False)
    print(f"\n=== Grid search ({len(df_grid)} candidates) ===")
    print("\nTop-15 by OOF_recency (no bias filter):")
    print(df_grid.sort_values("OOF_recency").head(15).to_string(index=False))

    for ceil in (1.0, 1.5, 2.0):
        surv = df_grid[df_grid["OOF_bias_recency_pct"].abs() <= ceil]
        if len(surv) > 0:
            BIAS_CEIL = ceil
            print(f"\nUsing bias ceiling = {ceil}%  ({len(surv)} candidates pass)")
            break
    else:
        surv = df_grid
        BIAS_CEIL = float("inf")

    champ_idx = surv["OOF_recency"].idxmin()
    champ = surv.loc[champ_idx]
    a_star = float(champ["a"])
    b_star = float(champ["b"])
    print(f"\nChampion: a={a_star:.3f}  b={b_star:.3f}  "
          f"OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}")

    val_out = val[KEY + ["target_qty"]].copy()
    val_out["prediction"] = ((1 - a_star - b_star) * val[BASE_TAG] +
                             a_star * val[HELP_A_TAG] +
                             b_star * val[HELP_B_TAG]).clip(lower=0).astype(np.float32)
    test_out = test[KEY + ["target_qty"]].copy()
    test_out["prediction"] = ((1 - a_star - b_star) * test[BASE_TAG] +
                              a_star * test[HELP_A_TAG] +
                              b_star * test[HELP_B_TAG]).clip(lower=0).astype(np.float32)

    sv = score_frame(val_out)
    st = score_frame(test_out)

    v11_final_val = score_frame(pd.read_csv(OUT / "preds_v11_final_val.csv"))
    v11_final_test = score_frame(pd.read_csv(OUT / "preds_v11_final_test.csv"))

    print(f"\n=== Existing V11 FINAL (a=0.225, b=0.000) ===")
    print(f"  val   SIMSCORE={v11_final_val['SIMSCORE']:.4f}  bias%={v11_final_val['Agg_Bias_pct']:+.2f}")
    print(f"  test  SIMSCORE={v11_final_test['SIMSCORE']:.4f}  bias%={v11_final_test['Agg_Bias_pct']:+.2f}")
    print(f"\n=== NEW V11 + Chronos (a={a_star:.3f}, b={b_star:.3f}) ===")
    print(f"  val   SIMSCORE={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  "
          f"bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"  test  SIMSCORE={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  "
          f"bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")

    improved = champ["OOF_recency"] < 0.4119  # V11 final's OOF
    if improved:
        val_out.to_csv(OUT / "preds_v11_chronos_blend_val.csv", index=False)
        test_out.to_csv(OUT / "preds_v11_chronos_blend_test.csv", index=False)
        print(f"\n  -> Chronos blend improves OOF — saved as preds_v11_chronos_blend_*")
    else:
        print(f"\n  -> Chronos blend does NOT improve OOF over V11_final "
              f"(this OOF={champ['OOF_recency']:.4f} vs V11_final OOF=0.4119)")

    summary = {
        "champion_a": a_star, "champion_b": b_star,
        "OOF_recency": float(champ["OOF_recency"]),
        "OOF_bias_pct": float(champ["OOF_bias_recency_pct"]),
        "bias_ceiling_used": BIAS_CEIL,
        "blend_val": sv,
        "blend_test": st,
        "v11_final_val": v11_final_val,
        "v11_final_test": v11_final_test,
        "improved_over_v11_final": bool(improved),
    }
    (V11 / "chronos_blend_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\nwrote v11/chronos_blend_grid.csv  +  v11/chronos_blend_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
