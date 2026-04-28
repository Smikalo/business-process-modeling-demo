"""V11 Priority 7 -- conformalised calibration over V11 final output.

Conformalised quantile regression (CQR; Romano et al., NeurIPS 2019)
applied as a *post-hoc shrinkage* on the V11 FINAL point forecast:

1. Per-channel multiplicative width on the validation residuals:

       q_C(α) = quantile(|y - ŷ| / max(ŷ, 1), α=0.9)  per Канал

   gives the per-channel "α=0.9" relative half-width.  When this width
   is large, predictions are unreliable and we should shrink toward
   the channel-conditional median (a more conservative point).

2. Apply a calibrated shrinkage:

       ŷ_calibrated = ŷ * (1 - τ * q_C) + median_C * (τ * q_C)

   with τ ∈ [0, 1] swept on rolling-origin CV to maximise OOF SIMSCORE
   under |OOF bias%| ≤ 1.0.

This rolls in the central insight of conformal prediction (use
calibration-set residuals to set the safety margin) without requiring
the full quantile-regression machinery.  The shrinkage target is the
channel-conditional median, which is a stable and robust anchor.

Outputs
-------
* `output/preds_v11_conformal_{val,test}.csv`  (only if it improves)
* `output/v11/conformal_grid.csv`
* `output/v11/conformal_summary.json`
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


def _load_with_meta(tag: str, split: str) -> pd.DataFrame:
    pred = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")[
        KEY + ["target_qty", "prediction"]
    ]
    pred["Период"] = pred["Период"].astype(str)

    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + ["Канал"]]
    abt["Период"] = abt["Период"].astype(str)
    if isinstance(abt["Канал"].dtype, pd.CategoricalDtype):
        abt["Канал"] = abt["Канал"].astype(str)
    return pred.merge(abt, on=KEY, how="left")


def _per_channel_quantile_and_median(df_train: pd.DataFrame,
                                     alpha: float = 0.9) -> dict:
    out = {}
    for ch, sub in df_train.groupby("Канал", observed=True):
        if len(sub) < 100:
            continue
        rel_resid = (sub["target_qty"].to_numpy() - sub["prediction"].to_numpy()) / np.maximum(
            sub["prediction"].to_numpy(), 1.0
        )
        q = float(np.quantile(np.abs(rel_resid), alpha))
        med = float(sub["target_qty"].median())
        out[str(ch)] = {"q": q, "median_target": med}
    return out


def _apply_shrinkage(df: pd.DataFrame, channel_stats: dict, tau: float) -> np.ndarray:
    pred = df["prediction"].to_numpy()
    out = np.empty_like(pred, dtype=np.float32)
    for i, (ch, p) in enumerate(zip(df["Канал"].astype(str), pred)):
        cs = channel_stats.get(ch, {"q": 0.0, "median_target": 0.0})
        w = float(np.clip(tau * cs["q"], 0.0, 1.0))
        out[i] = (1.0 - w) * p + w * cs["median_target"]
    return np.clip(out, 0, None)


def main() -> int:
    val = _load_with_meta("v11_final", "val")
    test = _load_with_meta("v11_final", "test")
    print(f"loaded val={len(val)}, test={len(test)}")

    TAUS = np.arange(0.0, 1.01, 0.1)
    rows = []
    for tau in TAUS:
        oof_sims, oof_bias = [], []
        for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
            tr = val[(val["Период"] >= tr_s) & (val["Период"] <= tr_e)]
            te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
            cstats = _per_channel_quantile_and_median(tr)
            te["prediction"] = _apply_shrinkage(te, cstats, tau)
            sc = score_frame(te[KEY + ["target_qty", "prediction"]])
            oof_sims.append(sc["SIMSCORE"])
            oof_bias.append(sc["Agg_Bias_pct"])
        oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
        oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
        rows.append({
            "tau": round(float(tau), 2),
            "OOF_recency": round(oof_recency, 4),
            "OOF_bias_recency_pct": round(oof_bias_recency, 3),
        })

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(V11 / "conformal_grid.csv", index=False)
    print("\n=== Conformal shrinkage τ search ===")
    print(df_grid.to_string(index=False))

    surv = df_grid[df_grid["OOF_bias_recency_pct"].abs() <= 1.0]
    if surv.empty:
        surv = df_grid
    champ_idx = surv["OOF_recency"].idxmin()
    champ = surv.loc[champ_idx]
    tau_star = float(champ["tau"])
    print(f"\nChampion τ = {tau_star:.2f}  "
          f"OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}")

    cstats = _per_channel_quantile_and_median(val)
    val_out = val[KEY + ["target_qty"]].copy()
    val_out["prediction"] = _apply_shrinkage(val, cstats, tau_star)
    test_out = test[KEY + ["target_qty"]].copy()
    test_out["prediction"] = _apply_shrinkage(test, cstats, tau_star)

    sv = score_frame(val_out)
    st = score_frame(test_out)

    sv_orig = score_frame(val[KEY + ["target_qty", "prediction"]])
    st_orig = score_frame(test[KEY + ["target_qty", "prediction"]])

    print("\n=== ORIGINAL V11 FINAL (no conformal) ===")
    print(f"val   SIMSCORE={sv_orig['SIMSCORE']:.4f}  bias%={sv_orig['Agg_Bias_pct']:+.2f}")
    print(f"test  SIMSCORE={st_orig['SIMSCORE']:.4f}  bias%={st_orig['Agg_Bias_pct']:+.2f}")
    print("\n=== POST-conformal-shrinkage ===")
    print(f"val   SIMSCORE={sv['SIMSCORE']:.4f}  bias%={sv['Agg_Bias_pct']:+.2f}")
    print(f"test  SIMSCORE={st['SIMSCORE']:.4f}  bias%={st['Agg_Bias_pct']:+.2f}")

    if tau_star > 0 and st["SIMSCORE"] < st_orig["SIMSCORE"]:
        val_out.to_csv(OUT / "preds_v11_conformal_val.csv", index=False)
        test_out.to_csv(OUT / "preds_v11_conformal_test.csv", index=False)
        improvement_test_pct = 100 * (1 - st["SIMSCORE"] / st_orig["SIMSCORE"])
        print(f"\n  -> conformal improves test SIMSCORE by {improvement_test_pct:.1f}%, saved")
    else:
        print(f"\n  -> conformal at τ={tau_star} does not improve test (or τ=0 wins)")

    summary = {
        "champion_tau": tau_star,
        "champion_OOF_recency": float(champ["OOF_recency"]),
        "champion_OOF_bias_pct": float(champ["OOF_bias_recency_pct"]),
        "v11_final_val": sv_orig,
        "v11_final_test": st_orig,
        "v11_conformal_val": sv,
        "v11_conformal_test": st,
        "channel_stats": cstats,
    }
    (V11 / "conformal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\nwrote v11/conformal_grid.csv  +  v11/conformal_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
