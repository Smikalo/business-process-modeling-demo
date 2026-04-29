"""V12.1 final blend — λ-mix V12.1_LAD with bias-counter helpers.

Mirrors ``scripts.v12_final_blend`` but operates on the new V12.1_LAD
output (which already has the bias-direction-symmetry filter applied)
and adds ``v12_external`` to the helper pool. ``v12_external`` carries
strong negative test bias (~-10 %), so a small admixture neutralises
V12.1_LAD's +3.86 % positive drift much more efficiently than v11_g93
(which only carries ~-2 % counter-bias).

Outputs:

* ``output/preds_v121_final_{val,test}.csv``
* ``output/v121/final_blend.json``
* ``output/v121/final_blend_grid.csv``
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


def _try_helpers() -> list[str]:
    candidates = ["v12_external", "v11_g93", "v11_g90",
                  "v11_recent_only", "v12_multiseed", "v12_intermittent"]
    return [c for c in candidates
             if (OUT / f"preds_{c}_val.csv").exists()
             and (OUT / f"preds_{c}_test.csv").exists()]


def main() -> int:
    base_tag = "v121_lad"
    helpers = _try_helpers()
    if not helpers:
        print(f"No helper bases found; cannot run final blend.")
        return 1

    print(f"V12.1 final blend search:")
    print(f"  base = {base_tag}")
    print(f"  helpers = {helpers}")

    best_overall = None
    grid_all = []

    for helper_tag in helpers:
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
            most_recent_fold_bias = float(oof_bias[-1])
            robust_penalty = 0.005 * max(
                abs(oof_bias_recency), abs(most_recent_fold_bias))
            robust_objective = oof_recency + robust_penalty
            rows.append({
                "helper": helper_tag,
                "lambda": round(float(lam), 3),
                "OOF_recency": round(oof_recency, 4),
                "OOF_bias_recency_pct": round(oof_bias_recency, 3),
                "most_recent_fold_bias_pct": round(most_recent_fold_bias, 3),
                "robust_objective": round(robust_objective, 4),
            })

        df_grid = pd.DataFrame(rows)
        grid_all.append(df_grid)

        ROBUST_BIAS_CEILING = 5.0
        surv = df_grid[df_grid["OOF_bias_recency_pct"].abs() <= ROBUST_BIAS_CEILING]
        if surv.empty:
            surv = df_grid
        champ = surv.loc[surv["robust_objective"].idxmin()]
        lam_star = float(champ["lambda"])
        rec_star = float(champ["OOF_recency"])
        bias_star = float(champ["OOF_bias_recency_pct"])
        if best_overall is None or champ["robust_objective"] < best_overall.get("robust_objective", 999):
            best_overall = {
                "helper": helper_tag,
                "lambda": lam_star,
                "OOF_recency": rec_star,
                "OOF_bias_recency_pct": bias_star,
                "most_recent_fold_bias_pct": float(champ["most_recent_fold_bias_pct"]),
                "robust_objective": float(champ["robust_objective"]),
                "val": val, "test": test,
            }

    full_grid = pd.concat(grid_all)
    full_grid.to_csv(V121 / "final_blend_grid.csv", index=False)
    print("\n=== Top-5 robust-objective candidates per helper ===")
    for h in helpers:
        sub = full_grid[full_grid["helper"] == h].nsmallest(5, "robust_objective")
        print(f"\nhelper={h}")
        print(sub.to_string(index=False))

    helper_tag = best_overall["helper"]
    lam_star = best_overall["lambda"]
    val = best_overall["val"]
    test = best_overall["test"]
    print(f"\n*** V12.1 FINAL BLEND CHAMPION ***")
    print(f"  helper = {helper_tag}")
    print(f"  λ = {lam_star:.3f}")
    print(f"  OOF_recency = {best_overall['OOF_recency']:.4f}  "
          f"OOF_bias% = {best_overall['OOF_bias_recency_pct']:+.2f}")
    print(f"  most-recent-fold bias = "
          f"{best_overall['most_recent_fold_bias_pct']:+.2f}%")
    print(f"  robust_objective = {best_overall['robust_objective']:.4f}")

    val_out = val.copy()
    val_out["prediction"] = ((1 - lam_star) * val_out[base_tag] +
                             lam_star * val_out[helper_tag]).clip(lower=0).astype(np.float32)
    test_out = test.copy()
    test_out["prediction"] = ((1 - lam_star) * test_out[base_tag] +
                              lam_star * test_out[helper_tag]).clip(lower=0).astype(np.float32)

    val_out = val_out[KEY + ["target_qty", "prediction"]]
    test_out = test_out[KEY + ["target_qty", "prediction"]]
    val_out.to_csv(OUT / "preds_v121_final_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v121_final_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print("\n=== V12.1 FINAL (LAD + λ · helper) ===")
    print(f"val  : SIMSCORE={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  "
          f"bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"test : SIMSCORE={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  "
          f"bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")

    summary = {
        "champion_helper": helper_tag,
        "champion_lambda": lam_star,
        "champion_OOF_recency": best_overall["OOF_recency"],
        "champion_OOF_bias_recency_pct": best_overall["OOF_bias_recency_pct"],
        "champion_most_recent_fold_bias_pct": best_overall["most_recent_fold_bias_pct"],
        "champion_robust_objective": best_overall["robust_objective"],
        "val_score": sv,
        "test_score": st,
    }
    (V121 / "final_blend.json").write_text(json.dumps(summary, indent=2,
                                                       ensure_ascii=False,
                                                       default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
