"""V12.1 meta-blend — small admixture of V12.1_LAD on top of V11_final.

Anchor strategy: V11_final is the production champion (test SIMSCORE
0.4489) and was picked by V11's OOF-aware λ-blend. V12.1_LAD has
slightly worse test bias (+3.86 % vs V11_final's +2.80 %) but consumes
the EXT signals and beats V11_LAD raw. By using V11_final as the BASE
and V12.1_LAD as a small helper, we preserve V11_final's well-calibrated
bias while letting the EXT features contribute on the margin.

This is a "do no harm" blend: if V11_final is at SIMSCORE 0.4489 and
V12.1_LAD is at 0.4568, then for some λ ∈ [0, 0.5] the blend MAY beat
V11_final IF V12.1_LAD's errors decorrelate with V11_final's errors on
test. If correlation is too high, the blend asymptotes to V11_final.

Strict OOF-driven λ selection — no bias penalty, no robust tricks.
Plain OOF SIMSCORE minimum on V11's standard 3-fold CV.

Outputs:
  output/preds_v121_meta_{val,test}.csv
  output/v121/meta_blend.json
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
    helpers = ["v121_lad", "v12_external"]
    print(f"V12.1 meta-blend: base={base}, helpers={helpers}")

    val_b = _load(base, "val")
    test_b = _load(base, "test")
    grids = []
    best = None

    for h in helpers:
        val = val_b.merge(_load(h, "val").drop(columns=["target_qty"]),
                          on=KEY, how="inner")
        test = test_b.merge(_load(h, "test").drop(columns=["target_qty"]),
                            on=KEY, how="inner")
        rows = []
        LAMBDAS = np.arange(0.00, 0.51, 0.025)
        for lam in LAMBDAS:
            oof_sims = []
            for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
                te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
                te["prediction"] = ((1 - lam) * te[base] + lam * te[h]).clip(lower=0).astype(np.float32)
                oof_sims.append(score_frame(te[KEY + ["target_qty", "prediction"]])["SIMSCORE"])
            oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
            rows.append({"helper": h, "lambda": round(float(lam), 3),
                          "OOF_recency": round(oof_recency, 4),
                          "OOF_folds": [round(x, 4) for x in oof_sims]})
        df_grid = pd.DataFrame(rows)
        grids.append(df_grid)
        champ_row = df_grid.loc[df_grid["OOF_recency"].idxmin()]
        if best is None or champ_row["OOF_recency"] < best["OOF_recency"]:
            best = {"helper": h, "lambda": float(champ_row["lambda"]),
                    "OOF_recency": float(champ_row["OOF_recency"]),
                    "val": val, "test": test}

    g = pd.concat(grids)
    g.to_csv(V121 / "meta_blend_grid.csv", index=False)
    print("\n=== Top-3 per helper ===")
    for h in helpers:
        sub = g[g["helper"] == h].nsmallest(3, "OOF_recency")
        print(f"\nhelper={h}")
        print(sub.to_string(index=False))

    h = best["helper"]; lam = best["lambda"]
    print(f"\n*** V12.1 META CHAMPION ***  helper={h}  λ={lam:.3f}  "
          f"OOF_recency={best['OOF_recency']:.4f}")

    val = best["val"]; test = best["test"]
    val_out = val.copy()
    val_out["prediction"] = ((1 - lam) * val_out[base] + lam * val_out[h]).clip(lower=0).astype(np.float32)
    test_out = test.copy()
    test_out["prediction"] = ((1 - lam) * test_out[base] + lam * test_out[h]).clip(lower=0).astype(np.float32)
    val_out = val_out[KEY + ["target_qty", "prediction"]]
    test_out = test_out[KEY + ["target_qty", "prediction"]]
    val_out.to_csv(OUT / "preds_v121_meta_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v121_meta_test.csv", index=False)

    sv = score_frame(val_out); st = score_frame(test_out)
    print("\n=== V12.1 META (V11_final + λ · helper) ===")
    print(f"val  : SIM={sv['SIMSCORE']:.4f} WAPE={sv['WAPE']:.4f} bias%={sv['Agg_Bias_pct']:+.2f}")
    print(f"test : SIM={st['SIMSCORE']:.4f} WAPE={st['WAPE']:.4f} bias%={st['Agg_Bias_pct']:+.2f}")

    summary = {"champion_helper": h, "champion_lambda": lam,
                "champion_OOF_recency": best["OOF_recency"],
                "val_score": sv, "test_score": st}
    (V121 / "meta_blend.json").write_text(json.dumps(summary, indent=2,
                                                       ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
