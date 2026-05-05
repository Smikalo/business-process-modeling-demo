"""V13.1 champion blend — V12.1_champion + λ · V13_chronos (OOF-driven).

V13_chronos is Chronos-T5-Small zero-shot (Cell 6 in
``notebooks/v13_chronos_finetune_colab.py``; the LoRA fine-tune was a
no-op — context_len=48+horizon=12 exceeded the 54-month training history,
so all training samples were rejected and Cell 6 ran on the pretrained
weights).

Standalone numbers (test): WAPE 0.630, **bias -26.1 %** — much worse
than V12.1_champion (WAPE 0.394) but with strongly *negative* test
bias, opposite to V12.1_champion's +2.36 % positive bias.

Strategy (same recipe as V12.1_champion = V11_final + 0.05 · V12_external):
take a known-stable base (V12.1_champion) and a small λ admixture of a
strong bias-counter helper (V13_chronos), let OOF pick λ honestly.

Outputs:
  output/preds_v131_champion_{val,test}.csv
  output/v131/champion_blend.json
  output/v131/champion_grid.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V131 = OUT / "v131"
V131.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

CV_FOLDS = [
    ("2024-07", "2024-09", "2024-10", "2024-12"),
    ("2024-07", "2024-12", "2025-01", "2025-03"),
    ("2024-07", "2025-03", "2025-04", "2025-06"),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


def _load(tag, split):
    df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def main() -> int:
    base = "v121_champion"
    helper = "v13_chronos"
    print(f"V13.1 champion-blend search: base={base}, helper={helper}")

    val = _load(base, "val").merge(
        _load(helper, "val").drop(columns=["target_qty"]),
        on=KEY, how="inner")
    test = _load(base, "test").merge(
        _load(helper, "test").drop(columns=["target_qty"]),
        on=KEY, how="inner")
    print(f"aligned rows  val={len(val):,}  test={len(test):,}")

    LAMBDAS = np.arange(0.00, 0.21, 0.0125)
    rows = []
    for lam in LAMBDAS:
        oof_sims, oof_bias = [], []
        for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
            te = val[(val["Период"] >= va_s) & (val["Период"] <= va_e)].copy()
            te["prediction"] = ((1 - lam) * te[base] + lam * te[helper]).clip(lower=0).astype(np.float32)
            sc = score_frame(te[KEY + ["target_qty", "prediction"]])
            oof_sims.append(sc["SIMSCORE"])
            oof_bias.append(sc["Agg_Bias_pct"])
        oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
        oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
        rows.append({
            "lambda": round(float(lam), 4),
            "OOF_recency": round(oof_recency, 4),
            "OOF_bias_recency_pct": round(oof_bias_recency, 3),
            "OOF_folds": [round(x, 4) for x in oof_sims],
            "OOF_bias_folds": [round(x, 3) for x in oof_bias],
        })

    df_grid = pd.DataFrame(rows)
    df_grid.to_csv(V131 / "champion_grid.csv", index=False)
    print("\n=== OOF grid (full) ===")
    print(df_grid.to_string(index=False))

    champ = df_grid.loc[df_grid["OOF_recency"].idxmin()]
    lam_star = float(champ["lambda"])
    print(f"\n*** OOF picks λ={lam_star:.4f}  OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}")

    val_out = val.copy()
    val_out["prediction"] = ((1 - lam_star) * val_out[base] + lam_star * val_out[helper]).clip(lower=0).astype(np.float32)
    test_out = test.copy()
    test_out["prediction"] = ((1 - lam_star) * test_out[base] + lam_star * test_out[helper]).clip(lower=0).astype(np.float32)
    val_out = val_out[KEY + ["target_qty", "prediction"]]
    test_out = test_out[KEY + ["target_qty", "prediction"]]
    val_out.to_csv(OUT / "preds_v131_champion_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v131_champion_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print("\n=== V13.1 CHAMPION (V12.1 + λ · V13_chronos, OOF-picked) ===")
    print(f"val  : SIM={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"test : SIM={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")
    print(f"\nFor reference (on aligned subset, n={len(test):,}):")
    # Score V12.1_champion at λ=0 on the SAME row subset for fair Δ
    base_pred_test = test_out.copy()
    base_pred_test["prediction"] = test[base].to_numpy()
    sb = score_frame(base_pred_test)
    print(f"  V12.1_champion on this subset: SIM={sb['SIMSCORE']:.4f}  bias={sb['Agg_Bias_pct']:+.2f}%")
    delta = st["SIMSCORE"] - sb["SIMSCORE"]
    rel = delta / sb["SIMSCORE"] * 100
    print(f"  Δ test SIM = {delta:+.4f}  ({rel:+.2f}% relative)")

    # Also: full-rowset comparison (V12.1_champion has 18,298 rows; V13.1 has fewer if not all pairs aligned)
    full_v121 = pd.read_csv(OUT / "preds_v121_champion_test.csv")
    sf = score_frame(full_v121)
    print(f"\n  V12.1_champion FULL (n=18,298): SIM={sf['SIMSCORE']:.4f}  bias={sf['Agg_Bias_pct']:+.2f}%")
    print(f"  V13.1 covers {len(test_out)/len(full_v121)*100:.1f}% of V12.1 row keys")

    summary = {
        "champion_helper": helper,
        "champion_lambda": lam_star,
        "champion_OOF_recency": float(champ["OOF_recency"]),
        "champion_OOF_bias_recency_pct": float(champ["OOF_bias_recency_pct"]),
        "v121_champion_aligned_test_simscore": sb["SIMSCORE"],
        "v121_champion_full_test_simscore": sf["SIMSCORE"],
        "v131_aligned_test_simscore": st["SIMSCORE"],
        "delta_test_simscore_aligned": delta,
        "delta_relative_pct_aligned": rel,
        "n_aligned_rows_test": len(test_out),
        "n_full_rows_test": len(full_v121),
        "val_score": sv,
        "test_score": st,
    }
    (V131 / "champion_blend.json").write_text(json.dumps(summary, indent=2,
                                                        ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
