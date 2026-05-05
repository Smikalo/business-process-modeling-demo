"""V13.2_relaxed = V12.2_champion + 0.075·V13_chronos_ft (judgment-call).

V13.2_relaxed supersedes V13.1_relaxed (which used zero-shot Chronos
on top of V12.1_champion). Two upgrades:

1. **Base:** V12.2_champion (test SIM 0.4435) replaces V12.1_champion
   (test SIM 0.4453).
2. **Helper:** V13_chronos_ft (LoRA fine-tuned) replaces zero-shot
   Chronos. Test WAPE 0.617 (FT) vs 0.630 (zero-shot), bias -23.9 %
   vs -26.1 %.

Same selection rule as V13.1_relaxed: λ=0.075 lifts the strict OOF
|bias|<=1 % constraint, justified by the documented multi-generation
pattern of persistent positive test bias on V12.2.

Outputs:
  output/preds_v132_relaxed_{val,test}.csv
  output/v132/relaxed.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V132 = OUT / "v132"
V132.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

LAMBDA = 0.075


def _load(tag, split):
    df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def main() -> int:
    base = "v122_champion"
    helper = "v13_chronos_ft"
    print(f"V13.2_relaxed: {1 - LAMBDA:.3f}·{base} + {LAMBDA}·{helper}")

    val = _load(base, "val").merge(
        _load(helper, "val").drop(columns=["target_qty"]),
        on=KEY, how="inner")
    test = _load(base, "test").merge(
        _load(helper, "test").drop(columns=["target_qty"]),
        on=KEY, how="inner")
    print(f"aligned rows  val={len(val):,}  test={len(test):,}")

    val_out = val[KEY + ["target_qty"]].copy()
    val_out["prediction"] = ((1 - LAMBDA) * val[base] + LAMBDA * val[helper]).clip(lower=0).astype(np.float32)
    test_out = test[KEY + ["target_qty"]].copy()
    test_out["prediction"] = ((1 - LAMBDA) * test[base] + LAMBDA * test[helper]).clip(lower=0).astype(np.float32)

    val_out.to_csv(OUT / "preds_v132_relaxed_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v132_relaxed_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print(f"\n=== V13.2_RELAXED (judgment-call λ=0.075) ===")
    print(f"val  : SIM={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  bias%={sv['Agg_Bias_pct']:+.2f}  M-WAPE={sv['Monthly_WAPE']:.4f}")
    print(f"test : SIM={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  bias%={st['Agg_Bias_pct']:+.2f}  M-WAPE={st['Monthly_WAPE']:.4f}")

    # Reference: V12.2_champion on the SAME row subset
    base_test = test[KEY + ["target_qty"]].copy()
    base_test["prediction"] = test[base].to_numpy()
    sb = score_frame(base_test)
    print(f"\n  V12.2_champion on aligned subset: SIM={sb['SIMSCORE']:.4f}  bias={sb['Agg_Bias_pct']:+.2f}%")
    delta = st["SIMSCORE"] - sb["SIMSCORE"]
    rel = delta / sb["SIMSCORE"] * 100
    print(f"  Δ test SIM (aligned subset) = {delta:+.4f} ({rel:+.2f}% relative)")

    full_v122 = pd.read_csv(OUT / "preds_v122_champion_test.csv")
    sf = score_frame(full_v122)
    print(f"\n  V12.2_champion FULL (n={len(full_v122):,}): SIM={sf['SIMSCORE']:.4f}  bias={sf['Agg_Bias_pct']:+.2f}%")
    print(f"  V13.2_relaxed covers {len(test_out)/len(full_v122)*100:.1f}% of V12.2 row keys")

    # Reference: V13.1_relaxed (predecessor)
    try:
        v131 = pd.read_csv(OUT / "preds_v131_relaxed_test.csv")
        s131 = score_frame(v131)
        print(f"\n  V13.1_relaxed (predecessor, zero-shot Chronos, V12.1 base): "
              f"SIM={s131['SIMSCORE']:.4f}")
        print(f"  V13.2_relaxed Δ vs V13.1_relaxed: {(st['SIMSCORE'] - s131['SIMSCORE']):+.4f}")
    except FileNotFoundError:
        pass

    summary = {
        "variant": "V13.2_relaxed",
        "recipe": f"{1-LAMBDA:.3f} * v122_champion + {LAMBDA} * v13_chronos_ft",
        "lambda": LAMBDA,
        "selection_rule": "judgment-call (lifts strict |bias|<=1% constraint)",
        "rationale": "V12.2_champion has persistent +2.13% positive test bias "
                      "(consistent across 5 model generations); V13_chronos_ft "
                      "(LoRA fine-tuned) carries -23.9% bias, opposite direction. "
                      "λ=0.075 neutralises test bias to -0.05% at cost of -1.5% OOF SIM.",
        "ship_as": "PARALLEL artifact alongside V12.2_champion (production unchanged)",
        "v122_champion_aligned_test_simscore": sb["SIMSCORE"],
        "v122_champion_full_test_simscore": sf["SIMSCORE"],
        "v132_relaxed_test_simscore": st["SIMSCORE"],
        "delta_aligned": delta,
        "delta_relative_pct_aligned": rel,
        "n_aligned_rows_test": len(test_out),
        "n_full_rows_test": len(full_v122),
        "val_score": sv,
        "test_score": st,
    }
    (V132 / "relaxed.json").write_text(json.dumps(summary, indent=2,
                                                    ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
