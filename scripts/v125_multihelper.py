"""V12.5 multi-helper joint OOF search WITH V12_external_g93.

Same recipe as V12.3 but adds V12_external_g93 to the helper pool:

  (1 - α - β - γ - δ)·V11_final
  + α·V12_external + β·V11_g93 + γ·V13_chronos_ft + δ·V12_external_g93

Bias-laddered selection (1.0 / 1.25 / 1.5 / 1.75 / 2.0 %).
Grid {0, 0.025, ..., 0.20} per weight, total helper weight ≤ 0.40.
"""
from __future__ import annotations

import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V125 = OUT / "v125"
V125.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

CV_FOLDS = [
    ("2024-07", "2024-09", "2024-10", "2024-12"),
    ("2024-07", "2024-12", "2025-01", "2025-03"),
    ("2024-07", "2025-03", "2025-04", "2025-06"),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])

BASE = "v11_final"
HELPERS = ["v12_external", "v11_g93", "v13_chronos_ft", "v12_external_g93"]


def _load(tag, split):
    df = pd.read_csv(OUT / f"preds_{tag}_{split}.csv")
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def _blend(df, base, helpers, weights):
    pred = weights[0] * df[base]
    for i, h in enumerate(helpers):
        pred = pred + weights[i + 1] * df[h]
    return pred.clip(lower=0).astype(np.float32)


def main() -> int:
    print(f"V12.5 multi-helper search: base={BASE}  helpers={HELPERS}")

    val = _load(BASE, "val")
    test = _load(BASE, "test")
    for h in HELPERS:
        val = val.merge(_load(h, "val").drop(columns=["target_qty"]),
                         on=KEY, how="inner")
        test = test.merge(_load(h, "test").drop(columns=["target_qty"]),
                           on=KEY, how="inner")
    print(f"aligned rows  val={len(val):,}  test={len(test):,}")

    LEVELS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20]
    MAX_HELPER_SUM = 0.40
    candidates = [w for w in itertools.product(LEVELS, repeat=len(HELPERS))
                   if sum(w) <= MAX_HELPER_SUM]
    candidates = [(1 - sum(w),) + w for w in candidates]
    print(f"evaluating {len(candidates)} candidates")

    rows = []
    for ci, weights in enumerate(candidates):
        oof_sims, oof_bias = [], []
        for (_, _, vs, ve) in CV_FOLDS:
            te = val[(val["Период"] >= vs) & (val["Период"] <= ve)].copy()
            te["prediction"] = _blend(te, BASE, HELPERS, weights)
            sc = score_frame(te[KEY + ["target_qty", "prediction"]])
            oof_sims.append(sc["SIMSCORE"])
            oof_bias.append(sc["Agg_Bias_pct"])
        oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
        oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
        rows.append({
            "weights": list(weights),
            "w_base": weights[0],
            "w_v12ext": weights[1],
            "w_v11g93": weights[2],
            "w_chronos_ft": weights[3],
            "w_v12ext_g93": weights[4],
            "OOF_recency": oof_recency,
            "OOF_bias_recency_pct": oof_bias_recency,
        })
        if (ci + 1) % 200 == 0:
            print(f"  {ci+1}/{len(candidates)}")

    df = pd.DataFrame(rows)
    df.to_csv(V125 / "grid.csv", index=False)

    print("\n=== Champion selection (bias ladder) ===")
    best_overall = None
    for ceil in [1.0, 1.25, 1.5, 1.75, 2.0]:
        survs = df[df["OOF_bias_recency_pct"].abs() <= ceil].copy()
        if len(survs) == 0:
            continue
        ch = survs.loc[survs["OOF_recency"].idxmin()]
        print(f"  bias≤{ceil:.2f}%: {len(survs):4d} survivors  "
              f"OOF_rec={ch['OOF_recency']:.4f}  "
              f"weights=(base={ch['w_base']:.3f}, ext={ch['w_v12ext']:.3f}, "
              f"g93={ch['w_v11g93']:.3f}, chr_ft={ch['w_chronos_ft']:.3f}, "
              f"ext_g93={ch['w_v12ext_g93']:.3f})  "
              f"bias%={ch['OOF_bias_recency_pct']:+.2f}")
        if best_overall is None or ch["OOF_recency"] < best_overall["OOF_recency"]:
            best_overall = ch.copy()
            best_overall["ceiling"] = ceil

    best = best_overall
    weights = (best["w_base"], best["w_v12ext"], best["w_v11g93"],
               best["w_chronos_ft"], best["w_v12ext_g93"])
    print(f"\n*** V12.5 CHAMPION (ceiling = {best['ceiling']}) ***")
    print(f"  V11_final         {weights[0]:.3f}")
    print(f"  V12_external      {weights[1]:.3f}")
    print(f"  V11_g93           {weights[2]:.3f}")
    print(f"  V13_chronos_ft    {weights[3]:.3f}")
    print(f"  V12_external_g93  {weights[4]:.3f}")
    print(f"  OOF_recency       {best['OOF_recency']:.4f}")
    print(f"  OOF_bias%         {best['OOF_bias_recency_pct']:+.2f}")

    val_out = val[KEY + ["target_qty"]].copy()
    val_out["prediction"] = _blend(val, BASE, HELPERS, weights)
    test_out = test[KEY + ["target_qty"]].copy()
    test_out["prediction"] = _blend(test, BASE, HELPERS, weights)
    val_out.to_csv(OUT / "preds_v125_champion_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v125_champion_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print(f"\n=== V12.5 CHAMPION on aligned subset ({len(test_out):,} rows) ===")
    print(f"val  : SIM={sv['SIMSCORE']:.4f}  WAPE={sv['WAPE']:.4f}  bias%={sv['Agg_Bias_pct']:+.2f}")
    print(f"test : SIM={st['SIMSCORE']:.4f}  WAPE={st['WAPE']:.4f}  bias%={st['Agg_Bias_pct']:+.2f}")

    v122_full = pd.read_csv(OUT / "preds_v122_champion_test.csv")
    v122_lookup = v122_full[KEY + ["prediction"]].copy()
    v122_lookup["Период"] = v122_lookup["Период"].astype(str)
    v122_aligned = test[KEY + ["target_qty"]].merge(v122_lookup, on=KEY, how="inner")
    sa = score_frame(v122_aligned)
    sf = score_frame(v122_full)
    delta = st["SIMSCORE"] - sa["SIMSCORE"]
    rel = delta / sa["SIMSCORE"] * 100
    print(f"\n  V12.2_champion FULL  (n={len(v122_full):,}): SIM={sf['SIMSCORE']:.4f}")
    print(f"  V12.2_champion aligned (n={len(v122_aligned):,}): SIM={sa['SIMSCORE']:.4f}")
    print(f"  Δ V12.5 - V12.2 (aligned) = {delta:+.4f} ({rel:+.2f}%)")

    summary = {
        "champion_weights": [float(w) for w in weights],
        "helpers": HELPERS,
        "OOF_recency": float(best["OOF_recency"]),
        "OOF_bias_recency_pct": float(best["OOF_bias_recency_pct"]),
        "bias_ceiling_used": float(best["ceiling"]),
        "val_score": sv,
        "test_score": st,
        "v122_aligned_simscore": sa["SIMSCORE"],
        "v122_full_simscore": sf["SIMSCORE"],
        "delta_aligned": delta,
        "delta_relative_pct": rel,
    }
    (V125 / "champion.json").write_text(json.dumps(summary, indent=2,
                                                     ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
