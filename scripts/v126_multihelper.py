"""V12.6 multi-helper joint OOF search WITH V14 GlobalNN.

Same recipe as V12.5 but adds V14_globalnn (Transformer-encoder
trained on Colab/Kaggle GPU) to the helper pool:

  (1 - α - β - γ - δ - ε)·V11_final
  + α·V12_external + β·V11_g93 + γ·V13_chronos_ft + δ·V12_external_g93
  + ε·V14_globalnn

Bias-laddered selection (ceilings 1.0/1.25/1.5/1.75/2.0%).
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
V126 = OUT / "v126"
V126.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

CV_FOLDS = [
    ("2024-07", "2024-09", "2024-10", "2024-12"),
    ("2024-07", "2024-12", "2025-01", "2025-03"),
    ("2024-07", "2025-03", "2025-04", "2025-06"),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])

BASE = "v11_final"
HELPERS = ["v12_external", "v11_g93", "v13_chronos_ft",
            "v12_external_g93", "v14_globalnn"]


def _load(tag, split):
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. For V14_globalnn: run "
            f"./scripts/v14_kaggle_check.sh merge after Kaggle kernel completes."
        )
    df = pd.read_csv(p)
    df = df[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": tag})
    df["Период"] = df["Период"].astype(str)
    return df


def _blend(df, base, helpers, weights):
    pred = weights[0] * df[base]
    for i, h in enumerate(helpers):
        pred = pred + weights[i + 1] * df[h]
    return pred.clip(lower=0).astype(np.float32)


def main() -> int:
    print(f"V12.6 multi-helper search: base={BASE}  helpers={HELPERS}")

    val = _load(BASE, "val")
    test = _load(BASE, "test")
    for h in HELPERS:
        val = val.merge(_load(h, "val").drop(columns=["target_qty"]),
                         on=KEY, how="inner")
        test = test.merge(_load(h, "test").drop(columns=["target_qty"]),
                           on=KEY, how="inner")
    print(f"aligned rows  val={len(val):,}  test={len(test):,}")

    # 5-helper grid is 8^5 = 32768 candidates — too many. Use coarser grid.
    LEVELS = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15]
    MAX_HELPER_SUM = 0.40
    candidates = [w for w in itertools.product(LEVELS, repeat=len(HELPERS))
                   if sum(w) <= MAX_HELPER_SUM]
    candidates = [(1 - sum(w),) + w for w in candidates]
    print(f"evaluating {len(candidates)} candidates "
          f"(grid {LEVELS}, max helper sum {MAX_HELPER_SUM})")

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
            "w_v14_nn": weights[5],
            "OOF_recency": oof_recency,
            "OOF_bias_recency_pct": oof_bias_recency,
        })
        if (ci + 1) % 500 == 0:
            print(f"  {ci+1}/{len(candidates)}")

    df = pd.DataFrame(rows)
    df.to_csv(V126 / "grid.csv", index=False)

    print("\n=== Champion selection (bias ladder) ===")
    best_overall = None
    for ceil in [1.0, 1.25, 1.5, 1.75, 2.0]:
        survs = df[df["OOF_bias_recency_pct"].abs() <= ceil].copy()
        if len(survs) == 0:
            continue
        ch = survs.loc[survs["OOF_recency"].idxmin()]
        print(f"  bias≤{ceil:.2f}%: {len(survs):4d} survivors  "
              f"OOF_rec={ch['OOF_recency']:.4f}")
        print(f"    weights: base={ch['w_base']:.3f}  "
              f"ext={ch['w_v12ext']:.3f}  g93={ch['w_v11g93']:.3f}  "
              f"chr={ch['w_chronos_ft']:.3f}  ext_g93={ch['w_v12ext_g93']:.3f}  "
              f"NN={ch['w_v14_nn']:.3f}  bias%={ch['OOF_bias_recency_pct']:+.2f}")
        if best_overall is None or ch["OOF_recency"] < best_overall["OOF_recency"]:
            best_overall = ch.copy()
            best_overall["ceiling"] = ceil

    best = best_overall
    weights = (best["w_base"], best["w_v12ext"], best["w_v11g93"],
               best["w_chronos_ft"], best["w_v12ext_g93"], best["w_v14_nn"])
    print(f"\n*** V12.6 CHAMPION (ceiling = {best['ceiling']}) ***")
    for name, w in zip(["V11_final"] + HELPERS, weights):
        print(f"  {name:18s} {w:.3f}")
    print(f"  OOF_recency        {best['OOF_recency']:.4f}")
    print(f"  OOF_bias%          {best['OOF_bias_recency_pct']:+.2f}")

    val_out = val[KEY + ["target_qty"]].copy()
    val_out["prediction"] = _blend(val, BASE, HELPERS, weights)
    test_out = test[KEY + ["target_qty"]].copy()
    test_out["prediction"] = _blend(test, BASE, HELPERS, weights)
    val_out.to_csv(OUT / "preds_v126_champion_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v126_champion_test.csv", index=False)

    sv = score_frame(val_out)
    st = score_frame(test_out)
    print(f"\n=== V12.6 CHAMPION on aligned subset ({len(test_out):,} rows) ===")
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
    print(f"  Δ V12.6 - V12.2 (aligned) = {delta:+.4f} ({rel:+.2f}%)")
    if delta < 0:
        print(f"\n  ✓ V12.6 BEATS V12.2 — promote to production!")
    else:
        print(f"\n  V12.6 does NOT beat V12.2. V14_GlobalNN earns "
              f"{weights[5]:.3f} weight under OOF.")

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
        "promote": delta < 0,
    }
    (V126 / "champion.json").write_text(json.dumps(summary, indent=2,
                                                     ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
