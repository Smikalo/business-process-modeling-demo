"""V12 LAD search with **bias ladder** + V12 base extensions.

Differences vs V11 LAD search:

1. **Bias ladder.** Instead of a fixed |OOF bias%| ≤ 1.0 ceiling, sweep
   {1.0, 1.5, 2.0, 2.5%} ceilings and pick the champion across all
   (lower bias preferred when ties on OOF SIMSCORE).
2. **Extended pool.** Includes V11 bases + new V12 bases:
   - v12_multiseed (multi-seed bagged V11_recent_only)
   - v12_intermittent (Croston/SBA/TSB specialist)
   - v12_anomaly (anomaly-downweighted re-train)
   Bases that don't have prediction files are silently skipped.
3. **External-feature awareness.** Logs whether ``abt_v12_external.parquet``
   exists; if so, downstream callers can re-train V11 bases on it.
   (This script itself does not retrain — it works at the prediction
   level, choosing a LAD blend over already-existing prediction CSVs.)

Anti-overfit: gap ≤ 0.05 (same as V11), bias swept via ladder.

Writes:
  * ``output/preds_v12_lad_{val,test}.csv``
  * ``output/preds_v12_final_{val,test}.csv`` (LAD + streaming-bias overlay)
  * ``output/v12/lad_champion.json``
  * ``output/v12/lad_cv.csv``
  * ``output/v12/bias_ladder_summary.csv``
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame
from scripts.v77_multi_reconcile import (
    KEY, META_AXES, fit_per_channel_tilted, multi_reconcile,
)
from src.streaming_calibrator import streaming_calibrate

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V12 = OUT / "v12"
V12.mkdir(parents=True, exist_ok=True)

# Same CV folds and recency weights as V11
CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])

# Bias ladder: {percent}: priority
BIAS_LADDER = [1.0, 1.5, 2.0, 2.5]


def _load_split(tag: str, split: str) -> pd.DataFrame:
    p = OUT / f"preds_{tag}_{split}.csv"
    return (
        pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
          .rename(columns={"prediction": tag})
    )


def _load_wide(split: str, tags: list[str]) -> pd.DataFrame:
    base = _load_split(tags[0], split).rename(columns={"target_qty": "y"})
    for t in tags[1:]:
        base = base.merge(
            _load_split(t, split).drop(columns=["target_qty"]),
            on=KEY, how="inner",
        )
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


def _score(df, pred):
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def eval_pipeline(name, fn, val):
    oof_sims = []
    oof_bias = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = fn(tr, te)
        sc = _score(te, pred)
        oof_sims.append(sc["SIMSCORE"])
        oof_bias.append(sc["Agg_Bias_pct"])
    in_pred, meta = fn(val, val)
    insim = _score(val, in_pred)["SIMSCORE"]
    oof_mean = float(np.mean(oof_sims))
    oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
    oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))
    return {
        "name": name,
        "OOF_mean": round(oof_mean, 4),
        "OOF_recency": round(oof_recency, 4),
        "OOF_bias_recency_pct": round(oof_bias_recency, 3),
        "OOF_folds": [round(x, 4) for x in oof_sims],
        "in_sample": round(insim, 4),
        "gap": round(oof_mean - insim, 4),
        "meta": meta,
    }


def main() -> int:
    # Same V10 base pool as V11
    V10_BASE = [
        "v4", "v5", "v6", "v7", "v71", "v72_champion",
        "v77_recent", "v77_quantile60",
        "v8", "v8_recent",
        "v9", "v9_recent", "v9_weekly",
        "v10", "v10_recent",
    ]

    # V11 + V12 candidate additions to the pool
    V11_ADDITIONS = ["v11_recent_only", "v11_g93", "v11_g90", "v11_chronos"]
    V12_ADDITIONS = [
        "v12_multiseed", "v12_intermittent", "v12_anomaly",
    ]

    optional = []
    for tag in V11_ADDITIONS + V12_ADDITIONS:
        if (OUT / f"preds_{tag}_val.csv").exists() and (OUT / f"preds_{tag}_test.csv").exists():
            optional.append(tag)
        else:
            print(f"[skip] {tag} predictions not present")

    # Build pool variants — incremental ablation-style
    POOLS = {"v10_baseline": V10_BASE}
    for t in optional:
        POOLS[f"v10+{t}"] = V10_BASE + [t]
    if optional:
        POOLS["v10+all_optional"] = V10_BASE + optional
        # V11-only pool (control)
        v11_only = [t for t in optional if t.startswith("v11")]
        if v11_only:
            POOLS["v10+all_v11"] = V10_BASE + v11_only
        # V12-only pool (control — does V12 help even without V11 additions?)
        v12_only = [t for t in optional if t.startswith("v12")]
        if v12_only:
            POOLS["v10+all_v12"] = V10_BASE + v12_only
        # V11_g93+V12_multiseed combo (the most likely winner)
        wishlist = [t for t in optional if t in {
            "v11_g93", "v11_recent_only",
            "v12_multiseed", "v12_intermittent", "v12_anomaly",
        }]
        if wishlist:
            POOLS["v10+v11g93+v12best"] = V10_BASE + wishlist

    AXES_OPTIONS = {
        "ch08": [(["Канал"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5),
                            (["Бренд"], 0.3)],
        "ch08_chABC_brand": [(["Канал"], 0.8),
                             (["Канал", "Сегмент_ABC"], 0.5),
                             (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52, 0.55]

    print(f"\nV12 LAD search: {len(POOLS)} pools | {len(TAUS)} taus | "
          f"{len(AXES_OPTIONS)} axes -> "
          f"{len(POOLS)*len(TAUS)*len(AXES_OPTIONS)} candidates total")

    val_w = {k: _load_wide("val", tags) for k, tags in POOLS.items()}
    tst_w = {k: _load_wide("test", tags) for k, tags in POOLS.items()}

    pipelines: dict[str, tuple] = {}
    for pool_name, tags in POOLS.items():
        for tau in TAUS:
            for axes_name, axes in AXES_OPTIONS.items():
                def make_pipe(_tags, _tau, _axes):
                    def fn(tr, te):
                        pt, meta = fit_per_channel_tilted(_tags, tr, te, tau=_tau)
                        pt_tr, _ = fit_per_channel_tilted(_tags, tr, tr, tau=_tau)
                        out = multi_reconcile(tr, te, pt_tr, pt, _axes)
                        return out, {"base": meta, "axes": str(_axes),
                                     "tau": _tau, "pool": _tags}
                    return fn
                key = f"{pool_name}_tau{tau}_{axes_name}"
                pipelines[key] = (make_pipe(tags, tau, axes),
                                  val_w[pool_name], tst_w[pool_name])

    rows = []
    cand: dict[str, tuple] = {}
    n = len(pipelines)
    for i, (name, (fn, v_, t_)) in enumerate(pipelines.items(), 1):
        r = eval_pipeline(name, fn, v_)
        rows.append(r)
        cand[name] = (fn, v_, t_)
        if i <= 25 or i % 10 == 0 or i == n:
            print(f"[{i:3d}/{n}] {name:55s}  OOF={r['OOF_mean']:.4f}  "
                  f"rec={r['OOF_recency']:.4f}  bias%={r['OOF_bias_recency_pct']:+.2f}  "
                  f"gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V12 / "lad_cv.csv", index=False)
    print("\n=== Top-15 by OOF_recency (PRE-bias-filter) ===")
    print(df[["name", "OOF_recency", "OOF_mean", "OOF_bias_recency_pct",
              "in_sample", "gap"]].head(15).to_string(index=False))

    # Bias ladder: try each ceiling, pick the lowest-OOF champion at the
    # tightest ceiling that has at least one survivor
    GAP_CEILING = 0.05
    ladder_summary = []
    champ_by_ceiling: dict[float, dict | None] = {}
    for bias_ceiling in BIAS_LADDER:
        survivors = [r for r in rows
                     if r["gap"] <= GAP_CEILING
                     and abs(r["OOF_bias_recency_pct"]) <= bias_ceiling]
        if survivors:
            ch = min(survivors,
                     key=lambda r: (r["OOF_recency"],
                                    abs(r["OOF_bias_recency_pct"])))
            champ_by_ceiling[bias_ceiling] = ch
            ladder_summary.append({
                "bias_ceiling_pct": bias_ceiling,
                "n_survivors": len(survivors),
                "champion": ch["name"],
                "champion_OOF_recency": ch["OOF_recency"],
                "champion_bias_pct": ch["OOF_bias_recency_pct"],
                "champion_gap": ch["gap"],
            })
        else:
            ladder_summary.append({
                "bias_ceiling_pct": bias_ceiling,
                "n_survivors": 0,
                "champion": None,
                "champion_OOF_recency": None,
                "champion_bias_pct": None,
                "champion_gap": None,
            })

    pd.DataFrame(ladder_summary).to_csv(V12 / "bias_ladder_summary.csv", index=False)
    print("\n=== Bias ladder summary ===")
    print(pd.DataFrame(ladder_summary).to_string(index=False))

    # Pick the production champion: the BEST OOF_recency across all
    # ceilings, but prefer tighter bias ceiling on OOF ties (within 0.001).
    valid_champs = [(c, ceil) for ceil, c in champ_by_ceiling.items() if c]
    if not valid_champs:
        # Last resort: relax gap too
        survivors = rows
        ch = min(survivors, key=lambda r: r["OOF_recency"])
        production_ceiling = None
    else:
        # Sort by OOF_recency asc, then by ceiling asc (prefer tighter bias)
        valid_champs.sort(key=lambda x: (x[0]["OOF_recency"], x[1]))
        ch, production_ceiling = valid_champs[0]

    champ_name = ch["name"]
    print(f"\n*** V12 LAD CHAMPION (bias ceiling={production_ceiling}): {champ_name} ***")
    print(f"    OOF_recency={ch['OOF_recency']:.4f}  "
          f"OOF={ch['OOF_mean']:.4f}  "
          f"bias%={ch['OOF_bias_recency_pct']:+.2f}  "
          f"gap={ch['gap']:+.4f}")

    # Generate the test predictions from the champion pipeline
    fn, v_, t_ = cand[champ_name]
    val_pred, meta = fn(v_, v_)
    test_pred, _ = fn(v_, t_)

    val_out = v_[KEY].copy()
    val_out["target_qty"] = v_["y"].to_numpy()
    val_out["prediction"] = np.clip(val_pred, 0, None)

    test_out = t_[KEY].copy()
    test_out["target_qty"] = t_["y"].to_numpy()
    test_out["prediction"] = np.clip(test_pred, 0, None)

    val_out.to_csv(OUT / "preds_v12_lad_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v12_lad_test.csv", index=False)

    val_score = score_frame(val_out)
    test_score = score_frame(test_out)
    print("\n=== V12 LAD raw scores ===")
    print(f"VAL : SIMSCORE={val_score['SIMSCORE']:.4f}  WAPE={val_score['WAPE']:.4f}  "
          f"bias%={val_score['Agg_Bias_pct']:+.2f}")
    print(f"TEST: SIMSCORE={test_score['SIMSCORE']:.4f}  WAPE={test_score['WAPE']:.4f}  "
          f"bias%={test_score['Agg_Bias_pct']:+.2f}")

    # Streaming bias overlay (V11 trick: per-channel EMA of monthly bias).
    # The calibrator needs axis columns — merge from V7 ABT.
    abt_axes = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt_axes["Период"] = abt_axes["Период"].astype(str)
    val_with_axes = val_out.merge(abt_axes, on=KEY, how="left")
    test_with_axes = test_out.merge(abt_axes, on=KEY, how="left")

    val_cal, test_cal, _ = streaming_calibrate(
        val_with_axes, test_with_axes,
        axes=["Канал"], beta=0.5, fold_in_test=False,
    )
    val_calibrated = val_cal[KEY + ["target_qty"]].copy()
    val_calibrated["prediction"] = val_cal["prediction_calibrated"]
    test_calibrated = test_cal[KEY + ["target_qty"]].copy()
    test_calibrated["prediction"] = test_cal["prediction_calibrated"]

    val_calibrated.to_csv(OUT / "preds_v12_final_val.csv", index=False)
    test_calibrated.to_csv(OUT / "preds_v12_final_test.csv", index=False)

    val_final_score = score_frame(val_calibrated)
    test_final_score = score_frame(test_calibrated)
    print("\n=== V12_final (LAD + streaming bias) scores ===")
    print(f"VAL : SIMSCORE={val_final_score['SIMSCORE']:.4f}  "
          f"WAPE={val_final_score['WAPE']:.4f}  "
          f"bias%={val_final_score['Agg_Bias_pct']:+.2f}")
    print(f"TEST: SIMSCORE={test_final_score['SIMSCORE']:.4f}  "
          f"WAPE={test_final_score['WAPE']:.4f}  "
          f"bias%={test_final_score['Agg_Bias_pct']:+.2f}")

    # Persist champion JSON
    champion_doc = {
        "champion_name": champ_name,
        "production_bias_ceiling_pct": production_ceiling,
        "OOF_recency": ch["OOF_recency"],
        "OOF_mean": ch["OOF_mean"],
        "OOF_bias_recency_pct": ch["OOF_bias_recency_pct"],
        "gap": ch["gap"],
        "test_simscore_raw_lad": test_score["SIMSCORE"],
        "test_simscore_v12_final": test_final_score["SIMSCORE"],
        "test_wape_v12_final": test_final_score["WAPE"],
        "test_bias_pct_v12_final": test_final_score["Agg_Bias_pct"],
        "pool": meta.get("pool", []),
        "axes": meta.get("axes"),
        "tau": meta.get("tau"),
    }
    (V12 / "lad_champion.json").write_text(json.dumps(champion_doc, indent=2,
                                                       ensure_ascii=False))
    print(f"\nWrote {V12 / 'lad_champion.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
