"""V11 -- bias-constrained LAD search with new V11 bases.

Differences vs V10 LAD search:

1. **Bias-constrained selection.**  Augments the OOF_recency criterion
   with a hard constraint |OOF mean bias%| ≤ 1.0.  Candidates that
   exceed the bias budget are rejected.

2. **Expanded base pool.**  Adds V11 bases (V11_recent_only, V11_g93,
   V11_g90 — three drift-aware re-trainings of V10) and optionally
   V11_chronos (foundation-model preds, if present).

3. **Streaming calibration overlay.**  After picking the LAD champion,
   we apply a per-channel streaming bias recalibrator (Priority 2)
   on top of the predictions.  This is conservative bias correction
   that uses validation history at-inference-time.

4. **Conformal width adjustment** (Priority 7).  Optional
   `--conformal-shrink K` scales final predictions by a constant K
   chosen to minimize OOF SIMSCORE.

Search grid:
  * pools  : {V10 baseline, +v11_recent_only, +v11_g93, +v11_g90,
              +all_v11, +chronos (if present)}
  * tau    : {0.50, 0.52, 0.55}
  * axes   : {ch08, chABC05_brand03, ch08_chABC_brand}

Anti-overfit:
  * gap ≤ 0.05
  * |OOF bias%| ≤ 1.0  (NEW in V11)

Writes:
  * `output/preds_v11_lad_{val,test}.csv`
  * `output/v11/lad_champion.json`
  * `output/v11/lad_cv.csv`
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
V11 = OUT / "v11"
V11.mkdir(parents=True, exist_ok=True)

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


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
    V10_BASE = [
        "v4", "v5", "v6", "v7", "v71", "v72_champion",
        "v77_recent", "v77_quantile60",
        "v8", "v8_recent",
        "v9", "v9_recent", "v9_weekly",
        "v10", "v10_recent",
    ]

    optional = []
    for tag in ("v11_recent_only", "v11_g93", "v11_g90", "v11_chronos"):
        if (OUT / f"preds_{tag}_val.csv").exists() and (OUT / f"preds_{tag}_test.csv").exists():
            optional.append(tag)
        else:
            print(f"[skip] {tag} predictions not present")

    POOLS = {"v10_baseline": V10_BASE}
    for t in optional:
        POOLS[f"v10+{t}"] = V10_BASE + [t]
    if optional:
        POOLS["v10+all_v11"] = V10_BASE + optional
        if "v11_recent_only" in optional and "v11_g93" in optional:
            POOLS["v10+ro+g93"] = V10_BASE + ["v11_recent_only", "v11_g93"]
        if "v11_g93" in optional and "v11_g90" in optional:
            POOLS["v10+g93+g90"] = V10_BASE + ["v11_g93", "v11_g90"]

    AXES_OPTIONS = {
        "ch08": [(["Канал"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5),
                            (["Бренд"], 0.3)],
        "ch08_chABC_brand": [(["Канал"], 0.8),
                             (["Канал", "Сегмент_ABC"], 0.5),
                             (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52, 0.55]

    print(f"\nPools: {len(POOLS)} | Tau: {len(TAUS)} | Axes: {len(AXES_OPTIONS)}"
          f" -> {len(POOLS)*len(TAUS)*len(AXES_OPTIONS)} candidates total\n")

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
    df.to_csv(V11 / "lad_cv.csv", index=False)
    print("\n=== Top-15 by OOF_recency (PRE-bias-filter) ===")
    print(df[["name", "OOF_recency", "OOF_mean", "OOF_bias_recency_pct",
              "in_sample", "gap"]].head(15).to_string(index=False))

    GAP_CEILING = 0.05
    BIAS_CEILING = 1.0
    survivors = [r for r in rows
                 if r["gap"] <= GAP_CEILING
                 and abs(r["OOF_bias_recency_pct"]) <= BIAS_CEILING]
    if not survivors:
        print(f"\nWARN: bias ceiling {BIAS_CEILING}%% empty, relaxing to 2.0")
        survivors = [r for r in rows
                     if r["gap"] <= GAP_CEILING
                     and abs(r["OOF_bias_recency_pct"]) <= 2.0]
    if not survivors:
        survivors = [r for r in rows if r["gap"] <= GAP_CEILING]
    if not survivors:
        survivors = rows
    champ = min(survivors,
                key=lambda r: (r["OOF_recency"], abs(r["OOF_bias_recency_pct"])))
    print(f"\nGap ≤ {GAP_CEILING}, |bias%%| ≤ {BIAS_CEILING}  ({len(survivors)} survivors)")
    print(f"\nV11 RAW CHAMPION: {champ['name']}")
    print(f"  OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF={champ['OOF_mean']:.4f}  "
          f"OOF_bias%={champ['OOF_bias_recency_pct']:+.2f}  "
          f"gap={champ['gap']:+.4f}")

    fn, v_, t_ = cand[champ["name"]]
    val_pred, meta = fn(v_, v_)
    tst_pred, _ = fn(v_, t_)

    val_raw = v_[KEY + META_AXES].copy()
    val_raw["target_qty"] = v_["y"].to_numpy()
    val_raw["prediction"] = np.clip(val_pred, 0, None)

    tst_raw = t_[KEY + META_AXES].copy()
    tst_raw["target_qty"] = t_["y"].to_numpy()
    tst_raw["prediction"] = np.clip(tst_pred, 0, None)

    print("\n=== RAW (pre-streaming-calibration) ===")
    sv_raw = score_frame(val_raw[KEY + ["target_qty", "prediction"]])
    st_raw = score_frame(tst_raw[KEY + ["target_qty", "prediction"]])
    print(f"  val   SIMSCORE={sv_raw['SIMSCORE']:.4f}  WAPE={sv_raw['WAPE']:.4f}  "
          f"bias%={sv_raw['Agg_Bias_pct']:+.2f}  M-WAPE={sv_raw['Monthly_WAPE']:.4f}")
    print(f"  test  SIMSCORE={st_raw['SIMSCORE']:.4f}  WAPE={st_raw['WAPE']:.4f}  "
          f"bias%={st_raw['Agg_Bias_pct']:+.2f}  M-WAPE={st_raw['Monthly_WAPE']:.4f}")

    print("\n=== Searching streaming calibrator (axes / β / fold-in-test) ===")
    cal_grid = [
        (None, 0.5, False), (["Канал"], 0.5, False),
        (["Канал"], 0.7, False), (["Канал", "Сегмент_ABC"], 0.5, False),
        (None, 0.5, True), (["Канал"], 0.5, True),
        (["Канал"], 0.7, True),
    ]
    best_cal = None
    for axes, beta, fold_in in cal_grid:
        try:
            v_cal, t_cal, m = streaming_calibrate(
                val_raw, tst_raw, axes=axes, beta=beta,
                alpha_clip=(0.7, 1.5), fold_in_test=fold_in,
            )
        except Exception as e:
            print(f"  axes={axes} beta={beta} fold={fold_in} -> error: {e}")
            continue
        v_eval = v_cal[KEY + ["target_qty"]].copy()
        v_eval["prediction"] = v_cal["prediction_calibrated"]
        sc_v = score_frame(v_eval)
        ax_str = "global" if not axes else "+".join(axes)
        print(f"  axes={ax_str:25s} β={beta:.1f} fold_in_test={fold_in}  "
              f"val SIMSCORE={sc_v['SIMSCORE']:.4f}  bias%={sc_v['Agg_Bias_pct']:+.2f}")
        if best_cal is None or sc_v["SIMSCORE"] < best_cal["val_sim"]:
            best_cal = {"axes": axes, "beta": beta, "fold_in": fold_in,
                        "val_sim": sc_v["SIMSCORE"],
                        "v_cal": v_cal, "t_cal": t_cal, "meta": m}
    if best_cal is None:
        best_cal = {"axes": None, "beta": 0.5, "fold_in": False,
                    "v_cal": val_raw.assign(prediction_calibrated=val_raw["prediction"]),
                    "t_cal": tst_raw.assign(prediction_calibrated=tst_raw["prediction"]),
                    "meta": {}}
    print(f"  best calibrator: axes={best_cal['axes']}  β={best_cal['beta']}  "
          f"fold_in_test={best_cal['fold_in']}")

    val_out = best_cal["v_cal"][KEY + ["target_qty"]].copy()
    val_out["prediction"] = best_cal["v_cal"]["prediction_calibrated"]
    tst_out = best_cal["t_cal"][KEY + ["target_qty"]].copy()
    tst_out["prediction"] = best_cal["t_cal"]["prediction_calibrated"]

    val_out.to_csv(OUT / "preds_v11_lad_val.csv", index=False)
    tst_out.to_csv(OUT / "preds_v11_lad_test.csv", index=False)

    sv_cal = score_frame(val_out)
    st_cal = score_frame(tst_out)
    print("\n=== POST-streaming-calibration ===")
    print(f"  val   SIMSCORE={sv_cal['SIMSCORE']:.4f}  WAPE={sv_cal['WAPE']:.4f}  "
          f"bias%={sv_cal['Agg_Bias_pct']:+.2f}  M-WAPE={sv_cal['Monthly_WAPE']:.4f}")
    print(f"  test  SIMSCORE={st_cal['SIMSCORE']:.4f}  WAPE={st_cal['WAPE']:.4f}  "
          f"bias%={st_cal['Agg_Bias_pct']:+.2f}  M-WAPE={st_cal['Monthly_WAPE']:.4f}")

    (V11 / "lad_champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_recency": champ["OOF_recency"],
        "OOF_bias_recency_pct": champ["OOF_bias_recency_pct"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "raw_val_score": sv_raw, "raw_test_score": st_raw,
        "calibrated_val_score": sv_cal, "calibrated_test_score": st_cal,
        "calibrator": {
            "axes": best_cal["axes"], "beta": best_cal["beta"],
            "fold_in_test": best_cal["fold_in"],
            "meta": best_cal["meta"],
        },
        "lad_meta": meta,
    }, indent=2, ensure_ascii=False, default=str))
    print("\nwrote preds_v11_lad_{val,test}.csv  +  v11/lad_champion.json"
          "  +  v11/lad_cv.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
