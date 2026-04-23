"""Train V7 — the stacked, residual-corrected, conformal-wrapped model.

Pipeline (end-to-end inside this script):

1. V7 base: `TwoStageForecaster` on V7 ABT with pinball quantile loss (α tunable).
2. Classifier calibration: isotonic regression on validation.
3. Per-segment residual corrector: per (Бренд, Канал) LGB on val residuals
   (mean-corrects each segment; graceful fallback for sparse segments).
4. V4/V5/V6/V7 stacked ridge meta-learner on val predictions.
5. Conformal prediction intervals per segment from val residuals.
6. Evaluation on test + UAH cost scorecard using per-SKU margin table.

Artefacts (under output/):
- model_v7.joblib                        : pickled dict with all components
- preds_v7_{val,test}.csv                : V7 (base + residual) predictions
- preds_v7_stacked_{val,test}.csv        : stacked predictions (V4+V5+V6+V7)
- preds_v7_lower_{val,test}.csv          : conformal lower bound
- preds_v7_upper_{val,test}.csv          : conformal upper bound
- v7_metrics.csv                          : WAPE/MAPE_nz/RMSE/Bias for each variant
- v7_vs_v6.md                             : narrative comparison
- feature_importance_v7.csv               : base-model FI (gain)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import compute_all_metrics, split_train_val_test  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster, encode_categoricals,
    filter_active_pairs, get_feature_columns_v2,
)
from src.v7_components import (  # noqa: E402
    IsotonicCalibrator, PerSegmentConformal, RidgeStacker,
    SegmentResidualCorrector,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v7")

OUT = _REPO_ROOT / "output"


def _load_v7() -> pd.DataFrame:
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet").pipe(encode_categoricals)
    return abt


def _load_baseline_preds(tag: str, split: str) -> pd.DataFrame | None:
    path = OUT / f"preds_{tag}_{split}.csv"
    if not path.exists():
        log.warning("missing %s", path); return None
    df = pd.read_csv(path)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    return df[["Период", "Партнер", "Артикул", "prediction"]].rename(
        columns={"prediction": f"pred_{tag}"}
    )


def _attach_baselines(target: pd.DataFrame, split: str) -> pd.DataFrame:
    """Left-join V4/V5/V6 predictions onto `target` (keyed by period/partner/sku)."""
    out = target.copy()
    out["Период"] = pd.PeriodIndex(out["Период"].astype(str), freq="M")
    for tag in ("v4", "v5", "v6"):
        p = _load_baseline_preds(tag, split)
        if p is not None:
            out = out.merge(p, on=["Период", "Партнер", "Артикул"], how="left")
    for tag in ("v4", "v5", "v6"):
        col = f"pred_{tag}"
        if col not in out.columns:
            out[col] = np.nan
    return out


def _fill_baselines(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Impute missing baseline preds with each model's column median.

    A missing value means the SKU/partner wasn't in that model's active set
    — substituting the column median is a gentle nudge toward "this model
    has no opinion" rather than using zero which the positive ridge would
    treat as a 'forecast 0' signal."""
    preds = {}
    for tag in ("v4", "v5", "v6"):
        col = f"pred_{tag}"
        vals = df[col].astype(float).to_numpy()
        if np.isnan(vals).all():
            med = 0.0
        else:
            med = float(np.nanmedian(vals))
        vals = np.where(np.isnan(vals), med, vals)
        preds[tag] = vals
    return preds


def _save_preds(df: pd.DataFrame, preds: np.ndarray, path: Path) -> None:
    out = df[["Период", "Партнер", "Артикул", "target_qty"]].copy()
    out["prediction"] = preds
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="target_qty_imputed")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="Pinball quantile level for the base regressor.")
    ap.add_argument("--optuna-params", default=None,
                    help="Optional JSON with LightGBM params (from Optuna best).")
    ap.add_argument("--num-boost-round", type=int, default=1200)
    ap.add_argument("--residual-rounds", type=int, default=200)
    ap.add_argument("--disable-residual", action="store_true",
                    help="Skip the per-segment residual corrector "
                         "(it overfits on short val histories).")
    ap.add_argument("--stacker-alpha", type=float, default=10.0,
                    help="Ridge regularisation strength for the stacker meta-learner.")
    args = ap.parse_args()

    t_all = time.time()
    abt = _load_v7()
    feats = get_feature_columns_v2(abt)
    feats = [c for c in feats if c != "target_qty_imputed"]

    log.info("V7 ABT: %d rows, %d features (V6+price+cohort)", len(abt), len(feats))

    df_train, df_val, df_test = split_train_val_test(abt)
    active = filter_active_pairs(df_train)
    keys = active[["Партнер", "Артикул"]].drop_duplicates()
    df_val = df_val.merge(keys, on=["Партнер", "Артикул"], how="inner")
    df_test = df_test.merge(keys, on=["Партнер", "Артикул"], how="inner")
    log.info("Split: train=%d val=%d test=%d", len(active), len(df_val), len(df_test))

    reg_params = {
        "num_leaves": 255,
        "learning_rate": 0.05,
        "min_child_samples": 20,
    }
    if args.optuna_params:
        tune = json.loads(Path(args.optuna_params).read_text())
        best = tune.get("best_params", tune)
        reg_params.update(best)
        log.info("Optuna tuned params merged: %s", best)

    base = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params=reg_params,
        reg_objective="pinball",
        reg_objective_kwargs={"alpha": args.alpha},
        target_col=args.target,
    )
    t0 = time.time()
    base.fit(active, df_val, feats, num_boost_round=args.num_boost_round, early_stopping=60)
    log.info("V7 base fitted in %.1fs", time.time() - t0)

    val_periods = sorted(df_val["Период"].unique())
    split_idx = max(1, int(round(len(val_periods) * 0.6)))
    corr_periods = set(val_periods[:split_idx])
    meta_periods = set(val_periods[split_idx:])
    df_val_corr = df_val[df_val["Период"].isin(corr_periods)].copy()
    df_val_meta = df_val[df_val["Период"].isin(meta_periods)].copy()
    log.info("val split: corrector set=%d rows (%d periods) | meta set=%d rows (%d periods)",
             len(df_val_corr), len(corr_periods), len(df_val_meta), len(meta_periods))

    p_base_val = base.predict(df_val)
    p_base_test = base.predict(df_test)

    calib = IsotonicCalibrator().fit(
        base.clf.predict(df_val_corr[feats]),
        (df_val_corr["target_qty"] > 0).astype(int).to_numpy(),
    )

    def _cal_pred(d: pd.DataFrame) -> np.ndarray:
        pc = calib.transform(base.clf.predict(d[feats]))
        rg = base.reg.predict(d[feats]).clip(min=0)
        return (pc * rg).clip(min=0).astype(np.float32)

    p_cal_val = _cal_pred(df_val)
    p_cal_test = _cal_pred(df_test)
    p_cal_val_corr = _cal_pred(df_val_corr)
    p_cal_val_meta = _cal_pred(df_val_meta)

    if args.disable_residual:
        corrector = None
        log.info("residual corrector disabled by flag")
        p_v7_val = p_cal_val.copy()
        p_v7_val_meta = p_cal_val_meta.copy()
        p_v7_test = p_cal_test.copy()
    else:
        resid_corr = df_val_corr["target_qty"].to_numpy() - p_cal_val_corr
        corrector = SegmentResidualCorrector(
            feature_cols=feats,
            lgb_params={
                "objective": "regression_l1",
                "num_leaves": 15,
                "learning_rate": 0.05,
                "feature_fraction": 0.7,
                "bagging_fraction": 0.7,
                "bagging_freq": 5,
                "min_child_samples": 50,
                "n_jobs": -1,
                "verbose": -1,
            },
        ).fit(df_val_corr, resid_corr, n_rounds=args.residual_rounds)
        delta_val = corrector.predict(df_val)
        delta_val_meta = corrector.predict(df_val_meta)
        delta_test = corrector.predict(df_test)
        p_v7_val = np.clip(p_cal_val + delta_val, 0, None)
        p_v7_val_meta = np.clip(p_cal_val_meta + delta_val_meta, 0, None)
        p_v7_test = np.clip(p_cal_test + delta_test, 0, None)

    meta_k = _attach_baselines(df_val_meta, "val")
    test_k = _attach_baselines(df_test, "test")
    meta_preds = _fill_baselines(meta_k); test_preds = _fill_baselines(test_k)
    meta_preds["v7"] = p_v7_val_meta; test_preds["v7"] = p_v7_test

    stacker = RidgeStacker(alpha=args.stacker_alpha).fit(
        meta_preds, df_val_meta["target_qty"].to_numpy()
    )
    p_stack_val_meta = stacker.predict(meta_preds)
    p_stack_test = stacker.predict(test_preds)

    val_k = _attach_baselines(df_val, "val")
    val_preds_full = _fill_baselines(val_k); val_preds_full["v7"] = p_v7_val
    p_stack_val = stacker.predict(val_preds_full)

    conformal = PerSegmentConformal(low=0.1, high=0.9).fit(
        df_val_meta, df_val_meta["target_qty"].to_numpy(), p_stack_val_meta
    )
    lo_val, hi_val = conformal.intervals(df_val, p_stack_val)
    lo_test, hi_test = conformal.intervals(df_test, p_stack_test)

    def _m(name: str, y: np.ndarray, p: np.ndarray, split: str) -> dict:
        met = compute_all_metrics(y, p)
        return {"model": name, "split": split, **met}

    y_val = df_val["target_qty"].to_numpy()
    y_test = df_test["target_qty"].to_numpy()

    metrics = [
        _m("V7_base", y_val, p_base_val, "val"),
        _m("V7_base", y_test, p_base_test, "test"),
        _m("V7_cal", y_val, p_cal_val, "val"),
        _m("V7_cal", y_test, p_cal_test, "test"),
        _m("V7", y_val, p_v7_val, "val"),
        _m("V7", y_test, p_v7_test, "test"),
        _m("V7_stacked", y_val, p_stack_val, "val"),
        _m("V7_stacked", y_test, p_stack_test, "test"),
    ]
    for m in metrics:
        log.info("  %-12s %-4s WAPE=%.4f MAPE_nz=%.4f RMSE=%.3f Bias=%+.3f",
                 m["model"], m["split"], m["WAPE"], m["MAPE_nz"], m["RMSE"], m["Bias"])

    pd.DataFrame(metrics).to_csv(OUT / "v7_metrics.csv", index=False)
    _save_preds(df_val, p_v7_val, OUT / "preds_v7_val.csv")
    _save_preds(df_test, p_v7_test, OUT / "preds_v7_test.csv")
    _save_preds(df_val, p_stack_val, OUT / "preds_v7_stacked_val.csv")
    _save_preds(df_test, p_stack_test, OUT / "preds_v7_stacked_test.csv")
    _save_preds(df_val, lo_val, OUT / "preds_v7_lower_val.csv")
    _save_preds(df_val, hi_val, OUT / "preds_v7_upper_val.csv")
    _save_preds(df_test, lo_test, OUT / "preds_v7_lower_test.csv")
    _save_preds(df_test, hi_test, OUT / "preds_v7_upper_test.csv")

    bundle = {
        "base": base,
        "calibrator": calib,
        "corrector": corrector,
        "stacker": stacker,
        "conformal": conformal,
        "feats": feats,
        "alpha": args.alpha,
        "reg_params": reg_params,
    }
    joblib.dump(bundle, OUT / "model_v7.joblib")

    fi = base.feature_importance()
    fi.to_csv(OUT / "feature_importance_v7.csv", index=False)

    v6m = pd.read_csv(OUT / "v6_metrics.csv") if (OUT / "v6_metrics.csv").exists() else None
    if v6m is not None:
        v6_val = v6m[(v6m.model == "V6") & (v6m.split == "val")].iloc[0].to_dict()
        v6_test = v6m[(v6m.model == "V6") & (v6m.split == "test")].iloc[0].to_dict()
    else:
        v6_val = v6_test = None

    lines = [
        "# V7 vs V6 — stacked ensemble + price + cohort + per-segment residual + conformal",
        "",
        f"Base: pinball α={args.alpha} on `{args.target}` | Features: V6 + price(7) + cohort(4)",
        "",
        "## Metrics",
        "",
        "| Split | Model | WAPE | MAPE_nz | RMSE | Bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m['split']} | {m['model']} | {m['WAPE']:.4f} | {m['MAPE_nz']:.4f} | "
            f"{m['RMSE']:.3f} | {m['Bias']:+.3f} |"
        )
    if v6_test is not None:
        lines += [
            "",
            "## V7_stacked vs V6 on test",
            "",
            f"- WAPE   Δ = **{metrics[-1]['WAPE'] - float(v6_test['WAPE']):+.4f}** "
            f"(V7_stacked {metrics[-1]['WAPE']:.4f} vs V6 {float(v6_test['WAPE']):.4f})",
            f"- MAPE_nz Δ = **{metrics[-1]['MAPE_nz'] - float(v6_test['MAPE_nz']):+.4f}**",
            f"- Bias   V7_stacked {metrics[-1]['Bias']:+.3f} / V6 {float(v6_test['Bias']):+.3f}",
        ]
    (OUT / "v7_vs_v6.md").write_text("\n".join(lines))

    log.info("Total time: %.1fs", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
