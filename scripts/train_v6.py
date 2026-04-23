"""Train the V6 forecaster: pinball-quantile TwoStage on the V6 ABT.

Outputs:
    output/model_v6.joblib
    output/v6_metrics.csv
    output/preds_v6_val.csv
    output/preds_v6_test.csv
    output/feature_importance_v6.csv
    output/v6_vs_v5.md
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import compute_all_metrics, split_train_val_test  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster,
    encode_categoricals,
    filter_active_pairs,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v6")

V6_ABT = _REPO_ROOT / "output" / "abt_v6_cached.parquet"
V5_ABT = _REPO_ROOT / "output" / "abt_v5_cached.parquet"
MODEL_OUT = _REPO_ROOT / "output" / "model_v6.joblib"
METRICS_OUT = _REPO_ROOT / "output" / "v6_metrics.csv"
PREDS_VAL = _REPO_ROOT / "output" / "preds_v6_val.csv"
PREDS_TEST = _REPO_ROOT / "output" / "preds_v6_test.csv"
FI_OUT = _REPO_ROOT / "output" / "feature_importance_v6.csv"
COMPARE_MD = _REPO_ROOT / "output" / "v6_vs_v5.md"


def _split_and_filter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train, df_val, df_test = split_train_val_test(df)
    active = filter_active_pairs(df_train)
    keys = active[["Партнер", "Артикул"]].drop_duplicates()
    df_val = df_val.merge(keys, on=["Партнер", "Артикул"], how="inner")
    df_test = df_test.merge(keys, on=["Партнер", "Артикул"], how="inner")
    return active, df_val, df_test


def _train_one(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feats: list[str],
    reg_objective: str | None,
    reg_kwargs: dict,
    target_col: str,
    num_boost_round: int = 1200,
    early_stopping: int = 60,
) -> TwoStageForecaster:
    model = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
        reg_objective=reg_objective,
        reg_objective_kwargs=reg_kwargs,
        target_col=target_col,
    )
    t0 = time.time()
    model.fit(df_train, df_val, feats, num_boost_round=num_boost_round, early_stopping=early_stopping)
    log.info("Fitted in %.1fs", time.time() - t0)
    return model


def _save_preds(df: pd.DataFrame, preds, path: Path) -> None:
    out = df[["Период", "Партнер", "Артикул", "target_qty"]].copy()
    out["prediction"] = preds
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="target_qty_imputed",
                    help="target_qty or target_qty_imputed")
    ap.add_argument("--reg-objective", default="pinball")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--cost-under", type=float, default=2.5)
    ap.add_argument("--cost-over", type=float, default=1.0)
    ap.add_argument("--num-boost-round", type=int, default=1200)
    args = ap.parse_args()

    if not V6_ABT.exists():
        raise FileNotFoundError(f"V6 ABT missing: {V6_ABT}")

    log.info("Loading V6 ABT → %s", V6_ABT)
    v6 = pd.read_parquet(V6_ABT).pipe(encode_categoricals)
    feats_v6 = get_feature_columns_v2(v6)

    # If target uses imputed, materialise it as the regressor label. But do
    # NOT feed target_qty_imputed as a feature.
    if args.target == "target_qty_imputed" and "target_qty_imputed" in feats_v6:
        feats_v6 = [c for c in feats_v6 if c != "target_qty_imputed"]
    # was_censored as a feature is fine (helps the classifier).

    reg_kwargs: dict = {}
    if args.reg_objective == "pinball":
        reg_kwargs = {"alpha": args.alpha}
    elif args.reg_objective == "asymmetric":
        reg_kwargs = {"cost_under": args.cost_under, "cost_over": args.cost_over}

    log.info("V6 features=%d | objective=%s %s | target=%s",
             len(feats_v6), args.reg_objective, reg_kwargs, args.target)

    v6_train, v6_val, v6_test = _split_and_filter(v6)
    log.info("Split sizes: train=%d, val=%d, test=%d",
             len(v6_train), len(v6_val), len(v6_test))

    model = _train_one(
        v6_train, v6_val, feats_v6,
        args.reg_objective, reg_kwargs, args.target,
        num_boost_round=args.num_boost_round,
    )

    # Predictions
    p_val = model.predict(v6_val)
    p_test = model.predict(v6_test)

    m_val = compute_all_metrics(v6_val["target_qty"].to_numpy(), p_val)
    m_test = compute_all_metrics(v6_test["target_qty"].to_numpy(), p_test)
    log.info("V6 val: %s", m_val)
    log.info("V6 test: %s", m_test)

    # V5 reference metrics (read from prior run if available)
    v5_val = v5_test = None
    v5_metrics_path = _REPO_ROOT / "output" / "v5_metrics.csv"
    if v5_metrics_path.exists():
        v5_metrics = pd.read_csv(v5_metrics_path)
        v5_val_row = v5_metrics[(v5_metrics.model == "V5") & (v5_metrics.split == "val")]
        v5_test_row = v5_metrics[(v5_metrics.model == "V5") & (v5_metrics.split == "test")]
        if not v5_val_row.empty:
            v5_val = v5_val_row.iloc[0].to_dict()
        if not v5_test_row.empty:
            v5_test = v5_test_row.iloc[0].to_dict()

    joblib.dump(model, MODEL_OUT)
    log.info("Saved V6 model → %s", MODEL_OUT)

    _save_preds(v6_val, p_val, PREDS_VAL)
    _save_preds(v6_test, p_test, PREDS_TEST)
    log.info("Saved predictions → %s, %s", PREDS_VAL, PREDS_TEST)

    # Feature importance
    fi = model.feature_importance()
    fi.to_csv(FI_OUT, index=False)
    log.info("Feature importance → %s (top: %s)", FI_OUT, fi.head(10)["feature"].tolist())

    # Metrics CSV
    rows = [
        {"model": "V6", "split": "val", **m_val},
        {"model": "V6", "split": "test", **m_test},
    ]
    pd.DataFrame(rows).to_csv(METRICS_OUT, index=False)

    # Comparison report
    lines = [
        "# V6 vs V5 — imputation + promo-lifecycle + cost-calibrated loss",
        "",
        f"Objective: `{args.reg_objective}` {reg_kwargs}  | Target: `{args.target}`",
        "",
        "## Metrics",
        "",
        "| Split | Model | WAPE | MAPE_nz | RMSE | Bias |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if v5_val:
        lines.append(
            f"| val  | V5 | {float(v5_val['WAPE']):.4f} | {float(v5_val['MAPE_nz']):.4f} | "
            f"{float(v5_val['RMSE']):.4f} | — |"
        )
    lines.append(
        f"| val  | V6 | {m_val['WAPE']:.4f} | {m_val['MAPE_nz']:.4f} | "
        f"{m_val['RMSE']:.4f} | {m_val['Bias']:+.3f} |"
    )
    if v5_test:
        lines.append(
            f"| test | V5 | {float(v5_test['WAPE']):.4f} | {float(v5_test['MAPE_nz']):.4f} | "
            f"{float(v5_test['RMSE']):.4f} | — |"
        )
    lines.append(
        f"| test | V6 | {m_test['WAPE']:.4f} | {m_test['MAPE_nz']:.4f} | "
        f"{m_test['RMSE']:.4f} | {m_test['Bias']:+.3f} |"
    )

    if v5_test:
        lines += [
            "",
            "## Deltas (V6 − V5, negative = better)",
            "",
            f"- val  WAPE Δ = **{m_val['WAPE'] - float(v5_val['WAPE']):+.4f}**",
            f"- test WAPE Δ = **{m_test['WAPE'] - float(v5_test['WAPE']):+.4f}**",
            f"- test MAPE_nz Δ = **{m_test['MAPE_nz'] - float(v5_test['MAPE_nz']):+.4f}**",
        ]
    lines += [
        "",
        "## Features added in V6",
        "",
        "`was_censored`, `promo_duration_months`, `promo_depth_pct_current`, "
        "`months_since_last_promo`, `months_until_next_promo`, "
        "`post_promo_depletion_flag`, `sku_promo_sensitivity`.",
    ]
    COMPARE_MD.write_text("\n".join(lines))
    log.info("Comparison → %s", COMPARE_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
