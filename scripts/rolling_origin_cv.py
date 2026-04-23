"""Rolling-origin cross-validation for the V6 forecaster.

For each origin month ``O`` in a configurable window we:
    * train on   Период <= O - 2
    * validate on Период == O - 1   (early-stopping)
    * test on    Период == O         (metrics recorded)

Aggregates per-origin WAPE / MAPE_nz / RMSE / bias and reports
``mean + k*std`` as the selection score.  Default configuration uses
8 origins spanning the last year of the V6 ABT.

Usage
-----
``python -m scripts.rolling_origin_cv \
        --abt output/abt_v6_cached.parquet \
        --n-origins 8 \
        --target target_qty_imputed \
        --reg-objective pinball --alpha 0.6 \
        --output output/v6_rolling_cv.json``

Outputs
-------
``output/v6_rolling_cv.json`` — per-origin metrics + aggregates
``output/v6_rolling_cv.md``   — human-readable summary table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import compute_all_metrics  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster,
    encode_categoricals,
    filter_active_pairs,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("rolling_origin_cv")


def _pick_origins(df: pd.DataFrame, n: int) -> list[pd.Period]:
    unique_months = sorted(df["Период"].unique())
    # Reserve at least 24 months of training data before the first origin
    candidates = unique_months[24:]
    if len(candidates) < n:
        raise ValueError(f"Not enough months for {n} origins (have {len(candidates)})")
    return list(candidates[-n:])


def run_single_origin(
    df: pd.DataFrame,
    origin: pd.Period,
    target_col: str,
    reg_objective: str | None,
    reg_objective_kwargs: dict,
    num_boost_round: int,
    early_stopping: int,
) -> dict:
    log.info("── origin=%s ──", origin)
    train_end = origin - 2
    val_month = origin - 1
    test_month = origin

    df_train = df[df["Период"] <= train_end].copy()
    df_val = df[df["Период"] == val_month].copy()
    df_test = df[df["Период"] == test_month].copy()

    # Filter active pairs using train only
    df_train_active = filter_active_pairs(df_train)
    df_val = df_val.merge(
        df_train_active[["Партнер", "Артикул"]].drop_duplicates(),
        on=["Партнер", "Артикул"], how="inner",
    )
    df_test_in = df_test.merge(
        df_train_active[["Партнер", "Артикул"]].drop_duplicates(),
        on=["Партнер", "Артикул"], how="inner",
    )

    feat_cols = get_feature_columns_v2(df_train_active)
    model = TwoStageForecaster(
        reg_objective=reg_objective,
        reg_objective_kwargs=reg_objective_kwargs,
        target_col=target_col,
    )
    t0 = time.time()
    model.fit(
        df_train_active, df_val, feat_cols,
        num_boost_round=num_boost_round, early_stopping=early_stopping,
    )
    preds = model.predict(df_test_in)
    metrics = compute_all_metrics(df_test_in["target_qty"].to_numpy(), preds)
    metrics.update({
        "origin": str(origin),
        "n_train": len(df_train_active),
        "n_test": len(df_test_in),
        "n_test_full": len(df_test),
        "actual_sum": float(df_test_in["target_qty"].sum()),
        "pred_sum": float(preds.sum()),
        "fit_seconds": round(time.time() - t0, 1),
    })
    log.info("origin=%s  WAPE=%.4f  MAPE_nz=%.4f  Bias=%.2f",
             origin, metrics["WAPE"], metrics["MAPE_nz"], metrics["Bias"])
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abt", default="output/abt_v6_cached.parquet")
    ap.add_argument("--target", default="target_qty",
                    help="Target column to regress on (target_qty or target_qty_imputed)")
    ap.add_argument("--reg-objective", default=None,
                    help="Regression objective: tweedie (default), pinball, asymmetric")
    ap.add_argument("--alpha", type=float, default=0.6, help="pinball alpha")
    ap.add_argument("--cost-under", type=float, default=2.5)
    ap.add_argument("--cost-over", type=float, default=1.0)
    ap.add_argument("--n-origins", type=int, default=8)
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--output-json", default="output/v6_rolling_cv.json")
    ap.add_argument("--output-md", default="output/v6_rolling_cv.md")
    args = ap.parse_args()

    abt_path = _REPO_ROOT / args.abt
    log.info("Loading ABT → %s", abt_path)
    df = pd.read_parquet(abt_path)
    df = encode_categoricals(df)

    reg_kwargs: dict = {}
    if args.reg_objective == "pinball":
        reg_kwargs = {"alpha": args.alpha}
    elif args.reg_objective == "asymmetric":
        reg_kwargs = {"cost_under": args.cost_under, "cost_over": args.cost_over}

    origins = _pick_origins(df, args.n_origins)
    log.info("Origins: %s", [str(o) for o in origins])

    per_origin: list[dict] = []
    for origin in origins:
        m = run_single_origin(
            df, origin, args.target, args.reg_objective, reg_kwargs,
            num_boost_round=args.num_boost_round, early_stopping=args.early_stopping,
        )
        per_origin.append(m)

    wapes = np.array([m["WAPE"] for m in per_origin])
    mape_nz = np.array([m["MAPE_nz"] for m in per_origin])
    rmses = np.array([m["RMSE"] for m in per_origin])
    biases = np.array([m["Bias"] for m in per_origin])

    summary = {
        "config": {
            "abt": str(abt_path),
            "target": args.target,
            "reg_objective": args.reg_objective or "tweedie",
            "reg_objective_kwargs": reg_kwargs,
            "n_origins": args.n_origins,
            "num_boost_round": args.num_boost_round,
        },
        "per_origin": per_origin,
        "aggregates": {
            "WAPE_mean": float(wapes.mean()),
            "WAPE_std":  float(wapes.std(ddof=0)),
            "WAPE_selection_score": float(wapes.mean() + 0.5 * wapes.std(ddof=0)),
            "MAPE_nz_mean": float(mape_nz.mean()),
            "RMSE_mean":   float(rmses.mean()),
            "Bias_mean":   float(biases.mean()),
            "Bias_std":    float(biases.std(ddof=0)),
        },
    }

    out_json = _REPO_ROOT / args.output_json
    out_md = _REPO_ROOT / args.output_md
    out_json.parent.mkdir(exist_ok=True, parents=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("JSON summary → %s", out_json)

    # Markdown report
    lines = ["# Rolling-origin CV summary", "", f"ABT: `{abt_path.name}`  ",
             f"Target: `{args.target}` | Objective: `{args.reg_objective or 'tweedie'}` {reg_kwargs}",
             "", "## Per-origin metrics", "",
             "| origin | n_train | n_test | WAPE | MAPE_nz | RMSE | Bias | sec |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in per_origin:
        lines.append(
            f"| {m['origin']} | {m['n_train']:,} | {m['n_test']:,} | "
            f"{m['WAPE']:.4f} | {m['MAPE_nz']:.4f} | {m['RMSE']:.2f} | {m['Bias']:+.2f} | {m['fit_seconds']} |"
        )
    agg = summary["aggregates"]
    lines += [
        "",
        "## Aggregates",
        f"- mean WAPE: **{agg['WAPE_mean']:.4f}**  (std {agg['WAPE_std']:.4f})",
        f"- selection score (mean + 0.5σ): **{agg['WAPE_selection_score']:.4f}**",
        f"- mean MAPE_nz: {agg['MAPE_nz_mean']:.4f}",
        f"- mean RMSE: {agg['RMSE_mean']:.2f}",
        f"- mean Bias: {agg['Bias_mean']:+.3f} (std {agg['Bias_std']:.3f})",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Markdown summary → %s", out_md)

    print(f"\n=> mean WAPE={agg['WAPE_mean']:.4f}  std={agg['WAPE_std']:.4f}  "
          f"score={agg['WAPE_selection_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
