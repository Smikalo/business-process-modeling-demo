"""Cost-based scorecard: compare V4 / V5 / V6 / naive forecasts in UAH.

Assumptions (tunable via CLI flags):
* Holding cost  = `holding_rate` * unit_price * over_forecast_qty
  (annualised holding of 22% → monthly share ≈ 0.022; we default to a
  full-year holding equivalent of 0.22 for readability)
* Lost-margin  = `margin_rate` * (1 - `recovery`) * unit_price * under_forecast_qty
  * recovery captures partial back-order fulfilment next month.

Price source: per-SKU mean `implied_unit_price` from the V6 ABT (falls back
to brand-level median when the SKU has too few price observations).

Usage
-----
python -m scripts.decision_cost_scorecard \
       --preds-v4 output/preds_v4_test.csv \
       --preds-v5 output/preds_v5_test.csv \
       --preds-v6 output/preds_v6_test.csv \
       --abt output/abt_v6_cached.parquet \
       --output output/cost_scorecard.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("cost_scorecard")

KEYCOLS = ["Период", "Партнер", "Артикул"]


def _load_preds(path: Path | None, label: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        log.warning("Predictions for %s not found at %s — skipping", label, path)
        return None
    df = pd.read_csv(path, parse_dates=False)
    # Normalise period
    if "Период" in df.columns:
        df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M")
    if "prediction" not in df.columns:
        # Try common alternate names
        for alt in ("pred", "forecast", "y_pred"):
            if alt in df.columns:
                df["prediction"] = df[alt]
                break
    return df[KEYCOLS + ["prediction"]]


def _price_lookup(abt: pd.DataFrame) -> pd.Series:
    mask = (abt["implied_unit_price"].fillna(0) > 0)
    per_sku = (
        abt.loc[mask].groupby("Артикул")["implied_unit_price"].median()
    )
    return per_sku


def _naive_y1(test_df: pd.DataFrame) -> np.ndarray:
    """Naive y(t) = y(t-1), read from ``lag_1`` already in the ABT."""
    return test_df["lag_1"].clip(lower=0).to_numpy()


def _compute_cost(
    actual: np.ndarray,
    pred: np.ndarray,
    price: np.ndarray,
    holding_rate,
    margin_rate,
    recovery: float,
) -> dict:
    over = np.clip(pred - actual, 0, None)
    under = np.clip(actual - pred, 0, None)
    holding_cost = (np.asarray(holding_rate) * over * price).sum()
    lost_margin = (np.asarray(margin_rate) * (1 - recovery) * under * price).sum()
    total = holding_cost + lost_margin
    return {
        "holding_cost_UAH": float(holding_cost),
        "lost_margin_UAH": float(lost_margin),
        "total_cost_UAH": float(total),
        "over_qty": float(over.sum()),
        "under_qty": float(under.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-v4", default="output/preds_v4_test.csv")
    ap.add_argument("--preds-v5", default="output/preds_v5_test.csv")
    ap.add_argument("--preds-v6", default="output/preds_v6_test.csv")
    ap.add_argument("--preds-v7", default="output/preds_v7_test.csv")
    ap.add_argument("--abt", default="output/abt_v6_cached.parquet")
    ap.add_argument("--holding-rate", type=float, default=0.22)
    ap.add_argument("--margin-rate", type=float, default=0.28)
    ap.add_argument("--recovery", type=float, default=0.50)
    ap.add_argument("--margin-table", default=None,
                    help="Path to sku_margin.parquet (per-SKU unit_price + margin_rate). "
                         "When set, the scorecard uses per-SKU rates and --margin-rate/--holding-rate are ignored.")
    ap.add_argument("--output", default="output/cost_scorecard.md")
    ap.add_argument("--output-json", default="output/cost_scorecard.json")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.abt)
    margin_tbl = None
    if args.margin_table:
        mt_path = _REPO_ROOT / args.margin_table
        if mt_path.exists():
            margin_tbl = pd.read_parquet(mt_path).set_index("Артикул")
            log.info("Using per-SKU margin table from %s (%d SKUs)", mt_path, len(margin_tbl))
        else:
            log.warning("margin table %s not found — falling back to flat rates", mt_path)
    price_by_sku = _price_lookup(abt)
    global_price = float(price_by_sku.median()) if len(price_by_sku) else 100.0

    models: dict[str, pd.DataFrame | None] = {
        "V4": _load_preds(_REPO_ROOT / args.preds_v4, "V4"),
        "V5": _load_preds(_REPO_ROOT / args.preds_v5, "V5"),
        "V6": _load_preds(_REPO_ROOT / args.preds_v6, "V6"),
        "V7": _load_preds(_REPO_ROOT / args.preds_v7, "V7"),
    }

    base = next((m for m in (models["V7"], models["V6"], models["V5"], models["V4"]) if m is not None), None)
    if base is None:
        log.error("No prediction files found — nothing to score.")
        return 1
    test_base = base[KEYCOLS].merge(
        abt[KEYCOLS + ["target_qty", "lag_1", "Бренд", "Канал"]],
        on=KEYCOLS, how="left",
    )
    test_base = test_base.dropna(subset=["target_qty"])
    if margin_tbl is not None:
        test_base["price"] = (
            test_base["Артикул"].map(margin_tbl["unit_price_uah"]).fillna(global_price)
        )
        test_base["holding_rate"] = (
            test_base["Артикул"].map(margin_tbl["holding_rate_annual"])
                                  .fillna(args.holding_rate)
        )
        test_base["margin_rate"] = (
            test_base["Артикул"].map(margin_tbl["margin_rate"]).fillna(args.margin_rate)
        )
    else:
        test_base["price"] = test_base["Артикул"].map(price_by_sku).fillna(global_price)
        test_base["holding_rate"] = args.holding_rate
        test_base["margin_rate"] = args.margin_rate
    actual = test_base["target_qty"].to_numpy()
    price = test_base["price"].to_numpy()
    holding_arr = test_base["holding_rate"].to_numpy()
    margin_arr = test_base["margin_rate"].to_numpy()

    rows = []
    # Naive
    naive_pred = _naive_y1(test_base)
    rows.append(
        {"model": "naive (y_t = y_{t-1})", **_compute_cost(
            actual, naive_pred, price, holding_arr, margin_arr, args.recovery
        )}
    )
    for name, df in models.items():
        if df is None:
            continue
        merged = test_base.merge(df, on=KEYCOLS, how="left")
        pred = merged["prediction"].fillna(0).to_numpy()
        rows.append(
            {"model": name, **_compute_cost(
                actual, pred, price, holding_arr, margin_arr, args.recovery
            )}
        )

    # Segment breakdown (per brand × channel) for the best model only
    best = min((r for r in rows if r["model"] != "naive (y_t = y_{t-1})"),
               key=lambda r: r["total_cost_UAH"], default=None)
    seg_rows = []
    if best is not None and models[best["model"]] is not None:
        pmodel = models[best["model"]]
        merged = test_base.merge(pmodel, on=KEYCOLS, how="left")
        merged["prediction"] = merged["prediction"].fillna(0)
        merged["over"] = (merged["prediction"] - merged["target_qty"]).clip(lower=0)
        merged["under"] = (merged["target_qty"] - merged["prediction"]).clip(lower=0)
        merged["holding_UAH"] = merged["holding_rate"] * merged["over"] * merged["price"]
        merged["lost_UAH"] = merged["margin_rate"] * (1 - args.recovery) * merged["under"] * merged["price"]
        seg = (
            merged.groupby(["Бренд", "Канал"], observed=True)
            .agg(
                rows=("target_qty", "size"),
                actual_qty=("target_qty", "sum"),
                pred_qty=("prediction", "sum"),
                holding_UAH=("holding_UAH", "sum"),
                lost_UAH=("lost_UAH", "sum"),
            )
            .reset_index()
        )
        seg["total_UAH"] = seg["holding_UAH"] + seg["lost_UAH"]
        seg_rows = seg.sort_values("total_UAH", ascending=False).head(15).to_dict(orient="records")

    # Output
    out_md = _REPO_ROOT / args.output
    out_json = _REPO_ROOT / args.output_json
    out_json.write_text(json.dumps(
        {
            "config": {
                "holding_rate": args.holding_rate,
                "margin_rate": args.margin_rate,
                "recovery": args.recovery,
                "n_test_rows": int(len(test_base)),
                "global_price_fallback": global_price,
            },
            "models": rows,
            "top_segments_best_model": seg_rows,
        },
        indent=2, ensure_ascii=False
    ))

    lines = [
        "# Cost scorecard (UAH)",
        "",
        f"holding_rate = {args.holding_rate} | margin_rate = {args.margin_rate} | recovery = {args.recovery}  ",
        f"test rows: {len(test_base):,}  ",
        "",
        "## Model totals",
        "",
        "| model | total UAH | holding | lost margin | over Q | under Q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: x["total_cost_UAH"]):
        lines.append(
            f"| {r['model']} | {r['total_cost_UAH']:,.0f} | "
            f"{r['holding_cost_UAH']:,.0f} | {r['lost_margin_UAH']:,.0f} | "
            f"{r['over_qty']:,.0f} | {r['under_qty']:,.0f} |"
        )
    if seg_rows:
        lines += [
            "",
            f"## Top-15 segments (brand × channel) under best model ({best['model']})",
            "",
            "| Бренд | Канал | rows | actual | pred | holding | lost | total UAH |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in seg_rows:
            lines.append(
                f"| {s['Бренд']} | {s['Канал']} | {int(s['rows']):,} | "
                f"{s['actual_qty']:,.0f} | {s['pred_qty']:,.0f} | "
                f"{s['holding_UAH']:,.0f} | {s['lost_UAH']:,.0f} | {s['total_UAH']:,.0f} |"
            )
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Scorecard → %s", out_md)
    log.info("JSON → %s", out_json)
    for r in sorted(rows, key=lambda x: x["total_cost_UAH"]):
        log.info("  %-28s total=%10.0f UAH  (holding %.0f / lost %.0f)",
                 r["model"], r["total_cost_UAH"], r["holding_cost_UAH"], r["lost_margin_UAH"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
