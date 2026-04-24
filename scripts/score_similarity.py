"""Score prediction file on the V7.3 "similarity to actuals" metric set.

Unlike the UAH cost scorecard, this file rewards symmetric accuracy:
per-row error magnitudes, unbiased aggregates, and monthly-total fidelity.

Pre-registered SIMSCORE = WAPE + 0.005 × |agg_bias_pct| + 0.5 × monthly_WAPE

Used by both ``decision_gate_v73.py`` (for CV) and by the final test-set
evaluation in Step 7.  DO NOT retune the weights after looking at test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SIMSCORE_LAMBDA = 0.005
SIMSCORE_MU = 0.5


def score_frame(df: pd.DataFrame) -> dict:
    """Compute all similarity metrics from a dataframe with columns
    ``Период`` / ``target_qty`` / ``prediction`` (plus key columns).
    """
    y = df["target_qty"].to_numpy(dtype=float)
    p = df["prediction"].to_numpy(dtype=float)
    n = len(df)
    total_y = float(y.sum()) or 1.0
    total_p = float(p.sum())

    wape = float(np.abs(y - p).sum() / total_y)
    mae = float(np.abs(y - p).mean())
    rmse = float(np.sqrt(((y - p) ** 2).mean()))
    bias_units = float((p - y).mean())
    agg_bias_pct = float((total_p / total_y - 1.0) * 100.0)

    nz = y > 0
    smape_nz = (
        float((2 * np.abs(p[nz] - y[nz]) / (np.abs(p[nz]) + np.abs(y[nz]) + 1e-12)).mean())
        if nz.any() else 0.0
    )

    # Fast monthly aggregate: avoid PeriodIndex / groupby overhead by hashing
    # the 'Период' column to integer codes via pandas factorize.
    codes, _ = pd.factorize(df["Период"], sort=False)
    y_by = np.bincount(codes, weights=y)
    p_by = np.bincount(codes, weights=p)
    valid = y_by > 0
    monthly_wape = float(
        (np.abs(p_by[valid] - y_by[valid]) / y_by[valid]).mean()
    ) if valid.any() else 0.0
    g = None  # unused; kept for backward compat

    sim = wape + SIMSCORE_LAMBDA * abs(agg_bias_pct) + SIMSCORE_MU * monthly_wape

    return {
        "n_rows": int(n),
        "WAPE": round(wape, 4),
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "Bias_units": round(bias_units, 4),
        "Agg_Bias_pct": round(agg_bias_pct, 3),
        "SMAPE_nz": round(smape_nz, 4),
        "Monthly_WAPE": round(monthly_wape, 4),
        "SIMSCORE": round(sim, 4),
    }


def score_file(path: Path) -> dict:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "target_qty" not in df.columns or "prediction" not in df.columns:
        raise ValueError(f"{path} missing target_qty/prediction columns")
    return score_frame(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="CSV with Период/target_qty/prediction")
    ap.add_argument("--tag", default="", help="Label for the report row")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    path = Path(args.preds)
    res = score_file(path)
    res["tag"] = args.tag or path.stem
    res["preds_path"] = str(path)

    print(f"\n== {res['tag']}  ({res['n_rows']} rows) ==")
    for k, v in res.items():
        if k in ("tag", "preds_path"): continue
        print(f"  {k:<14} {v}")

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
