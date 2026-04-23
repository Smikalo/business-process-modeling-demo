"""V7.2 monthly calibrator experiment.

The V7.1 champion systematically under-forecasts every month in validation
(−0.7 % to −15.7 %).  That is an artefact of the α=0.45 pinball loss which
intentionally biases predictions down to trade holding-cost for
lost-margin in the newsvendor objective.

This script tests whether a **per-month multiplicative correction** learned
from validation residuals improves annual UAH cost on the test set.

For each month m:
    c_m = sum(y_val_m) / sum(yhat_val_m)

Then apply to test predictions as:
    yhat_test_corrected = c_{month(test)} * yhat_test

If the α=0.45 bias is truly cost-optimal, this should INCREASE UAH cost.
If not, it's a free win.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "output"


def _score(preds_path: Path, tag: str) -> dict:
    cmd = [
        sys.executable, "-m", "scripts.decision_cost_scorecard",
        "--margin-table", "output/sku_margin.parquet",
        "--preds-v7", str(preds_path.relative_to(_REPO)),
        "--output", f"output/cost_scorecard_{tag}.md",
        "--output-json", f"output/cost_scorecard_{tag}.json",
    ]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    data = json.loads((OUT / f"cost_scorecard_{tag}.json").read_text())
    v7_row = next(m for m in data["models"] if m["model"] == "V7")
    return v7_row


def build_monthly_factors(preds_val_path: Path) -> dict[int, float]:
    """c_m = sum(actual_m) / sum(pred_m), per month-of-year."""
    df = pd.read_csv(preds_val_path)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M").to_timestamp()
    df["month"] = df["Период"].dt.month
    g = df.groupby("month").agg(actual=("target_qty", "sum"),
                                pred=("prediction", "sum"))
    return {int(m): float(a / p) if p > 0 else 1.0
            for m, (a, p) in g.iterrows()}


def apply_correction(preds_path: Path, factors: dict[int, float], out_path: Path) -> None:
    df = pd.read_csv(preds_path)
    df["Период"] = pd.PeriodIndex(df["Период"].astype(str), freq="M").to_timestamp()
    df["month"] = df["Период"].dt.month
    df["prediction"] = df["prediction"] * df["month"].map(factors).astype(float)
    df["prediction"] = df["prediction"].clip(lower=0)
    df[["Период", "Партнер", "Артикул", "target_qty", "prediction"]].to_csv(
        out_path, index=False
    )


def main() -> int:
    champions = [
        ("v71_baseline", "preds_v71_channels_val.csv", "preds_v71_channels_test.csv"),
        ("v72_seasonal", "preds_v72_val.csv",          "preds_v72_test.csv"),
    ]

    rows = []
    for champ_tag, val_name, test_name in champions:
        val_path = OUT / val_name
        test_path = OUT / test_name
        if not val_path.exists() or not test_path.exists():
            print(f"skip {champ_tag}: missing preds"); continue

        uncal = _score(test_path, f"{champ_tag}_uncalibrated_tmp")
        rows.append({"champion": champ_tag, "variant": "raw",
                     **{k: uncal[k] for k in ("total_cost_UAH","holding_cost_UAH","lost_margin_UAH")}})

        factors = build_monthly_factors(val_path)
        print(f"\n{champ_tag} monthly factors (learned on val):")
        for m in range(1, 13):
            print(f"  month={m:2d}  c={factors.get(m, 1.0):.4f}")

        calibrated_path = OUT / f"preds_{champ_tag}_monthcal_test.csv"
        apply_correction(test_path, factors, calibrated_path)
        cal = _score(calibrated_path, f"{champ_tag}_monthcal_tmp")
        rows.append({"champion": champ_tag, "variant": "month_calibrated",
                     **{k: cal[k] for k in ("total_cost_UAH","holding_cost_UAH","lost_margin_UAH")}})

        delta = cal["total_cost_UAH"] - uncal["total_cost_UAH"]
        print(f"\n{champ_tag} total UAH: raw={uncal['total_cost_UAH']:>12,}  "
              f"calibrated={cal['total_cost_UAH']:>12,}  Δ={delta:>+10,}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / "v72_monthcal_results.csv", index=False)
    print("\n", tbl.to_string(index=False), sep="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
