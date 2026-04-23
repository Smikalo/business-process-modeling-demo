"""Sweep pinball quantile α for V7 and collect WAPE + UAH cost on test."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "output"


def run_alpha(alpha: float) -> dict:
    subprocess.run([
        sys.executable, "-m", "scripts.train_v7",
        "--alpha", str(alpha),
        "--disable-residual",
        "--stacker-alpha", "10.0",
        "--num-boost-round", "1200",
    ], check=True, cwd=_REPO, capture_output=True)

    subprocess.run([
        sys.executable, "-m", "scripts.decision_cost_scorecard",
        "--margin-table", "output/sku_margin.parquet",
        "--output", f"output/cost_scorecard_v7_a{alpha}.md",
        "--output-json", f"output/cost_scorecard_v7_a{alpha}.json",
    ], check=True, cwd=_REPO, capture_output=True)

    metrics = pd.read_csv(OUT / "v7_metrics.csv")
    cal_test = metrics[(metrics["model"] == "V7_cal") & (metrics["split"] == "test")].iloc[0]

    cost = json.loads((OUT / f"cost_scorecard_v7_a{alpha}.json").read_text())
    v7_cost = next(m for m in cost["models"] if m["model"] == "V7")

    return {
        "alpha": alpha,
        "test_WAPE": float(cal_test["WAPE"]),
        "test_Bias": float(cal_test["Bias"]),
        "holding_UAH": v7_cost["holding_cost_UAH"],
        "lost_UAH": v7_cost["lost_margin_UAH"],
        "total_UAH": v7_cost["total_cost_UAH"],
    }


if __name__ == "__main__":
    rows = []
    for alpha in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        r = run_alpha(alpha)
        print(f"α={alpha:.2f}  WAPE={r['test_WAPE']:.4f}  Bias={r['test_Bias']:+.3f}  "
              f"cost={r['total_UAH']:>11,.0f} UAH")
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "v7_alpha_sweep.csv", index=False)
    print("saved output/v7_alpha_sweep.csv")
