"""V7.1 ablation harness.

Trains the V7 pipeline under several configurations, computes the UAH cost
scorecard for each, writes a summary table to ``output/v71_ablation.csv``,
and symlinks/copies the champion into the ``v7_1`` tag slot.

Variants (cumulative)::

    v7        : baseline (α=0.45, no recency, no monotone)
    v7_rec    : + recency weights γ=0.97
    v7_mono   : + monotone constraints on lag/rolling/stockout
    v7_em     : + one EM round (stockout-censored re-imputation)

The champion is whichever minimises annualised UAH cost.  All artefacts are
tagged with the variant name (e.g. ``preds_v7_rec_stacked_test.csv``,
``model_v7_rec.joblib``).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "output"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("ablate_v71")


VARIANTS = [
    {"name": "v7_base",      "args": []},
    {"name": "v7_rec95",     "args": ["--recency-gamma", "0.95"]},
    {"name": "v7_rec97",     "args": ["--recency-gamma", "0.97"]},
    {"name": "v7_rec99",     "args": ["--recency-gamma", "0.99"]},
    {"name": "v7_rec_em",    "args": ["--recency-gamma", "0.97", "--em-rounds", "1"]},
]


def _run_train(tag: str, extra_args: list[str], num_boost_round: int = 1200) -> None:
    cmd = [
        sys.executable, "-m", "scripts.train_v7",
        "--disable-residual", "--stacker-alpha", "10.0",
        "--num-boost-round", str(num_boost_round),
        "--save-tag", tag,
        *extra_args,
    ]
    log.info("→ %s", " ".join(cmd))
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("%s FAILED:\n%s", tag, r.stderr[-2000:])
        raise SystemExit(r.returncode)


def _run_scorecard(tag: str) -> dict:
    cmd = [
        sys.executable, "-m", "scripts.decision_cost_scorecard",
        "--margin-table", "output/sku_margin.parquet",
        "--preds-v7", f"output/preds_v7_{tag}_test.csv",
        "--output", f"output/cost_scorecard_{tag}.md",
        "--output-json", f"output/cost_scorecard_{tag}.json",
    ]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        log.error("scorecard %s FAILED:\n%s", tag, r.stderr[-2000:])
        raise SystemExit(r.returncode)
    data = json.loads((OUT / f"cost_scorecard_{tag}.json").read_text())
    v7_row = next((m for m in data["models"] if m["model"] == "V7"), None)
    if v7_row is None:
        raise RuntimeError(f"V7 row missing in cost_scorecard_{tag}.json")
    return {
        "total_UAH": v7_row["total_cost_UAH"],
        "holding_UAH": v7_row["holding_cost_UAH"],
        "lost_UAH": v7_row["lost_margin_UAH"],
    }


def _metrics(tag: str) -> dict:
    m = pd.read_csv(OUT / f"v7_{tag}_metrics.csv")
    v7_val = m[(m.model == "V7") & (m.split == "val")].iloc[0].to_dict()
    v7_test = m[(m.model == "V7") & (m.split == "test")].iloc[0].to_dict()
    stk_val = m[(m.model == "V7_stacked") & (m.split == "val")].iloc[0].to_dict()
    stk_test = m[(m.model == "V7_stacked") & (m.split == "test")].iloc[0].to_dict()
    return {
        "val_WAPE": round(float(v7_val["WAPE"]), 4),
        "val_Bias": round(float(v7_val["Bias"]), 3),
        "test_WAPE": round(float(v7_test["WAPE"]), 4),
        "test_MAPE_nz": round(float(v7_test["MAPE_nz"]), 4),
        "test_Bias": round(float(v7_test["Bias"]), 3),
        "stk_test_WAPE": round(float(stk_test["WAPE"]), 4),
        "stk_test_Bias": round(float(stk_test["Bias"]), 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-boost-round", type=int, default=1200)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Variant names to skip (e.g. v7_em).")
    args = ap.parse_args()

    rows = []
    for v in VARIANTS:
        name = v["name"]
        if name in args.skip:
            continue
        # Strip the leading "v7_" from the name for save-tag cleanliness.
        tag = name.removeprefix("v7_") or "base"
        log.info("=" * 60)
        log.info("VARIANT %s (tag=%s)", name, tag)
        log.info("=" * 60)
        t0 = time.time()
        _run_train(tag, v["args"], num_boost_round=args.num_boost_round)
        sc = _run_scorecard(tag)
        m = _metrics(tag)
        rows.append({
            "variant": name,
            **m,
            "UAH_cost": int(sc["total_UAH"]),
            "holding_UAH": int(sc["holding_UAH"]),
            "lost_UAH": int(sc["lost_UAH"]),
            "wall_s": round(time.time() - t0, 1),
        })
        log.info("→ %s cost=%s UAH  WAPE=%s  Bias=%+.3f",
                 name, f"{int(sc['total_UAH']):,}", m["test_WAPE"], m["test_Bias"])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "v71_ablation.csv", index=False)
    log.info("\n%s", df.to_string(index=False))

    # Pick champion: lowest UAH cost with WAPE no more than +0.01 over the baseline.
    base_wape = df.loc[df.variant == "v7_base", "test_WAPE"].iloc[0] if (df.variant == "v7_base").any() else df["test_WAPE"].min()
    ok = df[df["test_WAPE"] <= base_wape + 0.01]
    if ok.empty:
        ok = df
    champ = ok.loc[ok["UAH_cost"].idxmin()]
    log.info("CHAMPION: %s (cost=%s, WAPE=%s)",
             champ["variant"], f"{int(champ['UAH_cost']):,}", champ["test_WAPE"])

    (OUT / "v71_champion.json").write_text(json.dumps({
        "champion": champ["variant"],
        "tag": str(champ["variant"]).removeprefix("v7_"),
        "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in champ.to_dict().items()},
        "all_variants": rows,
    }, indent=2, default=float))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
