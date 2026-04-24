"""Retrain the V7 base model with SYMMETRIC regression objectives.

The V7.2 / V7.1 / V7 family all use pinball-quantile loss (α=0.45 or α=0.6)
which biases predictions toward over- or under-forecasting.  In the V7.3
NNLS stack they get zero weight on some channels.  This script trains
two extra base learners with *symmetric* loss (Tweedie, MAE) using the
same V7 feature set, then saves their predictions to the stacker pool.

Outputs:
    output/preds_v7sym_tweedie_{val,test}.csv
    output/preds_v7sym_mae_{val,test}.csv

The script is intentionally minimal (no stacking, no calibration, no
residual corrector) to keep the runtime bounded on laptop CPU.

Usage:
    OMP_NUM_THREADS=4 python -m scripts.v75_train_symmetric \
        --objective tweedie --rounds 250
    OMP_NUM_THREADS=4 python -m scripts.v75_train_symmetric \
        --objective mae --rounds 250
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "4")

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation import split_train_val_test  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster,
    encode_categoricals,
    filter_active_pairs,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v75_sym")

OUT = _REPO / "output"


OBJECTIVES = {
    "tweedie": {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
    },
    "mae": {
        "objective": "regression_l1",
    },
    "huber": {
        "objective": "huber",
        "alpha": 0.9,
    },
    "l2": {
        "objective": "regression",
    },
}


def _save_preds(df: pd.DataFrame, preds: np.ndarray, path: Path) -> None:
    out = df[["Период", "Партнер", "Артикул", "target_qty"]].copy()
    out["prediction"] = np.clip(preds, 0, None)
    out.to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=sorted(OBJECTIVES), default="tweedie")
    ap.add_argument("--abt", default="abt_v7_cached.parquet")
    ap.add_argument("--rounds", type=int, default=250)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--lr", type=float, default=0.07)
    ap.add_argument("--target", default="target_qty_imputed")
    ap.add_argument("--tag", default=None,
                    help="override output tag (default: objective name)")
    args = ap.parse_args()

    tag = args.tag or args.objective
    t_all = time.time()

    abt_path = OUT / args.abt if not Path(args.abt).is_absolute() else Path(args.abt)
    abt = pd.read_parquet(abt_path).pipe(encode_categoricals)
    feats = [c for c in get_feature_columns_v2(abt) if c != "target_qty_imputed"]
    log.info("ABT: %d rows  |  %d features  |  objective=%s", len(abt), len(feats), args.objective)

    df_train, df_val, df_test = split_train_val_test(abt)
    active = filter_active_pairs(df_train)
    keys = active[["Партнер", "Артикул"]].drop_duplicates()
    df_val = df_val.merge(keys, on=["Партнер", "Артикул"], how="inner")
    df_test = df_test.merge(keys, on=["Партнер", "Артикул"], how="inner")
    log.info("train=%d  val=%d  test=%d", len(active), len(df_val), len(df_test))

    reg_params = {
        "num_leaves": args.num_leaves,
        "learning_rate": args.lr,
        "min_child_samples": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "num_threads": 4,
        "force_col_wise": True,
        "verbose": -1,
        **OBJECTIVES[args.objective],
    }

    model = TwoStageForecaster(
        clf_params={"num_leaves": 63, "learning_rate": 0.05,
                    "min_child_samples": 30,
                    "num_threads": 4, "force_col_wise": True, "verbose": -1},
        reg_params=reg_params,
        reg_objective="",
        target_col=args.target,
    )
    t0 = time.time()
    model.fit(active, df_val, feats,
              num_boost_round=args.rounds, early_stopping=args.early_stop)
    log.info("fit in %.1fs", time.time() - t0)

    p_val = model.predict(df_val)
    p_test = model.predict(df_test)
    _save_preds(df_val, p_val, OUT / f"preds_v7sym_{tag}_val.csv")
    _save_preds(df_test, p_test, OUT / f"preds_v7sym_{tag}_test.csv")

    def _metric(y, p):
        wape = float(np.abs(y - p).sum() / max(float(y.sum()), 1.0))
        bias = float((p.sum() - y.sum()) / max(float(y.sum()), 1.0) * 100.0)
        return wape, bias

    y_val = df_val["target_qty"].to_numpy()
    y_test = df_test["target_qty"].to_numpy()
    vw, vb = _metric(y_val, p_val)
    tw, tb = _metric(y_test, p_test)
    log.info("v7sym_%s  val WAPE=%.4f bias%%=%+.2f  |  test WAPE=%.4f bias%%=%+.2f",
             tag, vw, vb, tw, tb)
    log.info("total %.1fs", time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
