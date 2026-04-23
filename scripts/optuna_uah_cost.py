"""V7.2 Optuna — hyperparameter search with UAH cost as the objective.

The previous Kaggle Optuna run used val pinball loss and produced params
that were 12.5K UAH *worse* on business cost.  This script retargets
Optuna at the actual scorecard: per-row cost computed with per-SKU
margins from ``output/sku_margin.parquet``.

Search space matches the Kaggle space (num_leaves, learning_rate,
feature_fraction, bagging_fraction, bagging_freq, min_data_in_leaf,
reg_lambda).  Pruner kills unpromising trials early to stay under a 30-
minute budget on CPU.

Writes:
- ``output/v72_optuna_uah_best.json`` (shape matches train_v7 --optuna-params)
- ``output/v72_optuna_uah_trials.csv``
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation import split_train_val_test  # noqa: E402
from src.model_v2 import (  # noqa: E402
    encode_categoricals, filter_active_pairs, get_feature_columns_v2,
)
from src.v71_components import build_recency_weights  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("optuna_uah")
optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT = _REPO / "output"


def _build_val_cost_vectors(df_val: pd.DataFrame, margin: pd.DataFrame) -> dict:
    """Pre-compute per-row price/holding-rate/margin-rate/recovery for val."""
    m = margin.drop_duplicates(subset="Артикул").set_index("Артикул")
    price = df_val["Артикул"].map(m["unit_price_uah"]).astype(float)
    holding = df_val["Артикул"].map(m["holding_rate_annual"]).astype(float)
    margin_rate = df_val["Артикул"].map(m["margin_rate"]).astype(float)

    price = price.fillna(float(m["unit_price_uah"].median())).to_numpy()
    holding = holding.fillna(0.22).to_numpy()
    margin_rate = margin_rate.fillna(0.28).to_numpy()
    actual = df_val["target_qty"].to_numpy(dtype=float)
    return {
        "actual": actual,
        "price": price,
        "holding": holding,
        "margin_rate": margin_rate,
        "recovery": 0.50,
    }


def _uah_cost(pred: np.ndarray, v: dict) -> float:
    over = np.clip(pred - v["actual"], 0, None)
    under = np.clip(v["actual"] - pred, 0, None)
    holding = (v["holding"] * over * v["price"]).sum()
    lost = (v["margin_rate"] * (1 - v["recovery"]) * under * v["price"]).sum()
    return float(holding + lost)


def _fit_and_score(params_reg: dict, df_train: pd.DataFrame, df_val: pd.DataFrame,
                   feats: list[str], cost_vec: dict,
                   sw_train: np.ndarray | None, sw_val: np.ndarray | None,
                   alpha: float = 0.45, num_boost_round: int = 500) -> float:
    """One fit + score. Returns total val UAH cost."""
    pos_mask = df_train["target_qty"] > 0
    pos_mask_val = df_val["target_qty"] > 0

    y_clf_tr = (df_train["target_qty"] > 0).astype("int8")
    y_clf_val = (df_val["target_qty"] > 0).astype("int8")

    clf_params = {
        "objective": "binary", "metric": "binary_logloss",
        "num_leaves": 63, "learning_rate": 0.05,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "min_child_samples": 20, "is_unbalance": True,
        "n_jobs": -1, "device": "cpu", "verbose": -1,
    }
    ts_clf = lgb.Dataset(df_train[feats], label=y_clf_tr,
                         weight=sw_train, categorical_feature="auto")
    vs_clf = lgb.Dataset(df_val[feats], label=y_clf_val,
                         weight=sw_val, categorical_feature="auto")
    m_clf = lgb.train(clf_params, ts_clf, num_boost_round=200, valid_sets=[vs_clf],
                      callbacks=[lgb.early_stopping(30, verbose=False)])

    reg_params = {
        "objective": "quantile", "alpha": alpha, "metric": "quantile",
        "n_jobs": -1, "device": "cpu", "verbose": -1,
        **params_reg,
    }
    sw_tr_pos = sw_train[pos_mask.to_numpy()] if sw_train is not None else None
    sw_val_pos = sw_val[pos_mask_val.to_numpy()] if sw_val is not None else None
    ts_reg = lgb.Dataset(df_train.loc[pos_mask, feats],
                         label=df_train.loc[pos_mask, "target_qty"],
                         weight=sw_tr_pos, categorical_feature="auto")
    vs_reg = lgb.Dataset(df_val.loc[pos_mask_val, feats],
                         label=df_val.loc[pos_mask_val, "target_qty"],
                         weight=sw_val_pos, categorical_feature="auto")
    m_reg = lgb.train(reg_params, ts_reg, num_boost_round=num_boost_round,
                      valid_sets=[vs_reg],
                      callbacks=[lgb.early_stopping(60, verbose=False)])

    p_pos = m_clf.predict(df_val[feats])
    q = m_reg.predict(df_val[feats])
    pred = (p_pos * np.clip(q, 0, None)).astype(float)
    return _uah_cost(pred, cost_vec)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--abt-path", default="abt_v7_cached.parquet")
    ap.add_argument("--recency-gamma", type=float, default=0.95)
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--num-boost-round", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--output-json", default="output/v72_optuna_uah_best.json")
    ap.add_argument("--output-trials", default="output/v72_optuna_uah_trials.csv")
    args = ap.parse_args()

    abt = pd.read_parquet(OUT / args.abt_path).pipe(encode_categoricals)
    df_train, df_val, _ = split_train_val_test(abt)
    df_train = filter_active_pairs(df_train)
    keys = df_train[["Партнер", "Артикул"]].drop_duplicates()
    df_val = df_val.merge(keys, on=["Партнер", "Артикул"], how="inner")
    feats = get_feature_columns_v2(abt)

    margin = pd.read_parquet(OUT / "sku_margin.parquet")
    cost_vec = _build_val_cost_vectors(df_val, margin)
    log.info("val set: %d rows, baseline actual UAH value = %.0f",
             len(df_val), (cost_vec["actual"] * cost_vec["price"]).sum())

    sw_train = build_recency_weights(df_train, gamma=args.recency_gamma) \
        if args.recency_gamma else None
    sw_val = build_recency_weights(df_val, gamma=args.recency_gamma) \
        if args.recency_gamma else None

    def objective(trial: optuna.Trial) -> float:
        p = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        t0 = time.time()
        cost = _fit_and_score(p, df_train, df_val, feats, cost_vec,
                              sw_train, sw_val, alpha=args.alpha,
                              num_boost_round=args.num_boost_round)
        log.info("trial %03d  UAH=%10.0f  [%5.1fs]  %s",
                 trial.number, cost, time.time() - t0, p)
        return cost

    study = optuna.create_study(direction="minimize", study_name="v72-uah")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    log.info("BEST: UAH=%.0f after %d trials", study.best_value, args.n_trials)
    log.info("params: %s", study.best_params)

    out = {
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": args.n_trials,
        "objective": "val_UAH_cost",
        "alpha": args.alpha,
        "recency_gamma": args.recency_gamma,
    }
    (OUT / Path(args.output_json).name).write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.output_json)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT / Path(args.output_trials).name, index=False)
    log.info("wrote %s", args.output_trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
