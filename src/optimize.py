"""Optuna hyperparameter optimization for LightGBM (CPU, zero cost)."""

from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from src.evaluation import get_feature_columns, wape

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
) -> float:
    feature_cols = get_feature_columns(df_train)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "n_jobs": -1,
        "device": "cpu",
        "verbose": -1,
    }

    train_set = lgb.Dataset(df_train[feature_cols], label=df_train["target_qty"])
    val_set = lgb.Dataset(df_val[feature_cols], label=df_val["target_qty"])

    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    preds = model.predict(df_val[feature_cols]).clip(min=0)
    return wape(df_val["target_qty"].values, preds)


def run_optimization(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_trials: int = 30,
) -> tuple[dict, optuna.Study]:
    """Run Optuna study and return best params + study."""
    study = optuna.create_study(direction="minimize", study_name="lgbm-wape")

    study.optimize(
        lambda trial: objective(trial, df_train, df_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    log.info("Optuna best WAPE=%.4f after %d trials: %s", study.best_value, n_trials, best)
    return best, study
