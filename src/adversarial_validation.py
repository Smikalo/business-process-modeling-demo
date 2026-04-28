"""V11 -- adversarial validation utilities.

Train a binary classifier to distinguish training-window rows from
recent-validation-window rows.  If the classifier achieves AUC > 0.55,
the two windows are distributionally distinct (= drift detected).

Two outputs:

1. **Drift importance**: per-feature contribution to the classifier's
   AUC -- ranks features by how much they identify "is this row from the
   recent regime?".  The top-K features ARE the drift indicators.

2. **Importance weights for retraining**: for each TRAIN row, the
   classifier's predicted probability `p_recent(x)`.  Use the
   density-ratio estimator
        w(x) = p_recent(x) / (1 - p_recent(x))   [clipped to [0.1, 10]]
   as a sample weight when retraining the demand model.  This pushes
   the model to *focus* on the regions of feature-space that look like
   the recent / test regime, without ever using test labels.

References
----------
Sugiyama et al., *Density Ratio Estimation in Machine Learning*, 2012.
The "covariate shift" literature is the canonical foundation; in Kaggle
practice this is just called "adversarial validation" since
~Pavel Pleskov (2017).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)


@dataclass
class DriftAudit:
    """Result of an adversarial-validation audit."""

    auc_oof: float
    auc_in_sample: float
    feature_importance: pd.DataFrame
    train_weights: pd.Series  # indexed by row id
    p_recent_on_train: np.ndarray
    n_train: int
    n_recent: int


def adversarial_audit(
    abt: pd.DataFrame,
    train_period_start: str,
    train_period_end: str,
    recent_period_start: str,
    recent_period_end: str,
    feature_cols: list[str],
    cat_cols: list[str] | None = None,
    n_splits: int = 4,
    weight_clip: tuple[float, float] = (0.1, 10.0),
    seed: int = 17,
) -> DriftAudit:
    """Run an adversarial-validation audit between two time windows.

    Parameters
    ----------
    abt
        Full ABT containing both `Период` (string period[M]) and the
        feature columns.
    train_period_start / train_period_end
        Inclusive YYYY-MM strings defining the "old" window.
    recent_period_start / recent_period_end
        Inclusive YYYY-MM strings defining the "recent" window.
    feature_cols
        Numeric or categorical columns to feed the adversarial classifier.
    cat_cols
        Subset of `feature_cols` that are categorical (LightGBM hint).
    n_splits
        Number of stratified CV folds for the OOF AUC and weights.
    weight_clip
        Lower/upper clip on the density-ratio weight to prevent runaway.

    Returns
    -------
    DriftAudit
    """
    cat_cols = cat_cols or []
    pers = abt["Период"].astype(str)
    train_mask = (pers >= train_period_start) & (pers <= train_period_end)
    recent_mask = (pers >= recent_period_start) & (pers <= recent_period_end)
    keep = train_mask | recent_mask
    df = abt.loc[keep].copy()
    df["__y_recent"] = recent_mask[keep].astype(int).to_numpy()
    log.info("adversarial audit: %d train rows  vs  %d recent rows",
             int(train_mask.sum()), int(recent_mask.sum()))

    X = df[feature_cols]
    y = df["__y_recent"].to_numpy()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(df), dtype=np.float64)
    fi = np.zeros(len(feature_cols), dtype=np.float64)

    for fold_ix, (tr, va) in enumerate(skf.split(X, y)):
        ds_tr = lgb.Dataset(X.iloc[tr], y[tr],
                            categorical_feature=cat_cols if cat_cols else "auto")
        ds_va = lgb.Dataset(X.iloc[va], y[va], reference=ds_tr,
                            categorical_feature=cat_cols if cat_cols else "auto")
        booster = lgb.train(
            params={
                "objective": "binary", "metric": "auc",
                "num_leaves": 63, "learning_rate": 0.05,
                "min_child_samples": 50, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "lambda_l2": 1.0, "verbose": -1, "seed": seed + fold_ix,
            },
            train_set=ds_tr, valid_sets=[ds_va], num_boost_round=600,
            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)],
        )
        oof[va] = booster.predict(X.iloc[va],
                                  num_iteration=booster.best_iteration)
        fi += booster.feature_importance(importance_type="gain")

    from sklearn.metrics import roc_auc_score
    auc_oof = float(roc_auc_score(y, oof))

    full_ds = lgb.Dataset(X, y,
                          categorical_feature=cat_cols if cat_cols else "auto")
    final = lgb.train(
        params={"objective": "binary", "metric": "auc",
                "num_leaves": 63, "learning_rate": 0.05,
                "min_child_samples": 50, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 5,
                "lambda_l2": 1.0, "verbose": -1, "seed": seed},
        train_set=full_ds, num_boost_round=400,
    )
    auc_full = float(roc_auc_score(y, final.predict(X)))

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "gain_total_avg_per_fold": fi / n_splits,
    }).sort_values("gain_total_avg_per_fold", ascending=False).reset_index(drop=True)

    train_idx = np.where(df["__y_recent"].to_numpy() == 0)[0]
    p_train = oof[train_idx]
    p_train_safe = np.clip(p_train, 1e-3, 1 - 1e-3)
    w_raw = p_train_safe / (1.0 - p_train_safe)
    w_clipped = np.clip(w_raw, *weight_clip).astype(np.float32)
    train_global_idx = df.iloc[train_idx].index
    train_weights = pd.Series(w_clipped, index=train_global_idx,
                              name="adv_weight")

    return DriftAudit(
        auc_oof=auc_oof,
        auc_in_sample=auc_full,
        feature_importance=fi_df,
        train_weights=train_weights,
        p_recent_on_train=p_train,
        n_train=int(train_mask.sum()),
        n_recent=int(recent_mask.sum()),
    )
