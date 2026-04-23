"""Generate predictions from saved V4/V5 models for cost-scorecard comparison.

Loads models from ``output/model_v4_final.joblib`` and ``output/model_v5.joblib``
(both are ``TwoStageForecaster`` instances), re-applies the same split as
``train_v5.py``, and writes ``output/preds_{v4,v5}_test.csv``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation import compute_all_metrics, split_train_val_test  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster,
    encode_categoricals,
    filter_active_pairs,
    get_feature_columns_v2,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gen_baseline_preds")


def generate(version: str, abt_path: Path, model_path: Path | None, out_test: Path, out_val: Path) -> None:
    if not abt_path.exists():
        log.warning("ABT %s missing — skipping %s", abt_path, version)
        return
    df = pd.read_parquet(abt_path).pipe(encode_categoricals)
    df_train, df_val, df_test = split_train_val_test(df)
    active = filter_active_pairs(df_train)
    keys = active[["Партнер", "Артикул"]].drop_duplicates()
    df_val = df_val.merge(keys, on=["Партнер", "Артикул"], how="inner")
    df_test = df_test.merge(keys, on=["Партнер", "Артикул"], how="inner")

    feats = get_feature_columns_v2(active)

    if model_path is not None and model_path.exists():
        model = joblib.load(model_path)
        if not hasattr(model, "predict"):
            log.info("%s model is not a TwoStageForecaster — retraining a plain baseline.", version)
            model = None
    else:
        model = None

    if model is None:
        log.info("Training plain %s TwoStage baseline for prediction dump...", version)
        model = TwoStageForecaster(
            clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
            reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
        )
        model.fit(active, df_val, feats, num_boost_round=1200, early_stopping=60)

    p_val = model.predict(df_val)
    p_test = model.predict(df_test)

    m_val = compute_all_metrics(df_val["target_qty"].to_numpy(), p_val)
    m_test = compute_all_metrics(df_test["target_qty"].to_numpy(), p_test)
    log.info("%s val: %s", version, m_val)
    log.info("%s test: %s", version, m_test)

    for path, split_df, preds in [(out_val, df_val, p_val), (out_test, df_test, p_test)]:
        out = split_df[["Период", "Партнер", "Артикул", "target_qty"]].copy()
        out["prediction"] = preds
        out.to_csv(path, index=False)
        log.info("Wrote %s", path)


def main() -> int:
    generate(
        "V4",
        _REPO_ROOT / "output" / "abt_v4_cached.parquet",
        _REPO_ROOT / "output" / "model_v4_final.joblib",
        _REPO_ROOT / "output" / "preds_v4_test.csv",
        _REPO_ROOT / "output" / "preds_v4_val.csv",
    )
    generate(
        "V5",
        _REPO_ROOT / "output" / "abt_v5_cached.parquet",
        _REPO_ROOT / "output" / "model_v5.joblib",
        _REPO_ROOT / "output" / "preds_v5_test.csv",
        _REPO_ROOT / "output" / "preds_v5_val.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
