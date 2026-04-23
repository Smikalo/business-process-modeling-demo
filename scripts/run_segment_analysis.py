"""Per-segment WAPE/MAPE_nz for baseline vs full-signal model.

Segments: Канал, Бренд, Группа_товара, volume_tier.
Output: output/segment_analysis.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
log = logging.getLogger("seg")

import numpy as np
import pandas as pd

from src.enrichment_external import enrich_with_external
from src.evaluation import compute_all_metrics, split_train_val_test, wape, mape_nonzero
from src.external_data import get_loader, import_default_loaders, list_loaders
from src.model_v2 import (
    TwoStageForecaster,
    encode_categoricals,
    get_feature_columns_v2,
)

OUTPUT = Path("output")
ABT_CACHE = OUTPUT / "abt_v4_cached.parquet"


def _train(df_train: pd.DataFrame, df_val: pd.DataFrame, feat: list[str]) -> TwoStageForecaster:
    m = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
    )
    m.fit(df_train, df_val, feat, num_boost_round=600, early_stopping=40)
    return m


def _segment_metrics(df: pd.DataFrame, actual: np.ndarray, pred: np.ndarray, keys: list[str]) -> pd.DataFrame:
    out = df[keys + ["target_qty"]].copy()
    out["actual"] = actual
    out["pred"] = pred
    rows: list[dict] = []
    for key in keys:
        grp = out.groupby(key)
        for name, g in grp:
            a = g["actual"].values
            p = g["pred"].values
            rows.append(
                {
                    "segment_key": key,
                    "segment_value": str(name),
                    "n_rows": len(g),
                    "volume": float(a.sum()),
                    "WAPE": round(wape(a, p), 4),
                    "MAPE_nz": round(mape_nonzero(a, p), 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    import_default_loaders()
    loader_names = list_loaders()

    abt = encode_categoricals(pd.read_parquet(ABT_CACHE))

    # Baseline
    log.info("Training baseline")
    df_train_b, df_val_b, df_test_b = split_train_val_test(abt)
    feat_b = get_feature_columns_v2(abt)
    m_base = _train(df_train_b, df_val_b, feat_b)
    preds_base = m_base.predict(df_test_b)

    # Full-signal
    log.info("Training with all external signals")
    abt_full = enrich_with_external(abt, [get_loader(n) for n in loader_names])
    df_train_f, df_val_f, df_test_f = split_train_val_test(abt_full)
    extras = [c for n in loader_names for c in get_loader(n).signal_cols]
    feat_f = list(dict.fromkeys(feat_b + [c for c in extras if c in abt_full.columns]))
    m_full = _train(df_train_f, df_val_f, feat_f)
    preds_full = m_full.predict(df_test_f)

    seg_keys = ["Канал", "Бренд", "Группа_товара"]

    actual = df_test_b["target_qty"].values
    base_df = _segment_metrics(df_test_b, actual, preds_base, seg_keys).rename(
        columns={"WAPE": "baseline_WAPE", "MAPE_nz": "baseline_MAPE_nz"}
    )
    full_df = _segment_metrics(df_test_f, df_test_f["target_qty"].values, preds_full, seg_keys).rename(
        columns={"WAPE": "full_WAPE", "MAPE_nz": "full_MAPE_nz"}
    )

    merged = base_df.merge(
        full_df[["segment_key", "segment_value", "full_WAPE", "full_MAPE_nz"]],
        on=["segment_key", "segment_value"],
        how="left",
    )
    merged["WAPE_delta"] = (merged["full_WAPE"] - merged["baseline_WAPE"]).round(4)
    merged["MAPE_nz_delta"] = (merged["full_MAPE_nz"] - merged["baseline_MAPE_nz"]).round(4)
    merged = merged.sort_values(["segment_key", "volume"], ascending=[True, False])
    merged.to_csv(OUTPUT / "segment_analysis.csv", index=False)

    print("\n=== Biggest WAPE improvements from adding external signals (TEST) ===")
    print(
        merged.sort_values("WAPE_delta")
        .head(15)
        [["segment_key", "segment_value", "n_rows", "volume", "baseline_WAPE", "full_WAPE", "WAPE_delta"]]
        .to_string(index=False)
    )
    print("\n=== Biggest regressions ===")
    print(
        merged.sort_values("WAPE_delta", ascending=False)
        .head(10)
        [["segment_key", "segment_value", "n_rows", "volume", "baseline_WAPE", "full_WAPE", "WAPE_delta"]]
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
