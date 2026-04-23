"""Train a single model with ALL external signals and report feature importance.

Saves:
  output/feature_importance_v5.csv             — native LightGBM gain per feature
  output/feature_importance_by_source.csv      — grouped by loader
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
log = logging.getLogger("fi")

import pandas as pd

from src.enrichment_external import enrich_with_external
from src.evaluation import split_train_val_test
from src.external_data import get_loader, import_default_loaders, list_loaders
from src.model_v2 import (
    TwoStageForecaster,
    encode_categoricals,
    get_feature_columns_v2,
)

OUTPUT = Path("output")
ABT_CACHE = OUTPUT / "abt_v4_cached.parquet"


def main() -> int:
    import_default_loaders()
    loader_names = list_loaders()
    log.info("Loading baseline ABT")
    abt = encode_categoricals(pd.read_parquet(ABT_CACHE))

    loaders = [get_loader(n) for n in loader_names]
    log.info("Enriching with %d loaders: %s", len(loaders), loader_names)
    abt = enrich_with_external(abt, loaders)

    extras = [c for L in loaders for c in L.signal_cols]
    base = get_feature_columns_v2(abt)
    feat_cols = list(dict.fromkeys(base + [c for c in extras if c in abt.columns]))

    df_train, df_val, _ = split_train_val_test(abt)
    model = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
    )
    model.fit(df_train, df_val, feat_cols, num_boost_round=800, early_stopping=50)

    fi = model.feature_importance()

    # Attribute each feature to its source loader (or "internal").
    col_to_source = {}
    for L in loaders:
        for c in L.signal_cols:
            col_to_source[c] = L.name
    fi["source"] = fi["feature"].map(lambda f: col_to_source.get(f, "internal"))

    fi.to_csv(OUTPUT / "feature_importance_v5.csv", index=False)

    # Grouped summary.
    grouped = (
        fi.groupby("source", as_index=False)
        .agg(
            n_features=("feature", "count"),
            total_gain=("gain_total", "sum"),
            mean_gain=("gain_total", "mean"),
            max_gain=("gain_total", "max"),
        )
        .sort_values("total_gain", ascending=False)
    )
    total = grouped["total_gain"].sum()
    grouped["pct_of_total"] = (grouped["total_gain"] / total * 100).round(2)
    grouped.to_csv(OUTPUT / "feature_importance_by_source.csv", index=False)

    print("\n=== Top 25 features by combined gain ===")
    print(fi.head(25).to_string(index=False))
    print("\n=== Gain contribution by source ===")
    print(grouped.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
