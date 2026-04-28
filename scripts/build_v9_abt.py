"""Build the V9 analytical base table.

V9 = V8 ABT + sales-as-leading-indicator features.

The V8 ABT contains `Количество_sales` (current-month sales qty) and
`lag_1_Выручка_sales` (revenue one month back) but no proper sales-
quantity lag features, no multi-month sales rolling, no sell-through
ratio at multiple horizons, no sales momentum, no sales-share-of-brand.

Sales is the *downstream* signal in the supply chain
(store→consumer); it leads shipments (supplier→distributor) by 1-3
weeks because retailers replenish based on sell-through.

Adds 12 sales-leading features (all lagged ≥ 1 month):
* sales_qty_lag_{1,2,3,6,12,13}
* sales_qty_rmean_{3,6,12}_lag1
* sales_qty_growth_lag1, sales_yoy_ratio_lag1
* sell_through_ratio_lag{1,2}
* sales_lead_signal_lag1, sales_share_of_brand_lag1

Writes ``output/abt_v9_cached.parquet`` and a feature manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.features_sales_leading import build_sales_features  # noqa: E402
from src.model_v2 import encode_categoricals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v9_abt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v8_cached.parquet")
    ap.add_argument("--output-abt", default="output/abt_v9_cached.parquet")
    ap.add_argument("--manifest", default="output/v9_feature_manifest.json")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.input_abt)
    log.info("loaded V8 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    cat_cols_to_restore = {}
    for c in ("Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения",
              "Группа_товара"):
        if c in abt.columns and isinstance(abt[c].dtype, pd.CategoricalDtype):
            cat_cols_to_restore[c] = abt[c].cat.categories
            abt[c] = abt[c].astype(str)

    before = set(abt.columns)
    abt = build_sales_features(abt)
    log.info("added sales-leading features: %s",
             sorted(set(abt.columns) - before))
    new_sales = sorted(set(abt.columns) - before)

    abt = encode_categoricals(abt)

    out_abt = _REPO_ROOT / args.output_abt
    abt.to_parquet(out_abt, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_abt, len(abt), abt.shape[1])

    manifest = {
        "rows": int(len(abt)),
        "cols": int(abt.shape[1]),
        "sales_leading_features": new_sales,
    }
    (_REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("manifest → %s", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
