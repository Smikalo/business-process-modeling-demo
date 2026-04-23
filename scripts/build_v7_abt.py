"""Build the V7 analytical base table.

V7 = V6 ABT + relative price/elasticity features + cohort/substitution
features. Writes `output/abt_v7_cached.parquet` and a feature manifest.
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

from src.features_price import add_price_features
from src.features_cohort import add_cohort_features
from src.model_v2 import encode_categoricals

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v7_abt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v6_cached.parquet")
    ap.add_argument("--output-abt", default="output/abt_v7_cached.parquet")
    ap.add_argument("--manifest", default="output/v7_feature_manifest.json")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.input_abt)
    log.info("loaded V6 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    before = set(abt.columns)
    abt = add_price_features(abt)
    log.info("added price features: %s", sorted(set(abt.columns) - before))

    before = set(abt.columns)
    abt = add_cohort_features(abt)
    log.info("added cohort features: %s", sorted(set(abt.columns) - before))

    abt = encode_categoricals(abt)

    out_abt = _REPO_ROOT / args.output_abt
    abt.to_parquet(out_abt, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_abt, len(abt), abt.shape[1])

    manifest = {
        "rows": int(len(abt)),
        "cols": int(abt.shape[1]),
        "price_features": [
            "price_lag1", "price_lag3", "price_vs_brand_median",
            "price_vs_channel_median", "price_vs_rrc",
            "price_change_3m_pct", "sku_price_elasticity",
        ],
        "cohort_features": [
            "cohort_demand_lag1", "cohort_stockout_share_lag1",
            "cohort_size", "cannibalisation_pressure",
        ],
    }
    (_REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("manifest → %s", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
