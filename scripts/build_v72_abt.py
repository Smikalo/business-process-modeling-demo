"""Build the V7.2 analytical base table.

V7.2 = V7 ABT + Q4 / seasonal-lift features. Writes
`output/abt_v72_cached.parquet` and a feature manifest.
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

from src.features_seasonal import add_seasonal_features
from src.model_v2 import encode_categoricals

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v72_abt")

SEASONAL_FEATURES = [
    "is_xmas_window", "month_of_year", "months_to_xmas",
    "sku_dec_lift_lag1y", "brand_channel_dec_lift", "y_lag12",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v7_cached.parquet")
    ap.add_argument("--output-abt", default="output/abt_v72_cached.parquet")
    ap.add_argument("--manifest", default="output/v72_feature_manifest.json")
    ap.add_argument("--train-cutoff", default=None,
                    help="ISO date; Dec-lift aggregates use rows strictly "
                         "before this.  Default: ABT_max − 20 months.")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.input_abt)
    log.info("loaded V7 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    before = set(abt.columns)
    abt = add_seasonal_features(abt, train_cutoff=args.train_cutoff)
    new_cols = sorted(set(abt.columns) - before)
    log.info("added seasonal features: %s", new_cols)

    abt = encode_categoricals(abt)

    out_abt = _REPO_ROOT / args.output_abt
    abt.to_parquet(out_abt, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_abt, len(abt), abt.shape[1])

    manifest = {
        "rows": int(len(abt)),
        "cols": int(abt.shape[1]),
        "seasonal_features": new_cols,
    }
    (_REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("manifest → %s", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
