"""Build the V10 analytical base table.

V10 = V9 ABT + receipts/stock leading-indicator features.

Adds 17 new lagged features mining three previously-untapped signal classes:
* central-warehouse receipts (4 lags + 2 rolling + 1 growth + 1 ratio)
* central-warehouse stock (3 lags + depletion + DOS + buildup flag)
* retail-trade stock (3 lags + velocity + tt/orc ratio)

Writes ``output/abt_v10_cached.parquet`` and a feature manifest.
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

from src.features_recv_stock_leading import (  # noqa: E402
    build_receipts_stock_features,
)
from src.model_v2 import encode_categoricals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v10_abt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v9_cached.parquet")
    ap.add_argument("--output-abt", default="output/abt_v10_cached.parquet")
    ap.add_argument("--manifest", default="output/v10_feature_manifest.json")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.input_abt)
    log.info("loaded V9 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    for c in ("Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения",
              "Группа_товара"):
        if c in abt.columns and isinstance(abt[c].dtype, pd.CategoricalDtype):
            abt[c] = abt[c].astype(str)

    before = set(abt.columns)
    abt = build_receipts_stock_features(abt)
    new_recv_stock = sorted(set(abt.columns) - before)
    log.info("added receipts/stock features: %s", new_recv_stock)

    abt = encode_categoricals(abt)

    out_abt = _REPO_ROOT / args.output_abt
    abt.to_parquet(out_abt, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_abt, len(abt), abt.shape[1])

    manifest = {
        "rows": int(len(abt)),
        "cols": int(abt.shape[1]),
        "recv_stock_features": new_recv_stock,
    }
    (_REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("manifest → %s", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
