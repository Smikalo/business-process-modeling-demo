"""Build the V8 analytical base table.

V8 = V7 ABT + within-month/weekly features (extracted from raw daily
shipment data) + re-introduced ``school_calendar_ua`` external loader.

Why these specific additions:

* **Within-month features** are *genuinely new information* — every prior
  ABT generation collapses to monthly grain at ingestion, throwing away
  the within-month timing signal entirely.  See
  ``src/features_within_month.py``.
* **school_ua** was rejected at V5 under WAPE only.  Since SIMSCORE
  weights monthly-WAPE heavily and back-to-school / winter-break dates
  are exactly monthly-level signals, this loader deserves a re-evaluation
  under the V7 backbone + SIMSCORE objective (per V7.7 final report's
  "what is now most promising" item #4).

Writes ``output/abt_v8_cached.parquet`` and a feature manifest.

The script does NOT touch the V7 ABT or any prior artefact — V7.7 / V7.8
predictions remain valid and available for the V8 LAD stack.
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

from src.features_within_month import add_within_month_features  # noqa: E402
from src.loaders.school_calendar_ua import SchoolCalendarLoader  # noqa: E402
from src.model_v2 import encode_categoricals  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v8_abt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v7_cached.parquet")
    ap.add_argument("--output-abt", default="output/abt_v8_cached.parquet")
    ap.add_argument("--manifest", default="output/v8_feature_manifest.json")
    ap.add_argument("--no-within-month", action="store_true",
                    help="Skip within-month features (debug only).")
    ap.add_argument("--no-school", action="store_true",
                    help="Skip school_calendar_ua (debug only).")
    args = ap.parse_args()

    abt = pd.read_parquet(_REPO_ROOT / args.input_abt)
    log.info("loaded V7 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    before = set(abt.columns)
    if not args.no_within_month:
        abt = add_within_month_features(abt)
        log.info("added within-month features: %s",
                 sorted(set(abt.columns) - before))
    new_wm = sorted(set(abt.columns) - before)

    before = set(abt.columns)
    if not args.no_school:
        loader = SchoolCalendarLoader()
        signals = loader.transform(loader.fetch_raw())
        abt["Период"] = abt["Период"].astype("period[M]")
        signals["Период"] = signals["Период"].astype("period[M]")
        abt = abt.merge(signals, on="Период", how="left")
        log.info("added school_ua features: %s",
                 sorted(set(abt.columns) - before))
    new_school = sorted(set(abt.columns) - before)

    abt = encode_categoricals(abt)

    out_abt = _REPO_ROOT / args.output_abt
    abt.to_parquet(out_abt, index=False)
    log.info("wrote %s (%d rows, %d cols)", out_abt, len(abt), abt.shape[1])

    manifest = {
        "rows": int(len(abt)),
        "cols": int(abt.shape[1]),
        "within_month_features": new_wm,
        "school_features": new_school,
    }
    (_REPO_ROOT / args.manifest).write_text(json.dumps(manifest, indent=2))
    log.info("manifest → %s", args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
