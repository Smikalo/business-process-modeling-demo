"""V11 Priority 3 -- recent-only-window ABT.

Creates a filtered V10 ABT containing only rows from 2023-01 onwards.
This is the "hyper-recent" training window: it gives up the long
historical context (~3 years of pre-2023 data) in exchange for
training on a regime that more closely matches the test distribution.

Validation and test windows are unchanged.

Outputs:
* `output/abt_v11_recent_only_cached.parquet`
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v11_recent_only")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-abt", default="output/abt_v10_cached.parquet")
    ap.add_argument("--output-abt",
                    default="output/abt_v11_recent_only_cached.parquet")
    ap.add_argument("--cutoff", default="2023-01",
                    help="YYYY-MM minimum period kept in training window")
    args = ap.parse_args()

    abt = pd.read_parquet(REPO / args.input_abt)
    log.info("loaded V10 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    pers = abt["Период"].astype(str)
    train_window_end = "2024-06"
    val_start = "2024-07"
    val_end = "2025-06"
    test_start = "2025-07"

    is_train = (pers >= args.cutoff) & (pers <= train_window_end)
    is_val = (pers >= val_start) & (pers <= val_end)
    is_test = pers >= test_start
    keep = is_train | is_val | is_test

    n_dropped = int((~keep).sum())
    abt2 = abt.loc[keep].copy()
    log.info("recent-only filter (cutoff=%s): %d rows kept, %d dropped "
             "(train kept %d -> %d)",
             args.cutoff, len(abt2), n_dropped,
             int(((pers >= "2020-01") & (pers <= train_window_end)).sum()),
             int(is_train.sum()))

    abt2.to_parquet(REPO / args.output_abt, index=False)
    log.info("wrote %s", args.output_abt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
