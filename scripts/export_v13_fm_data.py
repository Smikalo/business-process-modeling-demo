"""Export per-pair monthly history for V13 foundation-model fine-tuning.

Foundation models (Chronos, TimesFM, Moirai) operate on per-series
monthly history (no exogenous features needed for the FM forward pass —
exogenous features re-enter at the LAD merge stage). This script
prepares the data the GPU notebooks expect.

Outputs:
  output/v13_fm/series_train.parquet — wide-format (pair × month) for
                                        2020-01 to 2024-06 (training context).
  output/v13_fm/series_oof.parquet   — wide-format for 2024-07 to 2026-01
                                        (12 val months + 7 test months).
  output/v13_fm/manifest.json        — pair count, period range, schema hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
DST = OUT / "v13_fm"
DST.mkdir(parents=True, exist_ok=True)


def main() -> int:
    abt_path = OUT / "abt_v10_cached.parquet"
    if not abt_path.exists():
        raise SystemExit(f"V11 ABT missing: {abt_path}")

    abt = pd.read_parquet(abt_path)
    keep = ["Период", "Партнер", "Артикул", "target_qty"]
    df = abt[keep].copy()
    df["Период"] = df["Период"].astype(str)
    df["pair"] = df["Партнер"].astype(str) + "||" + df["Артикул"].astype(str)

    # Pivot to wide: rows = pair, columns = month, values = qty
    wide = df.pivot_table(index="pair", columns="Период",
                          values="target_qty", aggfunc="sum",
                          fill_value=0.0).sort_index(axis=1)
    print(f"wide shape: {wide.shape}  (pairs × months)")

    # Filter to "active" pairs: at least 6 nonzero months in 2023-2024
    active_mask_cols = [c for c in wide.columns
                         if c >= "2023-01" and c <= "2024-12"]
    n_nonzero = (wide[active_mask_cols] > 0).sum(axis=1)
    keep_pairs = n_nonzero >= 6
    wide_active = wide[keep_pairs]
    print(f"active pairs (≥6 nonzero months in 2023-2024): "
          f"{len(wide_active):,} / {len(wide):,}")

    train_cols = [c for c in wide_active.columns if c <= "2024-06"]
    oof_cols   = [c for c in wide_active.columns if c >= "2024-07"]

    train_wide = wide_active[train_cols].reset_index()
    oof_wide   = wide_active[oof_cols].reset_index()

    train_wide.to_parquet(DST / "series_train.parquet", index=False)
    oof_wide.to_parquet(DST / "series_oof.parquet", index=False)

    manifest = {
        "n_pairs": int(len(wide_active)),
        "train_months": train_cols,
        "oof_months": oof_cols,
        "n_train_months": len(train_cols),
        "n_oof_months": len(oof_cols),
        "schema": "wide: pair (str), <YYYY-MM> (float)",
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nwrote {DST / 'series_train.parquet'} "
          f"({(DST / 'series_train.parquet').stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {DST / 'series_oof.parquet'} "
          f"({(DST / 'series_oof.parquet').stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {DST / 'manifest.json'}")
    print(f"\n=== UPLOAD INSTRUCTIONS ===")
    print(f"Drag both parquet files to Google Drive at:")
    print(f"   /MyDrive/v13_fm_data/")
    print(f"…or for Kaggle:")
    print(f"   create dataset slug 'v13-fm-data' and upload both files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
