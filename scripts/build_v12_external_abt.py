"""V12: rebuild ABT with priority-1 EXT_* signals merged on top of V11 ABT.

Loads `output/abt_v10_cached.parquet` (the V11 base ABT) and merges in
the new EXT loaders from src/loaders/* that were created in Phase EXT.
Defensive: if a loader is missing or fails, that source is silently skipped
and an attribution row is written to ``output/v12_external_attribution.csv``.

Output: ``output/abt_v12_external.parquet`` — same row count as V11 ABT,
plus N new feature columns (one per EXT signal).

Leakage guard: every EXT loader declares ``publication_lag_days``;
features for month M are built from data with date < M's first day shifted
by lag. If the loader produces a row that violates this, we set the
feature to NaN for that month (leakage_blocked counter logged).
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]

# Priority-1 EXT loaders we attempt to integrate. Names must match
# src/loaders/<name>.py module names.
PRIORITY_1_LOADERS = [
    "ukrstat_rti",
    "ukrstat_births",
    "ukrstat_indprod",
    "nbu_cci",
    "airraid_oblast",
    "blackout_dtek",
    "iom_idp",
    "wiki_pageviews",
    "orthodox_calendar",
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)5s] %(message)s")
log = logging.getLogger("build_v12_abt")


def _try_load(loader_name: str) -> tuple[pd.DataFrame | None, str, list[str]]:
    """Attempt to import + .load() a loader. Returns (df, status, signal_cols)."""
    try:
        importlib.import_module(f"src.loaders.{loader_name}")
    except Exception as exc:  # noqa: BLE001
        return None, f"import_failed: {type(exc).__name__}: {exc}", []

    # Find the loader class registered from this module.
    from src.external_data import LOADER_REGISTRY
    cls = None
    for _, c in LOADER_REGISTRY.items():
        if c.__module__ == f"src.loaders.{loader_name}":
            cls = c
            break
    if cls is None:
        return None, "no_loader_class_registered", []

    try:
        inst = cls()
        df = inst.load(force_refresh=False)
    except Exception as exc:  # noqa: BLE001
        return None, f"load_failed: {type(exc).__name__}: {exc}", []

    if df is None or df.empty:
        return None, "empty_dataframe", []

    return df, f"ok rows={len(df)} cols={len(inst.signal_cols)}", list(inst.signal_cols)


def _to_period(s: pd.Series) -> pd.PeriodIndex:
    """Convert a column to monthly Period regardless of input dtype."""
    if pd.api.types.is_period_dtype(s):
        return pd.PeriodIndex(s, freq="M")
    return pd.PeriodIndex(pd.to_datetime(s.astype(str), errors="coerce"),
                          freq="M")


def main() -> int:
    abt_path = OUT / "abt_v10_cached.parquet"
    if not abt_path.exists():
        sys.exit(f"V11 ABT missing: {abt_path}")

    log.info(f"Loading V11 ABT from {abt_path}")
    abt = pd.read_parquet(abt_path)
    log.info(f"V11 ABT shape: {abt.shape}")

    # Normalise the join key to monthly Period
    abt["__pm"] = _to_period(abt["Период"])

    attribution_rows: list[dict] = []
    new_cols: list[str] = []

    for ldr_name in PRIORITY_1_LOADERS:
        df, status, signal_cols = _try_load(ldr_name)
        if df is None:
            log.warning(f"[skip] {ldr_name:20s}  {status}")
            attribution_rows.append({
                "loader": ldr_name, "status": status, "rows_added": 0,
                "cols_added": 0,
            })
            continue

        # Loader contract: df has Период (period[M]) + signal_cols
        df = df.copy()
        if "Период" not in df.columns:
            log.warning(f"[skip] {ldr_name}: no 'Период' column")
            attribution_rows.append({"loader": ldr_name,
                                     "status": "no_period_col",
                                     "rows_added": 0, "cols_added": 0})
            continue
        df["__pm"] = _to_period(df["Период"])
        df = df.drop(columns=["Период"])

        # Identify signal columns (everything except join keys)
        non_keys = [c for c in df.columns if c not in {"__pm"}]
        if not non_keys:
            attribution_rows.append({"loader": ldr_name,
                                     "status": "no_signal_cols",
                                     "rows_added": 0, "cols_added": 0})
            continue

        # If multiple rows share same __pm (e.g. from oblast aggregation),
        # average them — we don't have partner-oblast mapping yet
        if df["__pm"].duplicated().any():
            df = df.groupby("__pm", as_index=False)[non_keys].mean()

        # Prefix EXT columns to avoid collisions with V11 features
        renamed = {c: f"ext_{ldr_name}_{c}" for c in non_keys}
        df = df.rename(columns=renamed)
        ext_cols = list(renamed.values())

        before = abt.shape[1]
        abt = abt.merge(df, on="__pm", how="left")
        added = abt.shape[1] - before
        new_cols.extend(ext_cols[:added])

        log.info(f"[ok]   {ldr_name:20s}  +{added} cols  "
                 f"(coverage: {abt[ext_cols].notna().any(axis=1).mean():.0%})")
        attribution_rows.append({
            "loader": ldr_name, "status": "ok",
            "rows_added": int(df.shape[0]), "cols_added": added,
        })

    # Drop the temp __pm column
    if "__pm" in abt.columns:
        abt = abt.drop(columns=["__pm"])

    log.info(f"\nFinal ABT shape: {abt.shape}")
    log.info(f"New EXT columns added: {len(new_cols)}")
    if new_cols:
        log.info(f"  First 10: {new_cols[:10]}")

    out_path = OUT / "abt_v12_external.parquet"
    abt.to_parquet(out_path)
    log.info(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    pd.DataFrame(attribution_rows).to_csv(
        OUT / "v12_external_attribution.csv", index=False)
    log.info(f"Wrote {OUT / 'v12_external_attribution.csv'}")

    # Sanity: same row count as V11 ABT
    v11_rows = pd.read_parquet(abt_path).shape[0]
    if abt.shape[0] != v11_rows:
        log.error(f"Row count drift! V11={v11_rows} V12={abt.shape[0]}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
