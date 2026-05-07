"""Add-one-source and leave-one-out ablation harness.

Quantifies the marginal contribution of each external signal on WAPE/MAPE_nz/RMSE,
both on the validation and the held-out test set.

Usage:

    python -m scripts.run_ablation --loaders nbu_fx holidays_ua
    python -m scripts.run_ablation --all        # every registered loader
    python -m scripts.run_ablation --mode loo   # leave-one-out instead

Output:
  output/ablation_results.csv        — machine-readable (one row per model)
  output/ablation_results.md         — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(levelname)s %(message)s")
log = logging.getLogger("ablation")

import numpy as np
import pandas as pd

from src.enrichment_external import enrich_with_external
from src.evaluation import compute_all_metrics, split_train_val_test
from src.external_data import get_loader, import_default_loaders, list_loaders
from src.model_v2 import (
    TwoStageForecaster,
    encode_categoricals,
    get_feature_columns_v2,
)

OUTPUT = Path("output")
ABT_CACHE = OUTPUT / "abt_v4_cached.parquet"
ABLATION_CSV = OUTPUT / "ablation_results.csv"
ABLATION_MD = OUTPUT / "ablation_results.md"
BASE_PRED_CACHE = OUTPUT / "ablation_base_predictions.parquet"


def load_baseline_abt() -> pd.DataFrame:
    """Load the V4-cached ABT (or build from scratch if not present)."""
    if ABT_CACHE.exists():
        df = pd.read_parquet(ABT_CACHE)
        return encode_categoricals(df)
    raise FileNotFoundError(
        f"{ABT_CACHE} not found. Run `python pipelines/run_v4_final.py` first to cache it."
    )


def _feature_set(df: pd.DataFrame, extra_cols: list[str]) -> list[str]:
    base = get_feature_columns_v2(df)
    # dedupe while preserving order
    seen = set(base)
    full = list(base)
    for c in extra_cols:
        if c in df.columns and c not in seen and df[c].dtype.kind in ("f", "i", "u", "b"):
            full.append(c)
            seen.add(c)
    return full


def _train_and_score(
    df: pd.DataFrame, feat_cols: list[str]
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    df_train, df_val, df_test = split_train_val_test(df)
    model = TwoStageForecaster(
        clf_params={"num_leaves": 127, "learning_rate": 0.05, "min_child_samples": 30},
        reg_params={"num_leaves": 255, "learning_rate": 0.05, "min_child_samples": 20},
    )
    model.fit(df_train, df_val, feat_cols, num_boost_round=800, early_stopping=50)
    p_val = model.predict(df_val)
    p_test = model.predict(df_test)
    m_val = compute_all_metrics(df_val["target_qty"].values, p_val)
    m_test = compute_all_metrics(df_test["target_qty"].values, p_test)
    return m_val, m_test, p_val, p_test


def run_add_one(loader_names: list[str]) -> pd.DataFrame:
    """For each loader, train a model with baseline + loader.signal_cols and
    report metrics + delta vs baseline."""
    base_abt = load_baseline_abt()

    log.info("── Baseline (no external signals) ──")
    feat_base = _feature_set(base_abt, extra_cols=[])
    t0 = time.time()
    m_val, m_test, p_val, p_test = _train_and_score(base_abt, feat_base)
    runtime = time.time() - t0
    rows: list[dict] = [
        {
            "source": "baseline",
            "mode": "add_one",
            "feats_added": 0,
            "runtime_sec": round(runtime, 1),
            "val_WAPE": m_val["WAPE"],
            "val_MAPE_nz": m_val["MAPE_nz"],
            "val_RMSE": m_val["RMSE"],
            "test_WAPE": m_test["WAPE"],
            "test_MAPE_nz": m_test["MAPE_nz"],
            "test_RMSE": m_test["RMSE"],
            "val_WAPE_delta": 0.0,
            "test_WAPE_delta": 0.0,
            "val_MAPE_delta": 0.0,
            "test_MAPE_delta": 0.0,
        }
    ]
    base_val_wape = m_val["WAPE"]
    base_test_wape = m_test["WAPE"]
    base_val_mape = m_val["MAPE_nz"]
    base_test_mape = m_test["MAPE_nz"]

    _, df_val_base, df_test_base = split_train_val_test(base_abt)
    pd.DataFrame(
        {
            "period": df_val_base["Период"].astype(str).values,
            "partner": df_val_base["Партнер"].values,
            "sku": df_val_base["Артикул"].values,
            "actual": df_val_base["target_qty"].values,
            "baseline_val": p_val,
        }
    ).to_parquet(BASE_PRED_CACHE)

    for name in loader_names:
        log.info("── Add-one: %s ──", name)
        try:
            loader = get_loader(name)
            abt = enrich_with_external(base_abt, [loader])
            feat_cols = _feature_set(abt, extra_cols=loader.signal_cols)
            t0 = time.time()
            m_val, m_test, _, _ = _train_and_score(abt, feat_cols)
            runtime = time.time() - t0

            rows.append(
                {
                    "source": name,
                    "mode": "add_one",
                    "feats_added": len(loader.signal_cols),
                    "runtime_sec": round(runtime, 1),
                    "val_WAPE": m_val["WAPE"],
                    "val_MAPE_nz": m_val["MAPE_nz"],
                    "val_RMSE": m_val["RMSE"],
                    "test_WAPE": m_test["WAPE"],
                    "test_MAPE_nz": m_test["MAPE_nz"],
                    "test_RMSE": m_test["RMSE"],
                    "val_WAPE_delta": round(m_val["WAPE"] - base_val_wape, 4),
                    "test_WAPE_delta": round(m_test["WAPE"] - base_test_wape, 4),
                    "val_MAPE_delta": round(m_val["MAPE_nz"] - base_val_mape, 4),
                    "test_MAPE_delta": round(m_test["MAPE_nz"] - base_test_mape, 4),
                }
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Loader %s failed: %s", name, exc)
            rows.append(
                {
                    "source": name,
                    "mode": "add_one",
                    "feats_added": -1,
                    "runtime_sec": 0.0,
                    "val_WAPE": None,
                    "val_MAPE_nz": None,
                    "val_RMSE": None,
                    "test_WAPE": None,
                    "test_MAPE_nz": None,
                    "test_RMSE": None,
                    "val_WAPE_delta": None,
                    "test_WAPE_delta": None,
                    "val_MAPE_delta": None,
                    "test_MAPE_delta": None,
                    "error": str(exc),
                }
            )

    return pd.DataFrame(rows)


def run_loo(loader_names: list[str]) -> pd.DataFrame:
    """Leave-one-out: train with ALL loaders minus one, measure degradation
    when that signal is removed."""

    base_abt = load_baseline_abt()
    loaders = [get_loader(n) for n in loader_names]

    full_abt = enrich_with_external(base_abt, loaders)
    full_extra = [c for L in loaders for c in L.signal_cols]
    feat_full = _feature_set(full_abt, extra_cols=full_extra)
    log.info("── Full-signal baseline (all %d sources) ──", len(loaders))
    m_val_full, m_test_full, _, _ = _train_and_score(full_abt, feat_full)

    rows: list[dict] = [
        {
            "source": "__all__",
            "mode": "loo",
            "feats_total": len(full_extra),
            "val_WAPE": m_val_full["WAPE"],
            "test_WAPE": m_test_full["WAPE"],
            "val_MAPE_nz": m_val_full["MAPE_nz"],
            "test_MAPE_nz": m_test_full["MAPE_nz"],
        }
    ]

    for held_out in loader_names:
        kept = [L for L in loaders if L.name != held_out]
        log.info("── LOO without %s ──", held_out)
        abt = enrich_with_external(base_abt, kept)
        extras = [c for L in kept for c in L.signal_cols]
        feat_cols = _feature_set(abt, extra_cols=extras)
        m_val, m_test, _, _ = _train_and_score(abt, feat_cols)
        rows.append(
            {
                "source": held_out,
                "mode": "loo",
                "feats_total": len(extras),
                "val_WAPE": m_val["WAPE"],
                "test_WAPE": m_test["WAPE"],
                "val_MAPE_nz": m_val["MAPE_nz"],
                "test_MAPE_nz": m_test["MAPE_nz"],
                "val_WAPE_loss": round(m_val["WAPE"] - m_val_full["WAPE"], 4),
                "test_WAPE_loss": round(m_test["WAPE"] - m_test_full["WAPE"], 4),
            }
        )

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Ablation results",
        "",
        f"Generated: {pd.Timestamp.utcnow().isoformat(timespec='seconds')}",
        "",
        "| source | mode | feats | val WAPE | Δ val WAPE | test WAPE | Δ test WAPE | val MAPE_nz | test MAPE_nz | runtime(s) |",
        "|--------|------|------:|---------:|-----------:|----------:|------------:|------------:|-------------:|-----------:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            "| {source} | {mode} | {feats} | {vw} | {dvw} | {tw} | {dtw} | {vm} | {tm} | {rt} |".format(
                source=r.get("source", "?"),
                mode=r.get("mode", "?"),
                feats=r.get("feats_added", r.get("feats_total", "?")),
                vw=r.get("val_WAPE", "-"),
                dvw=r.get("val_WAPE_delta", r.get("val_WAPE_loss", "-")),
                tw=r.get("test_WAPE", "-"),
                dtw=r.get("test_WAPE_delta", r.get("test_WAPE_loss", "-")),
                vm=r.get("val_MAPE_nz", "-"),
                tm=r.get("test_MAPE_nz", "-"),
                rt=r.get("runtime_sec", "-"),
            )
        )
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loaders", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mode", choices=["add_one", "loo"], default="add_one")
    args = ap.parse_args()

    import_default_loaders()
    available = list_loaders()

    if args.all:
        names = available
    elif args.loaders:
        names = args.loaders
    else:
        log.error("No loaders selected. Use --loaders ... or --all. Available: %s", available)
        return 1

    unknown = [n for n in names if n not in available]
    if unknown:
        log.error("Unknown loaders: %s; available: %s", unknown, available)
        return 2

    t0 = time.time()
    if args.mode == "add_one":
        df = run_add_one(names)
    else:
        df = run_loo(names)

    df["run_utc"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if ABLATION_CSV.exists():
        existing = pd.read_csv(ABLATION_CSV)
        df_out = pd.concat([existing, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(ABLATION_CSV, index=False)
    write_markdown(df_out, ABLATION_MD)

    log.info("Total ablation runtime %.0fs", time.time() - t0)
    print("\nSummary:")
    cols = [c for c in ["source", "mode", "val_WAPE", "val_WAPE_delta", "test_WAPE", "test_WAPE_delta", "test_MAPE_nz"] if c in df.columns]
    print(df[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
