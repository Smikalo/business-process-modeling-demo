"""V12 Phase 1 - Intermittent-demand specialist forecaster.

Trains a Croston/SBA/TSB ensemble (selected per pair via the
Syntetos-Boylan-Croston classification) on the V11 validation/test
window and writes its predictions in the standard
``Период,Партнер,Артикул,target_qty,prediction`` schema so it can
slot into the V11 stacker as a new base learner.

The motivation: V11's gradient-boosted bases are trained with
squared/quantile losses on a flat panel.  Long zero stretches
dominate that loss for intermittent SKUs and the trees regress
positive bursts toward zero.  Per-pair classical smoothers ignore the
zeros where they should (Croston/SBA) and adapt their occurrence
probability over time (TSB), and we get a complementary signal.

Outputs
-------
* ``output/preds_v12_intermittent_val.csv``
* ``output/preds_v12_intermittent_test.csv``
* ``output/v12_intermittent_summary.csv``  (per-class counts and
  mean WAPE so we can see which buckets benefit most)

Run::

    PYTHONPATH=. .venv/bin/python -m scripts.train_v12_intermittent
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.intermittent import (  # noqa: E402  (sys.path mutation above)
    croston,
    expanding_mean,
    sba,
    sbc_classify,
    tsb,
)
from scripts.score_similarity import score_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v12_intermittent")


# Validation / test cutoffs are derived from the V11 base preds at
# load time but we keep these as fall-back defaults that match the
# project conventions (see ``build_v11_recent_only.py``).
DEFAULT_VAL_START = "2024-07"
DEFAULT_TEST_START = "2025-07"

# Per-class smoother dispatch.  The mapping is conservative and
# follows the literature: smooth = SBA (low-bias variant of
# exponential smoothing), intermittent = TSB (best for non-stationary
# demand rates), lumpy = TSB with a faster occurrence-probability
# tracker, erratic = expanding mean (Croston-family does poorly when
# CV² of demand sizes is high).
CLASS_TO_METHOD = {
    "smooth":       ("sba",  {"alpha": 0.4}),
    "intermittent": ("tsb",  {"alpha": 0.4, "beta": 0.1}),
    "lumpy":        ("tsb",  {"alpha": 0.4, "beta": 0.2}),
    "erratic":      ("mean", {}),
}


# ---------------------------------------------------------------------------
# Per-pair worker.
# ---------------------------------------------------------------------------


def _smoother_predict(
    y: np.ndarray,
    method: str,
    alpha: float = 0.4,
    beta: float = 0.1,
    h: int = 1,
) -> float:
    """Dispatch to the right smoother and return a scalar h-step point.

    Defensive guards:
        * empty / all-zero history → 0
        * <3 observations          → mean of available history
    """
    n = y.size
    if n == 0:
        return 0.0
    if not np.any(y > 0):
        return 0.0
    if n < 3:
        return float(y.mean())

    method = method.lower()
    if method == "croston":
        return float(croston(y, alpha=alpha, h=h)[h - 1])
    if method == "sba":
        return float(sba(y, alpha=alpha, h=h)[h - 1])
    if method == "mean":
        return float(expanding_mean(y, h=h)[h - 1])
    return float(tsb(y, alpha=alpha, beta=beta, h=h)[h - 1])


def forecast_one_pair(arg: tuple) -> dict:
    """Multiprocessing-friendly worker.

    ``arg`` is::

        (partner, sku, y_full, periods_full, target_periods, val_start)

    where:
        * ``y_full`` / ``periods_full`` are the monolithic chronological
          history for the pair (full ABT history, sorted ascending).
        * ``target_periods`` is the set of ``Период`` strings we have to
          forecast (val ∪ test for that pair).
        * ``val_start`` is the ``YYYY-MM`` cutoff used for SBC class
          assignment - we classify on the *training* history alone so
          val/test months don't leak into the dispatcher.

    Returns a dict with keys ``rows`` (list of (period, prediction))
    and ``cls`` (the SBC class string).
    """
    partner, sku, y_full, periods_full, target_periods, val_start = arg
    y_full = np.asarray(y_full, dtype=float)
    n = y_full.size

    # ---- 1. classify using only pre-val_start history --------------
    train_mask = np.array([p < val_start for p in periods_full])
    train_y = y_full[train_mask]
    cls = sbc_classify(train_y)
    method, kwargs = CLASS_TO_METHOD[cls]

    # ---- 2. produce one-step-ahead point forecasts ----------------
    rows: list[tuple[str, float]] = []
    target_set = set(target_periods)
    for i in range(n):
        p = periods_full[i]
        if p in target_set:
            hist = y_full[:i]
            fc = _smoother_predict(hist, method, h=1, **kwargs)
            # Predictions are non-negative point forecasts of demand
            # quantity; clip just in case any numerical drift occurs.
            rows.append((p, max(0.0, fc)))

    return {"partner": partner, "sku": sku, "cls": cls, "rows": rows}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def _load_v11_keys(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the V11-final val and test prediction frames.

    We anchor V12 to *exactly* the same (Период, Партнер, Артикул)
    keys as the V11 final stacker so the row counts and aggregates are
    directly comparable and the CSV can be plugged into the existing
    SIMSCORE pipeline without reshaping.
    """
    val = pd.read_csv(repo / "output/preds_v11_final_val.csv")
    test = pd.read_csv(repo / "output/preds_v11_final_test.csv")
    for df in (val, test):
        df["Период"] = df["Период"].astype(str)
        df["Партнер"] = df["Партнер"].astype(str)
        df["Артикул"] = df["Артикул"].astype(str)
    return val, test


def _build_pair_args(
    abt: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_start: str,
) -> list[tuple]:
    """Materialise per-pair argument tuples for the worker pool."""
    abt = abt.copy()
    abt["Период"] = abt["Период"].astype(str)
    abt["Партнер"] = abt["Партнер"].astype(str)
    abt["Артикул"] = abt["Артикул"].astype(str)
    abt = abt[["Период", "Партнер", "Артикул", "target_qty"]]
    abt = abt.sort_values(["Партнер", "Артикул", "Период"])

    target_keys = pd.concat(
        [val[["Партнер", "Артикул", "Период"]],
         test[["Партнер", "Артикул", "Период"]]],
        ignore_index=True,
    )
    targets_per_pair = (
        target_keys
        .groupby(["Партнер", "Артикул"])["Период"]
        .apply(lambda s: tuple(sorted(set(s.tolist()))))
        .to_dict()
    )

    args: list[tuple] = []
    for (partner, sku), grp in abt.groupby(["Партнер", "Артикул"], sort=False):
        if (partner, sku) not in targets_per_pair:
            continue
        periods_full = grp["Период"].tolist()
        y_full = grp["target_qty"].fillna(0.0).to_numpy(dtype=float)
        args.append((
            partner,
            sku,
            y_full,
            periods_full,
            targets_per_pair[(partner, sku)],
            val_start,
        ))
    return args


def _materialise_predictions(
    results: list[dict],
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reshape worker results into V11-style val / test prediction
    frames and a per-class SBC summary.
    """
    pred_records = []
    cls_records = []
    for r in results:
        for period, fc in r["rows"]:
            pred_records.append({
                "Период":   period,
                "Партнер": r["partner"],
                "Артикул": r["sku"],
                "prediction": fc,
            })
        cls_records.append({
            "Партнер": r["partner"],
            "Артикул": r["sku"],
            "sbc_class": r["cls"],
        })

    preds = pd.DataFrame.from_records(pred_records)
    cls_df = pd.DataFrame.from_records(cls_records)

    val_out = val.merge(
        preds, on=["Период", "Партнер", "Артикул"], how="left",
        suffixes=("_v11", "")
    )
    val_out = val_out[["Период", "Партнер", "Артикул", "target_qty",
                       "prediction"]]
    val_out["prediction"] = val_out["prediction"].fillna(0.0)

    test_out = test.merge(
        preds, on=["Период", "Партнер", "Артикул"], how="left",
        suffixes=("_v11", "")
    )
    test_out = test_out[["Период", "Партнер", "Артикул", "target_qty",
                         "prediction"]]
    test_out["prediction"] = test_out["prediction"].fillna(0.0)

    # Per-class summary: counts + mean WAPE on val (unweighted across
    # pairs).  Useful diagnostic for which class actually benefits.
    full = val_out.merge(cls_df, on=["Партнер", "Артикул"], how="left")
    rows = []
    for cls, grp in full.groupby("sbc_class", dropna=False):
        n_pairs = grp[["Партнер", "Артикул"]].drop_duplicates().shape[0]
        y = grp["target_qty"].to_numpy(dtype=float)
        p = grp["prediction"].to_numpy(dtype=float)
        denom = float(np.abs(y).sum()) or 1.0
        wape = float(np.abs(y - p).sum() / denom)
        rows.append({
            "sbc_class": cls,
            "n_pairs": n_pairs,
            "n_rows_val": len(grp),
            "wape_val": round(wape, 4),
            "mean_target_qty": round(float(y.mean()), 4),
            "mean_prediction": round(float(p.mean()), 4),
        })
    summary = pd.DataFrame(rows).sort_values("n_pairs", ascending=False)
    return val_out, test_out, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abt", default="output/abt_v10_cached.parquet",
                    help="Source ABT parquet (V10 cached has all history)")
    ap.add_argument("--val-preds", default="output/preds_v11_final_val.csv")
    ap.add_argument("--test-preds", default="output/preds_v11_final_test.csv")
    ap.add_argument("--out-val", default="output/preds_v12_intermittent_val.csv")
    ap.add_argument("--out-test", default="output/preds_v12_intermittent_test.csv")
    ap.add_argument("--out-summary", default="output/v12_intermittent_summary.csv")
    ap.add_argument("--val-start", default=DEFAULT_VAL_START,
                    help="First val month - history before this is the "
                         "training set used by the SBC classifier")
    ap.add_argument("--workers", type=int, default=6,
                    help="multiprocessing pool size (each pair is "
                         "embarrassingly parallel)")
    args = ap.parse_args()

    t0 = time.time()
    log.info("loading ABT %s", args.abt)
    abt = pd.read_parquet(REPO / args.abt)
    log.info("ABT %d rows, %d unique pairs", len(abt),
             abt[["Партнер", "Артикул"]].drop_duplicates().shape[0])

    val, test = _load_v11_keys(REPO)
    log.info("V11 anchor: %d val rows, %d test rows, %d pairs",
             len(val), len(test),
             val[["Партнер", "Артикул"]].drop_duplicates().shape[0])

    pair_args = _build_pair_args(abt, val, test, args.val_start)
    log.info("dispatching %d pairs to %d workers",
             len(pair_args), args.workers)

    if args.workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.workers) as pool:
            results = pool.map(forecast_one_pair, pair_args, chunksize=64)
    else:
        results = [forecast_one_pair(a) for a in pair_args]

    log.info("smoothers done in %.1fs", time.time() - t0)

    val_out, test_out, summary = _materialise_predictions(results, val, test)

    out_val = REPO / args.out_val
    out_test = REPO / args.out_test
    out_summary = REPO / args.out_summary
    out_val.parent.mkdir(parents=True, exist_ok=True)
    val_out.to_csv(out_val, index=False)
    test_out.to_csv(out_test, index=False)
    summary.to_csv(out_summary, index=False)
    log.info("wrote %s (%d rows)", out_val, len(val_out))
    log.info("wrote %s (%d rows)", out_test, len(test_out))
    log.info("wrote %s", out_summary)

    # ---- score (SIMSCORE) ------------------------------------------
    val_metrics = score_frame(val_out)
    test_metrics = score_frame(test_out)

    print("\n== V12_intermittent VAL ==")
    for k, v in val_metrics.items():
        print(f"  {k:<14} {v}")
    print("\n== V12_intermittent TEST ==")
    for k, v in test_metrics.items():
        print(f"  {k:<14} {v}")

    print("\n== SBC class distribution (per-pair) ==")
    print(summary.to_string(index=False))

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
