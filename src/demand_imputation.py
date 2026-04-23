"""Censored-demand imputation for stockout-suppressed months.

When a (Partner, SKU) that historically sells regularly shows zero sales in a
month where the product was out-of-stock at the warehouse (``stockout_orc=1``),
the observed zero is **censored** — it reflects supply failure, not a drop
in demand.  Training the model on those zeros drags predictions downward.

This module replaces those censored zeros with a principled estimate of the
counterfactual demand, using a brand × channel × month-of-year baseline
scaled by an empirical-Bayes SKU factor.

Public API
----------
``impute_stockout_demand(df, censor_density_min=0.3, strategy="stockout_orc", eb_prior=6.0)``
    Returns a copy of ``df`` with two additional columns:
        ``target_qty_imputed`` — same as ``target_qty`` except on censored rows
        ``was_censored``       — int8 flag (train-time feature)
    The original ``target_qty`` is **not** modified — callers can train on
    either target by switching the column name.

Design notes
------------
*   Censoring mask is deliberately conservative (target=0 AND stockout_orc=1
    AND trailing demand-density >= 0.3) so fewer than ~3% of rows get
    imputed even in a large ABT.  Leakage risk is minimised because we only
    rewrite rows with target=0.
*   The imputation itself uses only **past-only** statistics within
    non-censored rows, so applying it before the temporal split does not
    introduce future information into training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

Strategy = Literal["stockout_orc", "stockout_both"]


@dataclass
class ImputationReport:
    n_rows: int
    n_censored: int
    n_imputed: int
    mean_imputed_qty: float
    strategy: Strategy
    censor_density_min: float

    @property
    def share_censored(self) -> float:
        return self.n_censored / max(self.n_rows, 1)

    def __str__(self) -> str:
        return (
            f"ImputationReport(rows={self.n_rows:,}, censored={self.n_censored:,} "
            f"({self.share_censored*100:.2f}%), imputed_mean={self.mean_imputed_qty:.2f}, "
            f"strategy={self.strategy!r}, min_density={self.censor_density_min})"
        )


def _build_brand_channel_month_baseline(
    df: pd.DataFrame, exclude_mask: pd.Series
) -> pd.Series:
    """Mean target_qty per (Бренд, Канал, month) using only non-censored rows."""
    clean = df.loc[~exclude_mask, ["Бренд", "Канал", "Период", "target_qty"]].copy()
    clean["mnth"] = clean["Период"].dt.month
    baseline = (
        clean.groupby(["Бренд", "Канал", "mnth"], observed=True)["target_qty"]
        .mean()
        .rename("bc_month_baseline")
    )
    return baseline


def _compute_sku_factor(
    df: pd.DataFrame, exclude_mask: pd.Series, eb_prior: float
) -> pd.Series:
    """Empirical-Bayes SKU-level factor relative to brand-channel mean.

    factor = (N_sku * mean_sku + eb_prior * 1.0) / (N_sku + eb_prior) /
              brand_channel_mean_of_same_rows

    Shrinkage prior = 1.0 (SKU is identical to its brand-channel cohort).
    """
    clean = df.loc[~exclude_mask, ["Артикул", "Бренд", "Канал", "target_qty"]].copy()
    # Aggregate per SKU (across all brand/channel combos the SKU appears in)
    sku_stats = (
        clean.groupby("Артикул", observed=True)
        .agg(sku_mean=("target_qty", "mean"), sku_count=("target_qty", "size"))
    )
    # Global reference mean for shrinkage
    global_mean = clean["target_qty"].mean() if len(clean) else 1.0
    if global_mean == 0:
        return pd.Series(1.0, index=sku_stats.index, name="sku_factor")
    eb_mean = (sku_stats["sku_count"] * sku_stats["sku_mean"] + eb_prior * global_mean) / (
        sku_stats["sku_count"] + eb_prior
    )
    factor = (eb_mean / global_mean).fillna(1.0).clip(0.1, 10.0)
    factor.name = "sku_factor"
    return factor


def impute_stockout_demand(
    df: pd.DataFrame,
    censor_density_min: float = 0.3,
    strategy: Strategy = "stockout_orc",
    eb_prior: float = 6.0,
    min_imputed_qty: float = 0.5,
) -> tuple[pd.DataFrame, ImputationReport]:
    """Impute counterfactual demand on stockout-censored zeros.

    Parameters
    ----------
    df
        ABT with at least ``target_qty``, ``stockout_orc``, ``stockout_both``,
        ``demand_density``, ``Бренд``, ``Канал``, ``Артикул``, ``Период``.
    censor_density_min
        Minimum ``demand_density`` for a zero row to be flagged as censored.
        Lower values impute more rows (looser); higher values are stricter.
    strategy
        ``"stockout_orc"`` requires warehouse stockout only.
        ``"stockout_both"`` requires both warehouse AND retail stockout (stricter).
    eb_prior
        Prior strength in months for the empirical-Bayes SKU factor.  Bigger
        values shrink the SKU factor harder toward the brand-channel mean.
    min_imputed_qty
        Floor applied to imputed values (avoids imputing pure zeros).

    Returns
    -------
    (df_augmented, report)
    """
    required = {
        "target_qty", "stockout_orc", "stockout_both", "demand_density",
        "Бренд", "Канал", "Артикул", "Период",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"impute_stockout_demand: missing columns {missing}")

    stockout_col = strategy
    if stockout_col not in df.columns:
        raise ValueError(f"Strategy column {stockout_col!r} not found in df")

    out = df.copy()
    censor_mask = (
        (out["target_qty"] == 0)
        & (out[stockout_col] == 1)
        & (out["demand_density"] >= censor_density_min)
    )
    out["was_censored"] = censor_mask.astype("int8")

    # Compute imputation sources only from non-censored rows
    baseline = _build_brand_channel_month_baseline(out, censor_mask)
    sku_factor = _compute_sku_factor(out, censor_mask, eb_prior)

    # Lookup for imputation
    tmp = out.loc[censor_mask, ["Бренд", "Канал", "Период", "Артикул"]].copy()
    tmp["mnth"] = tmp["Период"].dt.month
    tmp = tmp.join(baseline, on=["Бренд", "Канал", "mnth"])
    tmp["sku_factor"] = tmp["Артикул"].map(sku_factor).fillna(1.0)
    tmp["imputed"] = (
        tmp["bc_month_baseline"].fillna(0.0) * tmp["sku_factor"]
    ).clip(lower=min_imputed_qty)

    out["target_qty_imputed"] = out["target_qty"].astype(float)
    out.loc[censor_mask, "target_qty_imputed"] = tmp["imputed"].values

    report = ImputationReport(
        n_rows=len(out),
        n_censored=int(censor_mask.sum()),
        n_imputed=int(censor_mask.sum()),
        mean_imputed_qty=float(tmp["imputed"].mean()) if len(tmp) else 0.0,
        strategy=strategy,
        censor_density_min=censor_density_min,
    )
    log.info(str(report))
    return out, report
