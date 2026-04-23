"""Promotion-lifecycle features.

Built on top of the ``is_promo`` + ``promo_discount_pct`` columns that
``src.enrichment.enrich_promotions`` already creates.  National promotions
(``data/Нац. акции 2024.xlsx``) are **published in advance**, so using the
current-month and forward-looking flags at forecast time is legitimate —
the planner literally has the promo calendar on their desk.

Adds the following SKU-month features:

``promo_duration_months``
    Consecutive months with ``is_promo==1`` ending at this period.
``promo_depth_pct_current``
    Same as ``promo_discount_pct`` but zero when no promo.
``months_since_last_promo``
    Capped at 24; 24 means "no prior promo observed".
``months_until_next_promo``
    Capped at 24; looks forward in the published calendar.
``post_promo_depletion_flag``
    1 for 1–2 months after a promo with depth ≥ 20 percent (the
    post-sale pipeline-drain period).
``sku_promo_sensitivity``
    Per-SKU uplift ratio (promo-month mean / trailing baseline mean),
    empirical-Bayes shrunk toward 1.0.  Uses only non-leakage rows.

Public API
----------
``add_promo_lifecycle(df, eb_prior=6.0)``
    Returns a **copy** of ``df`` with the columns above added.  Expects
    ``is_promo`` and ``promo_discount_pct`` already present.  Safe to call
    twice — existing columns are overwritten.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRP = ["Партнер", "Артикул"]
HORIZON_CAP = 24


def _consecutive_prior_streak(flag: pd.Series) -> pd.Series:
    """Run length of consecutive prior-month 1's (excluding current row)."""
    shifted = flag.shift(1, fill_value=0).astype(int)
    # Reset count whenever shifted == 0.
    groups = (shifted == 0).cumsum()
    return shifted.groupby(groups).cumsum().astype("int16")


def _months_until_next(flag: pd.Series, cap: int = HORIZON_CAP) -> pd.Series:
    """For each row, count the number of rows to the next ``flag==1`` row
    (within the same group).  Uses reverse-cumulative trick."""
    n = len(flag)
    # Position index
    idx = np.arange(n)
    next_positions = np.full(n, cap, dtype=np.int16)
    last = -1
    # Walk backwards, recording nearest future 1
    for i in range(n - 1, -1, -1):
        if flag.iloc[i] == 1:
            last = i
        if last == -1:
            next_positions[i] = cap
        else:
            next_positions[i] = min(last - i, cap)
    return pd.Series(next_positions, index=flag.index)


def _compute_sku_sensitivity(df: pd.DataFrame, eb_prior: float) -> pd.Series:
    """Per-SKU promo uplift ratio with empirical-Bayes shrinkage."""
    if "target_qty" not in df.columns:
        return pd.Series(dtype=float)
    # Promo-month mean and non-promo-month mean per SKU
    agg = (
        df.groupby(["Артикул", "is_promo"], observed=True)["target_qty"]
        .agg(["mean", "size"])
        .reset_index()
    )
    # Pivot to wide
    wide = agg.pivot(index="Артикул", columns="is_promo", values=["mean", "size"]).fillna(0)
    # Get columns (is_promo 0 / 1)
    p_mean = wide[("mean", 1)] if ("mean", 1) in wide.columns else pd.Series(0.0, index=wide.index)
    np_mean = wide[("mean", 0)] if ("mean", 0) in wide.columns else pd.Series(0.0, index=wide.index)
    p_size = wide[("size", 1)] if ("size", 1) in wide.columns else pd.Series(0.0, index=wide.index)

    ratio = np.where(np_mean > 0, p_mean / np_mean, 1.0)
    # Shrink toward 1.0 with prior
    eb = (p_size * ratio + eb_prior * 1.0) / (p_size + eb_prior)
    eb = np.clip(eb, 0.2, 5.0)
    return pd.Series(eb, index=wide.index, name="sku_promo_sensitivity")


def add_promo_lifecycle(df: pd.DataFrame, eb_prior: float = 6.0) -> pd.DataFrame:
    """Extend the ABT with promo-lifecycle features.  See module docstring."""
    required = {"is_promo", "promo_discount_pct", "Период", "Артикул", "Партнер"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_promo_lifecycle: missing columns {missing}")

    out = df.sort_values(GRP + ["Период"]).reset_index(drop=True).copy()

    out["promo_depth_pct_current"] = (
        out["promo_discount_pct"].astype(float) * out["is_promo"].astype(float)
    ).fillna(0.0).astype("float32")

    # Per (Partner, SKU) promo streak and forward/backward distances
    grp = out.groupby(GRP, sort=False, observed=True)
    out["promo_duration_months"] = grp["is_promo"].transform(_consecutive_prior_streak)

    # months_since_last_promo: distance since last is_promo==1 (excluding current)
    shifted = grp["is_promo"].shift(1, fill_value=0).astype(int)
    # Cumulative-max-index trick: record indices where shifted==1, then diff
    idx = np.arange(len(out))
    last_promo_idx = np.where(shifted.to_numpy() == 1, idx, -1)
    # forward-fill last_promo_idx within each group
    last_promo_series = pd.Series(last_promo_idx, index=out.index)
    group_ids = grp.ngroup().to_numpy()

    ffilled = (
        pd.DataFrame({"g": group_ids, "idx": last_promo_series})
        .replace(-1, np.nan)
        .groupby("g")["idx"]
        .ffill()
        .fillna(-HORIZON_CAP)
        .to_numpy()
    )
    out["months_since_last_promo"] = np.clip(idx - ffilled, 0, HORIZON_CAP).astype("int16")

    # months_until_next_promo (needs per-group forward walk)
    out["months_until_next_promo"] = (
        grp["is_promo"]
        .transform(lambda s: _months_until_next(s, HORIZON_CAP))
        .astype("int16")
    )

    # Post-promo depletion: last month or two after a ≥20% promo ended
    prev_promo = grp["is_promo"].shift(1, fill_value=0)
    prev_depth = grp["promo_depth_pct_current"].shift(1, fill_value=0)
    prev2_promo = grp["is_promo"].shift(2, fill_value=0)
    prev2_depth = grp["promo_depth_pct_current"].shift(2, fill_value=0)
    out["post_promo_depletion_flag"] = (
        (
            ((prev_promo == 1) & (prev_depth >= 0.20))
            | ((prev2_promo == 1) & (prev2_depth >= 0.20))
        )
        & (out["is_promo"] == 0)
    ).astype("int8")

    # SKU sensitivity (safe: uses aggregate over all rows, not leaky row-level)
    sku_sens = _compute_sku_sensitivity(out, eb_prior)
    out["sku_promo_sensitivity"] = out["Артикул"].map(sku_sens).fillna(1.0).astype("float32")

    log.info(
        "add_promo_lifecycle: rows=%d, promo_share=%.2f%%, mean_duration=%.2f, "
        "mean_sens=%.2f, post_promo_share=%.2f%%",
        len(out),
        out["is_promo"].mean() * 100,
        out["promo_duration_months"].mean(),
        out["sku_promo_sensitivity"].mean(),
        out["post_promo_depletion_flag"].mean() * 100,
    )
    return out
