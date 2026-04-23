"""Temporal leakage guard utilities for external data signals.

Every external signal declares a ``publication_lag_days`` that describes how long
after end-of-month its monthly value becomes observable.  When joining the signal
to the ABT for forecast month *t*, we must use a version of the signal that was
observable at the start of *t* — otherwise the training set leaks information
from the future.

The guard is responsible for:
- shifting signal values forward by the required number of months, and
- asserting that the final joined DataFrame contains no rows that would have
  required future-dated knowledge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

MONTH_DAYS = 30.0


@dataclass
class LagConfig:
    publication_lag_days: int
    forecast_horizon_months: int = 1


def _lag_in_months(publication_lag_days: int, forecast_horizon_months: int) -> int:
    """Return the integer number of months the signal must be shifted forward.

    A publication lag of *L* days means the value for month *m* is only
    observable during month ``m + ceil(L/30)``.  When predicting month *t* we
    need a value known at the start of *t*, which is the signal timestamped for
    month ``t - (ceil(L/30) + forecast_horizon_months - 1)``.

    We collapse that to a simple forward shift: shift the signal column by
    ``ceil(L/30) + forecast_horizon_months - 1`` months so that joining on
    ``Период == t`` picks up the correct observable value.
    """

    if publication_lag_days < 0:
        raise ValueError("publication_lag_days must be non-negative; negatives are future leaks")
    if forecast_horizon_months < 1:
        raise ValueError("forecast_horizon_months must be >= 1")

    publication_months = math.ceil(publication_lag_days / MONTH_DAYS)
    return publication_months + (forecast_horizon_months - 1)


def apply_publication_lag(
    df: pd.DataFrame,
    publication_lag_days: int,
    signal_cols: list[str],
    period_col: str = "Период",
    group_cols: list[str] | None = None,
    forecast_horizon_months: int = 1,
) -> pd.DataFrame:
    """Shift signal values forward so they respect their publication delay.

    Parameters
    ----------
    df : DataFrame
        Long-form frame containing ``period_col`` and ``signal_cols``.
    publication_lag_days : int
        Days between end-of-month and the moment the value is observable.
    signal_cols : list[str]
        Columns to be lagged.
    period_col : str
        Name of the ``period[M]`` column.
    group_cols : list[str] or None
        Additional grouping keys (e.g. oblast, SKU) so lagging happens per group.
    forecast_horizon_months : int
        1 for next-month forecasts; increase for multi-step horizons.

    Returns
    -------
    DataFrame
        Copy of ``df`` with ``signal_cols`` shifted forward in time by the
        computed number of months.  The earliest months will therefore be NaN.
    """

    shift = _lag_in_months(publication_lag_days, forecast_horizon_months)
    if shift == 0:
        return df.copy()

    out = df.copy()
    sort_cols = (group_cols or []) + [period_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    if group_cols:
        out[signal_cols] = out.groupby(group_cols)[signal_cols].shift(shift)
    else:
        out[signal_cols] = out[signal_cols].shift(shift)

    return out


def assert_no_future_leak(
    abt: pd.DataFrame,
    signal_cols: list[str],
    publication_lag_days: int,
    period_col: str = "Период",
    max_training_period: pd.Period | str | None = None,
    forecast_horizon_months: int = 1,
) -> None:
    """Raise ``AssertionError`` if any row in the training window carries a signal
    value that could not have been observed in time.

    This is a defensive check — ``apply_publication_lag`` is the intended way to
    make a signal safe; this function verifies the result.

    The heuristic: for every non-null signal row in the training window, compute
    the earliest month in which that value was observable and assert it is not
    after the row's forecast period.
    """

    if max_training_period is None:
        return
    max_training_period = pd.Period(max_training_period, freq="M")

    shift = _lag_in_months(publication_lag_days, forecast_horizon_months)
    if shift == 0:
        return

    train = abt[abt[period_col] <= max_training_period]
    for col in signal_cols:
        if col not in train.columns:
            continue
        non_null = train[col].notna().sum()
        if non_null == 0:
            continue
        earliest_possible_period = train[period_col].min() + shift
        leaked = train[(train[col].notna()) & (train[period_col] < earliest_possible_period)]
        n_leaked = len(leaked)
        assert n_leaked == 0, (
            f"Leakage detected in column {col!r}: {n_leaked} training rows carry "
            f"signal values before earliest observable period {earliest_possible_period}."
        )
