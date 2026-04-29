"""Intermittent-demand classical forecasters (Croston / SBA / TSB).

Tree-based models like the LightGBM stack used in V11 underfit the
many-zero / occasional-burst cells in our data: their loss is
dominated by the long zero stretches and they regress positive bursts
toward the mean.  Classical Croston-family smoothers, fit
independently per (Партнер, Артикул) pair, often beat them on the
intermittent long tail.

This module implements three closely related smoothers from first
principles plus the Syntetos-Boylan-Croston (SBC) demand-pattern
classifier that selects between them.

References
----------
Croston, J.D. (1972) "Forecasting and stock control for intermittent
    demands."  Operational Research Quarterly 23(3): 289-303.
Syntetos, A.A., Boylan, J.E. (2001) "On the bias of intermittent
    demand estimates."  IJPE 71(1-3): 457-466.
Syntetos, A.A., Boylan, J.E., Croston, J.D. (2005) "On the
    categorization of demand patterns."  JORS 56(5): 495-503.
Teunter, R.H., Syntetos, A.A., Babai, M.Z. (2011) "Intermittent
    demand: Linking forecasting to inventory obsolescence."  EJOR
    214(3): 606-615.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "croston",
    "sba",
    "tsb",
    "expanding_mean",
    "fit_predict_per_pair",
    "sbc_classify",
]


# ---------------------------------------------------------------------------
# Core smoothers.  Each returns the level forecast for h steps ahead;
# Croston-family forecasts are flat in h.
# ---------------------------------------------------------------------------


def _initial_size_interval(y: np.ndarray) -> tuple[float, float, int]:
    """Return (initial demand size z0, initial interval x0, idx_first_nz).

    We use the first non-zero observation as the initialization for the
    demand-size smoother and set the initial interval to the index
    distance from t=0 to the first non-zero observation (clamped to 1).
    """
    nz_idx = np.flatnonzero(y > 0)
    if nz_idx.size == 0:
        return 0.0, 1.0, -1
    first = int(nz_idx[0])
    z0 = float(y[first])
    x0 = float(first + 1) if first > 0 else 1.0
    return z0, x0, first


def croston(y: np.ndarray, alpha: float = 0.4, h: int = 1) -> np.ndarray:
    """Standard Croston (1972).

    Maintains two SES recursions: one on the non-zero demand sizes
    and one on the inter-demand intervals.  The point forecast is
    ``z / x`` and is flat across horizons.

    Parameters
    ----------
    y : 1-D array of historical demand (zeros allowed).
    alpha : smoothing parameter, common to size and interval.
    h : forecast horizon (length of returned array).

    Returns
    -------
    np.ndarray of length ``h`` containing the (flat) point forecast.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.any(y > 0):
        return np.zeros(h, dtype=float)

    z, x, first = _initial_size_interval(y)
    q = 1  # periods since last non-zero (counter)
    for t in range(first + 1, y.size):
        if y[t] > 0:
            z = alpha * y[t] + (1.0 - alpha) * z
            x = alpha * q + (1.0 - alpha) * x
            q = 1
        else:
            q += 1

    fcst = z / x if x > 0 else 0.0
    return np.full(h, fcst, dtype=float)


def sba(y: np.ndarray, alpha: float = 0.4, h: int = 1) -> np.ndarray:
    """Syntetos-Boylan Approximation (2001).

    Croston is biased upward by approximately ``alpha / (2 - alpha)``;
    SBA multiplies the Croston forecast by ``(1 - alpha / 2)`` to
    remove the leading-order term of that bias.
    """
    base = croston(y, alpha=alpha, h=h)
    return (1.0 - alpha / 2.0) * base


def tsb(
    y: np.ndarray,
    alpha: float = 0.4,
    beta: float = 0.1,
    h: int = 1,
) -> np.ndarray:
    """Teunter-Syntetos-Babai (2011).

    Smooths the demand *size* on non-zero periods (with rate ``alpha``)
    and the *probability of occurrence* every period (with rate
    ``beta``).  The forecast is ``p * z`` and is flat across horizons.
    Unlike Croston/SBA, TSB updates ``p`` even when no demand occurs,
    which makes it the standard choice for non-stationary series whose
    demand rate is drifting.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n == 0:
        return np.zeros(h, dtype=float)
    if not np.any(y > 0):
        return np.zeros(h, dtype=float)

    nz_idx = np.flatnonzero(y > 0)
    first = int(nz_idx[0])
    z = float(y[first])
    # Initialise probability with the empirical fraction of non-zero
    # periods seen up to and including the first hit; this is more
    # stable than starting from 1.0 when the warm-up is long.
    p = float((y[: first + 1] > 0).mean())

    for t in range(first + 1, n):
        if y[t] > 0:
            z = alpha * y[t] + (1.0 - alpha) * z
            p = beta * 1.0 + (1.0 - beta) * p
        else:
            p = beta * 0.0 + (1.0 - beta) * p

    fcst = p * z
    return np.full(h, fcst, dtype=float)


def expanding_mean(y: np.ndarray, h: int = 1) -> np.ndarray:
    """Trivial expanding-mean baseline.

    Used as the recommended forecaster for the *erratic* SBC class
    (high size variability but frequent demand) where Croston-family
    smoothers perform poorly.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return np.zeros(h, dtype=float)
    return np.full(h, float(y.mean()), dtype=float)


# ---------------------------------------------------------------------------
# SBC classification.
# ---------------------------------------------------------------------------


def sbc_classify(y: np.ndarray) -> str:
    """Syntetos-Boylan-Croston demand-pattern classification.

    Uses two diagnostics computed on the historical series:

    * ADI - average inter-demand interval, i.e. the mean gap (in
      periods) between successive non-zero observations.
    * CV² - squared coefficient of variation of the *non-zero* demand
      sizes (sample std, ddof=1).

    Cut-offs (Syntetos, Boylan, Croston, 2005):

    ===========   ==============   =================
    ADI           CV²              class
    ===========   ==============   =================
    ``<= 1.32``   ``<= 0.49``      ``smooth``
    ``> 1.32``    ``<= 0.49``      ``intermittent``
    ``<= 1.32``   ``> 0.49``       ``erratic``
    ``> 1.32``    ``> 0.49``       ``lumpy``
    ===========   ==============   =================

    Edge cases
    ----------
    * Empty / all-zero / single-non-zero series are classified as
      ``intermittent`` (ADI = +inf, CV² conventionally 0); we still
      pick a reasonable bucket so the dispatcher has a method.
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0 or not np.any(y > 0):
        return "intermittent"

    nz = y[y > 0]
    nz_idx = np.flatnonzero(y > 0)

    if nz_idx.size >= 2:
        intervals = np.diff(nz_idx)
        adi = float(intervals.mean())
    else:
        adi = float(y.size)

    if nz.size >= 2:
        mean_nz = float(nz.mean())
        std_nz = float(nz.std(ddof=1))
        cv2 = (std_nz / mean_nz) ** 2 if mean_nz > 0 else 0.0
    else:
        cv2 = 0.0

    smooth_adi = adi <= 1.32
    smooth_cv2 = cv2 <= 0.49

    if smooth_adi and smooth_cv2:
        return "smooth"
    if not smooth_adi and smooth_cv2:
        return "intermittent"
    if smooth_adi and not smooth_cv2:
        return "erratic"
    return "lumpy"


# ---------------------------------------------------------------------------
# High-level dispatcher.
# ---------------------------------------------------------------------------


def fit_predict_per_pair(
    history: pd.Series,
    h: int,
    method: str = "tsb",
    alpha: float = 0.4,
    beta: float = 0.1,
) -> float:
    """Fit ``method`` on ``history`` and return the h-step-ahead point.

    Parameters
    ----------
    history : ``pd.Series`` of demand sorted in chronological order.
        The index is unused but typically holds the ``Период``.
    h : forecast horizon (>=1).  Croston-family forecasts are flat,
        so this just selects the length of the returned vector; we
        return the scalar at offset ``h-1``.
    method : one of ``{"croston", "sba", "tsb", "mean"}``.
    alpha, beta : smoothing rates passed through.

    Returns
    -------
    float : the point forecast at horizon ``h``.

    Defensive fallbacks
    -------------------
    * ``len(history) < 3``  → mean of available data.
    * ``history`` all zero  → 0.
    * unknown ``method``    → ``"tsb"``.
    """
    y = np.asarray(history.to_numpy() if isinstance(history, pd.Series)
                   else history, dtype=float)

    if y.size == 0:
        return 0.0
    if not np.any(y > 0):
        return 0.0
    if y.size < 3:
        return float(y.mean())

    method = (method or "tsb").lower()
    if method == "croston":
        out = croston(y, alpha=alpha, h=h)
    elif method == "sba":
        out = sba(y, alpha=alpha, h=h)
    elif method == "mean":
        out = expanding_mean(y, h=h)
    else:  # default tsb
        out = tsb(y, alpha=alpha, beta=beta, h=h)

    return float(out[h - 1])
