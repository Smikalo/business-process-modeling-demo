"""V7.3 ensemble / calibration sweep over saved val + test predictions.

Anti-overfit discipline:

*  VAL (2024-07 … 2025-06) is used to **fit** blend weights / scalars /
   per-month multipliers.  SIMSCORE on VAL drives candidate selection.
*  TEST (2025-07 … 2026-02) is evaluated **only once, at the very end**, and
   only for the single winner.  No decision is made on test numbers.

Candidates considered (none require retraining):

*  bare_v5, bare_v6, bare_v7, bare_v71, bare_v72  — sanity benchmarks
*  scalar_v7x_valfit   — ``V7.2 × (Σy_val / Σp_val)`` — zero aggregate bias on val
*  monthly_v7x         — per-month scalar fit on val (1 dof per month)
*  linear_blend_V7V5   — weight grid on V5 / V7.2  (blend = α·V5 + (1-α)·V7.2)
*  linear_blend_V7V6   — weight grid on V6 / V7.2
*  nnls_stack_all      — non-negative least-squares stack of V5/V6/V7/V71/V72

Outputs:
    output/v73/stacker_sweep.csv       — every candidate's val SIMSCORE
    output/v73/stacker_sweep_preds/*.csv  — val + test preds for each candidate

The runner uses linear / analytic solvers only, so a full sweep finishes in
under a minute.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V73 = OUT / "v73"
V73.mkdir(parents=True, exist_ok=True)
PREDS_DIR = V73 / "stacker_sweep_preds"
PREDS_DIR.mkdir(parents=True, exist_ok=True)

KEY_COLS = ["Период", "Партнер", "Артикул"]

BASE_MODELS = ["v5", "v6", "v7", "v71", "v72_champion"]


def _load_model_split(tag: str, split: str) -> pd.DataFrame:
    path = OUT / f"preds_{tag}_{split}.csv"
    df = pd.read_csv(path)
    df = df[KEY_COLS + ["target_qty", "prediction"]].copy()
    df.rename(columns={"prediction": tag}, inplace=True)
    return df


def _wide(split: str) -> pd.DataFrame:
    """Wide dataframe: one column per model's prediction, plus target."""
    base = _load_model_split(BASE_MODELS[0], split)
    out = base[KEY_COLS + ["target_qty"]].rename(columns={"target_qty": "y"})
    for tag in BASE_MODELS:
        m = _load_model_split(tag, split)
        out = out.merge(m[KEY_COLS + [tag]], on=KEY_COLS, how="inner")
    return out


def _score(df: pd.DataFrame, pred: np.ndarray, tag: str) -> dict:
    out = df[KEY_COLS].copy()
    out["target_qty"] = df["y"].to_numpy()
    out["prediction"] = np.clip(pred, 0, None)
    r = score_frame(out)
    r["tag"] = tag
    return r, out


def _save(preds_df: pd.DataFrame, tag: str, split: str) -> None:
    p = PREDS_DIR / f"{tag}_{split}.csv"
    preds_df.to_csv(p, index=False)


def main() -> int:
    val = _wide("val")
    tst = _wide("test")
    print(f"val: {len(val)} rows   test: {len(tst)} rows")

    rows = []

    # -- bare models (reference) -------------------------------------------
    for tag in BASE_MODELS:
        r, v_p = _score(val, val[tag].to_numpy(), f"bare_{tag}_val")
        _save(v_p, f"bare_{tag}", "val")
        rows.append({**r, "family": "bare"})
        _r, t_p = _score(tst, tst[tag].to_numpy(), f"bare_{tag}_test")
        _save(t_p, f"bare_{tag}", "test")

    # -- scalar correction on V7.2 ----------------------------------------
    for base in ("v72_champion", "v71", "v7", "v6"):
        s = float(val["y"].sum() / val[base].sum())
        val_pred = val[base].to_numpy() * s
        tst_pred = tst[base].to_numpy() * s
        r, v_p = _score(val, val_pred, f"scalar_{base}")
        _save(v_p, f"scalar_{base}", "val")
        r["scale"] = round(s, 4)
        rows.append({**r, "family": "scalar"})
        _save(_score(tst, tst_pred, f"scalar_{base}")[1],
              f"scalar_{base}", "test")

    # -- per-month scalar (fit on val, applied in test by calendar month) --
    for base in ("v72_champion", "v7", "v6"):
        val_ = val.copy()
        val_["month"] = pd.PeriodIndex(val_["Период"].astype(str), freq="M").month
        mscale = (val_.groupby("month")["y"].sum()
                  / val_.groupby("month")[base].sum()).to_dict()
        val_pred = val_[base].to_numpy() * val_["month"].map(mscale).to_numpy()
        tst_ = tst.copy()
        tst_["month"] = pd.PeriodIndex(tst_["Период"].astype(str), freq="M").month
        tst_pred = tst_[base].to_numpy() * tst_["month"].map(mscale).fillna(1.0).to_numpy()
        r, v_p = _score(val_, val_pred, f"monthly_{base}")
        _save(v_p, f"monthly_{base}", "val")
        r["scale"] = {int(k): round(float(v), 3) for k, v in mscale.items()}
        rows.append({**r, "family": "monthly"})
        _save(_score(tst_, tst_pred, f"monthly_{base}")[1],
              f"monthly_{base}", "test")

    # -- two-model linear blends ------------------------------------------
    pairs = [("v5", "v72_champion"), ("v6", "v72_champion"),
             ("v5", "v7"), ("v6", "v7"), ("v71", "v72_champion")]
    for a, b in pairs:
        best = None
        for w in np.arange(0.0, 1.01, 0.05):
            p = w * val[a].to_numpy() + (1 - w) * val[b].to_numpy()
            r, _ = _score(val, p, f"blend_{a}_{b}_w{w:.2f}")
            if best is None or r["SIMSCORE"] < best[0]["SIMSCORE"]:
                best = (r, w, p)
        r, w, v_p_arr = best
        r["tag"] = f"blend_{a}_{b}"
        r["weight_a"] = round(float(w), 3)
        _, v_p = _score(val, v_p_arr, r["tag"])
        _save(v_p, r["tag"], "val")
        rows.append({**r, "family": "blend"})
        tst_pred = w * tst[a].to_numpy() + (1 - w) * tst[b].to_numpy()
        _save(_score(tst, tst_pred, r["tag"])[1], r["tag"], "test")

    # -- NNLS stack over all base models ----------------------------------
    X = val[BASE_MODELS].to_numpy()
    y = val["y"].to_numpy()
    coef, _ = nnls(X, y)
    coef = coef / coef.sum() if coef.sum() > 0 else coef
    val_pred = X @ coef
    r, v_p = _score(val, val_pred, "nnls_stack_all")
    _save(v_p, "nnls_stack_all", "val")
    r["coef"] = {m: round(float(c), 3) for m, c in zip(BASE_MODELS, coef)}
    rows.append({**r, "family": "stack"})
    tst_pred = tst[BASE_MODELS].to_numpy() @ coef
    _save(_score(tst, tst_pred, "nnls_stack_all")[1], "nnls_stack_all", "test")

    # Also: NNLS stack constrained to symmetric group {v4,v5,v6}  (avoid over-fcst)
    # reload v4 separately
    v4v = _load_model_split("v4", "val")
    v4t = _load_model_split("v4", "test")
    val2 = val.merge(v4v[KEY_COLS + ["v4"]], on=KEY_COLS)
    tst2 = tst.merge(v4t[KEY_COLS + ["v4"]], on=KEY_COLS)
    SYM = ["v4", "v5", "v6"]
    X2 = val2[SYM].to_numpy()
    y2 = val2["y"].to_numpy()
    coef2, _ = nnls(X2, y2)
    coef2 = coef2 / coef2.sum() if coef2.sum() > 0 else coef2
    val_pred = X2 @ coef2
    r, v_p = _score(val2, val_pred, "nnls_stack_symmetric")
    _save(v_p, "nnls_stack_symmetric", "val")
    r["coef"] = {m: round(float(c), 3) for m, c in zip(SYM, coef2)}
    rows.append({**r, "family": "stack"})
    tst_pred = tst2[SYM].to_numpy() @ coef2
    _save(_score(tst2, tst_pred, "nnls_stack_symmetric")[1],
          "nnls_stack_symmetric", "test")

    # -- summary ----------------------------------------------------------
    df = pd.DataFrame(rows)
    summary_cols = ["tag", "family", "WAPE", "SMAPE_nz", "Monthly_WAPE",
                    "Agg_Bias_pct", "Bias_units", "RMSE", "SIMSCORE"]
    df = df.sort_values("SIMSCORE")
    print("\n=== stacker sweep — val SIMSCORE ranking ===")
    print(df[summary_cols].to_string(index=False))
    df.to_csv(V73 / "stacker_sweep.csv", index=False)

    # best by SIMSCORE on val
    winner = df.iloc[0].to_dict()
    (V73 / "stacker_sweep_winner.json").write_text(
        json.dumps(winner, indent=2, ensure_ascii=False, default=str)
    )
    print(f"\nval winner: {winner['tag']}  SIMSCORE={winner['SIMSCORE']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
