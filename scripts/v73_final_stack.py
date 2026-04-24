"""Fit the final V7.3 NNLS stack, with broadened model pool, under rolling CV.

All decisions are made on the VAL window (2024-07 … 2025-06).  The 3-fold
rolling-origin CV within val chooses between:

* ``compact``     — V4, V5, V6, V7, V71, V72.champion   (6 models)
* ``broad``       — compact + V7-rec95, V7.1-channels-tuned  (8 models)
* ``SIMSCORE-opt``— compact + projected-gradient minimum of val-SIMSCORE

The verdict is: ``mean OOF SIMSCORE`` on 3 folds, with in-sample → OOF gap
≤ 0.05 as the anti-overfit guard.

The final weights are re-fit on the full val window; predictions are written
for val + test but **not** scored on test (Step 7 does that once).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V73 = OUT / "v73"
V73.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]

COMPACT = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
EXTRA_TAGS = {
    "v7_rec95": "preds_v7_rec95_test.csv",
    "v71_channels_tuned": "preds_v71_channels_tuned_test.csv",
    "v7_v72_uahopt": "preds_v7_v72_uahopt_test.csv",
}


def _load_split(tag: str, split: str) -> pd.DataFrame | None:
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    cols = [c for c in ["Период", "Партнер", "Артикул",
                        "target_qty", "prediction"] if c in d.columns]
    d = d[cols].rename(columns={"prediction": tag})
    return d


def _wide(split: str, tags: list[str]) -> pd.DataFrame:
    out = None
    for t in tags:
        d = _load_split(t, split)
        if d is None:
            continue
        if out is None:
            out = d.rename(columns={"target_qty": "y"})
        else:
            out = out.merge(d.drop(columns=["target_qty"]), on=KEY, how="inner")
    return out


def _score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def nnls_fit(X: np.ndarray, y: np.ndarray, normalize: bool = False):
    coef, _ = nnls(X, y)
    if normalize and coef.sum() > 0:
        coef = coef / coef.sum()
    return coef


def simscore_opt(X: np.ndarray, df: pd.DataFrame, n: int) -> np.ndarray:
    """Fast surrogate minimizer of SIMSCORE on the provided rows.

    SIMSCORE = WAPE + λ·|Agg_Bias_pct| + μ·Monthly_WAPE.  We optimize a
    differentiable proxy using pre-computed monthly and global aggregates,
    starting from the NNLS solution.  The surrogate matches the true
    SIMSCORE up to the (smooth) absolute-value approximation.
    """
    y = df["y"].to_numpy()
    coef0 = nnls_fit(X, y)
    month = pd.PeriodIndex(df["Период"].astype(str), freq="M").month
    months = np.unique(month)
    month_mask = [month == m for m in months]
    y_total = max(float(y.sum()), 1.0)
    y_by_m = np.array([y[mm].sum() for mm in month_mask])

    LAMBDA, MU = 0.005, 0.5

    def loss(w):
        p = X @ w
        wape = np.abs(y - p).sum() / y_total
        agg_bias_pct = (p.sum() / y_total - 1.0) * 100.0
        p_by_m = np.array([p[mm].sum() for mm in month_mask])
        mw = np.mean(np.abs(p_by_m - y_by_m) / np.maximum(y_by_m, 1e-6))
        return float(wape + LAMBDA * abs(agg_bias_pct) + MU * mw)

    cons = [
        {"type": "ineq", "fun": lambda w: w},
        {"type": "ineq", "fun": lambda w: 1.20 - w.sum()},
    ]
    res = minimize(loss, coef0, method="COBYLA", constraints=cons,
                   options={"rhobeg": 0.02, "maxiter": 150, "catol": 1e-6})
    return np.clip(res.x, 0, None)


def fold_eval(val: pd.DataFrame, tags: list[str], fit_fn) -> tuple[float, float, list[float]]:
    val["Период_p"] = pd.PeriodIndex(val["Период"].astype(str), freq="M")
    splits = [
        (pd.Period("2024-09", "M"), pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
        (pd.Period("2024-12", "M"), pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
        (pd.Period("2025-03", "M"), pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
    ]
    oof = []
    for (tr_e, va_s, va_e) in splits:
        tr = val[val["Период_p"] <= tr_e]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        coef = fit_fn(tr[tags].to_numpy(), tr)
        pred = te[tags].to_numpy() @ coef
        oof.append(_score(te, pred)["SIMSCORE"])
    in_coef = fit_fn(val[tags].to_numpy(), val)
    in_sim = _score(val, val[tags].to_numpy() @ in_coef)["SIMSCORE"]
    return float(np.mean(oof)), in_sim, oof


def main() -> int:
    val_c = _wide("val", COMPACT)
    broad_tags = COMPACT + [t for t in EXTRA_TAGS if
                            (OUT / f"preds_{t}_val.csv").exists()]
    val_b = _wide("val", broad_tags)
    print(f"compact tags ({len(COMPACT)}): {COMPACT}")
    print(f"broad tags   ({len(broad_tags)}): {broad_tags}")
    print(f"compact rows: {len(val_c)}   broad rows: {len(val_b)}")

    def nnls_raw(X, df): return nnls_fit(X, df["y"].to_numpy())
    def nnls_norm(X, df): return nnls_fit(X, df["y"].to_numpy(), normalize=True)
    def simopt(X, df): return simscore_opt(X, df, df.shape[0])

    candidates = []
    for tag_label, tags, df_val, fit_fn in [
        ("compact_nnls",      COMPACT,    val_c, nnls_raw),
        ("compact_nnls_norm", COMPACT,    val_c, nnls_norm),
        ("broad_nnls",        broad_tags, val_b, nnls_raw),
        ("broad_nnls_norm",   broad_tags, val_b, nnls_norm),
        ("compact_simopt",    COMPACT,    val_c, simopt),
        ("broad_simopt",      broad_tags, val_b, simopt),
    ]:
        mean_oof, in_sim, oof = fold_eval(df_val, tags, fit_fn)
        coef = fit_fn(df_val[tags].to_numpy(), df_val)
        candidates.append({
            "label": tag_label,
            "tags": tags,
            "OOF_mean": round(mean_oof, 4),
            "OOF_folds": [round(x, 4) for x in oof],
            "in_sample": round(in_sim, 4),
            "gap": round(mean_oof - in_sim, 4),
            "weights": {t: round(float(c), 4) for t, c in zip(tags, coef)},
            "weight_sum": round(float(coef.sum()), 4),
        })

    df = pd.DataFrame([{k: v for k, v in c.items() if k != "tags"}
                       for c in candidates])
    df = df.sort_values("OOF_mean")
    print("\n=== final stack candidates ===")
    print(df.to_string(index=False))
    df.to_csv(V73 / "final_stack_cv.csv", index=False)

    eligible = [c for c in candidates if c["gap"] <= 0.05]
    champ = min(eligible or candidates, key=lambda c: c["OOF_mean"])
    print(f"\nCHAMPION: {champ['label']}  OOF={champ['OOF_mean']:.4f}  "
          f"gap={champ['gap']:.4f}  weights={champ['weights']}")

    # Re-fit on full val and materialize predictions
    if "broad" in champ["label"]:
        tags = broad_tags
        df_val_use = val_b
    else:
        tags = COMPACT
        df_val_use = val_c
    fit_fn = {
        "compact_nnls":      nnls_raw,
        "compact_nnls_norm": nnls_norm,
        "broad_nnls":        nnls_raw,
        "broad_nnls_norm":   nnls_norm,
        "compact_simopt":    simopt,
        "broad_simopt":      simopt,
    }[champ["label"]]
    coef = fit_fn(df_val_use[tags].to_numpy(), df_val_use)
    print("final coefficients:", {t: round(float(c), 4) for t, c in zip(tags, coef)})

    tst = _wide("test", tags)
    out_val = df_val_use[KEY].copy()
    out_val["target_qty"] = df_val_use["y"].to_numpy()
    out_val["prediction"] = np.clip(df_val_use[tags].to_numpy() @ coef, 0, None)
    out_tst = tst[KEY].copy()
    out_tst["target_qty"] = tst["y"].to_numpy()
    out_tst["prediction"] = np.clip(tst[tags].to_numpy() @ coef, 0, None)

    out_val.to_csv(OUT / "preds_v73_test.csv", index=False)  # placeholder; overwritten below
    out_val.to_csv(OUT / "preds_v73_val.csv", index=False)
    out_tst.to_csv(OUT / "preds_v73_test.csv", index=False)

    meta = {
        "label": champ["label"],
        "tags_used": tags,
        "weights": {t: float(c) for t, c in zip(tags, coef)},
        "weight_sum": float(coef.sum()),
        "OOF_SIMSCORE_mean": champ["OOF_mean"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "OOF_fold_SIMSCORE": champ["OOF_folds"],
    }
    (V73 / "final_stack_meta.json").write_text(json.dumps(meta, indent=2,
                                                          ensure_ascii=False))
    print(f"\nwrote preds_v73_val.csv, preds_v73_test.csv, "
          f"v73/final_stack_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
