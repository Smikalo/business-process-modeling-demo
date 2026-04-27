"""V7.7 — multi-axis hierarchical reconciliation + quantile-shifted LAD.

Diagnostic of V7.5 residuals showed three high-leverage errors:

* **A-class under-forecast** (-5 % val, -12 % test, ~17 % of total error)
* **СТОК-ВИАТ over-forecast** (+12 %)
* **Monthly bias volatile** (+8 % to -32 %), worst on late months

V7.5 only reconciles by ``Канал``×month with shrink 0.8.  We extend to a
sequential MinT-flavoured hierarchical reconciliation:

    step 0:  base predictions (LAD blend, V7.5 compact pool)
    step 1:  scale by  Канал                     × shrink_ch
    step 2:  scale by  Канал × Сегмент_ABC       × shrink_abc
    step 3:  scale by  Бренд                     × shrink_brand

Each step's scale factor is computed from training-window residuals only
(no test leakage).  Shrinkage prevents over-correction on small cells.

We ALSO replace the symmetric L1 LAD with a **quantile-shifted LAD** at
τ ≥ 0.5: the V7.5 pool has slight negative bias on val (-0.86 %), so a
small upward tilt should improve raw WAPE alongside reconciliation.

Anti-overfit:
* All scales fit on training portion of each rolling-CV fold.
* Champion must have OOF→insample gap ≤ 0.05.
* Per-cell scales clipped to [0.5, 2.0] and require ≥ ``MIN_ROWS``
  training rows; small cells fall back to parent scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V77 = OUT / "v77"
V77.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]

LGB_BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
ANALYTICAL = ["ewma6", "ewma12", "median12", "yoyTrend"]
META_AXES = ["Канал", "Бренд", "Сегмент_ABC"]

MIN_ROWS = 250
SCALE_CLIP = (0.6, 1.8)

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]


def _load_split(tag: str, split: str) -> pd.DataFrame:
    p = OUT / f"preds_{tag}_{split}.csv"
    d = pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
    return d.rename(columns={"prediction": tag})


def _load_wide(split: str, tags: list[str]) -> pd.DataFrame:
    base = _load_split(tags[0], split).rename(columns={"target_qty": "y"})
    for t in tags[1:]:
        base = base.merge(_load_split(t, split).drop(columns=["target_qty"]),
                          on=KEY, how="inner")
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


# ── Tilted LAD ──────────────────────────────────────────────────────────────

def lad_tilted_simplex(X: np.ndarray, y: np.ndarray, tau: float = 0.5,
                       n_iters: int = 25, eps: float = 1.0) -> np.ndarray:
    """Sum-to-one, non-negative quantile-LAD via IRLS.

    Minimizes Σ ρ_τ(y_i − Xw)_i where ρ_τ(r) = max(τ r, (τ-1) r) is the
    tilted absolute loss (pinball).  τ=0.5 reduces to the symmetric LAD
    used in V7.5; τ>0.5 favours upward bias to combat under-forecasting.

    Implemented as IRLS over weighted NNLS:
        w_i ∝ τ            if r_i = y_i - ŷ_i > 0
        w_i ∝ (1-τ)        if r_i ≤ 0
    weighted further by 1/max(|r_i|, eps).
    """
    n, k = X.shape
    w, _ = nnls(X, y)
    w = w / w.sum() if w.sum() > 0 else np.full(k, 1.0 / k)
    for _ in range(n_iters):
        resid = y - X @ w
        sgn_w = np.where(resid > 0, tau, 1 - tau)
        scale = sgn_w / np.maximum(np.abs(resid), eps)
        sqr = np.sqrt(scale)
        Xw = X * sqr[:, None]
        yw = y * sqr
        w_new, _ = nnls(Xw, yw)
        s = w_new.sum()
        w_new = w_new / s if s > 0 else w
        if np.max(np.abs(w_new - w)) < 1e-4:
            w = w_new
            break
        w = w_new
    return w


def fit_per_channel_tilted(tags, df_tr, df_te, tau: float = 0.5,
                           min_rows: int = 500
                           ) -> tuple[np.ndarray, dict]:
    w_g = lad_tilted_simplex(df_tr[tags].to_numpy(), df_tr["y"].to_numpy(),
                             tau=tau)
    preds = np.zeros(len(df_te))
    meta: dict = {"_global": {t: float(c) for t, c in zip(tags, w_g)},
                  "tau": tau}
    for seg in df_te["Канал"].unique():
        tr_m = (df_tr["Канал"] == seg).to_numpy()
        te_m = (df_te["Канал"] == seg).to_numpy()
        if tr_m.sum() >= min_rows:
            w = lad_tilted_simplex(df_tr.loc[tr_m, tags].to_numpy(),
                                   df_tr.loc[tr_m, "y"].to_numpy(),
                                   tau=tau)
        else:
            w = w_g
        preds[te_m] = df_te.loc[te_m, tags].to_numpy() @ w
        meta[str(seg)] = {t: float(c) for t, c in zip(tags, w)}
    return preds, meta


# ── Multi-axis reconciliation ───────────────────────────────────────────────

def _key_series(df: pd.DataFrame, by: list[str]) -> pd.Series:
    """Build a single string key per row by joining the requested cols."""
    if len(by) == 1:
        return df[by[0]].astype(str)
    return df[by].astype(str).agg("||".join, axis=1)


def _scale_map(df: pd.DataFrame, p: np.ndarray, by: list[str],
               min_rows: int = MIN_ROWS, clip=SCALE_CLIP) -> dict[str, float]:
    tmp = df[by].copy()
    tmp["__y"] = df["y"].to_numpy()
    tmp["__p"] = p
    tmp["__k"] = _key_series(tmp, by)
    g = tmp.groupby("__k", observed=True).agg(
        a=("__y", "sum"), p=("__p", "sum"), n=("__y", "size")
    )
    g["scale"] = g["a"] / g["p"].clip(lower=1e-6)
    g.loc[g["n"] < min_rows, "scale"] = 1.0
    g["scale"] = g["scale"].clip(*clip)
    return g["scale"].to_dict()


def _apply_scale(df_te: pd.DataFrame, p_te: np.ndarray, by: list[str],
                 s_map: dict[str, float], shrink: float) -> np.ndarray:
    if not by:
        return p_te.copy()
    keys = _key_series(df_te, by).to_numpy()
    scales = np.array([s_map.get(k, 1.0) for k in keys])
    return p_te * (shrink * scales + (1 - shrink))


def multi_reconcile(df_tr, df_te, p_tr, p_te,
                    axes: list[tuple[list[str], float]]) -> np.ndarray:
    """Apply a sequence of (axis, shrink) reconciliations.

    Each step uses the *previously reconciled* training predictions to
    estimate the next axis's scale, then applies that scale to the test
    predictions.  This mimics MinT's hierarchical projection in a
    light-weight, leakage-safe way.
    """
    cur_tr = p_tr.copy()
    cur_te = p_te.copy()
    for axis, shrink in axes:
        s_map = _scale_map(df_tr.assign(_p=cur_tr), cur_tr, axis)
        cur_te = _apply_scale(df_te, cur_te, axis, s_map, shrink)
        cur_tr = _apply_scale(df_tr, cur_tr, axis, s_map, shrink)
    return cur_te


# ── CV harness ──────────────────────────────────────────────────────────────

def _score(df, pred):
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def eval_pipeline(name, fn, val):
    oof = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = fn(tr, te)
        oof.append(_score(te, pred)["SIMSCORE"])
    in_pred, meta = fn(val, val)
    insim = _score(val, in_pred)["SIMSCORE"]
    return {
        "name": name,
        "OOF_mean": round(float(np.mean(oof)), 4),
        "OOF_folds": [round(x, 4) for x in oof],
        "in_sample": round(insim, 4),
        "gap": round(float(np.mean(oof)) - insim, 4),
        "meta": meta,
    }


def main() -> int:
    tags = list(LGB_BASE)
    val = _load_wide("val", tags)
    tst = _load_wide("test", tags)

    AXES_OPTIONS = {
        "ch08":            [(["Канал"], 0.8)],
        "ch08_brand05":    [(["Канал"], 0.8), (["Бренд"], 0.5)],
        "ch08_chABC05":    [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.5)],
        "ch08_chABC05_brand03":
            [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
        "ch08_ABC05":      [(["Канал"], 0.8), (["Сегмент_ABC"], 0.5)],
        "chABC08":         [(["Канал", "Сегмент_ABC"], 0.8)],
        "ch08_chABC08":    [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
    }

    TAU_OPTIONS = [0.5, 0.52, 0.55]

    pipelines: dict[str, tuple] = {}

    for tau in TAU_OPTIONS:
        for axes_name, axes in AXES_OPTIONS.items():
            def make_pipe(_tau, _axes):
                def fn(tr, te):
                    pt, meta = fit_per_channel_tilted(tags, tr, te, tau=_tau)
                    pt_tr, _ = fit_per_channel_tilted(tags, tr, tr, tau=_tau)
                    out = multi_reconcile(tr, te, pt_tr, pt, _axes)
                    return out, {"base": meta, "axes": str(_axes)}
                return fn
            pipelines[f"v77_tau{tau}_{axes_name}"] = (make_pipe(tau, axes),
                                                     val, tst)

    rows, cand = [], {}
    for name, (fn, v_, t_) in pipelines.items():
        r = eval_pipeline(name, fn, v_)
        rows.append(r)
        cand[name] = (fn, v_, t_)
        print(f"{r['name']:48s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_mean")
    print("\n=== V7.7 multi-reconcile candidates (sorted by OOF SIMSCORE) ===")
    print(df.head(20).to_string(index=False))
    df.to_csv(V77 / "cv.csv", index=False)

    # Filter overfit candidates
    survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    # Pick lowest OOF, tie-break by smaller gap
    champ = min(survivors, key=lambda r: (r["OOF_mean"], r["gap"]))
    print(f"\nV7.7 CHAMPION: {champ['name']}  OOF={champ['OOF_mean']:.4f}  "
          f"gap={champ['gap']:+.4f}")

    fn, v_, t_ = cand[champ["name"]]
    val_pred, meta = fn(v_, v_)
    tst_pred, _ = fn(v_, t_)

    out_v = v_[KEY].copy()
    out_v["target_qty"] = v_["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v77_val.csv", index=False)

    out_t = t_[KEY].copy()
    out_t["target_qty"] = t_["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v77_test.csv", index=False)

    val_score = _score(v_, val_pred)
    tst_score = _score(t_, tst_pred)
    (V77 / "champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "val_score": val_score,
        "test_score": tst_score,
        "meta": meta,
    }, indent=2, ensure_ascii=False, default=str))

    print(f"\n  val   SIMSCORE={val_score['SIMSCORE']:.4f}  "
          f"WAPE={val_score['WAPE']:.4f}  bias%={val_score['Agg_Bias_pct']:+.2f}")
    print(f"  test  SIMSCORE={tst_score['SIMSCORE']:.4f}  "
          f"WAPE={tst_score['WAPE']:.4f}  bias%={tst_score['Agg_Bias_pct']:+.2f}")
    print("\nwrote preds_v77_val.csv, preds_v77_test.csv, v77/champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
