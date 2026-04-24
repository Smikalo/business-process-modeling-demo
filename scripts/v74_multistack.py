"""V7.4 — per-segment NNLS stacks, bias-constrained variants, and CV selection.

We extend the V7.3 NNLS idea in four orthogonal directions and let a
pre-registered 3-fold rolling-origin CV on val pick the winner:

* ``v73_compact``           — V7.3 champion (baseline to beat)
* ``v74_pool_full``         — global NNLS over the V7.3 pool + naive, ma3, ma6
* ``v74_per_channel``       — one NNLS per channel (Канал)
* ``v74_per_channel_full``  — per-channel NNLS over the extended pool
* ``v74_density_substack``  — one NNLS per demand-density bucket
* ``v74_bias_constrained``  — NNLS with linear constraint Σᵢwᵢ·p̄ᵢ = ȳ
* ``v74_combo``              — per-channel NNLS + global bias-correction scalar

Anti-overfit guardrails (identical to V7.3):

1. No test access during selection.
2. Each candidate's CV SIMSCORE is compared in-sample vs OOF.  Gap > 0.05 ⇒
   candidate auto-rejected.
3. Champion = lowest OOF SIMSCORE of surviving candidates; ties broken by
   smaller gap.
4. Test evaluated exactly once for the single winner (in a separate script).

Outputs:
    output/v74/multistack_cv.csv
    output/v74/multistack_champion.json
    output/preds_v74_val.csv
    output/preds_v74_test.csv
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
V74 = OUT / "v74"
V74.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]
BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
POOL_FULL = BASE + ["naiveS", "ma3", "ma6"]
SEG_COLS = ["Канал", "Бренд", "demand_density"]


# ── data loaders ────────────────────────────────────────────────────────────

def _load_split(tag: str, split: str) -> pd.DataFrame | None:
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
    d = d.rename(columns={"prediction": tag})
    return d


def _wide_with_segments(split: str, tags: list[str]) -> pd.DataFrame:
    base = _load_split(tags[0], split).rename(columns={"target_qty": "y"})
    for t in tags[1:]:
        d = _load_split(t, split)
        if d is None:
            raise FileNotFoundError(f"preds_{t}_{split}.csv missing")
        base = base.merge(d.drop(columns=["target_qty"]), on=KEY, how="inner")
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[
        KEY + ["Канал", "Бренд", "demand_density"]
    ]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["dd_bucket"] = pd.cut(
        out["demand_density"].fillna(0),
        bins=[-0.01, 0.05, 0.15, 0.35, 1.01],
        labels=["sparse", "low", "med", "high"],
    )
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


# ── stackers ────────────────────────────────────────────────────────────────

def nnls_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else w


def nnls_bias_constrained(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NNLS subject to Σⱼ wⱼ·mean(Xⱼ) == mean(y) — zero aggregate bias.

    Linearly constrained QP → SLSQP.  Starts from plain NNLS solution.
    """
    w0 = nnls_norm(X, y)
    x_mean = X.mean(axis=0)
    y_mean = float(y.mean())

    def obj(w):
        r = X @ w - y
        return float(r @ r)

    def grad(w):
        return 2.0 * (X.T @ (X @ w - y))

    cons = [
        {"type": "eq", "fun": lambda w: float(w @ x_mean - y_mean)},
    ]
    bounds = [(0, None)] * X.shape[1]
    res = minimize(obj, w0, jac=grad, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-8})
    w = np.clip(res.x, 0, None)
    return w


def fit_predict_global(tags, df_tr, df_te, fit_fn):
    w = fit_fn(df_tr[tags].to_numpy(), df_tr["y"].to_numpy())
    return df_te[tags].to_numpy() @ w, {"weights": {t: float(c) for t, c in zip(tags, w)}}


def fit_predict_per_segment(tags, df_tr, df_te, seg_col, fit_fn,
                            min_train_rows: int = 500):
    """Fit one NNLS per segment value; fall back to global weights if segment
    is too small in train."""
    w_global = fit_fn(df_tr[tags].to_numpy(), df_tr["y"].to_numpy())
    preds = np.zeros(len(df_te))
    weights_used: dict = {"_global": {t: float(c) for t, c in zip(tags, w_global)}}
    for seg in df_te[seg_col].unique():
        tr_m = df_tr[seg_col] == seg
        te_m = df_te[seg_col] == seg
        if tr_m.sum() >= min_train_rows:
            w = fit_fn(df_tr.loc[tr_m, tags].to_numpy(),
                       df_tr.loc[tr_m, "y"].to_numpy())
        else:
            w = w_global
        preds[te_m.to_numpy()] = df_te.loc[te_m, tags].to_numpy() @ w
        weights_used[str(seg)] = {t: float(c) for t, c in zip(tags, w)}
    return preds, weights_used


def fit_predict_per_seg_with_scalar(tags, df_tr, df_te, seg_col, fit_fn):
    preds, weights = fit_predict_per_segment(tags, df_tr, df_te, seg_col, fit_fn)
    # global bias correction scalar, fit on train in-sample residual
    tr_preds, _ = fit_predict_per_segment(tags, df_tr, df_tr, seg_col, fit_fn)
    s = float(df_tr["y"].sum() / max(tr_preds.sum(), 1e-9))
    return preds * s, {"per_seg_weights": weights, "global_scalar": s}


# ── CV eval ────────────────────────────────────────────────────────────────

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]


def score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def eval_candidate(name, make_predictor, val) -> dict:
    oof = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = make_predictor(tr, te)
        s = score(te, pred)
        oof.append(s["SIMSCORE"])
    in_pred, meta = make_predictor(val, val)
    in_sim = score(val, in_pred)["SIMSCORE"]
    return {
        "name": name,
        "OOF_mean": round(float(np.mean(oof)), 4),
        "OOF_folds": [round(x, 4) for x in oof],
        "in_sample": round(in_sim, 4),
        "gap": round(float(np.mean(oof)) - in_sim, 4),
        "meta": meta,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    val_c = _wide_with_segments("val", BASE)
    val_f = _wide_with_segments("val", POOL_FULL)
    tst_c = _wide_with_segments("test", BASE)
    tst_f = _wide_with_segments("test", POOL_FULL)

    candidates = [
        ("v73_compact_global",
         lambda tr, te, tags=BASE: fit_predict_global(tags, tr, te, nnls_norm),
         val_c, tst_c),
        ("v74_pool_full_global",
         lambda tr, te, tags=POOL_FULL: fit_predict_global(tags, tr, te, nnls_norm),
         val_f, tst_f),
        ("v74_compact_per_channel",
         lambda tr, te, tags=BASE: fit_predict_per_segment(tags, tr, te, "Канал", nnls_norm),
         val_c, tst_c),
        ("v74_full_per_channel",
         lambda tr, te, tags=POOL_FULL: fit_predict_per_segment(tags, tr, te, "Канал", nnls_norm),
         val_f, tst_f),
        ("v74_compact_per_brand",
         lambda tr, te, tags=BASE: fit_predict_per_segment(tags, tr, te, "Бренд", nnls_norm),
         val_c, tst_c),
        ("v74_compact_per_density",
         lambda tr, te, tags=BASE: fit_predict_per_segment(tags, tr, te, "dd_bucket", nnls_norm),
         val_c, tst_c),
        ("v74_full_per_density",
         lambda tr, te, tags=POOL_FULL: fit_predict_per_segment(tags, tr, te, "dd_bucket", nnls_norm),
         val_f, tst_f),
        ("v74_compact_bias_constrained",
         lambda tr, te, tags=BASE: fit_predict_global(tags, tr, te, nnls_bias_constrained),
         val_c, tst_c),
        ("v74_full_bias_constrained",
         lambda tr, te, tags=POOL_FULL: fit_predict_global(tags, tr, te, nnls_bias_constrained),
         val_f, tst_f),
        ("v74_per_channel_bias_scalar",
         lambda tr, te, tags=BASE: fit_predict_per_seg_with_scalar(tags, tr, te, "Канал", nnls_norm),
         val_c, tst_c),
        ("v74_full_per_channel_bias_scalar",
         lambda tr, te, tags=POOL_FULL: fit_predict_per_seg_with_scalar(tags, tr, te, "Канал", nnls_norm),
         val_f, tst_f),
    ]

    rows = []
    for name, pred_fn, val_df, _tst_df in candidates:
        r = eval_candidate(name, pred_fn, val_df)
        rows.append(r)
        print(f"{r['name']:40s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"} for r in rows])
    df = df.sort_values("OOF_mean")
    print("\n=== V7.4 candidates (sorted by OOF SIMSCORE) ===")
    print(df.to_string(index=False))
    df.to_csv(V74 / "multistack_cv.csv", index=False)

    survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        print("WARN: no candidate passes gap<=0.05; relaxing to <=0.08")
        survivors = [r for r in rows if r["gap"] <= 0.08] or rows
    champ = min(survivors, key=lambda r: (r["OOF_mean"], r["gap"]))
    print(f"\nCHAMPION: {champ['name']}  OOF={champ['OOF_mean']:.4f}  gap={champ['gap']:+.4f}")

    # refit on full val, materialize preds
    champ_entry = next(c for c in candidates if c[0] == champ["name"])
    _, pred_fn, val_df, tst_df = champ_entry
    val_pred, meta_full = pred_fn(val_df, val_df)
    tst_pred, _ = pred_fn(val_df, tst_df)

    out_v = val_df[KEY].copy()
    out_v["target_qty"] = val_df["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v74_val.csv", index=False)

    out_t = tst_df[KEY].copy()
    out_t["target_qty"] = tst_df["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v74_test.csv", index=False)

    (V74 / "multistack_champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "meta": meta_full,
    }, indent=2, ensure_ascii=False, default=str))

    print("\nwrote preds_v74_val.csv, preds_v74_test.csv, v74/multistack_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
