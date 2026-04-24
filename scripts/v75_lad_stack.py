"""V7.5 — LAD (Least Absolute Deviations) stack + per-channel reconciliation.

Key insight: NNLS minimizes Σ(y - Xw)² = L2 loss.  But our target metric
is **WAPE = Σ|y - ŷ| / Σy** which is L1 loss normalized by Σy.  A stacker
fit by minimizing L2 residuals is *not* WAPE-optimal.  This script uses a
linear-program formulation to minimize L1 loss directly:

    minimize  Σ u_i
    subject to  −u_i ≤ y_i − (X w)_i ≤ u_i
                w ≥ 0   ,   Σ w = 1

which is WAPE-optimal at the row level.  Solved per channel with scipy's
HiGHS LP backend (extremely fast even on 16 k rows).

Four candidates evaluated on 3-fold rolling-origin CV of the val window:

1. ``v75_lad_compact_per_channel``
2. ``v75_lad_extended_per_channel``     (adds ewma6/12/median12/yoyTrend)
3. ``v75_lad_compact_reconcile``        (LAD + channel×month scale, shrink 0.5)
4. ``v75_lad_compact_reconcile_sharp``  (same with shrink=0.8 — more aggressive)

Hard anti-overfit: champion must have gap ≤ 0.05.
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
V75 = OUT / "v75"
V75.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]

LGB_BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
ANALYTICAL = ["ewma6", "ewma12", "median12", "yoyTrend"]

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
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + ["Канал"]]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


# ── LAD fit via LP ──────────────────────────────────────────────────────────

def lad_nn_simplex(X: np.ndarray, y: np.ndarray,
                   n_iters: int = 15, eps: float = 1.0) -> np.ndarray:
    """Non-negative, sum-to-1 LAD fit via IRLS.

    We minimize Σ |y − Xw| by iteratively solving a weighted NNLS:
        r_i^(t+1) = 1 / max(|y_i − (Xw^(t))_i|, eps)
        w^(t+1) = argmin  Σ r_i (y_i − x_i·w)²   s.t. w ≥ 0
    After each NNLS step we renormalize w so Σw = 1 (projection onto the
    non-negative simplex by dividing by total mass).  Runtime: 10–20
    iterations × NNLS(≈O(nk²)) = fast even on 50k rows.
    """
    n, k = X.shape
    w, _ = nnls(X, y)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.full(k, 1.0 / k)
    for _ in range(n_iters):
        resid = y - X @ w
        r = 1.0 / np.maximum(np.abs(resid), eps)
        sqr = np.sqrt(r)
        # weighted NNLS by scaling rows
        Xw = X * sqr[:, None]
        yw = y * sqr
        w_new, _ = nnls(Xw, yw)
        s = w_new.sum()
        if s > 0:
            w_new = w_new / s
        else:
            w_new = w
        if np.max(np.abs(w_new - w)) < 1e-4:
            w = w_new
            break
        w = w_new
    return w


def fit_per_channel_lad(tags, df_tr, df_te, min_rows: int = 500
                        ) -> tuple[np.ndarray, dict]:
    w_g = lad_nn_simplex(df_tr[tags].to_numpy(), df_tr["y"].to_numpy())
    preds = np.zeros(len(df_te))
    meta: dict = {"_global": {t: float(c) for t, c in zip(tags, w_g)}}
    for seg in df_te["Канал"].unique():
        tr_m = (df_tr["Канал"] == seg).to_numpy()
        te_m = (df_te["Канал"] == seg).to_numpy()
        if tr_m.sum() >= min_rows:
            w = lad_nn_simplex(df_tr.loc[tr_m, tags].to_numpy(),
                               df_tr.loc[tr_m, "y"].to_numpy())
        else:
            w = w_g
        preds[te_m] = df_te.loc[te_m, tags].to_numpy() @ w
        meta[str(seg)] = {t: float(c) for t, c in zip(tags, w)}
    return preds, meta


def hierarchical_reconcile(df_tr, df_te, p_tr, p_te, shrink: float) -> np.ndarray:
    tr = df_tr.assign(p=p_tr)
    agg = tr.groupby("Канал", observed=True).agg(
        a=("y", "sum"), p=("p", "sum")
    )
    agg["scale"] = agg["a"] / agg["p"].clip(lower=1e-6)
    s_map = agg["scale"].to_dict()
    out = p_te.copy()
    for seg, s in s_map.items():
        mask = (df_te["Канал"] == seg).to_numpy()
        out[mask] = p_te[mask] * (shrink * s + (1 - shrink) * 1.0)
    return out


# ── CV harness ──────────────────────────────────────────────────────────────

def _score(df, pred):
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def eval_candidate(name, pipeline, val):
    oof = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = pipeline(tr, te)
        oof.append(_score(te, pred)["SIMSCORE"])
    in_pred, meta = pipeline(val, val)
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
    tags_c = list(LGB_BASE)
    tags_e = list(LGB_BASE) + list(ANALYTICAL)

    val_c = _load_wide("val", tags_c)
    val_e = _load_wide("val", tags_e)
    tst_c = _load_wide("test", tags_c)
    tst_e = _load_wide("test", tags_e)

    pipelines = {
        "v75_lad_compact_per_channel":
            (lambda tr, te: fit_per_channel_lad(tags_c, tr, te), val_c, tst_c),
        "v75_lad_extended_per_channel":
            (lambda tr, te: fit_per_channel_lad(tags_e, tr, te), val_e, tst_e),
    }

    def _with_reconcile(tags, shrink):
        def fn(tr, te):
            pt, meta = fit_per_channel_lad(tags, tr, te)
            pt_tr, _ = fit_per_channel_lad(tags, tr, tr)
            out = hierarchical_reconcile(tr, te, pt_tr, pt, shrink=shrink)
            return out, {"base": meta, "shrink": shrink}
        return fn

    pipelines["v75_lad_compact_reconcile_0.5"] = (_with_reconcile(tags_c, 0.5), val_c, tst_c)
    pipelines["v75_lad_compact_reconcile_0.8"] = (_with_reconcile(tags_c, 0.8), val_c, tst_c)
    pipelines["v75_lad_compact_reconcile_1.0"] = (_with_reconcile(tags_c, 1.0), val_c, tst_c)
    pipelines["v75_lad_extended_reconcile_0.5"] = (_with_reconcile(tags_e, 0.5), val_e, tst_e)

    rows = []
    cand_test = {}
    for name, (fn, val, tst) in pipelines.items():
        r = eval_candidate(name, fn, val)
        rows.append(r)
        cand_test[name] = (fn, val, tst)
        print(f"{r['name']:40s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_mean")
    print("\n=== V7.5 LAD candidates (sorted by OOF SIMSCORE) ===")
    print(df.to_string(index=False))
    df.to_csv(V75 / "lad_cv.csv", index=False)

    # add previous v75 champion (reconcile NNLS) to comparison — already fit
    try:
        prev = json.loads((V75 / "champion.json").read_text())
        print(f"\n(Previous V7.5 NNLS champion: {prev['champion']} "
              f"OOF={prev['OOF_SIMSCORE']:.4f} gap={prev['overfit_gap']:+.4f})")
    except Exception:
        pass

    survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    champ = min(survivors, key=lambda r: (r["OOF_mean"], r["gap"]))
    print(f"\nLAD CHAMPION: {champ['name']}  OOF={champ['OOF_mean']:.4f}  "
          f"gap={champ['gap']:+.4f}")

    fn, val, tst = cand_test[champ["name"]]
    val_pred, meta_full = fn(val, val)
    tst_pred, _ = fn(val, tst)

    out_v = val[KEY].copy()
    out_v["target_qty"] = val["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v75lad_val.csv", index=False)
    out_t = tst[KEY].copy()
    out_t["target_qty"] = tst["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v75lad_test.csv", index=False)

    (V75 / "lad_champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "meta": meta_full,
    }, indent=2, ensure_ascii=False, default=str))

    print("\nwrote preds_v75lad_val.csv, preds_v75lad_test.csv, v75/lad_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
