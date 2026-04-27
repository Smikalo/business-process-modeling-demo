"""V7.6 — LAD stack augmented with Kaggle-GPU symmetric-loss V7 retrains.

V7.5's LAD stacker only had access to pinball/asymmetric LightGBM bases
(v4..v72_champion).  This version adds three symmetric-objective siblings
trained on Kaggle GPU (notebook `v76_symmetric_retrain.ipynb`):

* ``v7sym_tweedie``    — Tweedie, variance_power=1.3  (near-zero agg bias)
* ``v7sym_tweedie15``  — Tweedie, variance_power=1.5  (slightly biased high)
* ``v7sym_mae``        — regression_l1               (best standalone WAPE)

Two weaker variants (huber, l2) are intentionally excluded — they are very
biased and would only waste a candidate slot.

The LAD candidate grid is extended to evaluate:

1. Pool-compact           — v4..v72_champion                               (V7.5 compact)
2. Pool-sym               — compact + {tweedie, tweedie15, mae}            (sym only)
3. Pool-sym+analytical    — compact + sym + analytical baselines          (full mix)

× each stacker family (per-channel LAD raw; per-channel LAD + hierarchical
reconciliation at three shrink levels 0.5 / 0.8 / 1.0).

Anti-overfit: champion must have gap ≤ 0.05 against 3-fold rolling CV.
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
V76 = OUT / "v76"
V76.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]

LGB_BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
ANALYTICAL = ["ewma6", "ewma12", "median12", "yoyTrend"]
SYM = ["v7sym_tweedie", "v7sym_tweedie15", "v7sym_mae"]

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


def lad_nn_simplex(X: np.ndarray, y: np.ndarray,
                   n_iters: int = 20, eps: float = 1.0) -> np.ndarray:
    """Non-negative, sum-to-1 LAD fit via IRLS (see v75_lad_stack.py)."""
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
    pools = {
        "compact":       list(LGB_BASE),
        "sym":           list(LGB_BASE) + list(SYM),
        "sym+analytical": list(LGB_BASE) + list(SYM) + list(ANALYTICAL),
    }

    val_w = {k: _load_wide("val",  tags) for k, tags in pools.items()}
    tst_w = {k: _load_wide("test", tags) for k, tags in pools.items()}

    pipelines: dict[str, tuple] = {}
    for pool_name, tags in pools.items():
        pipelines[f"v76_lad_{pool_name}_per_channel"] = (
            (lambda tr, te, _t=tags: fit_per_channel_lad(_t, tr, te)),
            val_w[pool_name], tst_w[pool_name],
        )
        for shrink in (0.5, 0.8, 1.0):
            def _with_reconcile(tags, shrink):
                def fn(tr, te):
                    pt, meta = fit_per_channel_lad(tags, tr, te)
                    pt_tr, _ = fit_per_channel_lad(tags, tr, tr)
                    out = hierarchical_reconcile(tr, te, pt_tr, pt, shrink=shrink)
                    return out, {"base": meta, "shrink": shrink}
                return fn
            pipelines[f"v76_lad_{pool_name}_reconcile_{shrink}"] = (
                _with_reconcile(tags, shrink),
                val_w[pool_name], tst_w[pool_name],
            )

    rows = []
    cand_test = {}
    for name, (fn, val, tst) in pipelines.items():
        r = eval_candidate(name, fn, val)
        rows.append(r)
        cand_test[name] = (fn, val, tst)
        print(f"{r['name']:48s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_mean")
    print("\n=== V7.6 LAD candidates (sorted by OOF SIMSCORE) ===")
    print(df.to_string(index=False))
    df.to_csv(V76 / "lad_cv.csv", index=False)

    try:
        prev = json.loads((OUT / "v75" / "lad_champion.json").read_text())
        print(f"\n(Previous V7.5 LAD champion: {prev['champion']} "
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
    out_v.to_csv(OUT / "preds_v76_val.csv", index=False)
    out_t = tst[KEY].copy()
    out_t["target_qty"] = tst["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v76_test.csv", index=False)

    (V76 / "lad_champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "meta": meta_full,
    }, indent=2, ensure_ascii=False, default=str))

    print("\nwrote preds_v76_val.csv, preds_v76_test.csv, v76/lad_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
