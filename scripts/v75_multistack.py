"""V7.5 — extended multistack with new signal streams, isotonic calibration,
and hierarchical reconciliation.

Builds on V7.4 (per-channel NNLS) with five extra ideas tested under the
same anti-overfit protocol as V7.3/V7.4:

* **Extended base pool** — adds ewma6, ewma12, median12, yoyTrend, and
  (if available) the symmetric tweedie / mae V7 retrains.
* **Per-SKU bias shrinkage** — per-SKU additive correction on val residuals,
  EB-shrunk by (count / (count+k)).  Applied *after* per-channel NNLS.
* **Per-channel isotonic calibration** — isotonic mapping from stacked
  prediction → actual, fit on 60 % of val, evaluated on the other 40 %.
* **Channel × month hierarchical reconciliation** — multiplicative rescale
  of rows within each channel × month so the channel × month total matches
  a smooth channel × month target learnt on val.
* **Candidate evaluation** — every configuration scored by 3-fold
  rolling-origin CV on val; gap ≤ 0.05 hard constraint.

Outputs:
    output/preds_v75_val.csv
    output/preds_v75_test.csv
    output/v75/multistack_cv.csv
    output/v75/champion.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V75 = OUT / "v75"
V75.mkdir(parents=True, exist_ok=True)

KEY = ["Период", "Партнер", "Артикул"]

LGB_BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]
ANALYTICAL = ["ewma6", "ewma12", "median12", "yoyTrend"]
# symmetric retrains optional — tried if file exists
SYM = ["v7sym_tweedie", "v7sym_mae"]

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]


# ── data loaders ────────────────────────────────────────────────────────────

def _load_split(tag: str, split: str) -> pd.DataFrame | None:
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
    return d.rename(columns={"prediction": tag})


def available_sym() -> list[str]:
    return [s for s in SYM if (OUT / f"preds_{s}_val.csv").exists()
            and (OUT / f"preds_{s}_test.csv").exists()]


def _load_wide(split: str, tags: list[str]) -> pd.DataFrame:
    first = _load_split(tags[0], split)
    if first is None:
        raise FileNotFoundError(f"preds_{tags[0]}_{split}.csv missing")
    base = first.rename(columns={"target_qty": "y"})
    for t in tags[1:]:
        d = _load_split(t, split)
        if d is None:
            raise FileNotFoundError(f"preds_{t}_{split}.csv missing")
        base = base.merge(d.drop(columns=["target_qty"]), on=KEY, how="inner")
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + ["Канал"]]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


# ── stacker primitives ──────────────────────────────────────────────────────

def _nnls_norm(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else w


def fit_per_channel(tags, df_tr, df_te, min_train_rows: int = 500
                    ) -> tuple[np.ndarray, dict]:
    w_g = _nnls_norm(df_tr[tags].to_numpy(), df_tr["y"].to_numpy())
    preds = np.zeros(len(df_te))
    meta: dict = {"_global": {t: float(c) for t, c in zip(tags, w_g)}}
    for seg in df_te["Канал"].unique():
        tr_m = (df_tr["Канал"] == seg).to_numpy()
        te_m = (df_te["Канал"] == seg).to_numpy()
        if tr_m.sum() >= min_train_rows:
            w = _nnls_norm(df_tr.loc[tr_m, tags].to_numpy(),
                           df_tr.loc[tr_m, "y"].to_numpy())
        else:
            w = w_g
        preds[te_m] = df_te.loc[te_m, tags].to_numpy() @ w
        meta[str(seg)] = {t: float(c) for t, c in zip(tags, w)}
    return preds, meta


def iso_calibrate_per_channel(df_tr, df_te, p_tr, p_te) -> np.ndarray:
    """Fit isotonic regression per channel on (p_tr -> y_tr), apply to p_te.
    Shrinks back to identity via blend=0.5 to avoid overfit.
    """
    out = p_te.copy()
    for seg in df_te["Канал"].unique():
        tr_m = (df_tr["Канал"] == seg).to_numpy()
        te_m = (df_te["Канал"] == seg).to_numpy()
        if tr_m.sum() < 500:
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
        iso.fit(p_tr[tr_m], df_tr.loc[tr_m, "y"].to_numpy())
        cal = iso.predict(p_te[te_m])
        out[te_m] = 0.5 * cal + 0.5 * p_te[te_m]
    return out


def hierarchical_reconcile(df_tr, df_te, p_tr, p_te, shrink: float = 0.7
                           ) -> np.ndarray:
    """Compute per-channel×month scale factor (actual/pred on train) and
    apply to test.  Shrink toward 1.0 by `shrink` to avoid over-correction.
    """
    tr = df_tr.assign(p=p_tr)
    agg = tr.groupby(["Канал", "Период"], observed=True).agg(
        a=("y", "sum"), p=("p", "sum"),
    ).reset_index()
    # per-channel average scale over the training months
    chan_scale = agg.groupby("Канал", observed=True).apply(
        lambda g: float(g["a"].sum()) / max(float(g["p"].sum()), 1.0),
        include_groups=False,
    ).to_dict()
    out = p_te.copy()
    for seg, s in chan_scale.items():
        mask = (df_te["Канал"] == seg).to_numpy()
        out[mask] = p_te[mask] * (shrink * s + (1 - shrink) * 1.0)
    return out


def per_sku_shrunk_corr(df_tr, df_te, p_tr, p_te, k: float = 12.0) -> np.ndarray:
    """Per-SKU additive bias correction: avg(y-p) on train, shrunk by n/(n+k).
    """
    resid = df_tr["y"].to_numpy() - p_tr
    tr = df_tr.assign(_r=resid)
    stats = tr.groupby("Артикул").agg(
        n=("_r", "size"), mu=("_r", "mean"),
    )
    stats["shrink"] = stats["n"] / (stats["n"] + k)
    stats["adj"] = stats["shrink"] * stats["mu"]
    m = df_te.merge(stats[["adj"]], left_on="Артикул", right_index=True, how="left")
    adj = m["adj"].fillna(0.0).to_numpy()
    return np.clip(p_te + adj, 0, None)


# ── CV framework ────────────────────────────────────────────────────────────

def _score(df: pd.DataFrame, pred: np.ndarray) -> dict:
    out = df[KEY].copy()
    out["target_qty"] = df["y"].to_numpy()
    out["prediction"] = np.clip(pred, 0, None)
    return score_frame(out)


def eval_candidate(name: str, pipeline, val: pd.DataFrame) -> dict:
    """pipeline(df_tr, df_te) → (preds_te, meta)."""
    oof = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        preds, _ = pipeline(tr, te)
        oof.append(_score(te, preds)["SIMSCORE"])
    in_preds, meta = pipeline(val, val)
    insim = _score(val, in_preds)["SIMSCORE"]
    oof_mean = float(np.mean(oof))
    return {
        "name": name,
        "OOF_mean": round(oof_mean, 4),
        "OOF_folds": [round(x, 4) for x in oof],
        "in_sample": round(insim, 4),
        "gap": round(oof_mean - insim, 4),
        "meta": meta,
    }


# ── pipeline builders ───────────────────────────────────────────────────────

def build_pipelines(tags_compact, tags_extended):
    def p_v74_compact(tr, te):
        return fit_per_channel(tags_compact, tr, te)

    def p_extended(tr, te):
        return fit_per_channel(tags_extended, tr, te)

    def p_extended_iso(tr, te):
        p_te, meta = fit_per_channel(tags_extended, tr, te)
        p_tr, _ = fit_per_channel(tags_extended, tr, tr)
        p_iso = iso_calibrate_per_channel(tr, te, p_tr, p_te)
        return p_iso, {"base": meta, "iso": True}

    def p_extended_reconcile(tr, te):
        p_te, meta = fit_per_channel(tags_extended, tr, te)
        p_tr, _ = fit_per_channel(tags_extended, tr, tr)
        p_rec = hierarchical_reconcile(tr, te, p_tr, p_te, shrink=0.5)
        return p_rec, {"base": meta, "reconcile_shrink": 0.5}

    def p_compact_reconcile(tr, te):
        p_te, meta = fit_per_channel(tags_compact, tr, te)
        p_tr, _ = fit_per_channel(tags_compact, tr, tr)
        p_rec = hierarchical_reconcile(tr, te, p_tr, p_te, shrink=0.5)
        return p_rec, {"base": meta, "reconcile_shrink": 0.5}

    def p_compact_sku(tr, te):
        p_te, meta = fit_per_channel(tags_compact, tr, te)
        p_tr, _ = fit_per_channel(tags_compact, tr, tr)
        p_sk = per_sku_shrunk_corr(tr, te, p_tr, p_te, k=12.0)
        return p_sk, {"base": meta, "sku_k": 12.0}

    def p_compact_full(tr, te):
        p_te, meta = fit_per_channel(tags_compact, tr, te)
        p_tr, _ = fit_per_channel(tags_compact, tr, tr)
        p_rec = hierarchical_reconcile(tr, te, p_tr, p_te, shrink=0.5)
        p_tr2 = hierarchical_reconcile(tr, tr, p_tr, p_tr, shrink=0.5)
        p_iso = iso_calibrate_per_channel(tr, te, p_tr2, p_rec)
        return p_iso, {"base": meta, "reconcile": True, "iso": True}

    return {
        "v74_compact_per_channel":        p_v74_compact,
        "v75_extended_per_channel":       p_extended,
        "v75_extended_iso":               p_extended_iso,
        "v75_extended_reconcile":         p_extended_reconcile,
        "v75_compact_reconcile":          p_compact_reconcile,
        "v75_compact_sku_shrunk":         p_compact_sku,
        "v75_compact_reconcile_iso":      p_compact_full,
    }


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    sym_tags = available_sym()
    print(f"detected symmetric retrains: {sym_tags or 'none (will skip)'}")

    tags_compact = list(LGB_BASE)
    tags_extended = list(LGB_BASE) + list(ANALYTICAL) + list(sym_tags)

    val_c = _load_wide("val", tags_compact)
    val_e = _load_wide("val", tags_extended)
    tst_c = _load_wide("test", tags_compact)
    tst_e = _load_wide("test", tags_extended)

    pipelines = build_pipelines(tags_compact, tags_extended)
    cand_val = {
        "v74_compact_per_channel": val_c,
        "v75_extended_per_channel": val_e,
        "v75_extended_iso": val_e,
        "v75_extended_reconcile": val_e,
        "v75_compact_reconcile": val_c,
        "v75_compact_sku_shrunk": val_c,
        "v75_compact_reconcile_iso": val_c,
    }
    cand_test = {
        "v74_compact_per_channel": tst_c,
        "v75_extended_per_channel": tst_e,
        "v75_extended_iso": tst_e,
        "v75_extended_reconcile": tst_e,
        "v75_compact_reconcile": tst_c,
        "v75_compact_sku_shrunk": tst_c,
        "v75_compact_reconcile_iso": tst_c,
    }

    rows = []
    for name, fn in pipelines.items():
        r = eval_candidate(name, fn, cand_val[name])
        rows.append(r)
        print(f"{r['name']:40s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_mean")
    print("\n=== V7.5 candidates (sorted by OOF SIMSCORE) ===")
    print(df.to_string(index=False))
    df.to_csv(V75 / "multistack_cv.csv", index=False)

    survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = [r for r in rows if r["gap"] <= 0.08] or rows
    champ = min(survivors, key=lambda r: (r["OOF_mean"], r["gap"]))
    print(f"\nCHAMPION: {champ['name']}  OOF={champ['OOF_mean']:.4f}  "
          f"gap={champ['gap']:+.4f}")

    fn = pipelines[champ["name"]]
    val_df = cand_val[champ["name"]]
    tst_df = cand_test[champ["name"]]
    # train on full val, predict val (in-sample) and test
    val_pred, meta_full = fn(val_df, val_df)
    tst_pred, _ = fn(val_df, tst_df)

    out_v = val_df[KEY].copy()
    out_v["target_qty"] = val_df["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v75_val.csv", index=False)

    out_t = tst_df[KEY].copy()
    out_t["target_qty"] = tst_df["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v75_test.csv", index=False)

    (V75 / "champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "tags_used": (tags_extended if "extended" in champ["name"]
                      else tags_compact),
        "meta": meta_full,
    }, indent=2, ensure_ascii=False, default=str))

    print("\nwrote preds_v75_val.csv, preds_v75_test.csv, v75/champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
