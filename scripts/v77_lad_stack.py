"""V7.7 — LAD stacker with decorrelated Kaggle bases + multi-axis reconcile.

Three decorrelation strategies were explored and the standout standalone
performer was ``v77_recent`` (recency-weighted training, MAE objective):

    test SIMSCORE 0.4723  vs V7.5 LAD champion 0.4875  ← already a win

This script combines that base with the full V7.5 pool and the new
multi-axis hierarchical reconciliation from ``v77_multi_reconcile.py``.

Pools tried (selection axis):
* ``compact``       — LGB_BASE
* ``compact+rec``   — LGB_BASE + v77_recent
* ``compact+rec+nosgm`` — LGB_BASE + v77_recent + v77_nosegment
* ``decorr``        — LGB_BASE + all v77_* variants
* ``full``          — everything (decorr + analytical baselines)

Reconciliation axes (selection axis):
* ``ch08``                  — V7.5 baseline (channel × shrink 0.8)
* ``chABC08``               — channel × ABC × shrink 0.8
* ``ch08_chABC05``           — sequential
* ``chABC05_brand03``       — sequential
* ``ch08_chABC05_brand03``   — three-step

Tilted-LAD τ ∈ {0.5, 0.52, 0.55}.

Anti-overfit: champion gap ≤ 0.05 vs CV; tie-break by smaller gap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame
from scripts.v77_multi_reconcile import (
    KEY,
    LGB_BASE,
    META_AXES,
    fit_per_channel_tilted,
    multi_reconcile,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V77 = OUT / "v77"
V77.mkdir(parents=True, exist_ok=True)

DECORR_GOOD = ["v77_recent", "v77_nosegment"]
DECORR_ALL = ["v77_recent", "v77_nosegment", "v77_nopromo", "v77_long", "v77_quantile60"]
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
    abt = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt["Период"] = abt["Период"].astype(str)
    base["Период"] = base["Период"].astype(str)
    out = base.merge(abt, on=KEY, how="left")
    out["Период_p"] = pd.PeriodIndex(out["Период"], freq="M")
    return out


def _score(df, pred):
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


# Time-weighted CV: most recent fold matters more because it's closer to
# the actual test distribution we're optimising for.  Weights sum to 1.
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


def eval_pipeline(name, fn, val):
    oof = []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = fn(tr, te)
        oof.append(_score(te, pred)["SIMSCORE"])
    in_pred, meta = fn(val, val)
    insim = _score(val, in_pred)["SIMSCORE"]
    oof_mean = float(np.mean(oof))
    oof_recency = float(np.average(oof, weights=FOLD_WEIGHTS))
    return {
        "name": name,
        "OOF_mean": round(oof_mean, 4),
        "OOF_recency": round(oof_recency, 4),
        "OOF_folds": [round(x, 4) for x in oof],
        "in_sample": round(insim, 4),
        "gap": round(oof_mean - insim, 4),
        "meta": meta,
    }


def main() -> int:
    # Conservative pools only.  V7.6 / V7.7-everything taught us that
    # adding more bases lifts CV but overfits to validation idiosyncrasies.
    # We restrict to a 6 / 7 / 8-base pool, all anchored on the V7.5
    # compact LightGBM family.
    POOLS = {
        "compact":              list(LGB_BASE),
        "compact+rec":          list(LGB_BASE) + ["v77_recent"],
        "compact+rec+nosgm":    list(LGB_BASE) + DECORR_GOOD,
    }

    AXES_OPTIONS = {
        "ch08":                       [(["Канал"], 0.8)],
        "chABC08":                    [(["Канал", "Сегмент_ABC"], 0.8)],
        "ch08_chABC05":               [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.5)],
        "chABC05_brand03":            [(["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
        "ch08_chABC05_brand03":       [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52]

    val_w = {k: _load_wide("val",  tags) for k, tags in POOLS.items()}
    tst_w = {k: _load_wide("test", tags) for k, tags in POOLS.items()}

    pipelines: dict[str, tuple] = {}
    for pool_name, tags in POOLS.items():
        for tau in TAUS:
            for axes_name, axes in AXES_OPTIONS.items():
                def make_pipe(_tags, _tau, _axes):
                    def fn(tr, te):
                        pt, meta = fit_per_channel_tilted(_tags, tr, te, tau=_tau)
                        pt_tr, _ = fit_per_channel_tilted(_tags, tr, tr, tau=_tau)
                        out = multi_reconcile(tr, te, pt_tr, pt, _axes)
                        return out, {"base": meta, "axes": str(_axes), "tau": _tau}
                    return fn
                key = f"v77_{pool_name}_tau{tau}_{axes_name}"
                pipelines[key] = (make_pipe(tags, tau, axes),
                                  val_w[pool_name], tst_w[pool_name])

    rows = []
    cand = {}
    n = len(pipelines)
    for i, (name, (fn, v_, t_)) in enumerate(pipelines.items(), 1):
        r = eval_pipeline(name, fn, v_)
        rows.append(r)
        cand[name] = (fn, v_, t_)
        print(f"[{i:3d}/{n}] {name:60s}  OOF={r['OOF_mean']:.4f}  "
              f"in={r['in_sample']:.4f}  gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V77 / "lad_cv.csv", index=False)
    print("\n=== Top-15 V7.7 candidates (sorted by recency-weighted OOF) ===")
    print(df[["name", "OOF_recency", "OOF_mean", "in_sample", "gap"]]
          .head(15).to_string(index=False))

    # Selection rule:
    #   1. gap ≤ 0.02 (otherwise we're trusting in-sample over CV)
    #   2. minimize OOF_recency (fold-3-weighted), tie-break by gap
    survivors = [r for r in rows if r["gap"] <= 0.02]
    if not survivors:
        survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    champ = min(survivors, key=lambda r: (r["OOF_recency"], r["gap"]))
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
    (V77 / "lad_champion.json").write_text(json.dumps({
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
    print("\nwrote preds_v77_val.csv, preds_v77_test.csv, v77/lad_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
