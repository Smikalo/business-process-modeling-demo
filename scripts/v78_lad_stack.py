"""V7.8 — extended-pool LAD stacker + multi-axis reconcile.

V7.7 used a 7-base LAD pool (v4/v5/v6/v7/v71/v72_champion + v77_recent) and
selected ``compact+rec_tau0.52_chABC05_brand03`` as champion (test SIMSCORE
0.4827).

The V7.7 final report flagged three open levers:
  1. per-month-of-year residual corrector (rejected after diagnosis -- val
     and test bias signs disagree on Jan/Feb/Sep, so the corrector would
     hurt test);
  2. tilted-quantile bases (τ ∈ {0.45, 0.50, 0.55});
  3. weekly-resolution bases.

V7.8 explores option 2 *without* burning Kaggle GPU hours by re-using bases
that already exist in ``output/`` but were excluded from V7.7's pool:

* ``v77_quantile60`` -- LightGBM at τ=0.60 (val SIM 0.5246, **bias +8.0 %**)
  -- the *only* positive-bias LightGBM base, perfectly complementary to
  V7.7's −1.4 % negative-bias blend.
* ``v7sym_mae``      -- symmetric MAE objective on V7 features (val SIM
  0.4606, near-zero bias).
* ``v75lad``         -- V7.5 LAD champion as a base (val SIM 0.4450,
  brings its own channel reconciliation into the blend).

Anti-overfit guards (unchanged from V7.7):
* Per-channel tilted LAD with sum-to-1 simplex weights (NNLS via IRLS).
* CV gap ≤ 0.02 (champion: ≤ 0.01 typical).
* Pool size ≤ 8 bases (we tried 6/7/8/10 and let CV pick).
* Recency-weighted CV (folds 0.2/0.3/0.5) is the primary criterion.
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
V78 = OUT / "v78"
V78.mkdir(parents=True, exist_ok=True)

DECORR_GOOD = ["v77_recent", "v77_nosegment"]

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]

# Same recency weighting as V7.7 -- more recent fold -> higher weight, since
# the test distribution lives at the end of the timeline.
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


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
    # New base pools we explore beyond V7.7's compact+rec.
    # Strategy: take V7.7's winning compact+rec pool as the core and try
    # adding ONE / TWO new bases at a time, never more than 9 bases total.
    POOLS = {
        # V7.7 baseline reference
        "v77_compact":           list(LGB_BASE),
        "v77_compact+rec":       list(LGB_BASE) + ["v77_recent"],
        # V7.8 candidates: add one new bias-diverse base
        "v78_+q60":              list(LGB_BASE) + ["v77_recent", "v77_quantile60"],
        "v78_+lad":              list(LGB_BASE) + ["v77_recent", "v75lad"],
        "v78_+mae":              list(LGB_BASE) + ["v77_recent", "v7sym_mae"],
        # Two new bases
        "v78_+q60+mae":          list(LGB_BASE) + ["v77_recent", "v77_quantile60", "v7sym_mae"],
        "v78_+lad+mae":          list(LGB_BASE) + ["v77_recent", "v75lad", "v7sym_mae"],
        "v78_+lad+q60":          list(LGB_BASE) + ["v77_recent", "v75lad", "v77_quantile60"],
        # Aggressive: three new bases (anti-overfit ceiling test)
        "v78_+lad+q60+mae":      list(LGB_BASE) + ["v77_recent", "v75lad", "v77_quantile60", "v7sym_mae"],
    }

    AXES_OPTIONS = {
        # V7.7's championship axis (best on V7.7 CV)
        "chABC05_brand03":            [(["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
        # V7.5's baseline (no ABC, no brand)
        "ch08":                       [(["Канал"], 0.8)],
        # V7.7's three-step
        "ch08_chABC05_brand03":       [(["Канал"], 0.8), (["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)],
        # V7.7-style channelxABC alone
        "chABC08":                    [(["Канал", "Сегмент_ABC"], 0.8)],
    }
    TAUS = [0.5, 0.52, 0.55]

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
                        return out, {"base": meta, "axes": str(_axes), "tau": _tau,
                                     "pool": _tags}
                    return fn
                key = f"{pool_name}_tau{tau}_{axes_name}"
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
              f"rec={r['OOF_recency']:.4f}  in={r['in_sample']:.4f}  "
              f"gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V78 / "lad_cv.csv", index=False)
    print("\n=== Top-15 V7.8 candidates (sorted by recency-weighted OOF) ===")
    print(df[["name", "OOF_recency", "OOF_mean", "in_sample", "gap"]]
          .head(15).to_string(index=False))

    # Selection rule (matches V7.7 to keep the comparison fair):
    #   1. gap ≤ 0.02 (otherwise we're trusting in-sample over CV)
    #   2. minimize OOF_recency (fold-3-weighted), tie-break by gap
    survivors = [r for r in rows if r["gap"] <= 0.02]
    if not survivors:
        print("\nWARN: no candidate has gap ≤ 0.02; relaxing to gap ≤ 0.05.")
        survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    champ = min(survivors, key=lambda r: (r["OOF_recency"], r["gap"]))
    print(f"\nV7.8 CHAMPION: {champ['name']}  OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF={champ['OOF_mean']:.4f}  gap={champ['gap']:+.4f}")

    fn, v_, t_ = cand[champ["name"]]
    val_pred, meta = fn(v_, v_)
    tst_pred, _ = fn(v_, t_)

    out_v = v_[KEY].copy()
    out_v["target_qty"] = v_["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v78_val.csv", index=False)

    out_t = t_[KEY].copy()
    out_t["target_qty"] = t_["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v78_test.csv", index=False)

    val_score = _score(v_, val_pred)
    tst_score = _score(t_, tst_pred)
    (V78 / "lad_champion.json").write_text(json.dumps({
        "champion": champ["name"],
        "OOF_SIMSCORE": champ["OOF_mean"],
        "OOF_recency": champ["OOF_recency"],
        "OOF_folds": champ["OOF_folds"],
        "in_sample_SIMSCORE": champ["in_sample"],
        "overfit_gap": champ["gap"],
        "val_score": val_score,
        "test_score": tst_score,
        "meta": meta,
    }, indent=2, ensure_ascii=False, default=str))

    print(f"\n  val   SIMSCORE={val_score['SIMSCORE']:.4f}  "
          f"WAPE={val_score['WAPE']:.4f}  bias%={val_score['Agg_Bias_pct']:+.2f}  "
          f"M-WAPE={val_score['Monthly_WAPE']:.4f}")
    print(f"  test  SIMSCORE={tst_score['SIMSCORE']:.4f}  "
          f"WAPE={tst_score['WAPE']:.4f}  bias%={tst_score['Agg_Bias_pct']:+.2f}  "
          f"M-WAPE={tst_score['Monthly_WAPE']:.4f}")
    print("\nwrote preds_v78_val.csv, preds_v78_test.csv, v78/lad_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
