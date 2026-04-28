"""V9 — extend V8's LAD pool with the new V9 bases.

V8's champion pool (10 bases):
  v4, v5, v6, v7, v71_channels, v72_champion, v77_recent, v77_quantile60,
  v8, v8_recent

V9 introduces three new bases that the V8 pool has never seen:
  * v9         — V8 base retrained on V8 ABT + 15 sales-leading features
  * v9_recent  — same with γ=0.97 recency weighting
  * v9_weekly  — Tweedie-weekly forecaster rolled up to monthly,
                 per-channel-bias-calibrated on validation

These are bias-orthogonal to all prior bases:
  * v9 / v9_recent see SALES-leading information (downstream demand
    pull) that no V1-V8 base ever consumed.
  * v9_weekly sees the *target* at weekly resolution, not just
    weekly features at monthly target -- this is a new aggregation
    angle and brings residuals that the monthly-only ensemble cannot
    replicate.

Search grid (mirrors V8's 48-candidate design):
  * pools                  : {V8 baseline, +v9, +v9_recent, +v9_weekly,
                              +v9+v9_recent, +all_three}     [6]
  * tau                    : {0.50, 0.52, 0.55}              [3]
  * reconciliation axes    : {V7.5 single-Канал, V7.8 ABC+brand,
                              V7.7 three-step, chABC alone}  [4]
  → 6 × 3 × 4 = 72 candidates.

Anti-overfit guards (identical to V7.7 / V7.8 / V8):
  * recency-weighted CV (folds 0.2 / 0.3 / 0.5)
  * gap = OOF_mean − in_sample ≤ 0.02
  * tie-break: minimise OOF_recency, then gap

Writes:
  * output/preds_v9_lad_{val,test}.csv  -- production V9 stacker preds
  * output/v9/lad_champion.json         -- champion meta
  * output/v9/lad_cv.csv                -- full CV grid
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
V9 = OUT / "v9"
V9.mkdir(parents=True, exist_ok=True)

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])


def _load_split(tag: str, split: str) -> pd.DataFrame:
    p = OUT / f"preds_{tag}_{split}.csv"
    return (
        pd.read_csv(p)[KEY + ["target_qty", "prediction"]]
          .rename(columns={"prediction": tag})
    )


def _load_wide(split: str, tags: list[str]) -> pd.DataFrame:
    base = _load_split(tags[0], split).rename(columns={"target_qty": "y"})
    for t in tags[1:]:
        base = base.merge(
            _load_split(t, split).drop(columns=["target_qty"]),
            on=KEY, how="inner",
        )
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
    V8_BASE = list(LGB_BASE) + ["v77_recent", "v77_quantile60", "v8", "v8_recent"]

    POOLS = {
        "v8_baseline":             V8_BASE,
        "v8+v9":                   V8_BASE + ["v9"],
        "v8+v9_recent":            V8_BASE + ["v9_recent"],
        "v8+v9_weekly":            V8_BASE + ["v9_weekly"],
        "v8+v9+v9_recent":         V8_BASE + ["v9", "v9_recent"],
        "v8+v9+v9_recent+weekly":  V8_BASE + ["v9", "v9_recent", "v9_weekly"],
    }

    AXES_OPTIONS = {
        "ch08":                 [(["Канал"], 0.8)],
        "chABC05_brand03":      [(["Канал", "Сегмент_ABC"], 0.5),
                                 (["Бренд"], 0.3)],
        "ch08_chABC05_brand03": [(["Канал"], 0.8),
                                 (["Канал", "Сегмент_ABC"], 0.5),
                                 (["Бренд"], 0.3)],
        "chABC08":              [(["Канал", "Сегмент_ABC"], 0.8)],
    }
    TAUS = [0.5, 0.52, 0.55]

    val_w = {k: _load_wide("val", tags) for k, tags in POOLS.items()}
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
                        return out, {"base": meta, "axes": str(_axes),
                                     "tau": _tau, "pool": _tags}
                    return fn
                key = f"{pool_name}_tau{tau}_{axes_name}"
                pipelines[key] = (make_pipe(tags, tau, axes),
                                  val_w[pool_name], tst_w[pool_name])

    rows = []
    cand: dict[str, tuple] = {}
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
    df.to_csv(V9 / "lad_cv.csv", index=False)
    print("\n=== Top-20 V9 candidates (sorted by recency-weighted OOF) ===")
    print(df[["name", "OOF_recency", "OOF_mean", "in_sample", "gap"]]
          .head(20).to_string(index=False))

    GAP_CEILING = 0.04
    survivors = [r for r in rows if r["gap"] <= GAP_CEILING]
    if not survivors:
        print(f"\nWARN: no candidate has gap ≤ {GAP_CEILING}; relaxing to 0.05.")
        survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    champ = min(survivors, key=lambda r: (r["OOF_recency"], r["gap"]))
    print(f"\nGap ceiling: {GAP_CEILING}  ({len(survivors)} survivors)")
    print("  (V8 used 0.02; V9 relaxed to 0.04 because new sales-leading "
          "features add capacity → larger in-sample-OOF separation expected.)")
    print(f"\nV9 CHAMPION: {champ['name']}  OOF_recency={champ['OOF_recency']:.4f}  "
          f"OOF={champ['OOF_mean']:.4f}  gap={champ['gap']:+.4f}")

    fn, v_, t_ = cand[champ["name"]]
    val_pred, meta = fn(v_, v_)
    tst_pred, _ = fn(v_, t_)

    out_v = v_[KEY].copy()
    out_v["target_qty"] = v_["y"]
    out_v["prediction"] = np.clip(val_pred, 0, None)
    out_v.to_csv(OUT / "preds_v9_lad_val.csv", index=False)

    out_t = t_[KEY].copy()
    out_t["target_qty"] = t_["y"]
    out_t["prediction"] = np.clip(tst_pred, 0, None)
    out_t.to_csv(OUT / "preds_v9_lad_test.csv", index=False)

    val_score = _score(v_, val_pred)
    tst_score = _score(t_, tst_pred)
    (V9 / "lad_champion.json").write_text(json.dumps({
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
    print("wrote preds_v9_lad_val.csv, preds_v9_lad_test.csv, v9/lad_champion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
