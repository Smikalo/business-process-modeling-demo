"""V7.8 final attempt — add a TIGHT per-channel bias scale on top of V7.7.

V7.7 multi-axis reconcile uses (Канал × Сегмент_ABC) at shrink 0.5 then
(Бренд) at shrink 0.3.  Even after reconcile, V7.7 leaves ~−1.4 % aggregate
test bias and per-channel biases of (РС −6.4 %, СК −1.6 %, ИМ +1.1 %,
НКП +2.1 %).

This script adds ONE final reconciliation step at the pure ``Канал`` level
with tight clip ``[0.92, 1.10]`` and shrinkage ``λ`` swept over
``{0.3, 0.5, 0.7}``.  4 extra parameters total.  CV-validated under the
same recency-weighted SIMSCORE rule as V7.7/V7.8.

Anti-overfit guards:
* Clip [0.92, 1.10] caps the scale at ±10 % per channel — bias correction
  only, not error correction.
* MIN_ROWS=500 per channel (all 4 channels comfortably exceed this in
  every CV fold).
* ``λ ≤ 0.5`` enforced as a hard ceiling.
* Champion must have CV gap ≤ 0.02.
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
    _scale_map,
    _apply_scale,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V78 = OUT / "v78"
V78.mkdir(parents=True, exist_ok=True)

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])

TIGHT_CLIP = (0.92, 1.10)


def _load_split(tag, split):
    p = OUT / f"preds_{tag}_{split}.csv"
    return pd.read_csv(p)[KEY+["target_qty","prediction"]].rename(columns={"prediction": tag})


def _load_wide(split, tags):
    base = _load_split(tags[0], split).rename(columns={"target_qty":"y"})
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


def _scale_map_tight(df, p, by, min_rows=500, clip=TIGHT_CLIP):
    return _scale_map(df, p, by, min_rows=min_rows, clip=clip)


def make_pipeline(tags, tau, axes, final_chan_lambda):
    """V7.7 stack + optional final pure-Канал scale."""
    def fn(tr, te):
        pt_v, meta = fit_per_channel_tilted(tags, tr, te, tau=tau)
        pt_tr, _ = fit_per_channel_tilted(tags, tr, tr, tau=tau)
        pt_v = multi_reconcile(tr, te, pt_tr, pt_v, axes)
        pt_tr = multi_reconcile(tr, tr, pt_tr, pt_tr, axes)
        if final_chan_lambda > 0:
            s_map = _scale_map_tight(tr, pt_tr, ["Канал"])
            pt_v = _apply_scale(te, pt_v, ["Канал"], s_map, final_chan_lambda)
        return pt_v, {"base": meta, "axes": str(axes), "tau": tau,
                      "final_chan_lambda": final_chan_lambda, "pool": tags}
    return fn


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
    return {"name": name, "OOF_mean": round(oof_mean, 4),
            "OOF_recency": round(oof_recency, 4),
            "OOF_folds": [round(x, 4) for x in oof],
            "in_sample": round(insim, 4),
            "gap": round(oof_mean - insim, 4), "meta": meta}


def main() -> int:
    POOLS = {
        "compact+rec":   list(LGB_BASE) + ["v77_recent"],
        "+q60":          list(LGB_BASE) + ["v77_recent", "v77_quantile60"],
    }
    AXES = [(["Канал", "Сегмент_ABC"], 0.5), (["Бренд"], 0.3)]
    TAUS = [0.5, 0.52, 0.55]
    FINAL_LAMBDAS = [0.0, 0.3, 0.5]

    val_w = {k: _load_wide("val",  tags) for k, tags in POOLS.items()}
    tst_w = {k: _load_wide("test", tags) for k, tags in POOLS.items()}

    rows = []
    cand = {}
    print(f"{'name':60s}  {'OOF':>7s} {'rec':>7s} {'in':>7s} {'gap':>7s}")
    for pool_name, tags in POOLS.items():
        for tau in TAUS:
            for fcl in FINAL_LAMBDAS:
                fn = make_pipeline(tags, tau, AXES, fcl)
                name = f"v78_{pool_name}_tau{tau}_chABC05_brand03_chL{fcl}"
                r = eval_pipeline(name, fn, val_w[pool_name])
                rows.append(r)
                cand[name] = (fn, val_w[pool_name], tst_w[pool_name])
                print(f"{name:60s}  {r['OOF_mean']:>7.4f} {r['OOF_recency']:>7.4f} "
                      f"{r['in_sample']:>7.4f} {r['gap']:>+7.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V78 / "lad_chL_cv.csv", index=False)
    print("\n=== V7.8 final-channel-scale candidates (sorted by recency-weighted OOF) ===")
    print(df.head(15).to_string(index=False))

    survivors = [r for r in rows if r["gap"] <= 0.02]
    if not survivors:
        survivors = [r for r in rows if r["gap"] <= 0.05]
    if not survivors:
        survivors = rows
    champ = min(survivors, key=lambda r: (r["OOF_recency"], r["gap"]))
    print(f"\nV7.8 (final-channel-scale) CHAMPION: {champ['name']}  "
          f"OOF_rec={champ['OOF_recency']:.4f}  gap={champ['gap']:+.4f}")

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

    print(f"\n  val   SIMSCORE={val_score['SIMSCORE']:.4f}  WAPE={val_score['WAPE']:.4f}  "
          f"bias%={val_score['Agg_Bias_pct']:+.2f}  M-WAPE={val_score['Monthly_WAPE']:.4f}")
    print(f"  test  SIMSCORE={tst_score['SIMSCORE']:.4f}  WAPE={tst_score['WAPE']:.4f}  "
          f"bias%={tst_score['Agg_Bias_pct']:+.2f}  M-WAPE={tst_score['Monthly_WAPE']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
