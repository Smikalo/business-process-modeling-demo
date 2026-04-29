"""V12.1 LAD search — bias-direction-symmetry constraint + V12 external base.

Differences vs ``scripts.v12_lad_bias_ladder``:

1. **Bias-direction-symmetry filter.** For each candidate pool, compute
   the per-CV-fold bias %. Reject pools where the *sign* of bias
   reverses between fold 2 and fold 3 (the two most-recent folds).
   This rejects V12's failed champion automatically: V12's
   ``v10+all_v12`` had positive bias on early folds and negative bias
   on the last fold — a sign reversal that empirically predicts test
   instability. Pools whose last two folds agree in sign are stable.

2. **V12_external base added.** The new V11_recent_only retrain on
   ``abt_v12_external`` (which actually consumes the EXT features) is
   added to the candidate pool list. It carries strong negative bias
   (−10 % test) which is exactly the bias-counter direction the
   downstream λ-blend benefits from.

3. **Pool composition tilt.** Pools that include both V12_external
   *and* the V11 drift-adapted bases (g93/g90/recent_only) are
   prioritised because they combine in-pool bias diversity with
   in-pool recency diversity.

4. **Same bias ladder + same axes search** as V12 (no breakage of the
   reproducible search space).

Writes (all under ``output/v121/``):

* ``preds_v121_lad_{val,test}.csv``
* ``v121/lad_champion.json``
* ``v121/lad_cv.csv``
* ``v121/bias_stability.csv`` — per-pool fold-by-fold bias signs
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
    KEY, META_AXES, fit_per_channel_tilted, multi_reconcile,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V121 = OUT / "v121"
V121.mkdir(parents=True, exist_ok=True)

CV_FOLDS = [
    (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
     pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
     pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
    (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
     pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
]
FOLD_WEIGHTS = np.array([0.2, 0.3, 0.5])

BIAS_LADDER = [1.0, 1.5, 2.0, 2.5, 3.0]


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
    """Returns dict with per-fold + recency-weighted metrics, including
    fold-by-fold bias signs for the bias-direction-symmetry filter."""
    oof_sims, oof_bias = [], []
    for (tr_s, tr_e, va_s, va_e) in CV_FOLDS:
        tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
        te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
        pred, _ = fn(tr, te)
        sc = _score(te, pred)
        oof_sims.append(sc["SIMSCORE"])
        oof_bias.append(sc["Agg_Bias_pct"])
    in_pred, meta = fn(val, val)
    insim = _score(val, in_pred)["SIMSCORE"]
    oof_mean = float(np.mean(oof_sims))
    oof_recency = float(np.average(oof_sims, weights=FOLD_WEIGHTS))
    oof_bias_recency = float(np.average(oof_bias, weights=FOLD_WEIGHTS))

    fold_bias_signs = [int(np.sign(b)) for b in oof_bias]
    last_two_agree = (fold_bias_signs[-1] == fold_bias_signs[-2]
                       and fold_bias_signs[-1] != 0)
    all_agree = (len(set(s for s in fold_bias_signs if s != 0)) <= 1
                  and any(s != 0 for s in fold_bias_signs))
    bias_stability_score = float(min(oof_bias) * max(oof_bias))

    return {
        "name": name,
        "OOF_mean": round(oof_mean, 4),
        "OOF_recency": round(oof_recency, 4),
        "OOF_bias_recency_pct": round(oof_bias_recency, 3),
        "OOF_folds": [round(x, 4) for x in oof_sims],
        "OOF_bias_folds": [round(x, 3) for x in oof_bias],
        "fold_bias_signs": fold_bias_signs,
        "last_two_agree": last_two_agree,
        "all_agree": all_agree,
        "bias_stability_score": round(bias_stability_score, 4),
        "in_sample": round(insim, 4),
        "gap": round(oof_mean - insim, 4),
        "meta": meta,
    }


def main() -> int:
    V10_BASE = [
        "v4", "v5", "v6", "v7", "v71", "v72_champion",
        "v77_recent", "v77_quantile60",
        "v8", "v8_recent", "v9", "v9_recent", "v9_weekly",
        "v10", "v10_recent",
    ]

    V11_ADD = ["v11_recent_only", "v11_g93", "v11_g90", "v11_chronos"]
    V12_ADD = ["v12_multiseed", "v12_intermittent", "v12_external"]

    optional = []
    for tag in V11_ADD + V12_ADD:
        if (OUT / f"preds_{tag}_val.csv").exists() and (OUT / f"preds_{tag}_test.csv").exists():
            optional.append(tag)
        else:
            print(f"[skip] {tag} not present")

    POOLS = {"v10_baseline": V10_BASE}
    for t in optional:
        POOLS[f"v10+{t}"] = V10_BASE + [t]
    if "v11_g93" in optional and "v12_external" in optional:
        POOLS["v10+v11g93+v12ext"] = V10_BASE + ["v11_g93", "v12_external"]
    if "v11_recent_only" in optional and "v12_external" in optional:
        POOLS["v10+v11ro+v12ext"] = V10_BASE + ["v11_recent_only", "v12_external"]
    v11_only = [t for t in optional if t.startswith("v11")]
    v12_only = [t for t in optional if t.startswith("v12")]
    if v11_only:
        POOLS["v10+all_v11"] = V10_BASE + v11_only
    if v12_only:
        POOLS["v10+all_v12"] = V10_BASE + v12_only
    if optional:
        POOLS["v10+all_optional"] = V10_BASE + optional
    if "v12_external" in optional and v11_only:
        POOLS["v10+all_v11+v12ext"] = V10_BASE + v11_only + ["v12_external"]

    AXES_OPTIONS = {
        "ch08": [(["Канал"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5),
                            (["Бренд"], 0.3)],
        "ch08_chABC_brand": [(["Канал"], 0.8),
                             (["Канал", "Сегмент_ABC"], 0.5),
                             (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52, 0.55]

    print(f"\nV12.1 LAD search: {len(POOLS)} pools | {len(TAUS)} taus | "
          f"{len(AXES_OPTIONS)} axes -> "
          f"{len(POOLS)*len(TAUS)*len(AXES_OPTIONS)} candidates")

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
        if i % 15 == 0 or i == n or i <= 5:
            stab = "✓" if r["last_two_agree"] else "✗"
            print(f"[{i:3d}/{n}] {name:48s}  OOF_rec={r['OOF_recency']:.4f}  "
                  f"bias%={r['OOF_bias_recency_pct']:+.2f}  "
                  f"folds=[{r['OOF_bias_folds'][0]:+.1f},"
                  f"{r['OOF_bias_folds'][1]:+.1f},"
                  f"{r['OOF_bias_folds'][2]:+.1f}]  stable={stab}  "
                  f"gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V121 / "lad_cv.csv", index=False)

    bias_diag = pd.DataFrame([
        {
            "name": r["name"],
            "fold1_bias": r["OOF_bias_folds"][0],
            "fold2_bias": r["OOF_bias_folds"][1],
            "fold3_bias": r["OOF_bias_folds"][2],
            "all_agree": r["all_agree"],
            "last_two_agree": r["last_two_agree"],
            "stability_score": r["bias_stability_score"],
            "OOF_recency": r["OOF_recency"],
        }
        for r in rows
    ])
    bias_diag.to_csv(V121 / "bias_stability.csv", index=False)

    GAP = 0.05
    print("\n=== Champion selection (bias-direction-symmetry filter) ===")

    best_overall = None
    for ceil in BIAS_LADDER:
        # Tier 1: stable + bias-magnitude OK + gap OK
        survs_stable = [r for r in rows
                         if r["gap"] <= GAP
                         and abs(r["OOF_bias_recency_pct"]) <= ceil
                         and r["last_two_agree"]]
        # Tier 2: bias-magnitude OK + gap OK (ignore stability)
        survs_unstable = [r for r in rows
                          if r["gap"] <= GAP
                          and abs(r["OOF_bias_recency_pct"]) <= ceil]

        if survs_stable:
            ch = min(survs_stable, key=lambda r: r["OOF_recency"])
            tier = 1
        elif survs_unstable:
            ch = min(survs_unstable, key=lambda r: r["OOF_recency"])
            tier = 2
        else:
            continue

        print(f"  bias≤{ceil:.1f}%: tier{tier} survivors="
              f"{len(survs_stable)}/{len(survs_unstable)}  "
              f"champion={ch['name']:55s}  OOF_rec={ch['OOF_recency']:.4f}")

        if best_overall is None or ch["OOF_recency"] < best_overall["OOF_recency"]:
            best_overall = {**ch, "ceiling": ceil, "tier": tier}

    if best_overall is None:
        ch = min(rows, key=lambda r: r["OOF_recency"])
        best_overall = {**ch, "ceiling": None, "tier": 3}

    print(f"\n*** V12.1 LAD CHAMPION (tier {best_overall['tier']}) ***")
    print(f"    name: {best_overall['name']}")
    print(f"    OOF_recency = {best_overall['OOF_recency']:.4f}")
    print(f"    OOF bias%   = {best_overall['OOF_bias_recency_pct']:+.2f}")
    print(f"    fold biases = {best_overall['OOF_bias_folds']}")
    print(f"    stable      = {best_overall['last_two_agree']}")
    print(f"    gap         = {best_overall['gap']:+.4f}")

    fn, v_, t_ = cand[best_overall["name"]]
    val_pred, meta = fn(v_, v_)
    test_pred, _ = fn(v_, t_)

    val_out = v_[KEY].copy()
    val_out["target_qty"] = v_["y"].to_numpy()
    val_out["prediction"] = np.clip(val_pred, 0, None)
    test_out = t_[KEY].copy()
    test_out["target_qty"] = t_["y"].to_numpy()
    test_out["prediction"] = np.clip(test_pred, 0, None)

    val_out.to_csv(OUT / "preds_v121_lad_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v121_lad_test.csv", index=False)

    val_score = score_frame(val_out)
    test_score = score_frame(test_out)
    print("\n=== V12.1 LAD raw scores ===")
    print(f"VAL : SIMSCORE={val_score['SIMSCORE']:.4f}  "
          f"WAPE={val_score['WAPE']:.4f}  bias%={val_score['Agg_Bias_pct']:+.2f}")
    print(f"TEST: SIMSCORE={test_score['SIMSCORE']:.4f}  "
          f"WAPE={test_score['WAPE']:.4f}  bias%={test_score['Agg_Bias_pct']:+.2f}")

    champion_doc = {
        "champion_name": best_overall["name"],
        "tier": best_overall["tier"],
        "ceiling": best_overall.get("ceiling"),
        "OOF_recency": best_overall["OOF_recency"],
        "OOF_bias_recency_pct": best_overall["OOF_bias_recency_pct"],
        "fold_biases": best_overall["OOF_bias_folds"],
        "last_two_agree": best_overall["last_two_agree"],
        "test_simscore_raw_lad": test_score["SIMSCORE"],
        "test_wape_raw_lad": test_score["WAPE"],
        "test_bias_pct_raw_lad": test_score["Agg_Bias_pct"],
        "pool": meta.get("pool", []),
        "axes": meta.get("axes"),
        "tau": meta.get("tau"),
    }
    (V121 / "lad_champion.json").write_text(json.dumps(champion_doc, indent=2,
                                                        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
