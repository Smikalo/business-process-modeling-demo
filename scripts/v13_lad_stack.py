"""V13 LAD search — merges fine-tuned foundation-model preds into V12 pool.

Reads (when present):
  output/preds_v13_chronos_{val,test}.csv
  output/preds_v13_timesfm_{val,test}.csv
  output/preds_v13_moirai_{val,test}.csv

Each FM that's present is added as a new candidate base in the LAD pool.
Same bias-ladder + axes search as V12. Picks the production champion
across the V12 pool ∪ FM additions.

Writes:
  output/preds_v13_lad_{val,test}.csv  (LAD raw)
  output/preds_v13_final_{val,test}.csv  (LAD + streaming bias)
  output/v13/lad_champion.json
  output/v13/lad_cv.csv
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame
from scripts.v12_lad_bias_ladder import (
    BIAS_LADDER, CV_FOLDS, FOLD_WEIGHTS,
    KEY, META_AXES, _load_split, _load_wide, _score, eval_pipeline,
)
from scripts.v77_multi_reconcile import fit_per_channel_tilted, multi_reconcile
from src.streaming_calibrator import streaming_calibrate

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V13 = OUT / "v13"
V13.mkdir(parents=True, exist_ok=True)


def main() -> int:
    V10_BASE = [
        "v4", "v5", "v6", "v7", "v71", "v72_champion",
        "v77_recent", "v77_quantile60",
        "v8", "v8_recent", "v9", "v9_recent", "v9_weekly",
        "v10", "v10_recent",
    ]
    V11_ADDITIONS = ["v11_recent_only", "v11_g93", "v11_g90"]
    V12_ADDITIONS = ["v12_multiseed", "v12_intermittent", "v12_anomaly"]
    V13_FM = ["v13_chronos", "v13_timesfm", "v13_moirai"]

    optional = []
    fms_present = []
    for tag in V11_ADDITIONS + V12_ADDITIONS + V13_FM:
        if (OUT / f"preds_{tag}_val.csv").exists() and (OUT / f"preds_{tag}_test.csv").exists():
            optional.append(tag)
            if tag in V13_FM:
                fms_present.append(tag)
        else:
            print(f"[skip] {tag} predictions not present")

    if not fms_present:
        print("\nNo V13 fine-tuned FM preds present. Either run the GPU")
        print("notebooks (notebooks/v13_chronos_finetune_colab.py) or drop")
        print("their predictions into output/. Falling through: V13 == V12.")
        # Make V12 final the V13 final (idempotent so audit_v13 doesn't crash)
        for split in ("val", "test"):
            src = OUT / f"preds_v12_final_{split}.csv"
            dst = OUT / f"preds_v13_final_{split}.csv"
            if src.exists():
                shutil.copy(src, dst)
                print(f"  {dst.name} = copy of {src.name}")
        return 0

    print(f"\nV13 FM bases present: {fms_present}")

    POOLS = {
        "v12_baseline": V10_BASE + [t for t in optional if not t.startswith("v13_")],
    }
    for fm in fms_present:
        POOLS[f"v12+{fm}"] = POOLS["v12_baseline"] + [fm]
    POOLS["v12+all_v13_fm"] = POOLS["v12_baseline"] + fms_present

    AXES_OPTIONS = {
        "ch08": [(["Канал"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5),
                            (["Бренд"], 0.3)],
        "ch08_chABC_brand": [(["Канал"], 0.8),
                             (["Канал", "Сегмент_ABC"], 0.5),
                             (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52, 0.55]

    print(f"\nV13 LAD search: {len(POOLS)} pools | {len(TAUS)} taus | "
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
        if i % 5 == 0 or i == n:
            print(f"[{i:3d}/{n}] {name:55s}  OOF={r['OOF_mean']:.4f}  "
                  f"rec={r['OOF_recency']:.4f}  bias%={r['OOF_bias_recency_pct']:+.2f}  "
                  f"gap={r['gap']:+.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "meta"}
                       for r in rows]).sort_values("OOF_recency")
    df.to_csv(V13 / "lad_cv.csv", index=False)

    GAP_CEILING = 0.05
    best_per_ceiling = {}
    for ceil in BIAS_LADDER:
        survivors = [r for r in rows
                     if r["gap"] <= GAP_CEILING
                     and abs(r["OOF_bias_recency_pct"]) <= ceil]
        if survivors:
            best_per_ceiling[ceil] = min(survivors,
                key=lambda r: (r["OOF_recency"], abs(r["OOF_bias_recency_pct"])))

    if not best_per_ceiling:
        ch = min(rows, key=lambda r: r["OOF_recency"])
        production_ceiling = None
    else:
        valid = [(c, ceil) for ceil, c in best_per_ceiling.items()]
        valid.sort(key=lambda x: (x[0]["OOF_recency"], x[1]))
        ch, production_ceiling = valid[0]

    champ_name = ch["name"]
    print(f"\n*** V13 LAD CHAMPION (bias ceiling={production_ceiling}): {champ_name} ***")

    fn, v_, t_ = cand[champ_name]
    val_pred, meta = fn(v_, v_)
    test_pred, _ = fn(v_, t_)

    val_out = v_[KEY].copy()
    val_out["target_qty"] = v_["y"].to_numpy()
    val_out["prediction"] = np.clip(val_pred, 0, None)
    test_out = t_[KEY].copy()
    test_out["target_qty"] = t_["y"].to_numpy()
    test_out["prediction"] = np.clip(test_pred, 0, None)

    val_out.to_csv(OUT / "preds_v13_lad_val.csv", index=False)
    test_out.to_csv(OUT / "preds_v13_lad_test.csv", index=False)

    abt_axes = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt_axes["Период"] = abt_axes["Период"].astype(str)
    val_with_axes = val_out.merge(abt_axes, on=KEY, how="left")
    test_with_axes = test_out.merge(abt_axes, on=KEY, how="left")

    val_cal, test_cal, _ = streaming_calibrate(
        val_with_axes, test_with_axes,
        axes=["Канал"], beta=0.5, fold_in_test=False,
    )
    val_calibrated = val_cal[KEY + ["target_qty"]].copy()
    val_calibrated["prediction"] = val_cal["prediction_calibrated"]
    test_calibrated = test_cal[KEY + ["target_qty"]].copy()
    test_calibrated["prediction"] = test_cal["prediction_calibrated"]
    val_calibrated.to_csv(OUT / "preds_v13_final_val.csv", index=False)
    test_calibrated.to_csv(OUT / "preds_v13_final_test.csv", index=False)

    val_score = score_frame(val_calibrated)
    test_score = score_frame(test_calibrated)
    print(f"\nV13_final val:  SIMSCORE={val_score['SIMSCORE']:.4f}  "
          f"WAPE={val_score['WAPE']:.4f}  bias%={val_score['Agg_Bias_pct']:+.2f}")
    print(f"V13_final test: SIMSCORE={test_score['SIMSCORE']:.4f}  "
          f"WAPE={test_score['WAPE']:.4f}  bias%={test_score['Agg_Bias_pct']:+.2f}")

    champion_doc = {
        "champion_name": champ_name,
        "production_bias_ceiling_pct": production_ceiling,
        "fms_used": fms_present,
        "OOF_recency": ch["OOF_recency"],
        "OOF_mean": ch["OOF_mean"],
        "OOF_bias_recency_pct": ch["OOF_bias_recency_pct"],
        "gap": ch["gap"],
        "test_simscore_v13_final": test_score["SIMSCORE"],
        "test_wape_v13_final": test_score["WAPE"],
        "test_bias_pct_v13_final": test_score["Agg_Bias_pct"],
        "pool": meta.get("pool", []),
        "axes": meta.get("axes"),
        "tau": meta.get("tau"),
    }
    (V13 / "lad_champion.json").write_text(json.dumps(champion_doc, indent=2,
                                                       ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
