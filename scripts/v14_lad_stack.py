"""V14 LAD search — merges GlobalNN + (optionally) MoE specialists.

Variants:
  --variant alpha : V13 pool + GlobalNN preds (post-Step 5d)
  --variant final : V13 pool + GlobalNN + 4 MoE per-cluster specialists

The MoE specialists are produced by ``scripts.v14_moe_specialists`` and
write four prediction CSVs:
  preds_v14_moe_smooth_{val,test}.csv         (smooth-demand pairs)
  preds_v14_moe_intermittent_{val,test}.csv   (intermittent pairs)
  preds_v14_moe_lumpy_{val,test}.csv          (lumpy pairs)
  preds_v14_moe_erratic_{val,test}.csv        (erratic pairs)
Each has full coverage of all keys (specialist's own pairs get its
prediction, others get NaN → fallback to V13_final at LAD time).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame
from scripts.v12_lad_bias_ladder import (
    BIAS_LADDER, KEY, META_AXES, _load_wide, _score, eval_pipeline,
)
from scripts.v77_multi_reconcile import fit_per_channel_tilted, multi_reconcile
from src.streaming_calibrator import streaming_calibrate

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V14 = OUT / "v14"
V14.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("alpha", "final"), default="final")
    args = ap.parse_args()

    V10_BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion",
                "v77_recent", "v77_quantile60",
                "v8", "v8_recent", "v9", "v9_recent", "v9_weekly",
                "v10", "v10_recent"]
    V11_ADD = ["v11_recent_only", "v11_g93", "v11_g90"]
    V12_ADD = ["v12_multiseed", "v12_intermittent", "v12_anomaly"]
    V13_FM = ["v13_chronos", "v13_timesfm", "v13_moirai"]
    V14_NN = ["v14_globalnn"]
    V14_MOE = ["v14_moe_smooth", "v14_moe_intermittent",
               "v14_moe_lumpy", "v14_moe_erratic"]

    base_pool = V10_BASE + V11_ADD + V12_ADD + V13_FM
    if args.variant == "alpha":
        candidates = V14_NN
    else:
        candidates = V14_NN + V14_MOE

    def _exists(tag):
        return ((OUT / f"preds_{tag}_val.csv").exists() and
                (OUT / f"preds_{tag}_test.csv").exists())

    base_present = [t for t in base_pool if _exists(t)]
    new_present  = [t for t in candidates if _exists(t)]
    print(f"V14 LAD ({args.variant}): base present={len(base_present)}/"
          f"{len(base_pool)}; new present={new_present}")

    if not new_present:
        print(f"\nNo V14 {args.variant} bases present. Falling through:")
        for split in ("val", "test"):
            src = OUT / f"preds_v13_final_{split}.csv"
            dst = OUT / f"preds_v14_{args.variant}_final_{split}.csv"
            if src.exists():
                shutil.copy(src, dst)
                print(f"  {dst.name} = copy of {src.name}")
        return 0

    POOLS = {f"v13_baseline": base_present}
    for n in new_present:
        POOLS[f"v13+{n}"] = base_present + [n]
    POOLS[f"v13+all_v14_{args.variant}"] = base_present + new_present

    AXES_OPTIONS = {
        "ch08": [(["Канал"], 0.8)],
        "chABC05_brand03": [(["Канал", "Сегмент_ABC"], 0.5),
                            (["Бренд"], 0.3)],
    }
    TAUS = [0.5, 0.52]

    val_w = {k: _load_wide("val", tags) for k, tags in POOLS.items()}
    tst_w = {k: _load_wide("test", tags) for k, tags in POOLS.items()}

    pipelines = {}
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
    cand = {}
    for i, (name, (fn, v_, t_)) in enumerate(pipelines.items(), 1):
        r = eval_pipeline(name, fn, v_)
        rows.append(r); cand[name] = (fn, v_, t_)
        print(f"[{i:2d}/{len(pipelines)}] {name:55s}  "
              f"OOF_rec={r['OOF_recency']:.4f}  bias%={r['OOF_bias_recency_pct']:+.2f}")

    # Bias ladder champion selection
    GAP = 0.05
    best = None
    best_ceil = None
    for ceil in BIAS_LADDER:
        survs = [r for r in rows if r["gap"] <= GAP and abs(r["OOF_bias_recency_pct"]) <= ceil]
        if survs:
            ch = min(survs, key=lambda r: (r["OOF_recency"], abs(r["OOF_bias_recency_pct"])))
            if best is None or ch["OOF_recency"] < best["OOF_recency"]:
                best, best_ceil = ch, ceil
    if best is None:
        best = min(rows, key=lambda r: r["OOF_recency"])
        best_ceil = None

    fn, v_, t_ = cand[best["name"]]
    val_pred, meta = fn(v_, v_)
    test_pred, _ = fn(v_, t_)

    val_out = v_[KEY].copy()
    val_out["target_qty"] = v_["y"].to_numpy()
    val_out["prediction"] = np.clip(val_pred, 0, None)
    test_out = t_[KEY].copy()
    test_out["target_qty"] = t_["y"].to_numpy()
    test_out["prediction"] = np.clip(test_pred, 0, None)

    val_out.to_csv(OUT / f"preds_v14_{args.variant}_lad_val.csv", index=False)
    test_out.to_csv(OUT / f"preds_v14_{args.variant}_lad_test.csv", index=False)

    abt_axes = pd.read_parquet(OUT / "abt_v7_cached.parquet")[KEY + META_AXES]
    abt_axes["Период"] = abt_axes["Период"].astype(str)
    val_with_axes = val_out.merge(abt_axes, on=KEY, how="left")
    test_with_axes = test_out.merge(abt_axes, on=KEY, how="left")
    val_cal, test_cal, _ = streaming_calibrate(
        val_with_axes, test_with_axes, axes=["Канал"], beta=0.5)
    val_calibrated = val_cal[KEY + ["target_qty"]].copy()
    val_calibrated["prediction"] = val_cal["prediction_calibrated"]
    test_calibrated = test_cal[KEY + ["target_qty"]].copy()
    test_calibrated["prediction"] = test_cal["prediction_calibrated"]
    val_calibrated.to_csv(OUT / f"preds_v14_{args.variant}_final_val.csv", index=False)
    test_calibrated.to_csv(OUT / f"preds_v14_{args.variant}_final_test.csv", index=False)

    val_score = score_frame(val_calibrated)
    test_score = score_frame(test_calibrated)
    print(f"\nV14_{args.variant}_final test:  SIMSCORE={test_score['SIMSCORE']:.4f}  "
          f"WAPE={test_score['WAPE']:.4f}  bias%={test_score['Agg_Bias_pct']:+.2f}")

    (V14 / f"lad_champion_{args.variant}.json").write_text(json.dumps({
        "champion_name": best["name"],
        "production_bias_ceiling_pct": best_ceil,
        "test_simscore": test_score["SIMSCORE"],
        "test_wape": test_score["WAPE"],
        "test_bias_pct": test_score["Agg_Bias_pct"],
        "pool": meta.get("pool", []),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
