"""V7.3 decision gate — rolling-origin CV + pre-registered pass/fail rule.

**Rules are frozen before any candidate is trained.  Do not mutate them.**

Pass criterion (a candidate "KEEPs"):
  - SIMSCORE improvement vs baseline on **≥ 3 of 5** CV folds
  - Mean CV SIMSCORE improvement ≥ 0.005
  - On **every fold** the candidate's SMAPE does not regress by > 0.01 vs the
    baseline, AND its Monthly_WAPE does not regress by > 0.01 vs the baseline

This script is intentionally thin: it delegates model training to
``src.model_v2.TwoStageForecaster`` configurable via CLI flags.  Each fold
produces a `preds_fold_k.csv` evaluated through ``score_similarity``.

Usage (example):

    python -m scripts.decision_gate_v73 \
        --abt-path output/abt_v7_cached.parquet \
        --tag v73_baseline_v7 \
        --reg-objective quantile --alpha 0.50 --recency-gamma 0.95 \
        --num-boost-round 500

    python -m scripts.decision_gate_v73 \
        --abt-path output/abt_v6_cached.parquet \
        --tag v6_baseline \
        --reg-objective tweedie --tweedie-variance-power 1.5 \
        --num-boost-round 500

The baseline CV numbers for V6 are computed once and cached in
``output/cv_baseline_v6.json``; subsequent runs compare to that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Avoid OpenMP thread-pool thrashing on macOS before LightGBM loads.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LIGHTGBM_NUM_THREADS", "4")

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation import rolling_cv_splits  # noqa: E402
from src.model_v2 import (  # noqa: E402
    TwoStageForecaster, encode_categoricals,
    filter_active_pairs, get_feature_columns_v2,
)
from src.v71_components import build_recency_weights  # noqa: E402
from scripts.score_similarity import score_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gate_v73")

OUT = _REPO / "output"

PASS_MIN_FOLDS = 3
PASS_MEAN_DELTA = 0.005
PASS_SMAPE_REGRESS_CAP = 0.01
PASS_MW_REGRESS_CAP = 0.01


def _train_one_fold(
    df_train_all: pd.DataFrame,
    df_val: pd.DataFrame,
    feats: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Fit one TwoStageForecaster and return val predictions.

    The last 3 months of ``df_train_all`` are carved off as an internal
    early-stopping validation set.  The outer ``df_val`` is never seen
    during training.
    """
    t_max = df_train_all["Период"].max()
    inner_cutoff = t_max - 3
    df_inner_tr = df_train_all[df_train_all["Период"] <= inner_cutoff].copy()
    df_inner_va = df_train_all[df_train_all["Период"] > inner_cutoff].copy()
    if len(df_inner_va) == 0:
        df_inner_tr = df_train_all
        df_inner_va = df_train_all.tail(1000)

    reg_params: dict = {
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "num_threads": args.num_threads,
        "force_col_wise": True,
        "verbose": -1,
    }
    if args.optuna_params:
        best = json.loads(Path(args.optuna_params).read_text())
        reg_params.update(best.get("best_params", best))

    if args.reg_objective == "tweedie":
        reg_params["objective"] = "tweedie"
        reg_params["tweedie_variance_power"] = args.tweedie_variance_power
        reg_obj = None
        reg_obj_kwargs: dict = {}
    elif args.reg_objective == "regression_l1":
        reg_params["objective"] = "regression_l1"
        reg_obj = None
        reg_obj_kwargs = {}
    elif args.reg_objective == "regression":
        reg_params["objective"] = "regression"
        reg_obj = None
        reg_obj_kwargs = {}
    elif args.reg_objective == "huber":
        reg_params["objective"] = "huber"
        reg_obj = None
        reg_obj_kwargs = {}
    elif args.reg_objective == "mape":
        reg_params["objective"] = "mape"
        reg_obj = None
        reg_obj_kwargs = {}
    else:
        reg_obj = "quantile"
        reg_obj_kwargs = {"alpha": args.alpha}

    tgt = "target_qty_imputed" if "target_qty_imputed" in df_inner_tr.columns else "target_qty"
    model = TwoStageForecaster(
        reg_params=reg_params,
        reg_objective=reg_obj,
        reg_objective_kwargs=reg_obj_kwargs,
        target_col=tgt,
    )

    sw_tr = sw_va = None
    if args.recency_gamma:
        anchor = t_max
        sw_tr = build_recency_weights(df_inner_tr, anchor=anchor,
                                      gamma=args.recency_gamma)
        sw_va = build_recency_weights(df_inner_va, anchor=anchor,
                                      gamma=args.recency_gamma)

    model.fit(
        df_inner_tr, df_inner_va, feats,
        num_boost_round=args.num_boost_round,
        sample_weight_train=sw_tr,
        sample_weight_val=sw_va,
    )
    preds = model.predict(df_val)

    out = df_val[["Период", "Партнер", "Артикул", "target_qty"]].copy()
    out["prediction"] = np.clip(preds, 0, None)
    out["Период"] = out["Период"].astype(str)
    return out


def run_cv(args: argparse.Namespace) -> dict:
    t0 = time.time()
    abt = pd.read_parquet(OUT / args.abt_path).pipe(encode_categoricals)
    feats = get_feature_columns_v2(abt)
    log.info("ABT: %d rows, %d features", len(abt), len(feats))

    folds = rolling_cv_splits(abt, n_folds=5, horizon_months=3,
                              final_train_end="2024-12")
    per_fold = []
    for i, (df_tr, df_va) in enumerate(folds, 1):
        df_tr_active = filter_active_pairs(df_tr)
        keys = df_tr_active[["Партнер", "Артикул"]].drop_duplicates()
        df_va_sub = df_va.merge(keys, on=["Партнер", "Артикул"], how="inner")
        log.info("Fold %d: train=%d (active=%d) → val=%d (%s)",
                 i, len(df_tr), len(df_tr_active), len(df_va_sub),
                 df_va_sub["Период"].min() if len(df_va_sub) else "∅")

        preds = _train_one_fold(df_tr_active, df_va_sub, feats, args)
        pf_scores = score_frame(preds)
        pf_scores["fold"] = i
        pf_scores["val_from"] = str(df_va_sub["Период"].min())
        pf_scores["val_to"] = str(df_va_sub["Период"].max())
        per_fold.append(pf_scores)
        preds.to_csv(OUT / f"cv_{args.tag}_fold{i}_preds.csv", index=False)
        log.info("  fold %d scores: SIMSCORE=%.4f  WAPE=%.4f  SMAPE=%.4f  "
                 "MW=%.4f  AggBias%%=%.2f",
                 i, pf_scores["SIMSCORE"], pf_scores["WAPE"],
                 pf_scores["SMAPE_nz"], pf_scores["Monthly_WAPE"],
                 pf_scores["Agg_Bias_pct"])

    df_folds = pd.DataFrame(per_fold)
    mean = df_folds[["WAPE", "SMAPE_nz", "Monthly_WAPE",
                     "Agg_Bias_pct", "SIMSCORE"]].mean()
    summary = {
        "tag": args.tag,
        "abt_path": args.abt_path,
        "cmd": vars(args),
        "per_fold": per_fold,
        "mean": {k: round(float(v), 4) for k, v in mean.items()},
        "runtime_s": round(time.time() - t0, 1),
    }

    out_json = OUT / f"cv_summary_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("wrote %s", out_json)

    log.info("\n=== %s (mean across %d folds) ===", args.tag, len(per_fold))
    log.info("  SIMSCORE   = %.4f", summary["mean"]["SIMSCORE"])
    log.info("  WAPE       = %.4f", summary["mean"]["WAPE"])
    log.info("  SMAPE_nz   = %.4f", summary["mean"]["SMAPE_nz"])
    log.info("  Monthly_WAPE = %.4f", summary["mean"]["Monthly_WAPE"])
    log.info("  Agg_Bias_pct = %.2f%%", summary["mean"]["Agg_Bias_pct"])
    log.info("  runtime: %.1fs", summary["runtime_s"])

    return summary


def compare_to_baseline(candidate: dict, baseline: dict) -> dict:
    """Apply the pre-registered decision rule."""
    per_fold_delta = []
    folds_improving = 0
    smape_regress_fails = []
    mw_regress_fails = []

    for c, b in zip(candidate["per_fold"], baseline["per_fold"]):
        d = {
            "fold": c["fold"],
            "val_from": c["val_from"],
            "val_to": c["val_to"],
            "d_SIMSCORE": round(c["SIMSCORE"] - b["SIMSCORE"], 4),
            "d_WAPE": round(c["WAPE"] - b["WAPE"], 4),
            "d_SMAPE": round(c["SMAPE_nz"] - b["SMAPE_nz"], 4),
            "d_MW": round(c["Monthly_WAPE"] - b["Monthly_WAPE"], 4),
            "d_AggBias": round(c["Agg_Bias_pct"] - b["Agg_Bias_pct"], 3),
        }
        per_fold_delta.append(d)
        if d["d_SIMSCORE"] < 0:
            folds_improving += 1
        if d["d_SMAPE"] > PASS_SMAPE_REGRESS_CAP:
            smape_regress_fails.append(d["fold"])
        if d["d_MW"] > PASS_MW_REGRESS_CAP:
            mw_regress_fails.append(d["fold"])

    mean_d_sim = candidate["mean"]["SIMSCORE"] - baseline["mean"]["SIMSCORE"]

    pass_folds = folds_improving >= PASS_MIN_FOLDS
    pass_mean = mean_d_sim <= -PASS_MEAN_DELTA
    pass_smape = len(smape_regress_fails) == 0
    pass_mw = len(mw_regress_fails) == 0
    verdict = "KEEP" if (pass_folds and pass_mean and pass_smape and pass_mw) else "KILL"

    return {
        "candidate_tag": candidate["tag"],
        "baseline_tag": baseline["tag"],
        "per_fold_delta": per_fold_delta,
        "mean_d_SIMSCORE": round(mean_d_sim, 4),
        "folds_improving_SIMSCORE": int(folds_improving),
        "smape_regress_fails_folds": smape_regress_fails,
        "mw_regress_fails_folds": mw_regress_fails,
        "pass_folds_rule": bool(pass_folds),
        "pass_mean_rule": bool(pass_mean),
        "pass_smape_rule": bool(pass_smape),
        "pass_mw_rule": bool(pass_mw),
        "verdict": verdict,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abt-path", default="abt_v7_cached.parquet")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reg-objective", default="quantile",
                    choices=["quantile", "tweedie", "regression_l1",
                             "regression", "huber", "mape"])
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--tweedie-variance-power", type=float, default=1.5)
    ap.add_argument("--recency-gamma", type=float, default=None)
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--num-threads", type=int, default=4,
                    help="LightGBM num_threads (macOS: 2-4 avoids OpenMP stalls)")
    ap.add_argument("--optuna-params", default=None)
    ap.add_argument("--baseline-cv", default=None,
                    help="Path to baseline cv_summary_*.json for decision gate. "
                         "If omitted, the candidate runs in isolation.")
    args = ap.parse_args()

    cand = run_cv(args)

    if args.baseline_cv:
        base = json.loads(Path(args.baseline_cv).read_text())
        decision = compare_to_baseline(cand, base)
        out_dec = OUT / f"decision_{args.tag}.json"
        out_dec.write_text(json.dumps(decision, indent=2, ensure_ascii=False))

        tag = f"{args.tag} vs {base['tag']}"
        log.info("\n=== DECISION GATE [%s] ===", tag)
        log.info("  mean ΔSIMSCORE   = %+0.4f  (threshold: ≤ -%.3f → %s)",
                 decision["mean_d_SIMSCORE"], PASS_MEAN_DELTA,
                 "✓" if decision["pass_mean_rule"] else "✗")
        log.info("  folds improving  = %d/5  (threshold: ≥%d → %s)",
                 decision["folds_improving_SIMSCORE"], PASS_MIN_FOLDS,
                 "✓" if decision["pass_folds_rule"] else "✗")
        log.info("  SMAPE regress    = folds %s  (any > +%.2f → %s)",
                 decision["smape_regress_fails_folds"] or "none",
                 PASS_SMAPE_REGRESS_CAP,
                 "✓" if decision["pass_smape_rule"] else "✗")
        log.info("  Monthly-WAPE reg = folds %s  (any > +%.2f → %s)",
                 decision["mw_regress_fails_folds"] or "none",
                 PASS_MW_REGRESS_CAP,
                 "✓" if decision["pass_mw_rule"] else "✗")
        log.info("  VERDICT          = %s", decision["verdict"])
        log.info("wrote %s", out_dec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
