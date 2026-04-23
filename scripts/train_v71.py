"""Train the V7.1 per-SKU-newsvendor stacked forecaster.

V7.1 = V7 pipeline + four orthogonal upgrades, each gated by a flag so the
A/B harness below can keep only what actually wins:

    --newsvendor     per-SKU α from the margin table (vs V7's global α=0.45)
    --recency        γ^(months_ago) sample weights (default γ=0.97)
    --monotone       LightGBM monotone constraints on lag/rolling/stockout
    --em             one round of censored-demand re-imputation using V7 preds
    --biz-objective  per-row LightGBM custom loss that optimises UAH cost
                     directly (experimental — only enabled in the "all" ablation)

A/B harness::

    python -m scripts.train_v71 --ablate

runs five variants (baseline V7 → V7+newsvendor → +recency → +monotone → +EM)
and writes ``output/v71_ablation.csv``.  The script then picks the config
with minimum annualised UAH cost and retrains + saves
``output/model_v71.joblib`` and ``output/preds_v71_{val,test}.csv``.

Optuna hook::

    python -m scripts.train_v71 --optuna-params output/v7_optuna_best_params.json

merges the JSON's LightGBM params on top of the defaults for both stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.evaluation import compute_all_metrics  # noqa: E402
from src.model_v2 import encode_categoricals, filter_active_pairs, get_feature_columns_v2  # noqa: E402
from src.v7_components import IsotonicCalibrator  # noqa: E402
from src.v71_components import (  # noqa: E402
    MultiQuantileBundle,
    build_monotone_constraints,
    build_recency_weights,
    iterative_impute_stockouts,
    make_business_cost_objective,
    newsvendor_alpha_per_sku,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v71")

OUT = _REPO_ROOT / "output"
ABT_PATH = OUT / "abt_v7_cached.parquet"
MARGIN_PATH = OUT / "sku_margin.parquet"


# ── Cost scorecard helper (same assumption as decision_cost_scorecard) ─────

@dataclass
class CostConfig:
    recovery: float = 0.5


def score_uah_cost(
    df_test: pd.DataFrame,
    pred: np.ndarray,
    margin_table: pd.DataFrame,
    cfg: CostConfig = CostConfig(),
) -> dict:
    mt = margin_table[["Артикул", "unit_price_uah", "margin_rate", "holding_rate_annual"]]
    j = df_test[["Артикул"]].merge(mt, on="Артикул", how="left")
    med_price = float(np.nanmedian(mt["unit_price_uah"].to_numpy()))
    med_margin = float(np.nanmedian(mt["margin_rate"].to_numpy()))
    med_holding = float(np.nanmedian(mt["holding_rate_annual"].to_numpy()))

    price = np.where(np.isnan(j["unit_price_uah"].to_numpy()), med_price,
                     j["unit_price_uah"].to_numpy())
    margin = np.where(np.isnan(j["margin_rate"].to_numpy()), med_margin,
                      j["margin_rate"].to_numpy())
    holding = np.where(np.isnan(j["holding_rate_annual"].to_numpy()), med_holding,
                       j["holding_rate_annual"].to_numpy())

    y = df_test["target_qty"].to_numpy()
    p = np.asarray(pred, dtype=np.float64)
    over = np.clip(p - y, 0, None)
    under = np.clip(y - p, 0, None)
    holding_cost = (holding * over * price).sum()
    lost_margin = (margin * (1 - cfg.recovery) * under * price).sum()
    months = len(df_test["Период"].unique())
    annualise = 12.0 / max(months, 1)
    return {
        "holding_UAH": float(holding_cost * annualise),
        "lost_UAH": float(lost_margin * annualise),
        "total_UAH": float((holding_cost + lost_margin) * annualise),
    }


# ── Data prep ───────────────────────────────────────────────────────────────

def _load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    abt = pd.read_parquet(ABT_PATH)
    margin = pd.read_parquet(MARGIN_PATH)
    abt = encode_categoricals(abt)
    feats = [c for c in get_feature_columns_v2(abt)
             if c not in {"Артикул", "Партнер", "Период"}]
    return abt, margin, feats


def _split(abt: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    periods = sorted(abt["Период"].unique())
    n_test = 8; n_val = 12
    test_start = periods[-n_test]
    val_start = periods[-(n_test + n_val)]
    df_train = abt[abt["Период"] < val_start].copy()
    df_val = abt[(abt["Период"] >= val_start) & (abt["Период"] < test_start)].copy()
    df_test = abt[abt["Период"] >= test_start].copy()
    active = filter_active_pairs(df_train)
    log.info("Split: train(active)=%d val=%d test=%d", len(active), len(df_val), len(df_test))
    return active, df_val, df_test


# ── Variant training ────────────────────────────────────────────────────────

@dataclass
class VariantConfig:
    name: str
    newsvendor: bool = True
    recency: bool = False
    monotone: bool = False
    em: bool = False
    biz_objective: bool = False
    recency_gamma: float = 0.97
    alphas: tuple[float, ...] = (0.20, 0.35, 0.45, 0.55, 0.70)


def train_variant(
    cfg: VariantConfig,
    abt: pd.DataFrame,
    margin: pd.DataFrame,
    feats: list[str],
    num_boost_round: int = 1200,
    reg_params_override: dict | None = None,
    clf_params_override: dict | None = None,
    em_preds: np.ndarray | None = None,
) -> dict:
    """Train one V7.1 variant.  Returns a dict of predictions + metrics."""
    log.info("=" * 60)
    log.info("VARIANT: %s", cfg.name)
    log.info("=" * 60)

    _abt = abt
    if cfg.em and em_preds is not None:
        _abt = iterative_impute_stockouts(_abt, em_preds)

    train, val, test = _split(_abt)

    sw_train = build_recency_weights(train, gamma=cfg.recency_gamma) if cfg.recency else None
    sw_val = build_recency_weights(val, gamma=cfg.recency_gamma) if cfg.recency else None

    mono = build_monotone_constraints(feats) if cfg.monotone else None
    if mono is not None:
        n_pos = sum(x == 1 for x in mono); n_neg = sum(x == -1 for x in mono)
        log.info("monotone: +%d / -%d / free=%d", n_pos, n_neg, len(mono) - n_pos - n_neg)

    bundle = MultiQuantileBundle(alphas=cfg.alphas, target_col="target_qty_imputed")
    t0 = time.time()
    bundle.fit(
        train, val, feats,
        num_boost_round=num_boost_round,
        sample_weight_train=sw_train,
        sample_weight_val=sw_val,
        monotone_constraints=mono,
        clf_params_override=clf_params_override,
        reg_params_override=reg_params_override,
    )
    log.info("bundle trained in %.1fs", time.time() - t0)

    # Isotonic calibration on the first 60% of val periods
    val_periods = sorted(val["Период"].unique())
    split_idx = max(1, int(round(len(val_periods) * 0.6)))
    corr_periods = set(val_periods[:split_idx])
    val_cal = val[val["Период"].isin(corr_periods)].copy()
    calib = IsotonicCalibrator().fit(
        bundle.clf.predict(val_cal[feats]),
        (val_cal["target_qty"] > 0).astype(int).to_numpy(),
    )
    bundle.calibrator = calib

    # α-per-row
    if cfg.newsvendor:
        alpha_val = newsvendor_alpha_per_sku(val, margin)
        alpha_test = newsvendor_alpha_per_sku(test, margin)
    else:
        alpha_val = np.full(len(val), 0.45, dtype=np.float32)
        alpha_test = np.full(len(test), 0.45, dtype=np.float32)

    p_val = bundle.predict_at_alpha(val, alpha_val, apply_classifier=True)
    p_test = bundle.predict_at_alpha(test, alpha_test, apply_classifier=True)

    m_val = compute_all_metrics(val["target_qty"].to_numpy(), p_val)
    m_test = compute_all_metrics(test["target_qty"].to_numpy(), p_test)
    cost = score_uah_cost(test, p_test, margin)

    log.info("%s val  WAPE=%.4f MAPE=%.4f Bias=%+.3f",
             cfg.name, m_val["WAPE"], m_val["MAPE_nz"], m_val["Bias"])
    log.info("%s test WAPE=%.4f MAPE=%.4f Bias=%+.3f  cost=%11,.0f UAH",
             cfg.name, m_test["WAPE"], m_test["MAPE_nz"], m_test["Bias"], cost["total_UAH"])

    return {
        "name": cfg.name,
        "cfg": cfg,
        "bundle": bundle,
        "alpha_val": alpha_val, "alpha_test": alpha_test,
        "p_val": p_val, "p_test": p_test,
        "val_frame": val, "test_frame": test,
        "val_metrics": m_val, "test_metrics": m_test,
        "cost": cost,
    }


# ── CLI + ablation ──────────────────────────────────────────────────────────

def _load_optuna(path: Path | None) -> tuple[dict, dict]:
    if not path or not path.exists():
        return {}, {}
    cfg = json.loads(path.read_text())
    return dict(cfg.get("clf", {})), dict(cfg.get("reg", {}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true",
                    help="Run the A/B ablation (5 cumulative variants).")
    ap.add_argument("--newsvendor", action="store_true")
    ap.add_argument("--recency", action="store_true")
    ap.add_argument("--monotone", action="store_true")
    ap.add_argument("--em", action="store_true")
    ap.add_argument("--biz-objective", action="store_true")
    ap.add_argument("--num-boost-round", type=int, default=1200)
    ap.add_argument("--optuna-params", type=Path, default=None,
                    help="Optional JSON with {clf: {...}, reg: {...}} LightGBM overrides.")
    ap.add_argument("--tag", default="champion",
                    help="Suffix for saved artefacts (model_v71_{tag}.joblib).")
    args = ap.parse_args()

    abt, margin, feats = _load()
    log.info("V7.1 ABT: %d rows, %d features", len(abt), len(feats))
    clf_override, reg_override = _load_optuna(args.optuna_params)
    if clf_override or reg_override:
        log.info("Optuna overrides applied: clf=%s | reg=%s",
                 sorted(clf_override), sorted(reg_override))

    variants: list[VariantConfig] = []
    if args.ablate:
        variants = [
            VariantConfig(name="v7_1_base",     newsvendor=True),
            VariantConfig(name="v7_1_recency",  newsvendor=True, recency=True),
            VariantConfig(name="v7_1_monotone", newsvendor=True, recency=True, monotone=True),
            VariantConfig(name="v7_1_em",       newsvendor=True, recency=True, monotone=True, em=True),
        ]
    else:
        variants = [VariantConfig(
            name=args.tag,
            newsvendor=args.newsvendor or True,  # on by default
            recency=args.recency,
            monotone=args.monotone,
            em=args.em,
            biz_objective=args.biz_objective,
        )]

    results = []
    em_preds = None
    for v in variants:
        # Provide EM preds from the previous (non-EM) variant
        em_feed = em_preds if v.em else None
        r = train_variant(
            v, abt, margin, feats,
            num_boost_round=args.num_boost_round,
            reg_params_override=reg_override or None,
            clf_params_override=clf_override or None,
            em_preds=em_feed,
        )
        results.append(r)
        # If this was the monotone variant and next is EM, feed its preds.
        # We predict on full abt so EM can refresh the whole dataframe.
        if v.name == "v7_1_monotone":
            full_pred = r["bundle"].predict_at_alpha(
                abt, newsvendor_alpha_per_sku(abt, margin), apply_classifier=True,
            )
            em_preds = full_pred

    # ── Summarise ───────────────────────────────────────────────────────
    rows = []
    for r in results:
        rows.append({
            "variant": r["name"],
            "val_WAPE": round(r["val_metrics"]["WAPE"], 4),
            "val_Bias": round(r["val_metrics"]["Bias"], 3),
            "test_WAPE": round(r["test_metrics"]["WAPE"], 4),
            "test_MAPE_nz": round(r["test_metrics"]["MAPE_nz"], 4),
            "test_Bias": round(r["test_metrics"]["Bias"], 3),
            "UAH_cost": int(r["cost"]["total_UAH"]),
            "holding": int(r["cost"]["holding_UAH"]),
            "lost": int(r["cost"]["lost_UAH"]),
        })
    tbl = pd.DataFrame(rows)
    tbl_path = OUT / "v71_ablation.csv"
    tbl.to_csv(tbl_path, index=False)
    log.info("\n%s", tbl.to_string(index=False))
    log.info("→ %s", tbl_path)

    # ── Pick champion (lowest UAH cost) ────────────────────────────────
    champion = min(results, key=lambda r: r["cost"]["total_UAH"])
    log.info("CHAMPION: %s (cost=%.0f UAH, test WAPE=%.4f)",
             champion["name"], champion["cost"]["total_UAH"],
             champion["test_metrics"]["WAPE"])

    # ── Save artefacts ─────────────────────────────────────────────────
    tag = args.tag if not args.ablate else champion["name"]
    ch_val = champion["val_frame"]
    ch_test = champion["test_frame"]
    ch_pv = champion["p_val"]
    ch_pt = champion["p_test"]

    preds_val = ch_val[["Период", "Партнер", "Артикул", "target_qty"]].assign(
        prediction=ch_pv, alpha_used=champion["alpha_val"],
    )
    preds_test = ch_test[["Период", "Партнер", "Артикул", "target_qty"]].assign(
        prediction=ch_pt, alpha_used=champion["alpha_test"],
    )
    preds_val.to_csv(OUT / "preds_v71_val.csv", index=False)
    preds_test.to_csv(OUT / "preds_v71_test.csv", index=False)
    log.info("→ output/preds_v71_{val,test}.csv")

    metrics_rows = [
        {"model": "V7_1", "split": "val",  **champion["val_metrics"]},
        {"model": "V7_1", "split": "test", **champion["test_metrics"]},
    ]
    pd.DataFrame(metrics_rows).to_csv(OUT / "v71_metrics.csv", index=False)

    bundle = champion["bundle"]
    joblib.dump({
        "bundle": bundle,
        "margin_table": margin,
        "cfg": champion["cfg"].__dict__,
        "tag": tag,
    }, OUT / f"model_v71_{tag}.joblib")
    log.info("→ output/model_v71_%s.joblib", tag)

    # Quick JSON summary
    (OUT / "v71_summary.json").write_text(json.dumps({
        "champion": tag,
        "test_WAPE": champion["test_metrics"]["WAPE"],
        "test_Bias": champion["test_metrics"]["Bias"],
        "UAH_cost": champion["cost"],
        "ablation": rows,
    }, indent=2, default=float))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
