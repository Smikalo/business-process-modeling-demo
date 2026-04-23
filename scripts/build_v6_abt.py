"""Build the V6 ABT by layering imputation + promo lifecycle on top of V5.

Inputs:
    output/abt_v5_cached.parquet

Outputs:
    output/abt_v6_cached.parquet
    output/v6_feature_manifest.json
    output/v6_imputation_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.demand_imputation import impute_stockout_demand
from src.features_promo import add_promo_lifecycle
from src.model_v2 import encode_categoricals, get_feature_columns_v2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v6_abt")

ROOT = Path(__file__).resolve().parent.parent
V5_ABT = ROOT / "output" / "abt_v5_cached.parquet"
V6_ABT = ROOT / "output" / "abt_v6_cached.parquet"
MANIFEST = ROOT / "output" / "v6_feature_manifest.json"
IMP_REPORT = ROOT / "output" / "v6_imputation_report.json"

NEW_FEATURE_COLS = [
    "was_censored",
    "promo_duration_months",
    "promo_depth_pct_current",
    "months_since_last_promo",
    "months_until_next_promo",
    "post_promo_depletion_flag",
    "sku_promo_sensitivity",
]


def main() -> int:
    if not V5_ABT.exists():
        raise FileNotFoundError(
            f"V5 ABT not found at {V5_ABT}. Run scripts/build_v5_abt.py first."
        )

    log.info("Loading V5 ABT → %s", V5_ABT)
    abt = pd.read_parquet(V5_ABT)
    log.info("V5 ABT rows=%d cols=%d", len(abt), abt.shape[1])

    # ── 1. Censored-demand imputation ──────────────────────────────────────
    log.info("Step 1/2: censored-demand imputation (stockout_orc strategy)")
    abt, imp_report = impute_stockout_demand(
        abt, censor_density_min=0.3, strategy="stockout_orc", eb_prior=6.0
    )
    IMP_REPORT.write_text(
        json.dumps(
            {
                "n_rows": imp_report.n_rows,
                "n_censored": imp_report.n_censored,
                "share_censored": imp_report.share_censored,
                "mean_imputed_qty": imp_report.mean_imputed_qty,
                "strategy": imp_report.strategy,
                "censor_density_min": imp_report.censor_density_min,
            },
            indent=2,
        )
    )
    log.info("Imputation report → %s (%s)", IMP_REPORT, imp_report)
    if imp_report.share_censored > 0.04:
        log.warning(
            "Share of imputed rows = %.2f%% exceeds the 4%% safety threshold; "
            "review imputation params before training.",
            imp_report.share_censored * 100,
        )

    # ── 2. Promo-lifecycle features ─────────────────────────────────────────
    log.info("Step 2/2: promo-lifecycle features")
    abt = add_promo_lifecycle(abt, eb_prior=6.0)

    # Re-encode categoricals because add_promo_lifecycle may reset dtypes
    abt = encode_categoricals(abt)

    v6_feats = get_feature_columns_v2(abt)
    added = [c for c in NEW_FEATURE_COLS if c in abt.columns]
    log.info("V6 added feature cols: %s", added)

    V6_ABT.parent.mkdir(exist_ok=True, parents=True)
    abt.to_parquet(V6_ABT, index=False)

    MANIFEST.write_text(
        json.dumps(
            {
                "parent": str(V5_ABT.name),
                "total_rows": int(len(abt)),
                "total_cols": int(abt.shape[1]),
                "feature_count": len(v6_feats),
                "new_features_added": added,
                "imputation": {
                    "strategy": imp_report.strategy,
                    "share_censored": imp_report.share_censored,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info(
        "V6 ABT written → %s (rows=%d cols=%d, features=%d)",
        V6_ABT, len(abt), abt.shape[1], len(v6_feats),
    )
    log.info("Manifest → %s", MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
