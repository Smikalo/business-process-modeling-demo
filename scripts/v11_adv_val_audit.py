"""V11 Priority 1 -- adversarial-validation drift audit.

Compares training rows (2020-01..2024-06) to the most recent validation
rows (2025-04..2025-06).  If AUC > 0.55, drift is real and we should
proceed with importance-weighted retraining.

Outputs:
* `output/v11/adv_audit_report.json` -- AUC + summary
* `output/v11/adv_drift_features.csv` -- top drifting features
* `output/v11/adv_train_weights.parquet` -- per-row importance weights
* prints the top-30 drifting features to stdout
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.adversarial_validation import adversarial_audit  # noqa: E402
from src.model_v2 import get_feature_columns_v2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v11_adv_val_audit")

OUT = REPO / "output"
V11 = OUT / "v11"
V11.mkdir(parents=True, exist_ok=True)


def main() -> int:
    abt = pd.read_parquet(OUT / "abt_v10_cached.parquet")
    abt["Период"] = abt["Период"].astype(str)
    log.info("loaded V10 ABT: %d rows, %d cols", len(abt), abt.shape[1])

    feats = get_feature_columns_v2(abt)
    EXCLUDE_ABS_TIME = {
        "year", "months_since_invasion", "days_since_train_start",
        "month_index", "abs_period", "war_phase", "trend_idx",
    }
    EXCLUDE_MACRO_PREFIXES = (
        "uah_", "wb_", "imf_", "nbu_", "conflict_", "weather_",
        "world_bank_", "fx_", "policy_rate", "inflation_",
        "cpi_", "ppi_", "gdp_",
    )
    EXCLUDE_MACRO_EXACT = {
        "policy_rate", "uah_usd_eom", "uah_eur_eom",
    }

    def _is_macro(c: str) -> bool:
        if c in EXCLUDE_MACRO_EXACT:
            return True
        return any(c.startswith(p) for p in EXCLUDE_MACRO_PREFIXES)

    feats = [c for c in feats
             if c != "target_qty_imputed"
             and not c.startswith("target_qty")
             and c != "was_censored"
             and c not in EXCLUDE_ABS_TIME
             and not _is_macro(c)]
    log.info("after exclusions: %d demand-relative features", len(feats))

    cat_cols = []
    for c in ("Канал", "Бренд", "Сегмент_ABC", "Тип_соглашения", "Группа_товара"):
        if c in abt.columns and isinstance(abt[c].dtype, pd.CategoricalDtype):
            cat_cols.append(c)
    log.info("features=%d, categorical=%d", len(feats), len(cat_cols))

    audit = adversarial_audit(
        abt=abt,
        train_period_start="2020-01", train_period_end="2024-06",
        recent_period_start="2025-04", recent_period_end="2025-06",
        feature_cols=feats, cat_cols=cat_cols,
        n_splits=4, weight_clip=(0.1, 10.0), seed=17,
    )

    log.info("AUC OOF       = %.4f", audit.auc_oof)
    log.info("AUC in-sample = %.4f", audit.auc_in_sample)
    log.info("n_train = %d  n_recent = %d", audit.n_train, audit.n_recent)
    log.info("p_recent on train: mean=%.4f  median=%.4f  q90=%.4f  q99=%.4f",
             audit.p_recent_on_train.mean(),
             np.median(audit.p_recent_on_train),
             np.quantile(audit.p_recent_on_train, 0.9),
             np.quantile(audit.p_recent_on_train, 0.99))

    print("\n=== Top-30 drifting features ===")
    print(audit.feature_importance.head(30).to_string(index=False))

    audit.feature_importance.to_csv(V11 / "adv_drift_features.csv", index=False)
    audit.train_weights.reset_index().rename(columns={"index": "row_id"}).to_parquet(
        V11 / "adv_train_weights.parquet", index=False,
    )

    summary = {
        "auc_oof": audit.auc_oof,
        "auc_in_sample": audit.auc_in_sample,
        "n_train_rows": audit.n_train,
        "n_recent_rows": audit.n_recent,
        "weight_stats": {
            "min": float(audit.train_weights.min()),
            "p25": float(audit.train_weights.quantile(0.25)),
            "median": float(audit.train_weights.median()),
            "p75": float(audit.train_weights.quantile(0.75)),
            "p99": float(audit.train_weights.quantile(0.99)),
            "max": float(audit.train_weights.max()),
        },
        "top_10_drift_features": audit.feature_importance.head(10).to_dict("records"),
    }
    (V11 / "adv_audit_report.json").write_text(
        json.dumps(summary, indent=2, default=float)
    )
    log.info("wrote v11/adv_audit_report.json + adv_drift_features.csv "
             "+ adv_train_weights.parquet")

    if audit.auc_oof < 0.55:
        log.warning("AUC < 0.55 -- drift is weak; importance-weighted "
                    "retraining may not help much.")
    elif audit.auc_oof > 0.7:
        log.warning("AUC > 0.70 -- drift is severe; importance weights "
                    "may overshoot.  Consider tighter weight clipping.")
    else:
        log.info("AUC in productive range [0.55, 0.70] -- proceed with "
                 "importance-weighted retraining.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
