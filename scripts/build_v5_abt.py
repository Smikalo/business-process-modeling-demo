"""Build the V5 enriched ABT using the decision-gate winners.

Inputs:
    output/abt_v4_cached.parquet   (V4 baseline ABT)
    output/decision_gate.csv       (chosen signals, optional override)

Outputs:
    output/abt_v5_cached.parquet   (V4 features + winning external signals)
    output/v5_feature_manifest.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.enrichment_external import enrich_with_external
from src.external_data import import_default_loaders
from src.model_v2 import encode_categoricals, get_feature_columns_v2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("build_v5_abt")

ROOT = Path(__file__).resolve().parent.parent
V4_ABT = ROOT / "output" / "abt_v4_cached.parquet"
V5_ABT = ROOT / "output" / "abt_v5_cached.parquet"
MANIFEST = ROOT / "output" / "v5_feature_manifest.json"

# Winners per output/decision_gate.md
V5_LOADERS: list[str] = [
    "conflict_ua",
    "gtrends_ua",
    "holidays_ua",
    "nbu_fx",
    "tmdb_movies",
    "world_bank_ua",
]


def main() -> int:
    import_default_loaders()
    log.info("Loading V4 ABT → %s", V4_ABT)
    abt = pd.read_parquet(V4_ABT)
    log.info("V4 ABT rows=%d cols=%d", len(abt), abt.shape[1])

    log.info("Enriching with V5 winners: %s", V5_LOADERS)
    abt_v5 = enrich_with_external(abt, V5_LOADERS, apply_lag=True)
    abt_v5 = encode_categoricals(abt_v5)

    feats = get_feature_columns_v2(abt_v5)
    # Capture the extra external-signal columns beyond the baseline
    base_feats = set(get_feature_columns_v2(pd.read_parquet(V4_ABT).pipe(encode_categoricals)))
    external_feats = sorted(c for c in abt_v5.columns if c not in base_feats and abt_v5[c].dtype.kind in ("f", "i", "u", "b"))

    V5_ABT.parent.mkdir(exist_ok=True, parents=True)
    abt_v5.to_parquet(V5_ABT, index=False)
    MANIFEST.write_text(
        json.dumps(
            {
                "loaders": V5_LOADERS,
                "baseline_feature_count": len(base_feats),
                "external_feature_count": len(external_feats),
                "external_features": external_feats,
                "total_rows": int(len(abt_v5)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    log.info("V5 ABT written → %s (rows=%d cols=%d)", V5_ABT, len(abt_v5), abt_v5.shape[1])
    log.info("External signal columns: %d", len(external_feats))
    log.info("Manifest → %s", MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
