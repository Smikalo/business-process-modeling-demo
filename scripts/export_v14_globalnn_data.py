"""Export training tensors for V14 GlobalNN (Transformer-encoder) Colab run.

GlobalNN takes a different input shape than the FMs of V13: it needs
the FULL feature space (engineered + EXT) tensorised per pair × month,
plus learned embedding indices for Партнер / Артикул / Бренд / Канал.

Outputs (all in output/v14_globalnn/):
  train.parquet  — 2020-01 to 2024-06 rows, all features + categorical
                   embedding indices.
  val.parquet    — 2024-07 to 2025-06.
  test.parquet   — 2025-07 onwards.
  vocab.json     — embedding vocabularies (str → int) for each
                   categorical column. Notebook re-uses to encode.
  manifest.json  — feature-list, n_pairs, n_partners, n_skus, n_brands,
                   n_channels, schema hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
DST = OUT / "v14_globalnn"
DST.mkdir(parents=True, exist_ok=True)


def _build_vocab(s: pd.Series) -> dict[str, int]:
    """Stable string→int vocab; unknown maps to 0."""
    uniq = sorted(s.dropna().astype(str).unique())
    return {"<UNK>": 0, **{v: i + 1 for i, v in enumerate(uniq)}}


def main() -> int:
    abt_path = OUT / "abt_v12_external.parquet"
    if not abt_path.exists():
        # Fall back to V11 ABT if V12 ABT not built yet
        abt_path = OUT / "abt_v10_cached.parquet"
        print(f"[warn] abt_v12_external missing; using {abt_path.name}")

    abt = pd.read_parquet(abt_path)
    print(f"loaded {abt_path.name}: {abt.shape}")

    # Build vocabs from the training period (avoid leakage via novel
    # categories appearing in val/test)
    train_mask = abt["Период"].astype(str) <= "2024-06"
    cats = ["Партнер", "Артикул", "Бренд", "Канал"]
    vocab = {c: _build_vocab(abt.loc[train_mask, c]) for c in cats}
    for c in cats:
        abt[f"{c}_idx"] = abt[c].astype(str).map(vocab[c]).fillna(0).astype(np.int32)

    # Drop string-categorical and other non-tensor-friendly columns
    drop_cols = [c for c in cats if c in abt.columns]
    drop_cols += [c for c in abt.columns if abt[c].dtype == "object"
                   and c not in ("Период",)]

    feature_cols = [c for c in abt.columns
                     if c not in drop_cols
                     and c not in ("target_qty", "Период")]
    print(f"feature_cols: {len(feature_cols)} (categoricals + numerics)")

    abt["Период_str"] = abt["Период"].astype(str)
    splits = {
        "train": abt[abt["Период_str"] <= "2024-06"],
        "val":   abt[(abt["Период_str"] >= "2024-07") &
                     (abt["Период_str"] <= "2025-06")],
        "test":  abt[abt["Период_str"] >= "2025-07"],
    }

    keep_cols = ["Период_str", "target_qty"] + feature_cols
    for name, sp in splits.items():
        out_path = DST / f"{name}.parquet"
        sp[keep_cols].to_parquet(out_path, index=False)
        print(f"wrote {out_path} ({len(sp):,} rows, "
              f"{out_path.stat().st_size / 1e6:.1f} MB)")

    (DST / "vocab.json").write_text(
        json.dumps({c: vocab[c] for c in cats}, ensure_ascii=False, indent=2))
    print(f"wrote {DST / 'vocab.json'}")

    manifest = {
        "src_abt": abt_path.name,
        "n_features": len(feature_cols),
        "n_partners": len(vocab["Партнер"]),
        "n_skus": len(vocab["Артикул"]),
        "n_brands": len(vocab["Бренд"]),
        "n_channels": len(vocab["Канал"]),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "feature_cols_first_30": feature_cols[:30],
    }
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {DST / 'manifest.json'}")

    print(f"\n=== UPLOAD INSTRUCTIONS ===")
    print(f"Drag the contents of {DST}/ to Google Drive at:")
    print(f"   /MyDrive/v14_globalnn_data/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
