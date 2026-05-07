"""V12.4 — V12_external base re-trained with stronger recency γ=0.93.

Same pattern as V11 → V11_g93: take a base that already works
(V12_external) and re-train with sharper recency weighting (γ=0.93,
~50% weight at 24 months ago vs γ=0.97's 50% at ~37 months) to
emphasize the recent test-window-like regime.

EXT signals (UA macro, Wikipedia attention, war intensity, etc.) have
stronger recency patterns than V11's lag features, so a sharper γ may
amplify the signal.

Outputs:
  output/preds_v7_v12_external_g93_seed{S}_{val,test}.csv
  output/preds_v12_external_g93_{val,test}.csv (5-seed bagged)
  output/v12_external_g93_summary.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.score_similarity import score_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("train_v12_external_g93")

OUT = REPO / "output"

SEEDS: list[int] = [42, 137, 271, 314, 1729]
BASE_TAG = "v12_external_g93"
ABT_PATH = "abt_v12_external_recent_only.parquet"
ALPHA = 0.45
GAMMA = 0.93

KEY_COLS = ["Период", "Партнер", "Артикул"]


def _seed_tag(seed: int) -> str:
    return f"{BASE_TAG}_seed{seed}"


def _train_one_seed(seed: int) -> dict:
    from scripts.train_v7 import main as train_v7_main

    save_tag = _seed_tag(seed)
    argv = [
        "--abt-path", ABT_PATH,
        "--save-tag", save_tag,
        "--alpha", str(ALPHA),
        "--recency-gamma", str(GAMMA),
        "--seed", str(seed),
    ]
    t0 = time.time()
    rc = train_v7_main(argv)
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"train_v7 returned {rc} for seed={seed}")
    return {
        "seed": seed,
        "save_tag": save_tag,
        "val_path": str(OUT / f"preds_v7_{save_tag}_val.csv"),
        "test_path": str(OUT / f"preds_v7_{save_tag}_test.csv"),
        "elapsed_s": round(dt, 1),
    }


def _aggregate(split: str, seed_results: list[dict]) -> pd.DataFrame:
    paths = [Path(r[f"{split}_path"]) for r in seed_results]
    frames = [pd.read_csv(p) for p in paths]
    base = frames[0].copy()
    for f, p in zip(frames[1:], paths[1:]):
        for col in KEY_COLS + ["target_qty"]:
            if not (base[col].astype(str).to_numpy() == f[col].astype(str).to_numpy()).all():
                raise RuntimeError(f"row-order mismatch on {col} between {paths[0].name} and {p.name}")
    pred_stack = np.column_stack([f["prediction"].to_numpy(dtype=np.float64) for f in frames])
    base["prediction"] = pred_stack.mean(axis=1).astype(np.float32)
    return base


def main() -> int:
    if not (OUT / ABT_PATH).exists():
        log.error("missing ABT %s", OUT / ABT_PATH)
        return 1

    workers = int(os.environ.get("BAGGING_WORKERS", "2"))
    log.info("V12.4 (g93) bagging: seeds=%s, workers=%d, gamma=%.2f",
             SEEDS, workers, GAMMA)

    t_all = time.time()
    seed_results: list[dict] = []
    if workers <= 1:
        for s in SEEDS:
            seed_results.append(_train_one_seed(s))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_train_one_seed, s): s for s in SEEDS}
            for fut in as_completed(futures):
                seed_results.append(fut.result())

    seed_results.sort(key=lambda r: r["seed"])

    summary = {"seeds": SEEDS, "abt_path": ABT_PATH, "gamma": GAMMA,
                "alpha": ALPHA, "per_seed": {}, "bagged": {},
                "wall_time_s": None}
    for split in ("val", "test"):
        for r in seed_results:
            summary["per_seed"].setdefault(str(r["seed"]), {})[split] = \
                score_frame(pd.read_csv(r[f"{split}_path"]))
        bagged = _aggregate(split, seed_results)
        out_path = OUT / f"preds_v12_external_g93_{split}.csv"
        bagged.to_csv(out_path, index=False)
        summary["bagged"][split] = score_frame(bagged)
        log.info("V12_external_g93 %s: %s", split, summary["bagged"][split])

    summary["wall_time_s"] = round(time.time() - t_all, 1)
    (OUT / "v12_external_g93_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Total wall-time: %.1f s", summary["wall_time_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
