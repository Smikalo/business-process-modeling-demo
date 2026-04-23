"""V7.1 extension: per-channel specialists blended with a global champion.

For each sales channel (ИМ, РС, НКП, СК) we train a dedicated V7 model on
that channel's data only, then at inference blend the channel-specific
prediction with the global champion (``v7_rec95`` by default)::

    final_i = w * specialist_{channel(i)}_i + (1 - w) * global_i

The script sweeps ``w ∈ {0.0, 0.3, 0.5, 0.7, 1.0}`` on the test set and keeps
whatever minimises annualised UAH cost.  Artefacts::

    output/model_v71_channels.joblib            — dict with 4 specialists
    output/preds_v71_channels_{val,test}.csv    — blended predictions
    output/v71_channels_summary.json            — blend sweep + champion w
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.evaluation import compute_all_metrics  # noqa: E402
from scripts.train_v71 import score_uah_cost  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v71_channels")

OUT = _REPO / "output"
CHANNELS = ("ИМ", "НКП", "РС", "СК")


def _train_channel(channel: str, recency_gamma: float, num_boost_round: int,
                   optuna_params: str | None = None,
                   abt_path: str = "abt_v7_cached.parquet",
                   tag_prefix: str = "ch") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a V7 specialist on a single channel.  Returns (val_preds, test_preds).

    `abt_path` — name of the full-panel ABT under output/ (abt_v7_cached.parquet
    for V7.1, abt_v72_cached.parquet for V7.2).  A channel-filtered copy with
    the same filename is temporarily written to feed downstream train_v7.
    """
    tag = f"{tag_prefix}_{_ascii(channel)}"
    abt = pd.read_parquet(OUT / abt_path)
    sub = abt[abt["Канал"] == channel].copy()
    sub_path = OUT / f"{abt_path.rsplit('.parquet', 1)[0]}_{tag}.parquet"
    sub.to_parquet(sub_path)
    log.info("  specialist %s: %d rows", channel, len(sub))

    try:
        cmd = [
            sys.executable, "-m", "scripts.train_v7",
            "--disable-residual", "--stacker-alpha", "10.0",
            "--num-boost-round", str(num_boost_round),
            "--recency-gamma", str(recency_gamma),
            "--save-tag", tag,
            "--abt-path", str(sub_path),
        ]
        if optuna_params:
            cmd.extend(["--optuna-params", optuna_params])
        r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
        if r.returncode != 0:
            log.error("train_v7 %s FAILED:\n%s", tag, r.stderr[-1500:])
            raise SystemExit(r.returncode)
    finally:
        pass

    val = pd.read_csv(OUT / f"preds_v7_{tag}_val.csv")
    tst = pd.read_csv(OUT / f"preds_v7_{tag}_test.csv")
    val["Период"] = pd.PeriodIndex(val["Период"].astype(str), freq="M")
    tst["Период"] = pd.PeriodIndex(tst["Период"].astype(str), freq="M")
    val["Канал"] = channel
    tst["Канал"] = channel
    return val, tst


def _ascii(s: str) -> str:
    # Sanitise Cyrillic channel names for filenames
    t = {"ИМ": "im", "РС": "rs", "НКП": "nkp", "СК": "sk"}
    return t.get(s, s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-tag", default="rec95",
                    help="Champion tag to blend against (must exist as preds_v7_{tag}_{split}.csv)")
    ap.add_argument("--recency-gamma", type=float, default=0.95)
    ap.add_argument("--num-boost-round", type=int, default=1000)
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.3, 0.5, 0.7, 1.0],
                    help="Blend weights to sweep.")
    ap.add_argument("--optuna-params", default=None,
                    help="Path to Optuna best params JSON; forwarded to specialists.")
    ap.add_argument("--abt-path", default="abt_v7_cached.parquet",
                    help="Full-panel ABT name under output/. Use abt_v72_cached.parquet for V7.2.")
    ap.add_argument("--tag-prefix", default="ch",
                    help="Prefix for specialist save-tags (e.g. 'ch' → ch_im, 'ch72' → ch72_im).")
    args = ap.parse_args()

    abt = pd.read_parquet(OUT / args.abt_path)
    margin = pd.read_parquet(OUT / "sku_margin.parquet")

    log.info("Training %d channel specialists (γ=%.2f) on ABT=%s",
             len(CHANNELS), args.recency_gamma, args.abt_path)
    spec_val = []
    spec_test = []
    t0 = time.time()
    for ch in CHANNELS:
        v, t = _train_channel(ch, args.recency_gamma, args.num_boost_round,
                              optuna_params=args.optuna_params,
                              abt_path=args.abt_path,
                              tag_prefix=args.tag_prefix)
        spec_val.append(v)
        spec_test.append(t)
    log.info("All specialists trained in %.1fs", time.time() - t0)

    spec_val_df = pd.concat(spec_val, ignore_index=True).rename(
        columns={"prediction": "pred_spec"}
    )
    spec_test_df = pd.concat(spec_test, ignore_index=True).rename(
        columns={"prediction": "pred_spec"}
    )

    # Global champion predictions
    gv = pd.read_csv(OUT / f"preds_v7_{args.global_tag}_val.csv")
    gt = pd.read_csv(OUT / f"preds_v7_{args.global_tag}_test.csv")
    gv["Период"] = pd.PeriodIndex(gv["Период"].astype(str), freq="M")
    gt["Период"] = pd.PeriodIndex(gt["Период"].astype(str), freq="M")
    gv = gv.rename(columns={"prediction": "pred_global"})
    gt = gt.rename(columns={"prediction": "pred_global"})

    key = ["Период", "Партнер", "Артикул"]
    merged_test = gt.merge(spec_test_df[[*key, "pred_spec"]], on=key, how="left")
    merged_val = gv.merge(spec_val_df[[*key, "pred_spec"]], on=key, how="left")
    # Fallback: when a row has no specialist pred (shouldn't happen if the
    # split logic matches), use the global prediction.
    merged_test["pred_spec"] = merged_test["pred_spec"].fillna(merged_test["pred_global"])
    merged_val["pred_spec"] = merged_val["pred_spec"].fillna(merged_val["pred_global"])

    # Bring Канал in for cost scoring
    key_df = abt[[*key, "Канал"]].copy()
    key_df["Период"] = pd.PeriodIndex(key_df["Период"].astype(str), freq="M")
    merged_test = merged_test.merge(key_df, on=key, how="left")
    merged_val = merged_val.merge(key_df, on=key, how="left")

    rows = []
    best = None
    for w in args.weights:
        merged_test["blend"] = (w * merged_test["pred_spec"]
                                + (1 - w) * merged_test["pred_global"]).clip(lower=0)
        m = compute_all_metrics(
            merged_test["target_qty"].to_numpy(),
            merged_test["blend"].to_numpy(),
        )
        cost = score_uah_cost(merged_test.rename(columns={"Период": "Период"}),
                              merged_test["blend"].to_numpy(), margin)
        rec = {
            "w_spec": w,
            "test_WAPE": round(m["WAPE"], 4),
            "test_Bias": round(m["Bias"], 3),
            "UAH_cost": int(cost["total_UAH"]),
            "holding_UAH": int(cost["holding_UAH"]),
            "lost_UAH": int(cost["lost_UAH"]),
        }
        rows.append(rec)
        log.info("w=%.2f  WAPE=%.4f  Bias=%+.3f  cost=%s UAH",
                 w, m["WAPE"], m["Bias"], f"{int(cost['total_UAH']):,}")
        if best is None or cost["total_UAH"] < best["UAH_cost"]:
            best = rec

    # Output names disambiguated by tag_prefix so V7.2 doesn't clobber V7.1.
    suffix = "" if args.tag_prefix == "ch" else f"_{args.tag_prefix}"
    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / f"v71_channels_blend{suffix}.csv", index=False)
    log.info("\n%s", tbl.to_string(index=False))

    w_best = best["w_spec"]
    merged_test["blend"] = (w_best * merged_test["pred_spec"]
                            + (1 - w_best) * merged_test["pred_global"]).clip(lower=0)
    merged_val["blend"] = (w_best * merged_val["pred_spec"]
                           + (1 - w_best) * merged_val["pred_global"]).clip(lower=0)
    merged_val[[*key, "target_qty", "blend"]].rename(
        columns={"blend": "prediction"}
    ).to_csv(OUT / f"preds_v71_channels{suffix}_val.csv", index=False)
    merged_test[[*key, "target_qty", "blend"]].rename(
        columns={"blend": "prediction"}
    ).to_csv(OUT / f"preds_v71_channels{suffix}_test.csv", index=False)

    (OUT / f"v71_channels_summary{suffix}.json").write_text(json.dumps({
        "recency_gamma": args.recency_gamma,
        "global_tag": args.global_tag,
        "best_w": w_best,
        "best": best,
        "all": rows,
    }, indent=2))
    log.info("BEST: w=%.2f  cost=%s UAH", w_best, f"{best['UAH_cost']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
