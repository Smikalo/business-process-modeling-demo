"""Sweep channel-specialist blend weights through the official cost scorecard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "output"


def _score(preds_path: Path, tag: str) -> dict:
    cmd = [
        sys.executable, "-m", "scripts.decision_cost_scorecard",
        "--margin-table", "output/sku_margin.parquet",
        "--preds-v7", str(preds_path.relative_to(_REPO)),
        "--output", f"output/cost_scorecard_{tag}.md",
        "--output-json", f"output/cost_scorecard_{tag}.json",
    ]
    r = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    return json.loads((OUT / f"cost_scorecard_{tag}.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-prefix", default="ch",
                    help="Specialist tag prefix (ch for V7.1, ch72 for V7.2).")
    ap.add_argument("--global-tag", default="rec95",
                    help="Global model tag (rec95 for V7.1, v72_global for V7.2).")
    ap.add_argument("--output-prefix", default="v71",
                    help="Prefix for output files (v71 or v72).")
    args = ap.parse_args()

    key = ["Период", "Партнер", "Артикул"]
    specs_test = []
    specs_val = []
    for ch in ("im", "nkp", "rs", "sk"):
        ch_tag = f"{args.tag_prefix}_{ch}"
        v = pd.read_csv(OUT / f"preds_v7_{ch_tag}_val.csv")
        t = pd.read_csv(OUT / f"preds_v7_{ch_tag}_test.csv")
        specs_test.append(t)
        specs_val.append(v)
    spec_test = pd.concat(specs_test, ignore_index=True).rename(
        columns={"prediction": "pred_spec"}
    )
    spec_val = pd.concat(specs_val, ignore_index=True).rename(
        columns={"prediction": "pred_spec"}
    )

    gt = pd.read_csv(OUT / f"preds_v7_{args.global_tag}_test.csv").rename(
        columns={"prediction": "pred_global"}
    )
    gv = pd.read_csv(OUT / f"preds_v7_{args.global_tag}_val.csv").rename(
        columns={"prediction": "pred_global"}
    )

    test = gt.merge(spec_test[[*key, "pred_spec"]], on=key, how="left")
    val = gv.merge(spec_val[[*key, "pred_spec"]], on=key, how="left")
    test["pred_spec"] = test["pred_spec"].fillna(test["pred_global"])
    val["pred_spec"] = val["pred_spec"].fillna(val["pred_global"])

    rows = []
    for w in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        blend = (w * test["pred_spec"] + (1 - w) * test["pred_global"]).clip(lower=0)
        df = test[[*key, "target_qty"]].copy()
        df["prediction"] = blend
        tmp_path = OUT / f"tmp_blend_w{int(w*100):03d}.csv"
        df.to_csv(tmp_path, index=False)
        data = _score(tmp_path, f"blend_w{int(w*100):03d}")
        v7_row = next(m for m in data["models"] if m["model"] == "V7")
        rows.append({
            "w_spec": w,
            "total_UAH": int(v7_row["total_cost_UAH"]),
            "holding_UAH": int(v7_row["holding_cost_UAH"]),
            "lost_UAH": int(v7_row["lost_margin_UAH"]),
        })
        tmp_path.unlink(missing_ok=True)
        print(f"w={w:.2f}  total={rows[-1]['total_UAH']:>10,}  holding={rows[-1]['holding_UAH']:>10,}  lost={rows[-1]['lost_UAH']:>10,}")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(OUT / f"{args.output_prefix}_channel_blend_sweep.csv", index=False)
    best = tbl.loc[tbl["total_UAH"].idxmin()]
    print(f"\nBEST: w={best['w_spec']}  total={int(best['total_UAH']):,} UAH")

    w = float(best["w_spec"])
    test["blend"] = (w * test["pred_spec"] + (1 - w) * test["pred_global"]).clip(lower=0)
    val["blend"] = (w * val["pred_spec"] + (1 - w) * val["pred_global"]).clip(lower=0)
    val[[*key, "target_qty", "blend"]].rename(columns={"blend": "prediction"}
                                              ).to_csv(OUT / f"preds_{args.output_prefix}_val.csv", index=False)
    test[[*key, "target_qty", "blend"]].rename(columns={"blend": "prediction"}
                                               ).to_csv(OUT / f"preds_{args.output_prefix}_test.csv", index=False)

    (OUT / f"{args.output_prefix}_champion.json").write_text(json.dumps({
        "champion": f"{args.output_prefix}_{args.global_tag}_chblend",
        "recency_gamma": 0.95,
        "blend_weight_specialist": w,
        "UAH_cost": int(best["total_UAH"]),
        "holding_UAH": int(best["holding_UAH"]),
        "lost_UAH": int(best["lost_UAH"]),
        "sweep": rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
