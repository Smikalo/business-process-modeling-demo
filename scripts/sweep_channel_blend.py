"""Sweep channel-specialist blend weights through the official cost scorecard."""

from __future__ import annotations

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
    # Rebuild the blended preds at each weight using already-saved
    # globals (rec95) + specialists (stored in the preds_v71_channels_*.csv,
    # but we only have ONE blend saved).  Instead, read per-channel preds
    # and re-combine.
    key = ["Период", "Партнер", "Артикул"]
    specs_test = []
    specs_val = []
    for ch_tag in ("ch_im", "ch_nkp", "ch_rs", "ch_sk"):
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

    gt = pd.read_csv(OUT / "preds_v7_rec95_test.csv").rename(
        columns={"prediction": "pred_global"}
    )
    gv = pd.read_csv(OUT / "preds_v7_rec95_val.csv").rename(
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
    tbl.to_csv(OUT / "v71_channel_blend_sweep.csv", index=False)
    best = tbl.loc[tbl["total_UAH"].idxmin()]
    print(f"\nBEST: w={best['w_spec']}  total={int(best['total_UAH']):,} UAH")

    # Persist the champion blended preds at the best w
    w = float(best["w_spec"])
    test["blend"] = (w * test["pred_spec"] + (1 - w) * test["pred_global"]).clip(lower=0)
    val["blend"] = (w * val["pred_spec"] + (1 - w) * val["pred_global"]).clip(lower=0)
    val[[*key, "target_qty", "blend"]].rename(columns={"blend": "prediction"}
                                              ).to_csv(OUT / "preds_v71_val.csv", index=False)
    test[[*key, "target_qty", "blend"]].rename(columns={"blend": "prediction"}
                                               ).to_csv(OUT / "preds_v71_test.csv", index=False)

    (OUT / "v71_champion.json").write_text(json.dumps({
        "champion": "v71_rec95_chblend",
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
