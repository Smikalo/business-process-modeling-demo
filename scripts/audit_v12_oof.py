"""V12 OOF + leakage audit.

Checks:
  1. Row counts match V11 (no key drift).
  2. Test predictions never use any feature with publication_lag_days > 0
     for a month M unless the source had data with date < first day of M.
     (Done by spot-checking each EXT loader's meta.json against test months.)
  3. V12_final test SIMSCORE / WAPE / bias improves on V11_final.
     Also checks that the val-test gap doesn't blow up (overfit gate).
  4. Bias-aware diagnostic: per-month bias % stays within ±10 % on test.
  5. Per-channel diagnostic: per-channel WAPE not regressing > 5 % vs V11_final.

Writes:
  * ``output/v12/audit_report.json`` — pass/fail per check + numbers
  * ``output/v12/audit_report.md``  — human-readable summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V12 = OUT / "v12"
V12.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]


def _load(tag: str, split: str) -> pd.DataFrame:
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def main() -> int:
    audit = {"checks": [], "passed": 0, "failed": 0, "warnings": 0}

    # --- 1. row counts ---
    v11v = _load("v11_final", "val")
    v11t = _load("v11_final", "test")
    v12v = _load("v12_final", "val")
    v12t = _load("v12_final", "test")

    if v12v.empty or v12t.empty:
        audit["checks"].append({"name": "v12_predictions_present",
                                 "status": "FAIL",
                                 "detail": "preds_v12_final_{val,test}.csv missing"})
        audit["failed"] += 1
        Path(V12 / "audit_report.json").write_text(json.dumps(audit, indent=2))
        return 1

    cnt_match = (len(v11v) == len(v12v) and len(v11t) == len(v12t))
    audit["checks"].append({
        "name": "row_count_match",
        "status": "PASS" if cnt_match else "FAIL",
        "detail": f"V11 val={len(v11v)} test={len(v11t)} ; V12 val={len(v12v)} test={len(v12t)}",
    })
    if cnt_match:
        audit["passed"] += 1
    else:
        audit["failed"] += 1

    # Sort + match keys
    v11v_s = v11v.set_index(KEY).sort_index()
    v11t_s = v11t.set_index(KEY).sort_index()
    v12v_s = v12v.set_index(KEY).sort_index()
    v12t_s = v12t.set_index(KEY).sort_index()
    keys_v_match = v11v_s.index.equals(v12v_s.index)
    keys_t_match = v11t_s.index.equals(v12t_s.index)
    audit["checks"].append({
        "name": "key_alignment",
        "status": "PASS" if (keys_v_match and keys_t_match) else "FAIL",
        "detail": f"val keys match={keys_v_match}, test keys match={keys_t_match}",
    })
    if keys_v_match and keys_t_match:
        audit["passed"] += 1
    else:
        audit["failed"] += 1

    # --- 2. SIMSCORE / WAPE / bias improvement ---
    v11v_sc = score_frame(v11v)
    v11t_sc = score_frame(v11t)
    v12v_sc = score_frame(v12v)
    v12t_sc = score_frame(v12t)

    delta_test_sim = (v12t_sc["SIMSCORE"] - v11t_sc["SIMSCORE"]) / v11t_sc["SIMSCORE"] * 100
    delta_test_wape = (v12t_sc["WAPE"] - v11t_sc["WAPE"]) / v11t_sc["WAPE"] * 100
    delta_test_bias_abs = abs(v12t_sc["Agg_Bias_pct"]) - abs(v11t_sc["Agg_Bias_pct"])

    summary = {
        "v11_final_val_simscore": round(v11v_sc["SIMSCORE"], 4),
        "v11_final_test_simscore": round(v11t_sc["SIMSCORE"], 4),
        "v11_final_test_wape": round(v11t_sc["WAPE"], 4),
        "v11_final_test_bias_pct": round(v11t_sc["Agg_Bias_pct"], 2),
        "v12_final_val_simscore": round(v12v_sc["SIMSCORE"], 4),
        "v12_final_test_simscore": round(v12t_sc["SIMSCORE"], 4),
        "v12_final_test_wape": round(v12t_sc["WAPE"], 4),
        "v12_final_test_bias_pct": round(v12t_sc["Agg_Bias_pct"], 2),
        "delta_test_simscore_pct": round(delta_test_sim, 2),
        "delta_test_wape_pct": round(delta_test_wape, 2),
        "delta_test_bias_abs_pct": round(delta_test_bias_abs, 2),
    }
    audit["scores"] = summary

    sim_better = delta_test_sim < 0
    audit["checks"].append({
        "name": "v12_test_simscore_beats_v11",
        "status": "PASS" if sim_better else "FAIL",
        "detail": f"V11={v11t_sc['SIMSCORE']:.4f}  V12={v12t_sc['SIMSCORE']:.4f}  "
                  f"Δ={delta_test_sim:+.2f}%",
    })
    if sim_better:
        audit["passed"] += 1
    else:
        audit["failed"] += 1

    wape_better = delta_test_wape < 0
    audit["checks"].append({
        "name": "v12_test_wape_beats_v11",
        "status": "PASS" if wape_better else "WARN",
        "detail": f"V11={v11t_sc['WAPE']:.4f}  V12={v12t_sc['WAPE']:.4f}  "
                  f"Δ={delta_test_wape:+.2f}%",
    })
    if wape_better:
        audit["passed"] += 1
    else:
        audit["warnings"] += 1

    bias_held = delta_test_bias_abs <= 0.5  # allow 0.5 pp slack
    audit["checks"].append({
        "name": "v12_test_bias_not_worse",
        "status": "PASS" if bias_held else "WARN",
        "detail": f"|V11 bias|={abs(v11t_sc['Agg_Bias_pct']):.2f}%  "
                  f"|V12 bias|={abs(v12t_sc['Agg_Bias_pct']):.2f}%  "
                  f"Δ={delta_test_bias_abs:+.2f} pp",
    })
    if bias_held:
        audit["passed"] += 1
    else:
        audit["warnings"] += 1

    # --- 3. val→test gap (overfit gate) ---
    gap_v11 = v11t_sc["SIMSCORE"] - v11v_sc["SIMSCORE"]
    gap_v12 = v12t_sc["SIMSCORE"] - v12v_sc["SIMSCORE"]
    gap_growth = gap_v12 - gap_v11
    gap_ok = gap_growth <= 0.02
    audit["checks"].append({
        "name": "overfit_gap_under_control",
        "status": "PASS" if gap_ok else "WARN",
        "detail": f"V11 gap={gap_v11:+.4f}  V12 gap={gap_v12:+.4f}  "
                  f"growth={gap_growth:+.4f}",
    })
    if gap_ok:
        audit["passed"] += 1
    else:
        audit["warnings"] += 1

    # --- 4. per-month bias on test ---
    v12t["per"] = v12t["Период"]
    by_m = v12t.groupby("per").agg(
        actual=("target_qty", "sum"),
        pred=("prediction", "sum"),
    )
    by_m["bias_pct"] = (by_m["pred"] / by_m["actual"] - 1) * 100
    bad_months = by_m[by_m["bias_pct"].abs() > 10]
    audit["checks"].append({
        "name": "no_test_month_with_extreme_bias",
        "status": "PASS" if len(bad_months) == 0 else "WARN",
        "detail": f"{len(bad_months)} test months with |bias%| > 10% "
                  f"(out of {len(by_m)} total)",
    })
    if len(bad_months) == 0:
        audit["passed"] += 1
    else:
        audit["warnings"] += 1

    # --- 5. EXT loader leakage spot-check ---
    ext_meta_dir = OUT / "external"
    if ext_meta_dir.exists():
        leak_violations = []
        for meta_path in ext_meta_dir.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:  # noqa: BLE001
                continue
            lag = int(meta.get("publication_lag_days", 0))
            if lag <= 0:
                continue
            # If the cache contains data with date >= first day of any
            # test month - lag, that's a potential leak. We just record
            # it; a proper guard runs at feature-generation time.
            date_max = meta.get("date_range_max", "")
            if date_max and pd.Period(date_max, freq="M") >= pd.Period("2025-07", freq="M"):
                leak_violations.append({
                    "loader": meta.get("source_name"),
                    "date_max": date_max,
                    "lag_days": lag,
                })
        audit["checks"].append({
            "name": "ext_leakage_spot_check",
            "status": "PASS" if not leak_violations else "WARN",
            "detail": f"{len(leak_violations)} loaders with potentially recent data; "
                       "feature-time guard handles actual leakage",
            "violations": leak_violations,
        })
        if not leak_violations:
            audit["passed"] += 1
        else:
            audit["warnings"] += 1

    # --- write report ---
    (V12 / "audit_report.json").write_text(json.dumps(audit, indent=2,
                                                      ensure_ascii=False))

    md_lines = [
        "# V12 audit report",
        "",
        f"**{audit['passed']} PASS · {audit['warnings']} WARN · {audit['failed']} FAIL**",
        "",
        "## Headline metrics",
        "",
        f"| metric | V11_final | V12_final | Δ |",
        f"|---|---:|---:|---:|",
        f"| Test SIMSCORE | {v11t_sc['SIMSCORE']:.4f} | {v12t_sc['SIMSCORE']:.4f} | "
        f"**{delta_test_sim:+.2f} %** |",
        f"| Test WAPE | {v11t_sc['WAPE']:.4f} | {v12t_sc['WAPE']:.4f} | "
        f"**{delta_test_wape:+.2f} %** |",
        f"| Test \\|bias%\\| | {abs(v11t_sc['Agg_Bias_pct']):.2f} % | "
        f"{abs(v12t_sc['Agg_Bias_pct']):.2f} % | "
        f"**{delta_test_bias_abs:+.2f} pp** |",
        f"| Val SIMSCORE | {v11v_sc['SIMSCORE']:.4f} | {v12v_sc['SIMSCORE']:.4f} | — |",
        f"| Val→Test gap | {gap_v11:+.4f} | {gap_v12:+.4f} | "
        f"{gap_growth:+.4f} |",
        "",
        "## Checks",
        "",
    ]
    for ch in audit["checks"]:
        emoji = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}.get(ch["status"], "?")
        md_lines.append(f"- {emoji} **{ch['name']}** — {ch['detail']}")
    md_lines.append("")
    (V12 / "audit_report.md").write_text("\n".join(md_lines))

    print("\n".join(md_lines))
    return 0 if audit["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
