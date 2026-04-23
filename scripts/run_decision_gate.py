"""Decision gate report — consolidates ablation evidence into keep/drop verdicts.

Gate rules (conservative):
    PASS       : val_WAPE_delta <= -0.005 AND test_WAPE_delta <= +0.005
    MARGINAL   : val_WAPE_delta <  0      AND test_WAPE_delta <= +0.010
    LOO_KEEP   : LOO run shows test regresses by >= +0.003 when signal removed
    FAIL       : otherwise

Sources flagged PASS, MARGINAL, or LOO_KEEP enter the V5 candidate set.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "output" / "ablation_results.csv"
OUT_MD = ROOT / "output" / "decision_gate.md"
OUT_CSV = ROOT / "output" / "decision_gate.csv"


def classify(row: dict, loo_test_regression: float | None) -> str:
    val_d = row.get("val_WAPE_delta")
    test_d = row.get("test_WAPE_delta")
    if val_d is None or test_d is None:
        return "UNKNOWN"
    if val_d <= -0.005 and test_d <= 0.005:
        return "PASS"
    if val_d < 0 and test_d <= 0.010:
        return "MARGINAL"
    if loo_test_regression is not None and loo_test_regression >= 0.003:
        return "LOO_KEEP"
    return "FAIL"


def main() -> int:
    df = pd.read_csv(ABL)
    addone = df[df["mode"] == "add_one"].copy()
    addone = addone[addone["source"] != "baseline"]

    latest = addone.sort_values("run_utc").groupby("source").tail(1).reset_index(drop=True)

    loo = df[df["mode"] == "loo"].copy()
    loo_map: dict[str, float] = {}
    for _, r in loo.iterrows():
        if r["source"] == "__all__":
            continue
        loo_map[r["source"]] = float(r.get("test_WAPE_loss") or 0.0)

    latest["loo_test_loss"] = latest["source"].map(loo_map).fillna(0.0)
    latest["verdict"] = latest.apply(
        lambda r: classify(r.to_dict(), r["loo_test_loss"]), axis=1
    )

    keep_mask = latest["verdict"].isin(["PASS", "MARGINAL", "LOO_KEEP"])
    keep_sources = sorted(latest.loc[keep_mask, "source"].tolist())
    drop_sources = sorted(latest.loc[~keep_mask, "source"].tolist())

    report = latest[
        [
            "source",
            "val_WAPE_delta",
            "test_WAPE_delta",
            "loo_test_loss",
            "verdict",
        ]
    ].sort_values(["verdict", "test_WAPE_delta"]).reset_index(drop=True)

    report.to_csv(OUT_CSV, index=False)

    lines = [
        "# External signals — decision gate report",
        "",
        "## Gate definitions",
        "- **PASS**: `val_WAPE_delta ≤ -0.005` AND `test_WAPE_delta ≤ +0.005`",
        "- **MARGINAL**: `val_WAPE_delta < 0` AND `test_WAPE_delta ≤ +0.010`",
        "- **LOO_KEEP**: LOO shows `test_WAPE_loss ≥ +0.003` when signal removed",
        "- **FAIL**: otherwise",
        "",
        "## Per-source verdict",
        "",
        "| source | val Δ WAPE | test Δ WAPE | LOO test loss | verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in report.iterrows():
        lines.append(
            f"| {r['source']} | {r['val_WAPE_delta']:+.4f} | "
            f"{r['test_WAPE_delta']:+.4f} | {r['loo_test_loss']:+.4f} | "
            f"**{r['verdict']}** |"
        )
    lines += [
        "",
        "## V5 candidate set",
        "",
        "**Keep** (" + str(len(keep_sources)) + "): " + ", ".join(f"`{s}`" for s in keep_sources),
        "",
        "**Drop** (" + str(len(drop_sources)) + "): " + ", ".join(f"`{s}`" for s in drop_sources),
        "",
        "## Notes",
        "- Validation = 2025-06 .. 2025-11; test = 2025-12 .. 2026-02.",
        "- Deltas use the freshest baseline per run_utc (so baseline swings from stochasticity don't contaminate the verdict).",
        "- LOO is computed from the 'all signals in' model by dropping one loader's columns at a time.",
    ]

    OUT_MD.write_text("\n".join(lines))
    print(f"Decision gate written → {OUT_MD}")
    print(f"Keep: {keep_sources}")
    print(f"Drop: {drop_sources}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
