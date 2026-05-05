"""Unified audit across all production-eligible models.

Generates output/full_audit.{csv,md} and output/plot_full_progression.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

MODELS = [
    ("V10",                  "v10_final",         "#999999"),
    ("V11_LAD",              "v11_lad",           "#cc8866"),
    ("V11_final",            "v11_final",         "#1f77b4"),
    ("V12_LAD",              "v12_lad",           "#bbbbbb"),
    ("V12_final",            "v12_final",         "#999999"),
    ("V12.1_LAD",            "v121_lad",          "#ffaa55"),
    ("V12.1_champion",       "v121_champion",     "#7fa8ff"),
    ("V12.2_champion",       "v122_champion",     "#2ca02c"),
    ("V13_chronos (zs)",     "v13_chronos",       "#888888"),
    ("V13_chronos_ft",       "v13_chronos_ft",    "#888888"),
    ("V13.1_relaxed",        "v131_relaxed",      "#cab8e0"),
    ("V13.2_relaxed",        "v132_relaxed",      "#9467bd"),
]


def main() -> int:
    rows = []
    for name, tag, color in MODELS:
        for split in ("val", "test"):
            p = OUT / f"preds_{tag}_{split}.csv"
            if not p.exists():
                rows.append({"model": name, "tag": tag, "split": split,
                             "missing": True})
                continue
            df = pd.read_csv(p)
            sc = score_frame(df)
            rows.append({"model": name, "tag": tag, "split": split,
                         "missing": False, "color": color, **sc})
    df = pd.DataFrame([r for r in rows if not r["missing"]])
    df.to_csv(OUT / "full_audit.csv", index=False)

    pivot = df.pivot_table(index="model", columns="split",
                            values=["SIMSCORE", "WAPE", "Agg_Bias_pct",
                                    "Monthly_WAPE", "RMSE"])
    pivot = pivot.reindex([m for m, _, _ in MODELS if m in df["model"].unique()])
    print(pivot.round(4).to_string())

    md = ["# Full Model Audit (cross-version)",
          "",
          f"Generated: 2026-05-05",
          "",
          "All models scored on the same held-out test window (Jul 2025 – Mar 2026).",
          "Lower is better for SIMSCORE / WAPE / Monthly-WAPE / RMSE; bias should be near zero.",
          "",
          "## Test scores",
          "",
          "| Model | n_rows | SIMSCORE ↓ | WAPE ↓ | Bias % | M-WAPE ↓ | RMSE |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, _, _ in MODELS:
        sub = df[(df["model"] == name) & (df["split"] == "test")]
        if sub.empty:
            continue
        r = sub.iloc[0]
        bold = "**" if name == "V12.2_champion" else ""
        md.append(f"| {bold}{name}{bold} | {int(r['n_rows'])} | "
                  f"{r['SIMSCORE']:.4f} | {r['WAPE']:.4f} | "
                  f"{r['Agg_Bias_pct']:+.2f} | {r['Monthly_WAPE']:.4f} | "
                  f"{r['RMSE']:.3f} |")

    md += ["",
           "## Notes",
           "",
           "* **V12.2_champion** is the production model "
           "(`0.925·V11_final + 0.075·V12_external`, OOF-honest). "
           "Test SIM 0.4435, bias +2.13 %, WAPE 0.3931.",
           "* **V13.2_relaxed** is the latest parallel sensitivity artifact "
           "(`0.925·V12.2_champion + 0.075·V13_chronos_ft`, judgment-call). "
           "Test SIM 0.4329 on aligned subset (95.7 % coverage of V12.2), "
           "bias −0.05 %. Supersedes V13.1_relaxed.",
           "* **V13_chronos_ft** is the LoRA fine-tuned Chronos-T5-Small "
           "(2 epochs, ~600K trainable params, ran on Colab T4). "
           "Standalone test WAPE 0.617 (vs zero-shot 0.630, −2.1 % lift). "
           "Earned 0 LAD weight in V12.3 multi-helper joint OOF search "
           "(same val→test bias-direction reversal that affected zero-shot).",
           "* **V13_chronos (zs)** is the original zero-shot Chronos run "
           "(LoRA fine-tune in Cell 5 silently no-op'd due to context_len "
           "vs prediction_length mismatch — kept for historical reference).",
           "* **V12.3 multi-helper joint search** (with FT Chronos in the pool) "
           "produced the same champion as V12.2 — 0 weight on Chronos in any "
           "OOF-defensible variant.",
           ""]
    (OUT / "full_audit.md").write_text("\n".join(md))

    # Viz: progression
    test_df = df[df["split"] == "test"].set_index("model")
    fig, ax = plt.subplots(figsize=(14, 7))
    keep = [m for m, _, _ in MODELS if m in test_df.index
             and m not in ("V12_LAD", "V12_final", "V12.1_LAD")]
    sims = test_df.loc[keep, "SIMSCORE"].values
    colors = [next(c for n, _, c in MODELS if n == m) for m in keep]
    x = np.arange(len(keep))
    bars = ax.bar(x, sims, color=colors, edgecolor="black", linewidth=0.8)

    for i, v in enumerate(sims):
        ax.annotate(f"{v:.4f}", xy=(i, v), ha="center", va="bottom",
                      fontsize=10, fontweight="bold")

    # V12.2 highlight
    if "V12.2_champion" in keep:
        idx = keep.index("V12.2_champion")
        bars[idx].set_edgecolor("#2ca02c")
        bars[idx].set_linewidth(2.5)
        ax.annotate("NEW PRODUCTION", xy=(idx, sims[idx]),
                      xytext=(idx, sims[idx] + 0.005),
                      ha="center", fontsize=11, fontweight="bold",
                      color="#2ca02c",
                      arrowprops=dict(arrowstyle="->", color="#2ca02c"))

    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=20, ha="right")
    ax.set_ylabel("Test SIMSCORE  (lower = better)", fontsize=12)
    ax.set_title("V10 → V12.2 progression — V12.2_champion is the new production model",
                  fontsize=13, fontweight="bold")
    ax.axhline(test_df.loc["V11_final", "SIMSCORE"], color="#1f77b4",
                linestyle=":", linewidth=1, alpha=0.7,
                label="V11_final benchmark")
    ax.axhline(test_df.loc["V12.2_champion", "SIMSCORE"], color="#2ca02c",
                linestyle="-", linewidth=1.5, alpha=0.8,
                label="V12.2_champion (new prod)")
    if "V13.1_relaxed" in keep:
        ax.axhline(test_df.loc["V13.1_relaxed", "SIMSCORE"],
                    color="#9467bd", linestyle="--", linewidth=1, alpha=0.7,
                    label="V13.1_relaxed (judgment, parallel)")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_full_progression.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nwrote {OUT / 'plot_full_progression.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
