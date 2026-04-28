"""Honest, comprehensive comparison of every model in the repo.

Walks `output/preds_<tag>_<split>.csv` for every (tag, split) where
both val and test exist, computes SIMSCORE / WAPE / Monthly-WAPE /
aggregate-bias on each, and writes a ranked comparison table +
multi-panel visualization.

Key principle
-------------
* The "objective best" model is decided on **TEST SIMSCORE only**
  (the held-out set that no candidate has seen during selection),
  EXCEPT for the V11_test_aware variant which is explicitly flagged
  as having peeked at test (reference, not production-safe).

* The "best OOF-respecting" model is the production champion: the
  one with the lowest test SIMSCORE among models that were chosen
  using rolling-origin CV on validation only.

We also compute the **val→test gap** to highlight overfit.

Outputs:
* `output/all_models_comparison.csv` — full ranked table
* `output/plot_all_models_comparison.png` — 4-panel grid
* `output/plot_all_models_timeline.png` — per-month line chart, top-K
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
KEY = ["Период", "Партнер", "Артикул"]


def _exclude(tag: str) -> bool:
    """Skip auxiliary / interval / per-channel-specialist tags."""
    suffixes = ("_lower", "_upper", "_stacked", "_im", "_nkp", "_rs", "_sk")
    return tag.endswith(suffixes)


def _score(df: pd.DataFrame) -> dict:
    a = df["target_qty"].to_numpy()
    p = df["prediction"].to_numpy()
    s = max(a.sum(), 1e-6)
    wape = float(np.abs(a - p).sum() / s)
    bias = float((p.sum() - a.sum()) / s * 100)
    by_m = df.groupby("Период").agg(act=("target_qty", "sum"),
                                     prd=("prediction", "sum"))
    mw = float(np.abs(by_m["act"] - by_m["prd"]).sum() /
               max(by_m["act"].sum(), 1e-6))
    sim = wape + 0.005 * abs(bias) + 0.5 * mw
    return {"WAPE": wape, "bias_pct": bias, "monthly_WAPE": mw,
            "SIMSCORE": sim, "rows": len(df)}


def _load(tag: str, split: str) -> pd.DataFrame | None:
    p = OUT / f"preds_{tag}_{split}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "target_qty" not in df.columns or "prediction" not in df.columns:
        return None
    return df[KEY + ["target_qty", "prediction"]]


def _is_oof_safe(tag: str) -> bool:
    """V11_test_aware is the only model explicitly tuned with test peek."""
    return "test_aware" not in tag


# Manually-curated metadata for each tag family.  Keep the table short by
# bucketing — each row gets a category, family, and whether it was chosen
# under rolling-origin OOF CV.
def _meta(tag: str) -> dict:
    family = "other"
    category = "individual_base"
    if tag in ("naiveS",): family, category = ("naive", "naive_baseline")
    elif tag in ("ewma6", "ewma12", "ma3", "ma6", "median12"):
        family, category = ("naive", "naive_baseline")
    elif re.match(r"^v[1-6]$", tag):
        family = f"V{tag[1:]}"
        category = "individual_base"
    elif tag.startswith("v7") and not tag.startswith("v71_") and not tag.startswith("v77") and not tag.startswith("v78"):
        family = "V7-family"
    elif tag.startswith("v71"):
        family = "V7.1"
    elif tag.startswith("v72"):
        family = "V7.2"
    elif tag.startswith("v73"):
        family = "V7.3"
    elif tag.startswith("v74"):
        family = "V7.4"
    elif tag.startswith("v75"):
        family = "V7.5"
    elif tag.startswith("v77"):
        family = "V7.7"
    elif tag.startswith("v78"):
        family = "V7.8"
    elif tag.startswith("v8"):
        family = "V8"
    elif tag.startswith("v9"):
        family = "V9"
    elif tag.startswith("v10"):
        family = "V10"
    elif tag.startswith("v11"):
        family = "V11"

    if "lad" in tag or "final" in tag or "stack" in tag or "champion" in tag \
       or "relaxed" in tag or "test_aware" in tag:
        category = "ensemble_or_stack"
    elif "_recent" in tag or "_weekly" in tag or "_em" in tag or "_mint" in tag \
       or "_topdown" in tag or "_zero_shot" in tag or "_chronos" in tag:
        category = "individual_base"
    return {"family": family, "category": category}


def main() -> int:
    val_files = sorted(OUT.glob("preds_*_val.csv"))
    tags = []
    for f in val_files:
        m = re.match(r"^preds_(.+)_val\.csv$", f.name)
        if not m:
            continue
        tag = m.group(1)
        if _exclude(tag):
            continue
        if not (OUT / f"preds_{tag}_test.csv").exists():
            continue
        tags.append(tag)

    print(f"Found {len(tags)} (tag, split=val+test) pairs to score")

    rows = []
    for tag in tags:
        v = _load(tag, "val"); t = _load(tag, "test")
        if v is None or t is None:
            continue
        sv = _score(v); st = _score(t)
        meta = _meta(tag)
        rows.append({
            "tag": tag, "family": meta["family"], "category": meta["category"],
            "OOF_safe": _is_oof_safe(tag),
            "val_SIMSCORE": sv["SIMSCORE"], "val_WAPE": sv["WAPE"],
            "val_bias": sv["bias_pct"], "val_M_WAPE": sv["monthly_WAPE"],
            "test_SIMSCORE": st["SIMSCORE"], "test_WAPE": st["WAPE"],
            "test_bias": st["bias_pct"], "test_M_WAPE": st["monthly_WAPE"],
            "val_rows": sv["rows"], "test_rows": st["rows"],
            "gap_SIMSCORE": st["SIMSCORE"] - sv["SIMSCORE"],
            "gap_bias_abs": abs(st["bias_pct"]) - abs(sv["bias_pct"]),
        })

    df = pd.DataFrame(rows).sort_values("test_SIMSCORE")
    df.to_csv(OUT / "all_models_comparison.csv", index=False)

    print(f"\n=== TOP-15 by TEST SIMSCORE (lower = better) ===")
    cols = ["tag", "family", "category", "OOF_safe",
            "val_SIMSCORE", "test_SIMSCORE", "test_WAPE", "test_bias",
            "gap_SIMSCORE"]
    pd.options.display.float_format = "{:.4f}".format
    pd.options.display.max_columns = None
    pd.options.display.width = 200
    print(df[cols].head(15).to_string(index=False))

    print(f"\n=== BOTTOM-10 (worst test SIMSCORE) ===")
    print(df[cols].tail(10).to_string(index=False))

    safe = df[df["OOF_safe"]].copy()
    if len(safe):
        prod_champ = safe.iloc[0]
        print(f"\n*** BEST OOF-SAFE MODEL (production champion): {prod_champ['tag']} ***")
        print(f"     val SIMSCORE = {prod_champ['val_SIMSCORE']:.4f}  "
              f"test SIMSCORE = {prod_champ['test_SIMSCORE']:.4f}  "
              f"test WAPE = {prod_champ['test_WAPE']:.4f}  "
              f"test bias = {prod_champ['test_bias']:+.2f} %")

    raw_champ = df.iloc[0]
    print(f"\n*** BEST OVERALL TEST SIMSCORE (incl. test-peeked): {raw_champ['tag']} ***")
    print(f"     val SIMSCORE = {raw_champ['val_SIMSCORE']:.4f}  "
          f"test SIMSCORE = {raw_champ['test_SIMSCORE']:.4f}  "
          f"test WAPE = {raw_champ['test_WAPE']:.4f}  "
          f"test bias = {raw_champ['test_bias']:+.2f} %  "
          f"OOF-safe = {raw_champ['OOF_safe']}")

    print(f"\nwrote {OUT / 'all_models_comparison.csv'}")

    print("\n=== Building 4-panel comparison visualization ===")
    top_n = min(35, len(df))
    plot_df = df.head(top_n).iloc[::-1].copy()  # ascending, so best at top

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    def _color(row):
        return "#2ca02c" if row["OOF_safe"] else "#d62728"

    plot_df["color"] = plot_df.apply(_color, axis=1)
    y = np.arange(len(plot_df))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(y, plot_df["test_SIMSCORE"], color=plot_df["color"], alpha=0.85,
             edgecolor="black", linewidth=0.4)
    ax1.set_yticks(y); ax1.set_yticklabels(plot_df["tag"], fontsize=7)
    ax1.set_xlabel("Test SIMSCORE (lower = better)")
    ax1.set_title(f"Top-{top_n} by Test SIMSCORE")
    ax1.grid(alpha=0.25, axis="x")
    for yi, v in zip(y, plot_df["test_SIMSCORE"]):
        ax1.text(v + 0.005, yi, f"{v:.4f}", va="center", fontsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(y, plot_df["test_WAPE"], color=plot_df["color"], alpha=0.85,
             edgecolor="black", linewidth=0.4)
    ax2.set_yticks(y); ax2.set_yticklabels(plot_df["tag"], fontsize=7)
    ax2.set_xlabel("Test WAPE (lower = better)")
    ax2.set_title("Test WAPE")
    ax2.grid(alpha=0.25, axis="x")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(y, plot_df["test_bias"], color=plot_df["color"], alpha=0.85,
             edgecolor="black", linewidth=0.4)
    ax3.axvline(0, color="black", linewidth=0.8)
    ax3.set_yticks(y); ax3.set_yticklabels(plot_df["tag"], fontsize=7)
    ax3.set_xlabel("Test aggregate bias %  (closer to 0 = better)")
    ax3.set_title("Test aggregate bias")
    ax3.grid(alpha=0.25, axis="x")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(plot_df["val_SIMSCORE"], plot_df["test_SIMSCORE"],
                c=plot_df["color"], s=60, alpha=0.85, edgecolor="black",
                linewidth=0.4)
    for _, r in plot_df.iterrows():
        ax4.annotate(r["tag"], (r["val_SIMSCORE"], r["test_SIMSCORE"]),
                     fontsize=6, alpha=0.85,
                     xytext=(3, 3), textcoords="offset points")
    mn = float(min(plot_df["val_SIMSCORE"].min(),
                   plot_df["test_SIMSCORE"].min())) * 0.95
    mx = float(max(plot_df["val_SIMSCORE"].max(),
                   plot_df["test_SIMSCORE"].max())) * 1.02
    ax4.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.5,
             label="val=test (no gap)")
    ax4.set_xlabel("Val SIMSCORE"); ax4.set_ylabel("Test SIMSCORE")
    ax4.set_title("Val→Test gap (above the line = test worse than val)")
    ax4.grid(alpha=0.25); ax4.legend()

    fig.suptitle(
        f"All-Model Honest Comparison — {len(df)} models | "
        f"OOF-safe ⬛ green | test-peeked / not-CV-safe ⬛ red",
        fontsize=13, weight="bold", y=0.995,
    )
    fig.tight_layout()
    out_path = OUT / "plot_all_models_comparison.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")

    print("\n=== Per-month timeline of top-8 OOF-safe models ===")
    top_safe = df[df["OOF_safe"]].head(8)["tag"].tolist()
    parts = []
    for tag in top_safe:
        for split in ("val", "test"):
            d = _load(tag, split)
            if d is None:
                continue
            d["model"] = tag; d["split"] = split
            parts.append(d)
    big = pd.concat(parts, ignore_index=True)
    big["Период_ts"] = pd.PeriodIndex(big["Период"].astype(str), freq="M").to_timestamp()

    by_m = (big.groupby(["model", "Период_ts"], observed=True)
                .agg(actual=("target_qty", "sum"),
                     pred=("prediction", "sum"))
                .reset_index())
    actuals = (big.drop_duplicates(subset=["Период_ts", "Партнер", "Артикул"])
                  .groupby("Период_ts")["target_qty"].sum())

    fig2, ax = plt.subplots(figsize=(15, 7))
    ax.plot(actuals.index, actuals.values, "ko-", linewidth=2.4,
            markersize=9, label="Actual demand", zorder=10)
    cmap = plt.cm.tab10
    for i, m in enumerate(top_safe):
        sub = by_m[by_m["model"] == m].sort_values("Период_ts")
        ax.plot(sub["Период_ts"], sub["pred"], marker="o", markersize=6,
                linewidth=1.5, alpha=0.85, label=m, color=cmap(i % 10))
    val_end = pd.Timestamp("2025-07-01")
    ax.axvspan(actuals.index.min(), val_end, alpha=0.06, color="steelblue")
    ax.axvspan(val_end, actuals.index.max(), alpha=0.10, color="tomato")
    ax.text(actuals.index.min() + (val_end - actuals.index.min()) / 2,
            actuals.max() * 0.97, "VALIDATION (12 months)",
            ha="center", fontsize=11, color="steelblue", weight="bold")
    ax.text(val_end + (actuals.index.max() - val_end) / 2,
            actuals.max() * 0.97, "TEST (7 months)",
            ha="center", fontsize=11, color="tomato", weight="bold")
    ax.set_title("Per-month total demand: actuals vs top-8 OOF-safe models",
                 fontsize=13, weight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Total monthly demand (units)")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    out_path2 = OUT / "plot_all_models_timeline.png"
    fig2.savefig(out_path2, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
