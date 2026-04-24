"""Rolling-origin CV of the top stacker candidates within the VAL window.

Protects against overfitting where a calibrator is fit and evaluated on the
same data.  Each of the 4 CV folds below splits the val window (2024-07 …
2025-06) into a calibration segment (everything before) and a held-out 3-month
block:

* fold 1 — calibrate on 2024-07…2024-09, eval on 2024-10…2024-12
* fold 2 — calibrate on 2024-07…2024-12, eval on 2025-01…2025-03
* fold 3 — calibrate on 2024-07…2025-03, eval on 2025-04…2025-06
* fold 4 — calibrate on all of val, eval on val (in-sample; for reference only)

For each candidate we report:

*  mean out-of-fold SIMSCORE across folds 1-3
*  in-sample vs out-of-fold gap (detects overfitting)

Writes ``output/v73/sweep_cv.csv``.

Final pick rule: lowest mean OOF SIMSCORE with gap ≤ 0.05.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from scripts.score_similarity import score_frame

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
V73 = OUT / "v73"
V73.mkdir(parents=True, exist_ok=True)
KEY = ["Период", "Партнер", "Артикул"]
BASE = ["v4", "v5", "v6", "v7", "v71", "v72_champion"]


def _wide(split: str) -> pd.DataFrame:
    out = None
    for t in BASE:
        p = OUT / f"preds_{t}_{split}.csv"
        d = pd.read_csv(p)
        d = d[KEY + ["target_qty", "prediction"]].rename(columns={"prediction": t})
        out = d if out is None else out.merge(
            d.drop(columns=["target_qty"]), on=KEY, how="inner")
    out = out.rename(columns={"target_qty": "y"})
    return out


def _score_sub(df: pd.DataFrame, pred: np.ndarray) -> dict:
    o = df[KEY].copy()
    o["target_qty"] = df["y"].to_numpy()
    o["prediction"] = np.clip(pred, 0, None)
    return score_frame(o)


def candidate_bare(train_df, test_df, base: str):
    return test_df[base].to_numpy()


def candidate_scalar(train_df, test_df, base: str):
    s = float(train_df["y"].sum() / train_df[base].sum())
    return test_df[base].to_numpy() * s


def candidate_monthly(train_df, test_df, base: str):
    tr = train_df.copy()
    tr["m"] = pd.PeriodIndex(tr["Период"].astype(str), freq="M").month
    mscale = (tr.groupby("m")["y"].sum() / tr.groupby("m")[base].sum()).to_dict()
    te = test_df.copy()
    te["m"] = pd.PeriodIndex(te["Период"].astype(str), freq="M").month
    fallback = float(tr["y"].sum() / tr[base].sum())
    return te[base].to_numpy() * te["m"].map(mscale).fillna(fallback).to_numpy()


def candidate_blend(train_df, test_df, a: str, b: str):
    best_w, best_s = 0.5, np.inf
    for w in np.arange(0.0, 1.01, 0.05):
        p = w * train_df[a].to_numpy() + (1 - w) * train_df[b].to_numpy()
        r = _score_sub(train_df, p)
        if r["SIMSCORE"] < best_s:
            best_s, best_w = r["SIMSCORE"], w
    return best_w * test_df[a].to_numpy() + (1 - best_w) * test_df[b].to_numpy()


def candidate_nnls(train_df, test_df, cols: list[str]):
    X = train_df[cols].to_numpy()
    y = train_df["y"].to_numpy()
    coef, _ = nnls(X, y)
    if coef.sum() > 0:
        coef = coef / coef.sum()
    return test_df[cols].to_numpy() @ coef


CANDIDATES = {
    "bare_v5":                   lambda tr, te: candidate_bare(tr, te, "v5"),
    "bare_v6":                   lambda tr, te: candidate_bare(tr, te, "v6"),
    "bare_v72_champion":         lambda tr, te: candidate_bare(tr, te, "v72_champion"),
    "scalar_v72_champion":       lambda tr, te: candidate_scalar(tr, te, "v72_champion"),
    "scalar_v6":                 lambda tr, te: candidate_scalar(tr, te, "v6"),
    "monthly_v72_champion":      lambda tr, te: candidate_monthly(tr, te, "v72_champion"),
    "monthly_v6":                lambda tr, te: candidate_monthly(tr, te, "v6"),
    "monthly_v7":                lambda tr, te: candidate_monthly(tr, te, "v7"),
    "blend_v6_v72":              lambda tr, te: candidate_blend(tr, te, "v6", "v72_champion"),
    "blend_v5_v72":              lambda tr, te: candidate_blend(tr, te, "v5", "v72_champion"),
    "nnls_all":                  lambda tr, te: candidate_nnls(tr, te, BASE),
    "nnls_symmetric":            lambda tr, te: candidate_nnls(tr, te, ["v4", "v5", "v6"]),
}


def fold_splits():
    return [
        (pd.Period("2024-07", "M"), pd.Period("2024-09", "M"),
         pd.Period("2024-10", "M"), pd.Period("2024-12", "M")),
        (pd.Period("2024-07", "M"), pd.Period("2024-12", "M"),
         pd.Period("2025-01", "M"), pd.Period("2025-03", "M")),
        (pd.Period("2024-07", "M"), pd.Period("2025-03", "M"),
         pd.Period("2025-04", "M"), pd.Period("2025-06", "M")),
    ]


def main() -> int:
    val = _wide("val")
    val["Период_p"] = pd.PeriodIndex(val["Период"].astype(str), freq="M")

    rows = []
    for name, fn in CANDIDATES.items():
        per_fold_sim = []
        for (tr_s, tr_e, va_s, va_e) in fold_splits():
            tr = val[(val["Период_p"] >= tr_s) & (val["Период_p"] <= tr_e)]
            te = val[(val["Период_p"] >= va_s) & (val["Период_p"] <= va_e)]
            pred = fn(tr, te)
            s = _score_sub(te, pred)
            per_fold_sim.append(s["SIMSCORE"])

        in_pred = fn(val, val)
        in_sim = _score_sub(val, in_pred)["SIMSCORE"]

        mean_oof = float(np.mean(per_fold_sim))
        rows.append({
            "candidate": name,
            "OOF_SIMSCORE_mean": round(mean_oof, 4),
            "OOF_f1": round(per_fold_sim[0], 4),
            "OOF_f2": round(per_fold_sim[1], 4),
            "OOF_f3": round(per_fold_sim[2], 4),
            "in_sample_SIMSCORE": round(in_sim, 4),
            "overfit_gap": round(mean_oof - in_sim, 4),
        })

    df = pd.DataFrame(rows).sort_values("OOF_SIMSCORE_mean")
    print("\n=== sweep CV (val internal rolling-origin) ===")
    print(df.to_string(index=False))
    df.to_csv(V73 / "sweep_cv.csv", index=False)

    eligible = df[df["overfit_gap"] <= 0.05]
    champ = (eligible.iloc[0] if len(eligible) else df.iloc[0]).to_dict()
    print(f"\nCV champion (OOF SIMSCORE, gap≤0.05): {champ['candidate']} "
          f"OOF={champ['OOF_SIMSCORE_mean']:.4f}  gap={champ['overfit_gap']:.4f}")
    (V73 / "sweep_cv_champion.txt").write_text(champ["candidate"] + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
