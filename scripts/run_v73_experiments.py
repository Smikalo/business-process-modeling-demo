"""Batch driver that kicks off all pre-registered V7.3 experiments.

Candidates are **fixed before any run**.  After the sweep is finished we pick
the one with the best mean CV SIMSCORE that also passes the decision gate
(see ``scripts/decision_gate_v73.py`` rules).  No re-tuning after the fact.

Every experiment writes:
    output/cv_summary_<tag>.json   — per-fold + mean metrics
    output/decision_<tag>.json     — pass/fail vs the chosen baseline

Tier A candidates (data-efficient):
    a1_alpha_050       quantile α=0.50                            (symmetric)
    a2_alpha_055       quantile α=0.55                            (mild upward)
    a3_tweedie_p14     tweedie vp=1.4                             (less zero-heavy)
    a4_tweedie_p15_rec tweedie vp=1.5, recency γ=0.95             (recency + symmetric)
    a5_mae             regression_l1                              (pure median)
    a6_huber           huber                                      (robust)

Tier B is run separately once the Tier A winner is known.

Usage:
    python -m scripts.run_v73_experiments --baseline-cv output/cv_summary_<chosen>.json
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("v73_exp")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"

TIER_A = [
    dict(tag="a1_alpha_050",       abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "quantile", "--alpha", "0.50"]),
    dict(tag="a2_alpha_055",       abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "quantile", "--alpha", "0.55"]),
    dict(tag="a3_tweedie_p14",     abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "tweedie",
                "--tweedie-variance-power", "1.4"]),
    dict(tag="a4_tweedie_rec095",  abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "tweedie",
                "--tweedie-variance-power", "1.5",
                "--recency-gamma", "0.95"]),
    dict(tag="a5_mae",             abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "regression_l1"]),
    dict(tag="a6_huber",            abt="abt_v7_cached.parquet",
         extra=["--reg-objective", "huber"]),
]


def run(tag: str, abt: str, extra: list[str], baseline_cv: str,
        num_boost_round: int = 400) -> int:
    cmd = [
        sys.executable, "-m", "scripts.decision_gate_v73",
        "--abt-path", abt,
        "--tag", tag,
        "--num-boost-round", str(num_boost_round),
        "--baseline-cv", baseline_cv,
        *extra,
    ]
    log.info("RUN %s  (%s)", tag, " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    log.info("FIN %s  rc=%d", tag, rc)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-cv", required=True,
                    help="Path to baseline cv_summary_*.json")
    ap.add_argument("--num-boost-round", type=int, default=400)
    ap.add_argument("--only", nargs="*", default=None,
                    help="Only run these tags")
    args = ap.parse_args()

    for spec in TIER_A:
        if args.only and spec["tag"] not in args.only:
            continue
        run(spec["tag"], spec["abt"], spec["extra"], args.baseline_cv,
            args.num_boost_round)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
