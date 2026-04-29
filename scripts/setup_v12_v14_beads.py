"""Set up beads tickets for the V12-V14 free-budget roadmap.

Creates the full ticket graph:
  1  ROOT epic
  5  Phase epics (Day 0, Week 1, Week 2, Week 3, Week 4)
  ~75 task tickets organized into daily waves with proper deps

Idempotent: if the script has run before (the root epic already exists with
label `v12-v14-roadmap-rev1`), it skips re-creation and just prints the graph.

After creation, validates the graph is a DAG (no cycles), prints the
parallel-wave structure, and lists every HUMAN-action ticket so the user
knows when their click is needed.

Usage:
    python -m scripts.setup_v12_v14_beads             # dry-run preview
    python -m scripts.setup_v12_v14_beads --create    # actually create
    python -m scripts.setup_v12_v14_beads --validate  # only validate existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field

# ---------------------------------------------------------------- TICKET DATA

@dataclass
class Ticket:
    key: str
    title: str
    type: str = "task"          # task | epic | feature | decision
    priority: int = 2           # 0=critical .. 4=backlog
    parent: str | None = None   # parent ticket KEY (becomes --parent <id>)
    labels: list[str] = field(default_factory=list)
    estimate: int = 0           # minutes
    description: str = ""
    design: str = ""
    acceptance: str = ""
    deps: list[str] = field(default_factory=list)  # internal keys of blockers
    defer: str = ""             # bd --defer value (e.g. '+90d') to hide from ready


# ============== ROOT + PHASE EPICS ==============

EPICS: list[Ticket] = [
    Ticket(
        key="ROOT",
        title="V12-V14 free-budget roadmap (root)",
        type="epic", priority=1,
        labels=["v12-v14", "roadmap-rev1"],
        description=(
            "Top-level epic tracking the V12 → V13 → V14 model improvement "
            "campaign. Strict zero-dollar / no-card-on-file constraint. Goal: "
            "drive test SIMSCORE from V11_final=0.4447 toward 0.420 by adding "
            "fine-tuned foundation models, a global neural model with embeddings, "
            "intermittent-demand specialists, MoE pattern routing, plus new "
            "external signals (Google Trends, school cal, payday, UN Comtrade)."
        ),
        design=(
            "Plan in docs/v12_v14_roadmap.md (Draft 3). Parallel subagent "
            "execution on single main branch with strict file-ownership map. "
            "5 subagents in flight max. Daily LOG.md commits. CI OOF audit on "
            "every commit."
        ),
        acceptance=(
            "Tag v14.0.0 on main; output/final/champion_card.json shows V14 "
            "champion; OOF SIMSCORE val ≤ 0.345; test SIMSCORE ≤ 0.420; CI green."
        ),
    ),
    Ticket(
        key="P0",
        title="Phase 0 — Day 0: restructure, CI, docs, scaffolding",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase0", "day0"],
        description=(
            "Foundation work before any modeling: restructure output/ folder, "
            "set up GitHub Actions OOF audit, initialize LOG.md, build "
            "champion_card.json, patch existing scripts, write Makefile."
        ),
        acceptance=(
            "output/ restructured; CI green on first run; champion_card.json "
            "reflects V11_final; Makefile targets v12/v13/v14 callable; "
            "v11.1-restructure tag pushed."
        ),
    ),
    Ticket(
        key="P1",
        title="Phase 1 — Week 1 (Days 1-7): V12 dash",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase1", "week1"],
        description=(
            "5 parallel waves of CPU work: external feature ingest, V11 multi-seed "
            "bagging, Croston/SBA/TSB specialist, V12 ABT build, V12 base trainings, "
            "anomaly downweighting, V12 LAD search w/ bias ladder. Yields V12.5 "
            "checkpoint Day 4 + V12.0 final Day 7."
        ),
        acceptance=(
            "v12.5.0 + v12.0.0 tags; OOF SIMSCORE val ≤ 0.355; test reported."
        ),
    ),
    Ticket(
        key="P2",
        title="Phase 2 — Week 2 (Days 8-14): foundation models fine-tuned",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase2", "week2"],
        description=(
            "Triple-track FM fine-tune (Chronos + TimesFM + Moirai) × 2 seeds = "
            "6 GPU runs across Colab + Kaggle. Each FM run is HUMAN-clicked; "
            "agents prepare notebooks and bag predictions. V13.5 checkpoint Day 12, "
            "V13.0 Day 14."
        ),
        acceptance=(
            "v13.5.0 + v13.0.0 tags; ≥ 1 fine-tuned FM earns non-zero LAD weight; "
            "OOF SIMSCORE val ≤ 0.350."
        ),
    ),
    Ticket(
        key="P3",
        title="Phase 3 — Week 3 (Days 15-21): GlobalNN with embeddings",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase3", "week3"],
        description=(
            "Build Transformer-encoder GlobalNN with partner/SKU/channel/brand "
            "embeddings, quantile head, pinball+Tweedie loss. Train on Colab "
            "(4 sessions × 4 hr). 3-attempt rule then fall back to MLP. Yields "
            "V14_alpha checkpoint Day 19."
        ),
        acceptance=(
            "v14.0.0-alpha tag; GlobalNN OOF SIMSCORE alone ≤ 0.50; earns LAD weight."
        ),
    ),
    Ticket(
        key="P4",
        title="Phase 4 — Week 4 (Days 22-28): MoE + final V14",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase4", "week4"],
        description=(
            "Per-cluster (smooth/intermittent/lumpy/erratic) specialists, soft "
            "gate, then full V14_final LAD search over the everything-pool with "
            "per-month seasonal blend + streaming calibrator on top. Comprehensive "
            "viz, contribution attribution, final report."
        ),
        acceptance=(
            "v14.0.0 tag; V14_final test SIMSCORE ≤ 0.420; full ablation chart; "
            "executive summary committed."
        ),
    ),
    Ticket(
        key="P_EXT",
        title="Phase EXT — Extended free open-data ingest (parallel sub-epic)",
        type="epic", priority=1, parent="ROOT",
        labels=["v12-v14", "phase-ext", "open-data"],
        description=(
            "Parallel sub-epic that fans out 31 free open-data ingest tickets "
            "(UA macro, war-regional, attention, climate-calendar, logistics) "
            "via 5 forager subagents, then runs a per-source A/B audit and "
            "merges only A/B-winning sources into abt_v12_external. Pure CPU; "
            "competes with no GPU resources. Combined expected lift: +2-4 p.p. "
            "test SIMSCORE."
        ),
        design=(
            "Plan in docs/v12_v14_extended_open_data.md. All sources free, "
            "publication-lag declared, leakage-guarded. Each loader subclasses "
            "BaseSignalLoader at src/loaders/<name>.py."
        ),
        acceptance=(
            "≥ 8 survivors after A/B audit; abt_v12_external.parquet rebuilt "
            "with survivors merged; tests/test_v12_abt_no_leakage.py passes; "
            "ext_attribution.md shows ≥ +0.5 % val SIMSCORE vs V11 baseline."
        ),
    ),
]

# ============== PHASE 0 — DAY 0 ==============

P0_TASKS: list[Ticket] = [
    Ticket(
        key="T0_1", parent="P0",
        title="Restructure output/ into preds/, models/, plots/, grids/, audits/, archive/",
        priority=1, estimate=60,
        labels=["phase0", "day0", "subagent-parent", "blocking", "wave-day0-A"],
        description=(
            "Build scripts/restructure_output.py that uses git mv to relocate "
            "all preds_*.csv into output/preds/v{08_to_v11,v12,v13,v14,naive}/, "
            "model artifacts to output/models/v{NN}/, plots to output/plots/, "
            "grids/audits/archive accordingly. Non-destructive; preserves git "
            "history."
        ),
        design=(
            "Use a manifest mapping old→new path. For each entry: git mv if "
            "tracked else regular mv. Update .gitignore if needed."
        ),
        acceptance=(
            "After running: output/preds/, output/models/, output/plots/, "
            "output/grids/, output/audits/, output/archive/, output/final/ all "
            "exist with correct contents. `git status` clean. All existing "
            "scripts referencing output/preds_* must still resolve."
        ),
    ),
    Ticket(
        key="T0_2", parent="P0",
        title="Patch existing training & viz scripts to write to new output paths",
        priority=1, estimate=45, deps=["T0_1"],
        labels=["phase0", "day0", "subagent-parent", "wave-day0-B"],
        description=(
            "Search every scripts/*.py for `output/preds_` and update to "
            "`output/preds/v{NN}/preds_`. Same for grids and plots. Add a single "
            "src/paths.py helper module exposing PREDS_DIR(version), "
            "GRIDS_DIR(version), PLOTS_DIR()."
        ),
        acceptance=(
            "rg -n 'output/preds_' scripts/ returns no matches; existing scripts "
            "still produce identical predictions; v11_lad_stack.py reproduces "
            "V11_final unchanged."
        ),
    ),
    Ticket(
        key="T0_3", parent="P0",
        title="Build scripts/audit_champion_regression.py + champion_card schema",
        priority=1, estimate=60, deps=["T0_2"],
        labels=["phase0", "day0", "subagent-parent", "wave-day0-C"],
        description=(
            "Loads output/final/champion_card.json (current V11_final stats), "
            "re-scores it against current preds, fails (exit 1) if val SIMSCORE "
            "regressed > 0.3 % unless commit message contains '[champion-bump]'. "
            "Schema: {tag, val_SIMSCORE, test_SIMSCORE, val_WAPE, test_WAPE, "
            "val_bias_pct, test_bias_pct, recipe, created_at}."
        ),
        acceptance=(
            "Running script with V11_final unchanged exits 0; running with a "
            "deliberately regressed champion exits 1; --bump flag promotes "
            "current to champion."
        ),
    ),
    Ticket(
        key="T0_4", parent="P0",
        title="Write .github/workflows/oof_audit.yml (CI)",
        priority=1, estimate=30, deps=["T0_3"],
        labels=["phase0", "day0", "subagent-parent", "wave-day0-C"],
        description=(
            "GitHub Actions workflow: trigger on push to main paths "
            "output/preds/**, output/grids/**, scripts/**. Steps: checkout w/ "
            "LFS, setup-python 3.11, pip install requirements.txt, run "
            "audit_champion_regression.py, upload output/audits/ as artifact."
        ),
        acceptance=(
            "On first push, workflow runs ≤ 5 min and is green. On a deliberately "
            "regressed commit, workflow goes red."
        ),
    ),
    Ticket(
        key="T0_5", parent="P0",
        title="Initialize output/final/champion_card.json with V11_final stats",
        priority=1, estimate=15, deps=["T0_1"],
        labels=["phase0", "day0", "subagent-parent", "wave-day0-A"],
        description=(
            "Run the all-models comparison script; extract V11_final's val/test "
            "SIMSCORE/WAPE/bias; write to output/final/champion_card.json with "
            "tag='v11_final', recipe=path-to-v11_chronos_blend.py, created_at=now."
        ),
        acceptance=(
            "Valid JSON; readable by audit_champion_regression.py; copied "
            "predictions in output/final/champion_predictions_{val,test}.csv."
        ),
    ),
    Ticket(
        key="T0_6", parent="P0",
        title="Initialize LOG.md with daily diary template",
        priority=2, estimate=15,
        labels=["phase0", "day0", "subagent-parent", "wave-day0-A"],
        description=(
            "Create LOG.md at repo root with: (1) intro pointer to roadmap and "
            "champion_card, (2) Day 0 entry, (3) commented template for daily "
            "entries (Goals, What ran, Numbers, Decisions, Risks, Tomorrow)."
        ),
        acceptance="LOG.md committed; Day 0 entry filled.",
    ),
    Ticket(
        key="T0_7", parent="P0",
        title="Create Makefile with v12/v12.5/v13/v13.5/v14/v14-alpha targets",
        priority=2, estimate=30, deps=["T0_2"],
        labels=["phase0", "day0", "subagent-parent", "wave-day0-D"],
        description=(
            "Each target chains the scripts that produce that release. Useful "
            "for reproducibility audits and CI. e.g. `make v12.5` runs "
            "build_v12_external_abt.py → train_v12_bases.py → "
            "v12_anomaly_weights.py → v12_lad_search.py."
        ),
        acceptance=(
            "make help lists all targets; running `make v12.5` from clean ABT "
            "reproduces V12.5 predictions byte-equal."
        ),
    ),
    Ticket(
        key="T0_8", parent="P0",
        title="Commit docs/v12_v14_roadmap.md (Draft 3 plan as committed truth)",
        priority=1, estimate=10,
        labels=["phase0", "day0", "subagent-parent", "wave-day0-A"],
        description=(
            "Write the full Draft 3 roadmap to docs/v12_v14_roadmap.md so any "
            "future agent/human can reconstruct intent without conversation context."
        ),
        acceptance="Doc committed; referenced from README.",
    ),
    Ticket(
        key="T0_9_HUMAN", parent="P0",
        title="[HUMAN] Confirm GitHub Actions enabled + Drive folder accessible",
        type="task", priority=1, estimate=5,
        labels=["phase0", "day0", "human-required", "blocking"],
        description=(
            "USER ACTION (≤ 5 min): "
            "1) Open repo Settings → Actions → General → 'Allow all actions' "
            "(if not already). "
            "2) Open https://drive.google.com/ and verify a folder /v11_chronos/ "
            "or /v13_fm_data/ exists you can write to. "
            "3) Close this ticket: bd close <id>."
        ),
        acceptance="User confirms via `bd close` with reason='confirmed'.",
    ),
    Ticket(
        key="T0_HOOKS", parent="P0",
        title="parent: bd hooks install (auto-prime + git hooks)",
        priority=2, estimate=10,
        labels=["phase0", "day0", "subagent-parent"],
        description=(
            "Run `bd hooks install` to auto-inject `bd prime` at session start "
            "and add the post-commit / pre-push git hooks beads needs. Verify "
            "with `bd doctor`."
        ),
        acceptance="`bd doctor` reports no missing hooks.",
    ),
    Ticket(
        key="T0_COMMIT_SETUP", parent="P0",
        title="parent: commit scripts/setup_v12_v14_beads.py + .beads/v12_v14_keymap.json",
        priority=2, estimate=10,
        labels=["phase0", "day0", "subagent-parent"],
        description=(
            "After running `python -m scripts.setup_v12_v14_beads --create`, "
            "commit the setup script + the keymap so anyone can reconstruct "
            "ticket↔key mapping (useful for resuming if conversation context "
            "is lost)."
        ),
        acceptance="Both files in git; commit message references plan.",
    ),
    Ticket(
        key="T0_10", parent="P0",
        title="Tag v11.1-restructure release",
        priority=1, estimate=10,
        deps=["T0_1", "T0_2", "T0_3", "T0_4", "T0_5", "T0_6", "T0_7",
              "T0_8", "T0_9_HUMAN", "T0_HOOKS", "T0_COMMIT_SETUP"],
        labels=["phase0", "day0", "subagent-parent"],
        description="git tag v11.1-restructure; git push --tags. Marks Phase 0 done.",
        acceptance="Tag visible on origin; CI green.",
    ),
    Ticket(
        key="T0_LOG", parent="P0", deps=["T0_10"],
        title="parent: LOG.md Day 0 + Phase 0 retrospective",
        priority=3, estimate=15,
        labels=["phase0", "day0", "subagent-parent", "log"],
        description="Daily diary: Phase 0 done, ready for Phase 1.",
        acceptance="LOG.md updated; commit pushed.",
    ),
]

# ============== PHASE 1 — WEEK 1 ==============

P1_TASKS: list[Ticket] = [
    # ----- DAY 1 Wave 1 (5 parallel) -----
    Ticket(
        key="T1_1", parent="P1", deps=["T0_10"],
        title="forager: Google Trends UA + FX volatility ingest",
        priority=2, estimate=180,
        labels=["phase1", "day1", "wave1", "subagent-forager"],
        description=(
            "Pull Google Trends UA monthly for 30 keywords (top-15 brand names "
            "from abt_v10_cached + 15 generic toy keywords). Range 2020-01 to "
            "2026-04. Build lag1, lag2, yoy features per keyword. Also pull NBU "
            "UAH/USD daily, compute rolling-30d volatility, monthly aggregate."
        ),
        design=(
            "Use pytrends (free, no auth). geo='UA'. Backoff on rate-limits. "
            "Save data/external/google_trends_ua.parquet + fx_volatility.parquet. "
            "Helper module src/external/trends.py with refresh()."
        ),
        acceptance=(
            "Both parquet files exist; ≥ 30 keywords × ≥ 70 monthly rows each; "
            "no NaNs in main columns; src/external/trends.py importable; unit "
            "test passes."
        ),
    ),
    Ticket(
        key="T1_2", parent="P1", deps=["T0_10"],
        title="forager: UA school calendar + pension/payday tables",
        priority=2, estimate=120,
        labels=["phase1", "day1", "wave1", "subagent-forager"],
        description=(
            "Hand-curate UA school holidays 2020-2026 from MES + Wikipedia. "
            "Build pension pay-day schedule (4-25th rolling) + salary schedule "
            "(15th, last day). Encode per month: school_days_count, "
            "is_winter_break, is_summer_break, easter_month, school_year_starts, "
            "payday_density_in_month, nearest_payday_distance_eom."
        ),
        acceptance=(
            "data/external/ua_school_calendar.csv (≥ 76 monthly rows) + "
            "data/external/payday_schedule.csv committed."
        ),
    ),
    Ticket(
        key="T1_3", parent="P1", deps=["T0_10"],
        title="trainer: V11 multi-seed bagging (5 seeds × 4 models)",
        priority=2, estimate=240,
        labels=["phase1", "day1", "wave1", "subagent-trainer"],
        description=(
            "Re-train v11_g93, v11_g90, v11_recent_only, v11_lad with seeds "
            "{41, 42, 43, 44, 45}. Average the 5 prediction sets per row → "
            "_bag versions. Use scripts/train_v7.py with --seed and "
            "--recency-gamma; for v11_lad, re-run scripts/v11_lad_stack.py."
        ),
        acceptance=(
            "8 CSV files at output/preds/v08_to_v11/preds_v11_*_bag_{val,test}.csv. "
            "Each bagged base must have OOF SIMSCORE ≤ its single-seed counterpart "
            "(within rounding)."
        ),
    ),
    Ticket(
        key="T1_4", parent="P1", deps=["T0_10"],
        title="specialist: Croston/SBA/TSB modules + SBC pattern classifier",
        priority=2, estimate=120,
        labels=["phase1", "day1", "wave1", "subagent-specialist"],
        description=(
            "Implement src/intermittent.py with croston(), sba(), tsb() (pure "
            "numpy, no deps beyond numpy). Implement src/sbc_classify.py with "
            "classify(history) → 'smooth'|'intermittent'|'lumpy'|'erratic' "
            "based on ADI and CV² thresholds (1.32 / 0.49). Add unit tests."
        ),
        acceptance=(
            "tests/test_intermittent.py passes (≥ 4 cases per method); "
            "tests/test_sbc_classify.py passes (≥ 4 cases for 4 patterns); "
            "modules importable."
        ),
    ),
    Ticket(
        key="T1_5", parent="P1", deps=["T0_10"],
        title="pilot: 3 FM notebook skeletons (Chronos, TimesFM, Moirai)",
        priority=2, estimate=240,
        labels=["phase1", "day1", "wave1", "subagent-pilot"],
        description=(
            "Create 3 notebooks with: install cell (using V11 Chronos recipe), "
            "Drive/Kaggle data load, FM weights load, fine-tune loop with 30-min "
            "checkpointing, prediction generation, save to Drive/Kaggle output. "
            "Skeletons only — DO NOT execute on GPU yet (Day 9-11 click action)."
        ),
        design=(
            "Chronos: HuggingFace amazon/chronos-t5-small + chronos-forecasting==1.5.2. "
            "TimesFM: google/timesfm-1.0-200m. Moirai: Salesforce/moirai-1.0-R-small. "
            "All checkpoints saved to /content/drive/MyDrive/v13_fm_data/ckpts/."
        ),
        acceptance=(
            "3 notebooks at notebooks/v13_{chronos,timesfm,moirai}_finetune_*.ipynb; "
            "each install cell tested in isolation by user (NOT a blocker — agents "
            "can verify by running install cell on local CPU as smoke test); "
            "documentation cells filled in."
        ),
    ),
    Ticket(
        key="T1_AUD", parent="P1",
        deps=["T1_1", "T1_2", "T1_3", "T1_4", "T1_5"],
        title="auditor: Day 1 deliverables review",
        priority=2, estimate=60,
        labels=["phase1", "day1", "subagent-auditor", "audit"],
        description=(
            "Read-only audit of Day 1 outputs: (1) verify external feature "
            "parquets are clean (no future leakage), (2) verify bagged preds "
            "respect training/val/test splits, (3) verify intermittent unit "
            "tests pass, (4) verify FM notebooks parse without syntax errors. "
            "Write output/audits/day1_audit.md."
        ),
        acceptance=(
            "audit doc committed; lists either ✅ all-pass or specific failures "
            "with remediation tickets created."
        ),
    ),
    Ticket(
        key="T1_LOG", parent="P1", deps=["T1_AUD"],
        title="parent: LOG.md Day 1 update",
        priority=3, estimate=15,
        labels=["phase1", "day1", "subagent-parent", "log"],
        description="Append Day 1 entry to LOG.md. Commit + push.",
        acceptance="LOG.md updated; commit pushed.",
    ),

    # ----- DAY 2 Wave 2 -----
    # NOTE: Day 2 tasks depend on the actual Day 1 deliverables they need
    # (e.g., bagged preds, Croston module), NOT on T1_LOG. LOG entries are
    # outputs of a day, not prerequisites for the next day.
    Ticket(
        key="T2_1", parent="P1", deps=["T0_10"],
        title="forager: UN Comtrade UA toy imports (HS 95) monthly",
        priority=2, estimate=120,
        labels=["phase1", "day2", "wave2", "subagent-forager"],
        description=(
            "Pull UN Comtrade UA imports HS chapter 95 monthly 2020-2026. Use "
            "free comtradeapi.un.org (free key signup). Aggregate to monthly UA "
            "import volume USD. Save data/external/un_comtrade_ua_toys.parquet."
        ),
        acceptance="parquet ≥ 70 monthly rows; importable; reasonable values.",
    ),
    Ticket(
        key="T2_2", parent="P1", deps=["T0_10"],
        title="forager: war events table (energy strikes, mobilization waves)",
        priority=3, estimate=90,
        labels=["phase1", "day2", "wave2", "subagent-forager"],
        description=(
            "Manually curate from Wikipedia + ISW timelines: monthly count of "
            "major energy infrastructure strikes, mobilization waves, "
            "blackout-days flag. Save data/external/war_events_ua.csv."
        ),
        acceptance="CSV with ≥ 76 monthly rows committed.",
    ),
    Ticket(
        key="T2_3", parent="P1", deps=["T1_3"],
        title="trainer: V12 seasonal (per-month-of-year) blend search script",
        priority=2, estimate=180,
        labels=["phase1", "day2", "wave2", "subagent-trainer"],
        description=(
            "Build scripts/v12_seasonal_blend.py: for each month-of-year m∈{1..12}, "
            "run (a, b) λ-grid over OOF rows where target month==m. EMA-smooth "
            "across months. Apply per-row at inference. Min 5 OOF rows per cell "
            "or fall back to global blend."
        ),
        acceptance=(
            "Script runs end-to-end in ≤ 30 min CPU; "
            "output/preds/v12/preds_v12_seasonal_{val,test}.csv produced; "
            "output/grids/v12/seasonal_grid.csv has all (a,b,month) eval rows; "
            "OOF SIMSCORE val improves ≥ 0.5 % over global blend baseline."
        ),
    ),
    Ticket(
        key="T2_4", parent="P1",
        deps=[
            "T1_1", "T1_2", "T2_1", "T2_2",
            # Priority-1 EXT sources fold into the initial V12 ABT directly.
            # Lower-priority sources arrive via EXT_SURVIVOR_MERGE (consumed by T4_3).
            "EXT_UKRSTAT_RTI", "EXT_UKRSTAT_BIRTHS", "EXT_UKRSTAT_INDPROD",
            "EXT_NBU_CCI", "EXT_AIRRAID_OBLAST", "EXT_BLACKOUT_DTEK",
            "EXT_IOM_IDP", "EXT_WIKI_PV", "EXT_ORTHODOX_CAL",
        ],
        title="trainer: Build abt_v12_external.parquet (V11 ABT + new external features)",
        priority=2, estimate=120,
        labels=["phase1", "day2", "wave2", "subagent-trainer"],
        description=(
            "scripts/build_v12_external_abt.py: load output/abt_v10_cached.parquet, "
            "merge in Google Trends, FX volatility, school cal, payday, Comtrade, "
            "war events, plus the 9 priority-1 EXT_* sources from Phase EXT "
            "(Ukrstat RTI/births/indprod, NBU CCI, oblast air-raids, DTEK blackouts, "
            "IOM IDP, Wikipedia pageviews, Orthodox calendar). Strict period cutoff "
            "(feature for month M built from data with date < first day of M) to "
            "prevent leakage. Lower-priority EXT sources are merged later via "
            "EXT_SURVIVOR_MERGE after the per-source A/B gate."
        ),
        acceptance=(
            "output/abt_v12_external.parquet exists; row count == V10 ABT row count; "
            "≥ 50 new feature columns (was ≥ 30 before EXT expansion); no NaN in "
            "macro-feature columns post-imputation; leakage test in "
            "tests/test_v12_abt_no_leakage.py passes."
        ),
    ),
    Ticket(
        key="T2_5", parent="P1", deps=["T1_4"],
        title="specialist: Croston specialist predictions for intermittent pairs",
        priority=2, estimate=120,
        labels=["phase1", "day2", "wave2", "subagent-specialist"],
        description=(
            "scripts/v12_croston_specialist.py: classify each pair via "
            "src/sbc_classify.py; for intermittent + lumpy pairs, predict via "
            "best-of (Croston, SBA, TSB) selected per-pair on training residuals; "
            "for smooth/erratic, predict 0 (LAD will weight)."
        ),
        acceptance=(
            "output/preds/v12/preds_v12_croston_{val,test}.csv produced; rows for "
            "non-intermittent pairs are 0; rows for intermittent pairs are non-zero."
        ),
    ),
    Ticket(
        key="T2_AUD", parent="P1",
        deps=["T2_1", "T2_2", "T2_3", "T2_4", "T2_5"],
        title="auditor: Day 2 deliverables review (incl. ABT leakage check)",
        priority=2, estimate=60,
        labels=["phase1", "day2", "subagent-auditor", "audit"],
        description=(
            "Read-only audit. Critical: verify abt_v12_external has zero forward "
            "feature leakage (all external features for month M derived from data "
            "with date < first day of M)."
        ),
        acceptance=(
            "output/audits/day2_audit.md committed; leakage check ✅ or remediation "
            "tickets created."
        ),
    ),
    Ticket(
        key="T2_LOG", parent="P1", deps=["T2_AUD"],
        title="parent: LOG.md Day 2 update", priority=3, estimate=15,
        labels=["phase1", "day2", "subagent-parent", "log"],
        description="Daily diary update.", acceptance="LOG.md updated.",
    ),

    # ----- DAY 3 Wave 3 -----
    Ticket(
        key="T3_1", parent="P1", deps=["T2_4"],
        title="trainer: Train v12_full (γ=0.85, all features incl. external)",
        priority=2, estimate=120,
        labels=["phase1", "day3", "wave3", "subagent-trainer"],
        description=(
            "Use scripts/train_v7.py --abt-path output/abt_v12_external.parquet "
            "--recency-gamma 0.85 --tag v12_full. Save preds + feature importance."
        ),
        acceptance=(
            "output/preds/v12/preds_v12_full_{val,test}.csv exist; "
            "output/grids/v12/v12_full_feat_importance.csv exists."
        ),
    ),
    Ticket(
        key="T3_2", parent="P1", deps=["T2_4"],
        title="trainer: Train v12_g93 + v12_recent_only on enriched ABT",
        priority=2, estimate=180,
        labels=["phase1", "day3", "wave3", "subagent-trainer"],
        description=(
            "Build output/abt_v12_external_recent_only.parquet (cutoff=2023-01) "
            "via scripts/build_v11_recent_only.py adapted. Train both bases."
        ),
        acceptance=(
            "preds_v12_g93_{val,test}.csv + preds_v12_recent_only_{val,test}.csv "
            "in output/preds/v12/."
        ),
    ),
    Ticket(
        key="T3_3", parent="P1", deps=["T3_1"],
        title="auditor: External-features A/B test on V12_full",
        priority=2, estimate=60,
        labels=["phase1", "day3", "wave3", "subagent-auditor", "decision"],
        description=(
            "Train v12_full_no_external (same recipe but using V11 ABT). Compare "
            "OOF val SIMSCORE. **Decision rule**: keep external features iff "
            "v12_full beats v12_full_no_external by ≥ 0.2 % OOF val. Otherwise "
            "drop them and rebuild ABT without losers."
        ),
        acceptance=(
            "output/audits/external_features_ab.md with verdict; "
            "if dropped, T3_3b ticket created to rebuild ABT."
        ),
    ),
    Ticket(
        key="T3_4", parent="P1", deps=["T3_1"],
        title="viz: V12 dashboard skeleton",
        priority=3, estimate=60,
        labels=["phase1", "day3", "wave3", "subagent-viz"],
        description="Adapt scripts/viz_v11_dashboard.py to V12 paths/tags.",
        acceptance="scripts/viz_v12_dashboard.py imports and runs (with placeholders).",
    ),
    Ticket(
        key="T3_AUD", parent="P1",
        deps=["T3_1", "T3_2", "T3_3", "T3_4"],
        title="auditor: Day 3 review",
        priority=2, estimate=30,
        labels=["phase1", "day3", "subagent-auditor", "audit"],
        description="Day 3 audit doc.",
        acceptance="output/audits/day3_audit.md committed.",
    ),
    Ticket(
        key="T3_LOG", parent="P1", deps=["T3_AUD"],
        title="parent: LOG.md Day 3 update", priority=3, estimate=15,
        labels=["phase1", "day3", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),

    # ----- DAY 4 Wave 4 → V12.5 checkpoint -----
    Ticket(
        key="T4_1", parent="P1", deps=["T3_1"],
        title="specialist: Build anomaly weights table (per-row + shock-month)",
        priority=2, estimate=120,
        labels=["phase1", "day4", "wave4", "subagent-specialist"],
        description=(
            "scripts/v12_anomaly_weights.py: per-row anomaly score = "
            "|residual / pred| against simple recency baseline. Top-1 % outliers → "
            "weight 0.3×. Shock months Feb-May 2022 → blanket 0.5×. Save "
            "output/grids/v12/anomaly_weights.parquet (key: KEY columns + weight)."
        ),
        acceptance="parquet exists; weight ∈ [0.3, 1.0]; weighted-row count audit OK.",
    ),
    Ticket(
        key="T4_2", parent="P1", deps=["T4_1", "T3_1", "T3_2"],
        title="trainer: Retrain V12 bases with anomaly weights",
        priority=2, estimate=240,
        labels=["phase1", "day4", "wave4", "subagent-trainer"],
        description=(
            "Re-run scripts/train_v7.py with --sample-weight-source "
            "output/grids/v12/anomaly_weights.parquet for v12_full, v12_g93, "
            "v12_recent_only → tags v12_full_aw, v12_g93_aw, v12_recent_only_aw."
        ),
        acceptance="3 _aw preds CSVs committed; OOF val ≥ 1 base improves vs non-aw.",
    ),
    Ticket(
        key="T4_3", parent="P1",
        deps=["T4_2", "T2_5", "T2_3", "T1_3", "EXT_SURVIVOR_MERGE"],
        title="trainer: V12 LAD search with bias ladder {1.0, 1.5, 2.0 %}",
        priority=1, estimate=180,
        labels=["phase1", "day4", "wave4", "subagent-trainer", "checkpoint"],
        description=(
            "scripts/v12_lad_search.py: pool = {V11 bagged bases, V12 bases, "
            "V12 _aw bases, Croston, V12 seasonal blend}. Bias ladder. Streaming "
            "calibrator. Uses the EXT-survivor-merged ABT (priority-1 EXT_* + "
            "any priority-2/3 sources that won the A/B gate). Output V12.5 final preds."
        ),
        acceptance=(
            "output/preds/v12/preds_v12.5_final_{val,test}.csv; "
            "output/grids/v12/lad_cv.csv; OOF val SIMSCORE ≤ 0.355."
        ),
    ),
    Ticket(
        key="T4_4", parent="P1", deps=["T4_3"],
        title="viz: V12.5 dashboard",
        priority=2, estimate=60,
        labels=["phase1", "day4", "wave4", "subagent-viz"],
        description="Run scripts/viz_v12_dashboard.py with V12.5 preds.",
        acceptance="output/plots/per_release/plot_v12.5_dashboard.png committed.",
    ),
    Ticket(
        key="T4_5", parent="P1", deps=["T4_3"],
        title="auditor: V12.5 OOF + leakage audit",
        priority=1, estimate=60,
        labels=["phase1", "day4", "wave4", "subagent-auditor", "audit"],
        description=(
            "Verify V12.5 selection used only val labels in OOF folds, no "
            "test labels touched at any selection step. Verify bias-ladder "
            "ceiling honored."
        ),
        acceptance=(
            "output/audits/v12.5_oof_audit.md = ✅; if any leakage flagged, "
            "block T4_6."
        ),
    ),
    Ticket(
        key="T4_6", parent="P1", deps=["T4_4", "T4_5"],
        title="parent: Tag v12.5.0 + bump champion_card if OOF beats V11",
        priority=1, estimate=30,
        labels=["phase1", "day4", "wave4", "subagent-parent", "checkpoint"],
        description=(
            "If V12.5 OOF SIMSCORE val < V11_final OOF SIMSCORE: bump "
            "champion_card.json with [champion-bump] commit. Tag v12.5.0 "
            "regardless. Push tags. CI must stay green."
        ),
        acceptance="Tag pushed; CI green; champion_card reflects new champ if applicable.",
    ),
    Ticket(
        key="T4_GATE_HUMAN", parent="P1", deps=["T4_6"],
        title="[HUMAN-DECISION] Approve V12.5 → continue to V12 polish?",
        type="decision", priority=2, estimate=10,
        labels=["phase1", "day4", "human-required", "gate-decision"],
        description=(
            "USER ACTION (≤ 5 min): "
            "Read output/audits/v12.5_oof_audit.md and "
            "output/plots/per_release/plot_v12.5_dashboard.png. "
            "Decide: (a) continue to Day 5-6 V12 polish, or "
            "(b) rollback (block subsequent tasks, ship V11). "
            "Close with reason='approved' or 'rollback'."
        ),
        acceptance="User closes with verdict.",
    ),
    Ticket(
        key="T4_LOG", parent="P1", deps=["T4_GATE_HUMAN"],
        title="parent: LOG.md Day 4 + V12.5 announcement",
        priority=3, estimate=15,
        labels=["phase1", "day4", "subagent-parent", "log"],
        description="Daily diary + V12.5 release notes.",
        acceptance="LOG.md updated.",
    ),

    # ----- DAY 5-6 V12 polish -----
    # T5_1 depends on the V12.5 LAD output (T4_3) and on the human gate
    # approval (T4_GATE_HUMAN), NOT on T4_LOG (which is async output).
    Ticket(
        key="T5_1", parent="P1", deps=["T4_3", "T4_GATE_HUMAN"],
        title="parent: Streaming calibrator → V12_final / V12_relaxed / V12_test_aware",
        priority=2, estimate=180,
        labels=["phase1", "day5", "subagent-parent"],
        description=(
            "Apply src/streaming_calibrator.py to V12.5 LAD predictions; search "
            "best beta + axes. Produce 3 variants (final = strict, relaxed = 1.5%, "
            "test_aware = test-peeked reference)."
        ),
        acceptance=(
            "preds_v12_{final,relaxed,test_aware}_{val,test}.csv in "
            "output/preds/v12/."
        ),
    ),
    Ticket(
        key="T5_2", parent="P1", deps=["T5_1"],
        title="viz: V12 vs V11 per-month timeline",
        priority=3, estimate=60,
        labels=["phase1", "day5", "subagent-viz"],
        description="Adapt scripts/viz_v11_vs_v10_timeline.py.",
        acceptance="output/plots/per_release/plot_v12_vs_v11_timeline.png.",
    ),
    Ticket(
        key="T5_3", parent="P1", deps=["T5_1", "T5_2"],
        title="viz: docs/v12_final_report.md",
        priority=2, estimate=120,
        labels=["phase1", "day6", "subagent-viz"],
        description="Full V12 narrative: gains attribution, variants table, recipe.",
        acceptance="docs/v12_final_report.md committed.",
    ),
    Ticket(
        key="T5_4", parent="P1", deps=["T5_3"],
        title="viz: README update with V12",
        priority=3, estimate=30,
        labels=["phase1", "day6", "subagent-viz"],
        description="Add V12 row to comparison table; update narrative.",
        acceptance="README updated.",
    ),
    Ticket(
        key="T5_5", parent="P1", deps=["T5_1"],
        title="auditor: Final V12 OOF audit",
        priority=2, estimate=45,
        labels=["phase1", "day6", "subagent-auditor", "audit"],
        description="Lock-in audit before tagging v12.0.0.",
        acceptance="output/audits/v12_final_audit.md = ✅.",
    ),
    Ticket(
        key="T5_LOG", parent="P1", deps=["T5_4", "T5_5"],
        title="parent: LOG.md Days 5-6 update", priority=3, estimate=15,
        labels=["phase1", "day6", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T7_1", parent="P1", deps=["T5_3", "T5_5"],
        title="parent: Tag v12.0.0",
        priority=1, estimate=10,
        labels=["phase1", "day7", "subagent-parent"],
        description="git tag v12.0.0; push tags; bump champion_card if applicable.",
        acceptance="Tag visible on origin; CI green.",
    ),
]

# ============== PHASE 2 — WEEK 2 (FM fine-tunes) ==============

P2_TASKS: list[Ticket] = [
    Ticket(
        key="T8_1", parent="P2", deps=["T7_1"],
        title="pilot: Build per-pair monthly history dataset for FM input",
        priority=2, estimate=120,
        labels=["phase2", "day8", "subagent-pilot"],
        description=(
            "scripts/build_fm_dataset.py: from abt_v12_external + raw history, "
            "produce a parquet of shape [num_series, history_length=36] tensors "
            "with metadata (series_id, partner, sku). Save data/fm/series.parquet."
        ),
        acceptance="parquet exists; ≥ 5000 series; uniform shape.",
    ),
    Ticket(
        key="T8_2", parent="P2", deps=["T1_5", "T8_1"],
        title="pilot: Finalize 3 FM notebooks (Chronos + TimesFM + Moirai) with checkpointing",
        priority=2, estimate=240,
        labels=["phase2", "day8", "subagent-pilot"],
        description=(
            "Open T1_5 skeletons, plug in dataset path, add checkpoint-every-30-min "
            "logic, verify install cells via Colab dry-run (just install + tiny "
            "fine-tune step)."
        ),
        acceptance=(
            "3 notebooks committed at notebooks/v13_*_finetune_*.ipynb; install "
            "cells verified."
        ),
    ),
    Ticket(
        key="T8_3", parent="P2", deps=["T8_2"],
        title="pilot: docs/v13_fm_runbook.md (step-by-step for each FM run)",
        priority=2, estimate=60,
        labels=["phase2", "day8", "subagent-pilot"],
        description=(
            "Mirror the V11 Chronos guide style; cover preemption-recovery, "
            "where to download preds CSVs, how to verify checkpoint integrity."
        ),
        acceptance="docs/v13_fm_runbook.md committed.",
    ),
    Ticket(
        key="T8_HUMAN_UPLOAD", parent="P2", deps=["T8_1"],
        title="[HUMAN] Upload data/fm/series.parquet to Drive /v13_fm_data/ + Kaggle dataset",
        type="task", priority=1, estimate=15,
        labels=["phase2", "day8", "human-required", "blocking"],
        description=(
            "USER ACTION (≤ 15 min): "
            "1) Open Google Drive, navigate to /v13_fm_data/ (create if missing). "
            "2) Upload data/fm/series.parquet. "
            "3) Open kaggle.com → Datasets → New Dataset, name 'v13-fm-series', "
            "upload series.parquet. "
            "4) Close this ticket. "
            "After this, the FM notebooks have data to fine-tune on."
        ),
        acceptance="User closes with reason='uploaded'.",
    ),
    Ticket(
        key="T8_LOG", parent="P2", deps=["T8_3", "T8_HUMAN_UPLOAD"],
        title="parent: LOG.md Day 8 update", priority=3, estimate=15,
        labels=["phase2", "day8", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),

    # ----- 6 GPU runs serialized per cloud -----
    # CLOUD CONSTRAINT: one Google account = ONE Colab GPU runtime at a time;
    # one Kaggle account = ONE GPU runtime at a time. So we serialize per cloud:
    #   Colab queue:  Chronos-S42 → TimesFM-S42 → Chronos-S1729 → Moirai-S1729
    #   Kaggle queue: Moirai-S42 → TimesFM-S1729
    # Cross-cloud is parallel (Day 9 Colab + Kaggle run side by side).
    Ticket(
        key="T9_HUMAN_CHRONOS_S42", parent="P2",
        deps=["T8_2", "T8_HUMAN_UPLOAD"],
        title="[HUMAN] Run Chronos seed=42 fine-tune on Colab (~5 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase2", "day9", "human-required", "gpu-run", "fm-chronos"],
        description=(
            "USER ACTION (≤ 5 min click + ~5 hr unattended): "
            "1) Open notebooks/v13_chronos_finetune_colab.ipynb in Colab. "
            "2) Runtime → Change runtime type → T4 GPU. "
            "3) Set notebook variable SEED=42. "
            "4) Runtime → Run all. "
            "5) Wait ~5 hr (preemption auto-resumes from checkpoint on re-Run). "
            "6) When 'preds_v13_chronos_ft_seed42_*.csv' lands on Drive, download "
            "to local output/preds/v13/. "
            "7) Close this ticket with reason='success' or 'fail-N' (where N = "
            "attempt number). 3 fails → escalate to T_MODAL_HUMAN."
        ),
        acceptance=(
            "output/preds/v13/preds_v13_chronos_ft_seed42_{val,test}.csv exist; "
            "ticket closed with reason."
        ),
    ),
    Ticket(
        key="T9_HUMAN_MOIRAI_S42", parent="P2",
        deps=["T8_2", "T8_HUMAN_UPLOAD"],
        title="[HUMAN] Run Moirai seed=42 fine-tune on Kaggle (~7 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase2", "day9", "human-required", "gpu-run", "fm-moirai"],
        description=(
            "USER ACTION: "
            "1) Open notebooks/v13_moirai_finetune_kaggle.ipynb (upload to Kaggle). "
            "2) Settings: GPU T4 / P100, internet on, dataset=v13-fm-series. "
            "3) Set SEED=42. Save & Run All. "
            "4) When done, download preds CSV → output/preds/v13/. "
            "5) Close with reason='success' or 'fail-N'."
        ),
        acceptance="preds_v13_moirai_ft_seed42_*.csv at output/preds/v13/.",
    ),
    Ticket(
        key="T10_HUMAN_TIMESFM_S42", parent="P2",
        # Colab queue: must wait for Chronos-S42 to free the GPU.
        deps=["T8_2", "T8_HUMAN_UPLOAD", "T9_HUMAN_CHRONOS_S42"],
        title="[HUMAN] Run TimesFM seed=42 fine-tune on Colab (~5 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase2", "day10", "human-required", "gpu-run", "fm-timesfm"],
        description=(
            "Same recipe as Chronos but with TimesFM notebook. SEED=42. After "
            "Chronos seed=42 has freed the Colab GPU."
        ),
        acceptance="preds_v13_timesfm_ft_seed42_*.csv at output/preds/v13/.",
    ),
    Ticket(
        key="T11_HUMAN_CHRONOS_S1729", parent="P2",
        # Colab queue: must wait for TimesFM-S42 to free the GPU.
        deps=["T9_HUMAN_CHRONOS_S42", "T10_HUMAN_TIMESFM_S42"],
        title="[HUMAN] Run Chronos seed=1729 fine-tune on Colab (~5 hr GPU)",
        type="task", priority=2, estimate=10,
        labels=["phase2", "day11", "human-required", "gpu-run", "fm-chronos"],
        description=(
            "USER ACTION: re-run notebooks/v13_chronos_finetune_colab.ipynb with "
            "SEED=1729 for bagging."
        ),
        acceptance="preds_v13_chronos_ft_seed1729_*.csv at output/preds/v13/.",
    ),
    Ticket(
        key="T11_HUMAN_TIMESFM_S1729", parent="P2",
        # Kaggle queue: must wait for Moirai-S42 to free the GPU.
        deps=["T9_HUMAN_MOIRAI_S42"],
        title="[HUMAN] Run TimesFM seed=1729 fine-tune on Kaggle (~6 hr GPU)",
        type="task", priority=2, estimate=10,
        labels=["phase2", "day11", "human-required", "gpu-run", "fm-timesfm"],
        description="USER ACTION: re-run TimesFM notebook on Kaggle with SEED=1729.",
        acceptance="preds_v13_timesfm_ft_seed1729_*.csv at output/preds/v13/.",
    ),
    Ticket(
        key="T11_HUMAN_MOIRAI_S1729", parent="P2",
        # Colab queue: must wait for Chronos-S1729 to free the GPU.
        deps=["T9_HUMAN_MOIRAI_S42", "T11_HUMAN_CHRONOS_S1729"],
        title="[HUMAN] Run Moirai seed=1729 fine-tune on Colab (~6 hr GPU)",
        type="task", priority=2, estimate=10,
        labels=["phase2", "day11", "human-required", "gpu-run", "fm-moirai"],
        description="USER ACTION: re-run Moirai notebook on Colab with SEED=1729.",
        acceptance="preds_v13_moirai_ft_seed1729_*.csv at output/preds/v13/.",
    ),

    # ----- Modal fallback gate (DEFERRED: only triggered by FM failures) -----
    # NOTE: This ticket has a soft dependency on FM-run tickets — it is meant
    # to be 'deferred' (bd defer ...) at create-time so it doesn't appear in
    # `bd ready` unless a parent agent explicitly un-defers it after recording
    # 3+3 documented FM-run failures.
    Ticket(
        key="T_MODAL_GATE_HUMAN", parent="P2",
        deps=["T8_2"],   # at minimum needs the runbook + notebooks ready
        defer="+90d",    # hidden from `bd ready` until parent un-defers
        title="[HUMAN-DECISION] Trigger Modal $0-cap fallback for failing FM?",
        type="decision", priority=4, estimate=15,
        labels=["phase2", "human-required", "gate-decision", "modal-fallback",
                "deferred-by-default"],
        description=(
            "Triggered only if a specific FM has failed 3 attempts on Colab AND "
            "3 on Kaggle (total 6 fails). DEFERRED at creation time — parent "
            "agent un-defers via `bd defer <id> --until=now` once "
            "output/audits/fm_failures.md documents 6 failures. USER ACTION: "
            "read fm_failures.md, decide go/no-go. If go: follow "
            "docs/modal_fallback_runbook.md (signup with $0 spend cap, run, "
            "delete workspace). If no-go: drop that FM from V13 pool."
        ),
        acceptance="User closes with reason='approved-modal' or 'skip-fm-X'.",
    ),
    Ticket(
        key="T_MODAL_RUNBOOK", parent="P2", deps=["T0_10"],
        title="pilot: docs/modal_fallback_runbook.md + scripts/modal_finetune.py",
        priority=3, estimate=120,
        labels=["phase2", "subagent-pilot", "modal-fallback"],
        description=(
            "Build the runbook EARLY in Phase 1 so it's ready if the gate fires. "
            "Step-by-step Modal signup with $0 spending cap, auto-recharge OFF; "
            "modal app definition for one-off FM fine-tune; cleanup steps."
        ),
        acceptance="Doc + script committed; never executed unless gate fires.",
    ),

    # ----- Bagging predictions -----
    Ticket(
        key="T9_BAG_CHRONOS", parent="P2",
        deps=["T9_HUMAN_CHRONOS_S42", "T11_HUMAN_CHRONOS_S1729"],
        title="pilot: Bag 2 Chronos seeds → preds_v13_chronos_ft_{val,test}.csv",
        priority=2, estimate=30,
        labels=["phase2", "day11", "subagent-pilot", "bagging"],
        description="Average per-row across 2 seed predictions.",
        acceptance="bagged CSV in output/preds/v13/.",
    ),
    Ticket(
        key="T9_BAG_TIMESFM", parent="P2",
        deps=["T10_HUMAN_TIMESFM_S42", "T11_HUMAN_TIMESFM_S1729"],
        title="pilot: Bag 2 TimesFM seeds → preds_v13_timesfm_ft_*.csv",
        priority=2, estimate=30,
        labels=["phase2", "day11", "subagent-pilot", "bagging"],
        description="Average per-row across 2 seed predictions.",
        acceptance="bagged CSV in output/preds/v13/.",
    ),
    Ticket(
        key="T9_BAG_MOIRAI", parent="P2",
        deps=["T9_HUMAN_MOIRAI_S42", "T11_HUMAN_MOIRAI_S1729"],
        title="pilot: Bag 2 Moirai seeds → preds_v13_moirai_ft_*.csv",
        priority=2, estimate=30,
        labels=["phase2", "day11", "subagent-pilot", "bagging"],
        description="Average per-row across 2 seed predictions.",
        acceptance="bagged CSV in output/preds/v13/.",
    ),

    # ----- DAY 12 V13.5 checkpoint -----
    Ticket(
        key="T12_1", parent="P2",
        deps=["T9_BAG_CHRONOS", "T9_BAG_TIMESFM", "T9_BAG_MOIRAI", "T7_1"],
        title="trainer: V13 LAD search (V12 + 3 bagged FMs)",
        priority=1, estimate=180,
        labels=["phase2", "day12", "subagent-trainer", "checkpoint"],
        description=(
            "scripts/v13_lad_search.py: pool = V12_final ingredients ∪ "
            "{v13_chronos_ft, v13_timesfm_ft, v13_moirai_ft}. Bias ladder. "
            "Streaming calibrator."
        ),
        acceptance=(
            "preds_v13.5_final_*.csv; OOF val SIMSCORE ≤ 0.350; ≥ 1 FM has weight > 0."
        ),
    ),
    Ticket(
        key="T12_2", parent="P2", deps=["T12_1"],
        title="viz: V13.5 dashboard",
        priority=2, estimate=60,
        labels=["phase2", "day12", "subagent-viz"],
        description="V13.5 dashboard.",
        acceptance="plot_v13.5_dashboard.png in output/plots/per_release/.",
    ),
    Ticket(
        key="T12_3", parent="P2", deps=["T12_1"],
        title="auditor: V13 OOF + leakage audit",
        priority=1, estimate=60,
        labels=["phase2", "day12", "subagent-auditor", "audit"],
        description="OOF discipline check for V13.5.",
        acceptance="output/audits/v13.5_oof_audit.md ✅.",
    ),
    Ticket(
        key="T12_4", parent="P2", deps=["T12_2", "T12_3"],
        title="parent: Tag v13.5.0",
        priority=1, estimate=15,
        labels=["phase2", "day12", "subagent-parent", "checkpoint"],
        description="Tag + push + bump champion_card if applicable.",
        acceptance="v13.5.0 visible on origin.",
    ),
    Ticket(
        key="T12_GATE_HUMAN", parent="P2", deps=["T12_4"],
        title="[HUMAN-DECISION] Approve V13.5 → continue to V13 polish?",
        type="decision", priority=2, estimate=10,
        labels=["phase2", "day12", "human-required", "gate-decision"],
        description=(
            "USER ACTION (≤ 5 min): review v13.5 dashboard + audit. "
            "Close with reason='approved' or 'rollback'."
        ),
        acceptance="User closes with verdict.",
    ),
    Ticket(
        key="T12_LOG", parent="P2", deps=["T12_GATE_HUMAN"],
        title="parent: LOG.md Day 12 + V13.5 announcement",
        priority=3, estimate=15,
        labels=["phase2", "day12", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),

    # ----- DAY 13-14 V13 polish -----
    Ticket(
        key="T13_1", parent="P2", deps=["T12_1", "T12_GATE_HUMAN"],
        title="parent: V13 streaming calibrator + V13_final / V13_relaxed / V13_test_aware",
        priority=2, estimate=180,
        labels=["phase2", "day13", "subagent-parent"],
        description="Mirror T5_1 for V13.5 LAD.",
        acceptance="3 V13 variants in output/preds/v13/.",
    ),
    Ticket(
        key="T13_2", parent="P2", deps=["T13_1"],
        title="viz: docs/v13_final_report.md",
        priority=2, estimate=120,
        labels=["phase2", "day13", "subagent-viz"],
        description="V13 narrative.",
        acceptance="docs/v13_final_report.md committed.",
    ),
    Ticket(
        key="T13_3", parent="P2", deps=["T13_1"],
        title="viz: V13 vs V12 timeline",
        priority=3, estimate=60,
        labels=["phase2", "day13", "subagent-viz"],
        description="Per-month timeline plot.",
        acceptance="plot_v13_vs_v12_timeline.png.",
    ),
    Ticket(
        key="T13_4", parent="P2", deps=["T13_2"],
        title="viz: README V13 update",
        priority=3, estimate=30,
        labels=["phase2", "day14", "subagent-viz"],
        description="README V13 update.", acceptance="README updated.",
    ),
    Ticket(
        key="T13_5", parent="P2", deps=["T13_1"],
        title="auditor: Final V13 OOF audit",
        priority=2, estimate=45,
        labels=["phase2", "day14", "subagent-auditor", "audit"],
        description="Lock-in audit.",
        acceptance="output/audits/v13_final_audit.md = ✅.",
    ),
    Ticket(
        key="T13_LOG", parent="P2", deps=["T13_4", "T13_5"],
        title="parent: LOG.md Days 13-14", priority=3, estimate=15,
        labels=["phase2", "day14", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T14_1", parent="P2", deps=["T13_2", "T13_5"],
        title="parent: Tag v13.0.0",
        priority=1, estimate=10,
        labels=["phase2", "day14", "subagent-parent"],
        description="Tag + push.",
        acceptance="v13.0.0 visible.",
    ),
]

# ============== PHASE 3 — WEEK 3 (GlobalNN) ==============

P3_TASKS: list[Ticket] = [
    # Phase 3 CODE work parallelizes with Phase 2 GPU runs:
    # the specialist subagent is free during Phase 2 (only pilots are busy
    # preparing FM notebooks). The GlobalNN model code only needs the V12
    # ABT format — it doesn't depend on V13 outputs at training time. So we
    # depend on T2_4 (V12 ABT exists) instead of T14_1 (V13 tag).
    # Phase 3 GPU work (T17_HUMAN_S1..S4) still waits for Phase 2 Colab to
    # free (enforced via T11_HUMAN_MOIRAI_S1729 dep).
    Ticket(
        key="T15_1", parent="P3", deps=["T2_4"],
        title="specialist: Build src/models/global_nn.py (Transformer-encoder)",
        priority=1, estimate=480,
        labels=["phase3", "day15", "subagent-specialist", "neural"],
        description=(
            "Architecture: embeddings (partner 256d, SKU 256d, channel 32d, brand "
            "64d), 4-head Transformer encoder (3 layers, d_model=384) over 24-month "
            "history with positional + temporal embeddings, static head over "
            "partner+SKU embedding dims, output = 3 quantiles (0.25, 0.5, 0.75) + "
            "Tweedie auxiliary. Loss = pinball + 0.1 × Tweedie."
        ),
        acceptance=(
            "src/models/global_nn.py importable; smoke-test forward pass on 100 "
            "synthetic series < 5 min CPU."
        ),
    ),
    Ticket(
        key="T15_2", parent="P3", deps=["T15_1"],
        title="specialist: tests/test_global_nn_smoke.py",
        priority=2, estimate=60,
        labels=["phase3", "day15", "subagent-specialist", "neural"],
        description="100-step CPU sanity test on synthetic data; loss decreases.",
        acceptance="pytest passes in < 5 min.",
    ),
    Ticket(
        key="T15_3", parent="P3", deps=["T15_1", "T15_2"],
        title="pilot: notebooks/v14_globalnn_colab.ipynb with checkpointing",
        priority=2, estimate=180,
        labels=["phase3", "day16", "subagent-pilot", "neural"],
        description=(
            "Colab notebook: install pytorch+wandb-lite, mount Drive, load training "
            "tensors, train with 30-min checkpointing, save best-val checkpoint."
        ),
        acceptance="Notebook committed; install cell verified locally.",
    ),
    Ticket(
        key="T15_4", parent="P3", deps=["T15_3"],
        title="pilot: Build training tensors + upload to Drive",
        priority=2, estimate=120,
        labels=["phase3", "day16", "subagent-pilot", "neural"],
        description=(
            "scripts/build_globalnn_tensors.py: from abt_v12_external + raw history → "
            "tensor data dict (X_history, X_static, y_target, embedding_idxs); "
            "save to data/globalnn/."
        ),
        acceptance="data/globalnn/*.pt or *.npz; size < 2 GB; loadable.",
    ),
    Ticket(
        key="T15_HUMAN_DRIVE", parent="P3", deps=["T15_4"],
        title="[HUMAN] Confirm /v14_globalnn_data/ Drive folder ready",
        type="task", priority=1, estimate=5,
        labels=["phase3", "day16", "human-required"],
        description=(
            "USER ACTION (≤ 3 min): "
            "1) Open Drive, create /v14_globalnn_data/ folder. "
            "2) Upload data/globalnn/* tensors to it. "
            "3) Close ticket."
        ),
        acceptance="User closes with reason='uploaded'.",
    ),
    Ticket(
        key="T15_LOG", parent="P3", deps=["T15_HUMAN_DRIVE"],
        title="parent: LOG.md Days 15-16", priority=3, estimate=15,
        labels=["phase3", "day16", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),

    # ----- 4 sequential GPU sessions for GlobalNN train -----
    Ticket(
        key="T17_HUMAN_S1", parent="P3",
        # Colab queue: must wait for the last Phase-2 Colab FM run to finish.
        deps=["T15_3", "T15_HUMAN_DRIVE", "T11_HUMAN_MOIRAI_S1729"],
        title="[HUMAN] GlobalNN training session 1 on Colab (~4 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase3", "day17", "human-required", "gpu-run", "globalnn"],
        description=(
            "USER ACTION: open notebooks/v14_globalnn_colab.ipynb in Colab T4, "
            "Run all. Auto-resumes from latest ckpt on re-Run after preemption. "
            "Close with reason='success' or 'fail-N'."
        ),
        acceptance="Checkpoint at /v14_globalnn_data/ckpts/ckpt_session1.pt.",
    ),
    Ticket(
        key="T17_HUMAN_S2", parent="P3", deps=["T17_HUMAN_S1"],
        title="[HUMAN] GlobalNN training session 2 (~4 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase3", "day17", "human-required", "gpu-run", "globalnn"],
        description="Continue training. Same notebook.",
        acceptance="ckpt_session2.pt produced.",
    ),
    Ticket(
        key="T18_HUMAN_S3", parent="P3", deps=["T17_HUMAN_S2"],
        title="[HUMAN] GlobalNN training session 3 (~4 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase3", "day18", "human-required", "gpu-run", "globalnn"],
        description="Continue training.",
        acceptance="ckpt_session3.pt produced.",
    ),
    Ticket(
        key="T18_HUMAN_S4", parent="P3", deps=["T18_HUMAN_S3"],
        title="[HUMAN] GlobalNN training session 4 + final inference (~4 hr GPU)",
        type="task", priority=1, estimate=10,
        labels=["phase3", "day18", "human-required", "gpu-run", "globalnn"],
        description=(
            "USER ACTION: continue training; if val converges, run final inference; "
            "save preds_v14_globalnn_*.csv to Drive; download to local."
        ),
        acceptance="output/preds/v14/preds_v14_globalnn_{val,test}.csv exist.",
    ),
    Ticket(
        key="T19_1", parent="P3", deps=["T18_HUMAN_S4", "T13_1"],
        title="trainer: V14_alpha LAD with GlobalNN added to V13 pool",
        priority=1, estimate=120,
        labels=["phase3", "day19", "subagent-trainer", "checkpoint"],
        description="scripts/v14_lad_search.py — full pool incl. GlobalNN.",
        acceptance=(
            "preds_v14_alpha_*.csv produced; if GlobalNN earns weight > 0, OOF "
            "val improves ≥ 0.3 % over V13_final."
        ),
    ),
    Ticket(
        key="T19_2", parent="P3", deps=["T19_1"],
        title="parent: Tag v14.0.0-alpha",
        priority=1, estimate=10,
        labels=["phase3", "day19", "subagent-parent"],
        description="Tag + push.",
        acceptance="v14.0.0-alpha visible.",
    ),
    Ticket(
        key="T19_LOG", parent="P3", deps=["T19_2"],
        title="parent: LOG.md Day 19", priority=3, estimate=15,
        labels=["phase3", "day19", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T20_1", parent="P3", deps=["T19_2"],
        title="pilot: prepare GlobalNN seed-2 retrain (optional bagging)",
        priority=3, estimate=30,
        labels=["phase3", "day20", "subagent-pilot", "neural"],
        description="Set SEED=1729 in notebook, prep for second run.",
        acceptance="Notebook updated; ready for second human click.",
    ),
    Ticket(
        key="T20_HUMAN_BAG", parent="P3", deps=["T20_1"],
        title="[HUMAN, OPTIONAL] GlobalNN seed=1729 retrain for bagging (~16 hr cumulative)",
        type="task", priority=3, estimate=10,
        labels=["phase3", "day20", "human-required", "gpu-run", "globalnn", "optional"],
        description=(
            "USER ACTION (optional): re-run GlobalNN with seed=1729 (4 sessions). "
            "Skip if you're tight on time — V14 still works with single seed. "
            "Close with reason='done' or 'skipped'."
        ),
        acceptance="Either bagged preds_v14_globalnn_bag_* OR closed='skipped'.",
    ),
    Ticket(
        key="T20_2", parent="P3", deps=["T19_2"],
        title="viz: V14_alpha report + dashboard",
        priority=3, estimate=120,
        labels=["phase3", "day21", "subagent-viz"],
        description="docs/v14_alpha_report.md + plot.",
        acceptance="Doc + plot committed.",
    ),
    Ticket(
        key="T21_LOG", parent="P3", deps=["T20_2"],
        title="parent: LOG.md Days 20-21", priority=3, estimate=15,
        labels=["phase3", "day21", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
]

# ============== PHASE 4 — WEEK 4 ==============

P4_TASKS: list[Ticket] = [
    Ticket(
        key="T22_1", parent="P4", deps=["T19_2"],
        title="specialist: cluster-SMOOTH specialist (V12-style LightGBM)",
        priority=2, estimate=180,
        labels=["phase4", "day22", "subagent-specialist", "moe"],
        description=(
            "Train a V12_full-style model only on rows whose pair is classified "
            "as 'smooth' by SBC; predict 0 elsewhere."
        ),
        acceptance="preds_v14_cluster_smooth_*.csv in output/preds/v14/.",
    ),
    Ticket(
        key="T22_2", parent="P4", deps=["T19_2"],
        title="specialist: cluster-INTERMITTENT specialist",
        priority=2, estimate=180,
        labels=["phase4", "day22", "subagent-specialist", "moe"],
        description="Same but for 'intermittent' cluster.",
        acceptance="preds_v14_cluster_intermittent_*.csv.",
    ),
    Ticket(
        key="T22_3", parent="P4", deps=["T19_2"],
        title="specialist: cluster-LUMPY specialist",
        priority=2, estimate=180,
        labels=["phase4", "day23", "subagent-specialist", "moe"],
        description="Same but for 'lumpy' cluster.",
        acceptance="preds_v14_cluster_lumpy_*.csv.",
    ),
    Ticket(
        key="T22_4", parent="P4", deps=["T19_2"],
        title="specialist: cluster-ERRATIC specialist",
        priority=2, estimate=180,
        labels=["phase4", "day23", "subagent-specialist", "moe"],
        description="Same but for 'erratic' cluster.",
        acceptance="preds_v14_cluster_erratic_*.csv.",
    ),
    Ticket(
        key="T22_AUD", parent="P4",
        deps=["T22_1", "T22_2", "T22_3", "T22_4"],
        title="auditor: per-cluster specialist beats global on its cluster?",
        priority=2, estimate=60,
        labels=["phase4", "day23", "subagent-auditor", "audit"],
        description=(
            "For each cluster, verify specialist OOF SIMSCORE on cluster-rows < "
            "v12_full OOF SIMSCORE on cluster-rows. Only specialists that pass "
            "this gate enter the V14 pool."
        ),
        acceptance=(
            "output/audits/cluster_specialists_audit.md with per-cluster verdict; "
            "specialists that fail are dropped from V14 pool."
        ),
    ),
    Ticket(
        key="T23_LOG", parent="P4", deps=["T22_AUD"],
        title="parent: LOG.md Days 22-23", priority=3, estimate=15,
        labels=["phase4", "day23", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T24_1", parent="P4", deps=["T22_AUD"],
        title="specialist: train soft gate (cluster posterior probabilities per pair)",
        priority=2, estimate=180,
        labels=["phase4", "day24", "subagent-specialist", "moe"],
        description=(
            "Multi-class LightGBM classifier on pair → cluster prob. Trained on "
            "training-period pair histories."
        ),
        acceptance=(
            "src/moe/gate.py + preds_v14_gate_probs.csv (one row per pair × 4 cluster cols)."
        ),
    ),
    Ticket(
        key="T24_2", parent="P4", deps=["T24_1"],
        title="specialist: build MoE ensemble pred (Σ gate_prob × specialist_pred)",
        priority=2, estimate=60,
        labels=["phase4", "day24", "subagent-specialist", "moe"],
        description="Weighted sum of cluster specialists by gate posterior.",
        acceptance="preds_v14_moe_*.csv at output/preds/v14/.",
    ),
    Ticket(
        key="T24_LOG", parent="P4", deps=["T24_2"],
        title="parent: LOG.md Day 24", priority=3, estimate=15,
        labels=["phase4", "day24", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T25_1", parent="P4", deps=["T19_2", "T24_2"],
        title="trainer: Final V14 LAD search (everything-pool)",
        priority=1, estimate=300,
        labels=["phase4", "day25", "subagent-trainer", "checkpoint"],
        description=(
            "scripts/v14_final_lad.py: pool = V12 bases (final, g93, recent, _aw) + "
            "3 fine-tuned FMs (bagged) + GlobalNN (bagged if available) + 4 cluster "
            "specialists (those that passed audit) + MoE pred + per-month seasonal "
            "blend + streaming calibrator on top. Bias ladder."
        ),
        acceptance=(
            "preds_v14_final_*.csv with OOF val SIMSCORE ≤ 0.345 and test ≤ 0.420 "
            "(or close — flagged for review if test regresses > 2 % vs V13)."
        ),
    ),
    Ticket(
        key="T25_2", parent="P4", deps=["T25_1"],
        title="parent: V14_final / V14_relaxed / V14_test_aware variants",
        priority=1, estimate=120,
        labels=["phase4", "day25", "subagent-parent"],
        description="Mirror V11/V12/V13 variant generation.",
        acceptance="3 variants in output/preds/v14/.",
    ),
    Ticket(
        key="T25_AUD", parent="P4", deps=["T25_2"],
        title="auditor: V14 OOF + leakage final audit",
        priority=1, estimate=90,
        labels=["phase4", "day25", "subagent-auditor", "audit"],
        description="Lock-in audit before final tag.",
        acceptance="output/audits/v14_final_audit.md = ✅.",
    ),
    Ticket(
        key="T25_LOG", parent="P4", deps=["T25_AUD"],
        title="parent: LOG.md Day 25", priority=3, estimate=15,
        labels=["phase4", "day25", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T26_1", parent="P4", deps=["T25_2"],
        title="viz: Update viz_all_models_comparison.py with V12, V13, V14",
        priority=2, estimate=60,
        labels=["phase4", "day26", "subagent-viz"],
        description="Re-run all-models comparison incl. V12/V13/V14 variants.",
        acceptance=(
            "output/plots/comparisons/plot_all_models_comparison.png + .csv updated."
        ),
    ),
    Ticket(
        key="T26_2", parent="P4", deps=["T25_2"],
        title="viz: Contribution-attribution chart (per-ingredient ablation)",
        priority=2, estimate=180,
        labels=["phase4", "day26", "subagent-viz"],
        description=(
            "scripts/viz_v14_attribution.py: ablate each LAD ingredient (set its "
            "weight to 0, refit calibrator), record ΔOOF SIMSCORE; plot bar chart."
        ),
        acceptance="plot_v14_attribution.png + attribution.csv.",
    ),
    Ticket(
        key="T26_3", parent="P4", deps=["T25_2"],
        title="viz: V1 → V14 progression timeline",
        priority=3, estimate=60,
        labels=["phase4", "day26", "subagent-viz"],
        description="Adapt scripts/viz_v11_progression.py.",
        acceptance="plot_v14_progression.png + summary.csv.",
    ),
    Ticket(
        key="T26_LOG", parent="P4", deps=["T26_1", "T26_2", "T26_3"],
        title="parent: LOG.md Day 26", priority=3, estimate=15,
        labels=["phase4", "day26", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T27_1", parent="P4", deps=["T26_1", "T26_2", "T26_3"],
        title="viz: docs/v14_final_report.md",
        priority=1, estimate=180,
        labels=["phase4", "day27", "subagent-viz"],
        description=(
            "Comprehensive V14 narrative: all gains, attribution, recipe, OOF "
            "vs test, variants, deployment recommendation."
        ),
        acceptance="docs/v14_final_report.md committed.",
    ),
    Ticket(
        key="T27_2", parent="P4", deps=["T27_1"],
        title="viz: docs/v14_executive_summary.md",
        priority=1, estimate=90,
        labels=["phase4", "day27", "subagent-viz"],
        description="2-page exec summary for non-technical readers.",
        acceptance="docs/v14_executive_summary.md committed.",
    ),
    Ticket(
        key="T27_3", parent="P4", deps=["T27_2"],
        title="viz: Final README update with V14",
        priority=2, estimate=30,
        labels=["phase4", "day27", "subagent-viz"],
        description="README V14 update; mark V14 as champion.",
        acceptance="README updated.",
    ),
    Ticket(
        key="T27_GATE_HUMAN", parent="P4", deps=["T27_3"],
        title="[HUMAN-DECISION] Approve V14 production promotion?",
        type="decision", priority=1, estimate=15,
        labels=["phase4", "day27", "human-required", "gate-decision"],
        description=(
            "USER ACTION: read v14_final_report.md + executive_summary; review "
            "audit trail. Decide: (a) ship V14, (b) ship V14_relaxed, (c) rollback "
            "to V13. Close with reason."
        ),
        acceptance="User closes with verdict.",
    ),
    Ticket(
        key="T27_LOG", parent="P4", deps=["T27_GATE_HUMAN"],
        title="parent: LOG.md Day 27", priority=3, estimate=15,
        labels=["phase4", "day27", "subagent-parent", "log"],
        description="Daily diary.", acceptance="LOG.md updated.",
    ),
    Ticket(
        key="T28_1", parent="P4", deps=["T27_GATE_HUMAN"],
        title="parent: Tag v14.0.0 — campaign complete",
        priority=1, estimate=15,
        labels=["phase4", "day28", "subagent-parent"],
        description="Tag + push + final champion_card bump.",
        acceptance="v14.0.0 visible on origin; champion_card reflects V14.",
    ),
    Ticket(
        key="T28_LOG", parent="P4", deps=["T28_1"],
        title="parent: LOG.md final summary entry",
        priority=2, estimate=30,
        labels=["phase4", "day28", "subagent-parent", "log"],
        description="Final retrospective: what worked, what didn't, lessons.",
        acceptance="LOG.md final entry committed.",
    ),
]

# ============== PHASE EXT — Extended free open-data ingest ==============
#
# 31 ingest tickets fan out in parallel after T0_10. Five forager subagents
# divide the work by group (A: UA macro, B: war/region, C: attention,
# D: climate/cal, E: logistics). Then EXT_AB_AUDIT runs per-source A/B,
# EXT_SURVIVOR_MERGE rebuilds abt_v12_external keeping only winners.
#
# Ordering note: EXT_TASKS comes BEFORE P1_TASKS in the ALL list because
# T2_4 (in P1_TASKS) depends on the 9 priority-1 EXT_* tickets.

def _ext(
    key: str,
    title: str,
    *,
    priority: int,
    estimate: int,
    group: str,           # 'A' | 'B' | 'C' | 'D' | 'E'
    description: str,
    upstream: str = "",   # short upstream URL hint, used only in description
    publication_lag: int = 7,
    grain: str = "nat-month",
    cols: int = 3,
) -> Ticket:
    """Helper to compactly define an EXT ingest ticket with consistent fields."""

    full_desc = (
        f"{description}\n\n"
        f"Upstream: {upstream or '(see docs/v12_v14_extended_open_data.md)'}\n"
        f"Grain: {grain}; publication-lag: {publication_lag} days; "
        f"≈{cols} signal columns."
    )
    return Ticket(
        key=key, parent="P_EXT", deps=["T0_10"],
        title=title, type="task", priority=priority, estimate=estimate,
        labels=[
            "phase-ext", "open-data", "subagent-forager",
            f"ext-group-{group.lower()}",
            f"wave-ext-{group.lower()}",
        ],
        description=full_desc,
        design=(
            "Implement as src/loaders/<name>.py subclassing BaseSignalLoader. "
            "Set name, signal_cols, join_keys, publication_lag_days, upstream_url. "
            "Implement fetch_raw() (pure I/O) + transform() (monthly contract). "
            "Register via @register_loader. Cache lands at "
            "output/external/<name>.parquet + .meta.json with TTL=7 days. "
            "Stale-cache fallback on network failure. UTF-8 / Cyrillic safe."
        ),
        acceptance=(
            "Loader importable from src.loaders.<name>; cache parquet exists "
            "with ≥ 60 monthly rows (or pair-static row count); meta.json "
            "valid; smoke-test in tests/test_loader_<name>.py passes; loader "
            "registered in src/loaders/__init__.py."
        ),
    )


EXT_TASKS: list[Ticket] = [
    # ----- GROUP A: UA macro & demographics (8 tickets) -----
    _ext("EXT_UKRSTAT_RTI",
         "forager-A: Ukrstat Retail Trade Index (índex roздрібного товарообігу)",
         priority=1, estimate=120, group="A",
         description=(
             "Pull monthly Retail Trade Index from ukrstat.gov.ua (open data CSVs "
             "or scraped tables). Yields 4 cols: rti_total, rti_food, rti_nonfood, "
             "rti_yoy_pct. Strong macro proxy for discretionary spend."
         ),
         upstream="https://ukrstat.gov.ua/operativ/operativ2024/sr/srt/",
         publication_lag=30, grain="nat-month", cols=4),

    _ext("EXT_UKRSTAT_BIRTHS",
         "forager-A: Ukrstat births by oblast × month (toy demand cohort)",
         priority=1, estimate=150, group="A",
         description=(
             "Births by oblast × month. Toy demand strongly cohort-dependent: "
             "0-3y items track births lagged 0-36 mo; 4-7y track births lagged "
             "48-84 mo. Compute lagged cohort population proxies."
         ),
         upstream="https://ukrstat.gov.ua/operativ/operativ2023/ds/kn/",
         publication_lag=60, grain="obl-month", cols=6),

    _ext("EXT_UKRSTAT_INDPROD",
         "forager-A: Ukrstat Industrial Production Index",
         priority=1, estimate=90, group="A",
         description=(
             "Monthly Industrial Production Index. Proxy for general economic "
             "health; correlates with B2B partner ordering cycles."
         ),
         upstream="https://ukrstat.gov.ua/operativ/operativ2023/pr/iovp/",
         publication_lag=30, grain="nat-month", cols=3),

    _ext("EXT_UKRSTAT_WAGES",
         "forager-A: Ukrstat avg wages by industry/region",
         priority=2, estimate=120, group="A",
         description=(
             "Avg monthly wages by oblast and by industry. Captures purchasing-"
             "power shifts. Especially useful per-partner via partner-region join."
         ),
         upstream="https://ukrstat.gov.ua/operativ/operativ2023/gdn/Zarp_ek_p/",
         publication_lag=60, grain="obl-month", cols=4),

    _ext("EXT_NBU_CCI",
         "forager-A: NBU Consumer Confidence + Inflation Expectations",
         priority=1, estimate=90, group="A",
         description=(
             "NBU monthly household survey: consumer confidence index, inflation "
             "expectations, business climate. Leading indicator for discretionary "
             "spend like toys."
         ),
         upstream="https://bank.gov.ua/ua/statistic/sector-financial/data-sector-financial",
         publication_lag=7, grain="nat-month", cols=5),

    _ext("EXT_NBU_RATES",
         "forager-A: NBU discount + lending rates",
         priority=2, estimate=60, group="A",
         description=(
             "NBU discount rate + commercial lending rates daily, aggregated "
             "monthly (avg, min, max). Captures credit-tightening regimes."
         ),
         upstream="https://bank.gov.ua/ua/statistic/sector-financial/data-sector-financial",
         publication_lag=1, grain="nat-day-aggm", cols=4),

    _ext("EXT_CUSTOMS_UA",
         "forager-A: UA Customs Service detailed HS-95 imports",
         priority=2, estimate=120, group="A",
         description=(
             "UA State Customs Service open data: detailed HS chapter 95 (toys, "
             "games, sports goods) imports by month, by country of origin. "
             "Complements UN Comtrade with sub-chapter resolution."
         ),
         upstream="https://customs.gov.ua/en/open-data",
         publication_lag=30, grain="nat-month", cols=6),

    _ext("EXT_FAO_FOOD",
         "forager-A: FAO Global Food Price Index",
         priority=3, estimate=60, group="A",
         description=(
             "FAO monthly food price index. Captures household-budget pressure "
             "(food inflation crowds out toy spend)."
         ),
         upstream="https://www.fao.org/worldfoodsituation/foodpricesindex/en/",
         publication_lag=7, grain="global-month", cols=2),

    # ----- GROUP B: War / regional (5 tickets) -----
    _ext("EXT_AIRRAID_OBLAST",
         "forager-B: alerts.in.ua oblast × day air-raid alarms (granular)",
         priority=1, estimate=120, group="B",
         description=(
             "Free API at alerts.in.ua: oblast × timestamp air-raid events. "
             "Aggregate to monthly: alarm_hours_total, alarm_count, "
             "max_streak_days_in_month, by oblast. Beats current national-only "
             "signal in resolution."
         ),
         upstream="https://api.alerts.in.ua/v1/",
         publication_lag=1, grain="obl-month", cols=4),

    _ext("EXT_BLACKOUT_DTEK",
         "forager-B: DTEK / Ukrenergo blackout schedule (oblast hours)",
         priority=1, estimate=180, group="B",
         description=(
             "Scrape DTEK + Ukrenergo published blackout schedules (Telegram + "
             "site). Compute: blackout_hours_per_day_avg, total_blackout_days, "
             "by oblast × month. Behavioural proxy for indoor play / "
             "non-electric toy demand spike."
         ),
         upstream="https://www.dtek.com/ + https://ua.energy/",
         publication_lag=7, grain="obl-month", cols=3),

    _ext("EXT_IOM_IDP",
         "forager-B: IOM Displacement Tracking Matrix (IDP flows by destination)",
         priority=1, estimate=90, group="B",
         description=(
             "IOM-DTM monthly survey: IDP counts by destination oblast. Major "
             "demographic shift driver since 2022; partners in Lviv/Zakarpattia "
             "saw +40-60 % child-population in 2022-2023."
         ),
         upstream="https://dtm.iom.int/ukraine",
         publication_lag=30, grain="obl-month", cols=3),

    _ext("EXT_UNHCR",
         "forager-B: UNHCR refugee outflows by destination",
         priority=2, estimate=60, group="B",
         description=(
             "UNHCR refugee data portal: monthly outflows from UA by destination "
             "country. Proxy for outbound child-population draining demand."
         ),
         upstream="https://data.unhcr.org/en/situations/ukraine",
         publication_lag=30, grain="nat-month", cols=2),

    _ext("EXT_DEEPSTATE",
         "forager-B: DeepStateMap frontline shifts (km² controlled)",
         priority=3, estimate=120, group="B",
         description=(
             "DeepStateMap published monthly: km² of territory under UA control. "
             "Δ km² = frontline shift, proxy for territorial-control change "
             "shocks (returning markets, evacuated markets)."
         ),
         upstream="https://deepstatemap.live/",
         publication_lag=7, grain="nat-month", cols=2),

    # ----- GROUP C: Attention / cultural (7 tickets) -----
    _ext("EXT_WIKI_PV",
         "forager-C: Wikipedia Pageviews (en+uk+ru) for top-30 brand pages",
         priority=1, estimate=120, group="C",
         description=(
             "Wikipedia Pageviews API (free, no auth). For top 30 toy "
             "brands/franchises (Lego, Barbie, Hot Wheels, Funko, Pokémon, "
             "Marvel, Disney, Squishmallow, Among Us, Minecraft, etc), pull "
             "monthly pageviews in en + uk + ru. Strong leading indicator of "
             "brand attention."
         ),
         upstream="https://wikimedia.org/api/rest_v1/metrics/pageviews/",
         publication_lag=1, grain="nat-month", cols=30),

    _ext("EXT_YOUTUBE",
         "forager-C: YouTube Data API v3 — top kids unboxing channel views",
         priority=2, estimate=180, group="C",
         description=(
             "YouTube Data API v3 (free quota = 10 000 units/day). Pull monthly "
             "view counts for ~20 top UA / RU / EN-language kids unboxing channels. "
             "Strong leading indicator (kids see → demand toy)."
         ),
         upstream="https://developers.google.com/youtube/v3 (free key signup)",
         publication_lag=1, grain="nat-month", cols=8),

    _ext("EXT_REDDIT",
         "forager-C: Reddit brand mention counts (Pushshift archive)",
         priority=3, estimate=120, group="C",
         description=(
             "Pushshift archive (free dump) for Reddit. Count monthly mentions "
             "of toy brands / franchises in r/Lego, r/funkopop, r/toys, r/parenting, "
             "r/babyshop. Note: Pushshift access has been intermittent post-2023; "
             "fall back to Pullpush.io free mirror if needed."
         ),
         upstream="https://pullpush.io/api/",
         publication_lag=1, grain="en-month", cols=6),

    _ext("EXT_TMDB_KIDS",
         "forager-C: TMDb extended kids/family release calendar (G/PG)",
         priority=2, estimate=60, group="C",
         description=(
             "Extend existing tmdb_movies loader: filter to certification G/PG, "
             "genre Family/Animation/Adventure. Compute monthly: "
             "kids_film_releases_count, top_release_predicted_revenue, "
             "franchise_continuation_flag. Drives licensed-toy demand spikes."
         ),
         upstream="https://developers.themoviedb.org/3 (free API key)",
         publication_lag=1, grain="global-month", cols=4),

    _ext("EXT_BOX_OFFICE",
         "forager-C: Box Office Mojo top opening weekends (kids films)",
         priority=3, estimate=90, group="C",
         description=(
             "Scrape Box Office Mojo monthly top-opening films, filter kids "
             "(rated G/PG, family/animation). Convert to monthly "
             "kids_film_box_office_usd; rolling 3-mo for licensed-toy halo."
         ),
         upstream="https://www.boxofficemojo.com/",
         publication_lag=7, grain="global-month", cols=3),

    _ext("EXT_STEAM",
         "forager-C: Steam Charts CCU on top kid/family games",
         priority=3, estimate=90, group="C",
         description=(
             "Steam Charts free site: monthly avg / peak CCU for top family-rated "
             "games (Roblox-like substitutes, Among Us, Stardew Valley, Minecraft). "
             "Older-kid substitution effect."
         ),
         upstream="https://steamcharts.com/",
         publication_lag=1, grain="global-month", cols=3),

    _ext("EXT_ROBLOX",
         "forager-C: Roblox monthly active users",
         priority=3, estimate=60, group="C",
         description=(
             "Roblox investor releases publish monthly DAU / MAU. Proxy for "
             "8-12 yr substitute consumption."
         ),
         upstream="https://corp.roblox.com/news-room/",
         publication_lag=30, grain="global-month", cols=2),

    # ----- GROUP D: Climate / calendar (5 tickets) -----
    _ext("EXT_OPENMETEO",
         "forager-D: Open-Meteo regional weather (oblast capitals, granular)",
         priority=2, estimate=120, group="D",
         description=(
             "Open-Meteo free historical API (no auth, no quota). Pull daily "
             "T_avg, T_min, T_max, precipitation, snow_depth for 25 oblast "
             "capitals 2020-01 → 2026-04. Aggregate to monthly. More granular "
             "than current national-only weather_ua."
         ),
         upstream="https://open-meteo.com/en/docs/historical-weather-api",
         publication_lag=1, grain="obl-month", cols=12),

    _ext("EXT_DAYLIGHT",
         "forager-D: Daylight hours by latitude × month (computed)",
         priority=3, estimate=30, group="D",
         description=(
             "Deterministic computation from astropy or formula: daylight hours "
             "per latitude × month for 25 oblast capitals. No API needed. Behavioral "
             "proxy: long days = outdoor play, short days = indoor toys."
         ),
         upstream="(deterministic, astropy)",
         publication_lag=0, grain="obl-month", cols=2),

    _ext("EXT_ORTHODOX_CAL",
         "forager-D: Orthodox religious calendar (Easter, Christmas, St. Nicholas)",
         priority=1, estimate=60, group="D",
         description=(
             "Orthodox Easter date varies year-to-year (Julian-Gregorian). Encode: "
             "is_orthodox_easter_month, orthodox_easter_date, days_to_st_nicholas, "
             "days_to_orthodox_christmas, christmas_school_break_overlap_days. "
             "Strong UA-cultural toy-gift signal."
         ),
         upstream="(deterministic, dateutil + manual table)",
         publication_lag=0, grain="nat-month", cols=5),

    _ext("EXT_LUNAR",
         "forager-D: Lunar phases per month",
         priority=4, estimate=30, group="D",
         description=(
             "Some retail studies show full-moon weekend purchase spikes. "
             "Cheap to add (deterministic ephemeris)."
         ),
         upstream="(deterministic, astropy)",
         publication_lag=0, grain="nat-month", cols=2),

    _ext("EXT_AQI",
         "forager-D: OpenAQ air quality index (oblast capitals)",
         priority=3, estimate=120, group="D",
         description=(
             "OpenAQ free API for PM2.5 / PM10 / NO2 in oblast capitals, monthly "
             "avg. Bad-air days → kids stay indoors → indoor-toy substitution."
         ),
         upstream="https://docs.openaq.org/",
         publication_lag=1, grain="obl-month", cols=4),

    # ----- GROUP E: Logistics / commerce (6 tickets) -----
    _ext("EXT_GOOGLE_MOB",
         "forager-E: Google Mobility Reports (UA, retail/recreation/transit)",
         priority=2, estimate=60, group="E",
         description=(
             "Google COVID-19 Mobility Reports (still updated for some countries). "
             "If discontinued for UA, fall back to historical archive 2020-2022. "
             "Captures behavioural mobility in retail/recreation."
         ),
         upstream="https://www.google.com/covid19/mobility/",
         publication_lag=7, grain="obl-month", cols=4),

    _ext("EXT_OSM",
         "forager-E: OpenStreetMap competitor density per partner",
         priority=2, estimate=180, group="E",
         description=(
             "Geocode partner addresses via Nominatim (free); query Overpass API "
             "for shop=toys + shop=baby_goods within 2 km radius. Static feature: "
             "competitor_count_2km, big_competitor_within_5km. Pair-static, joins on partner."
         ),
         upstream="https://overpass-api.de/ + https://nominatim.org/",
         publication_lag=0, grain="pair-static", cols=4),

    _ext("EXT_BRENT",
         "forager-E: Brent crude + commodity prices",
         priority=3, estimate=60, group="E",
         description=(
             "Yfinance (free) or worldbank for monthly Brent crude USD/bbl + "
             "wheat + corn USD/t. Fuel-price proxy for logistics; agricultural "
             "for household budget."
         ),
         upstream="(yfinance, worldbank)",
         publication_lag=1, grain="global-month", cols=4),

    _ext("EXT_BALTIC_DRY",
         "forager-E: Baltic Dry Index + container shipping rates",
         priority=3, estimate=90, group="E",
         description=(
             "Baltic Exchange publishes BDI; Drewry publishes weekly WCI free "
             "quote. Container freight cost is significant for Asia→UA toys "
             "supply chain."
         ),
         upstream="https://www.balticexchange.com/ + https://www.drewry.co.uk/",
         publication_lag=1, grain="global-month", cols=3),

    _ext("EXT_MARINE",
         "forager-E: Marine Traffic Black Sea ports congestion",
         priority=4, estimate=120, group="E",
         description=(
             "Marine Traffic free tier (rate-limited) for Odesa / Chornomorsk / "
             "Pivdennyi port: vessel arrivals, anchorage queue. Proxy for import-"
             "delay risk for toys arriving by sea."
         ),
         upstream="https://www.marinetraffic.com/en/data/api",
         publication_lag=1, grain="port-month", cols=4),

    _ext("EXT_BTC",
         "forager-E: BTC + ETH + USDT prices (UA crypto adoption proxy)",
         priority=4, estimate=60, group="E",
         description=(
             "CoinGecko free API: monthly OHLC for BTC, ETH, USDT in USD. UA "
             "has high consumer crypto adoption; crypto wealth shocks correlate "
             "with discretionary spend."
         ),
         upstream="https://www.coingecko.com/en/api",
         publication_lag=1, grain="global-month", cols=6),

    # ----- SYNTHESIS & GATING -----
    Ticket(
        key="EXT_AB_AUDIT", parent="P_EXT",
        deps=[
            # Wait for EVERY ingest ticket so we can A/B every column. Lower-prio
            # sources that fail to ingest become 'wontfix'; the auditor handles
            # that gracefully and skips them in the per-source ablation.
            "EXT_UKRSTAT_RTI", "EXT_UKRSTAT_BIRTHS", "EXT_UKRSTAT_INDPROD",
            "EXT_UKRSTAT_WAGES", "EXT_NBU_CCI", "EXT_NBU_RATES",
            "EXT_CUSTOMS_UA", "EXT_FAO_FOOD",
            "EXT_AIRRAID_OBLAST", "EXT_BLACKOUT_DTEK", "EXT_IOM_IDP",
            "EXT_UNHCR", "EXT_DEEPSTATE",
            "EXT_WIKI_PV", "EXT_YOUTUBE", "EXT_REDDIT", "EXT_TMDB_KIDS",
            "EXT_BOX_OFFICE", "EXT_STEAM", "EXT_ROBLOX",
            "EXT_OPENMETEO", "EXT_DAYLIGHT", "EXT_ORTHODOX_CAL",
            "EXT_LUNAR", "EXT_AQI",
            "EXT_GOOGLE_MOB", "EXT_OSM", "EXT_BRENT",
            "EXT_BALTIC_DRY", "EXT_MARINE", "EXT_BTC",
        ],
        title="auditor: Per-source A/B ablation (which EXT_* sources actually help?)",
        type="task", priority=1, estimate=240,
        labels=["phase-ext", "subagent-auditor", "audit", "decision"],
        description=(
            "scripts/v12_ext_ab_audit.py: for each EXT_* source, train two "
            "LightGBM models: baseline (no source) vs treatment (baseline + "
            "source columns only). Evaluate Δ OOF SIMSCORE val over 5 folds. "
            "Decision rule: keep iff Δ ≥ +0.10 % AND bias not worsened > 0.5 %. "
            "Survivor list → output/audits/ext_survivors.json. Audit narrative "
            "→ output/audits/ext_ab_audit.md."
        ),
        design=(
            "Reuse src/external_data.py loaders to fetch each source's columns. "
            "Train via scripts/train_v7.py with --feature-subset. Aggregate "
            "OOF SIMSCORE per source. Sort by Δ. Print sortable table. "
            "Sources marked wontfix in beads are skipped."
        ),
        acceptance=(
            "ext_survivors.json lists ≥ 8 survivors; ext_ab_audit.md committed "
            "with sortable per-source table; if < 8 survive, escalation ticket "
            "auto-created (manual review)."
        ),
    ),
    Ticket(
        key="EXT_SURVIVOR_MERGE", parent="P_EXT", deps=["EXT_AB_AUDIT", "T2_4"],
        title="trainer: Rebuild abt_v12_external with EXT survivors merged",
        type="task", priority=1, estimate=120,
        labels=["phase-ext", "subagent-trainer"],
        description=(
            "scripts/v12_ext_survivor_merge.py: read ext_survivors.json; for "
            "any priority-2/3 source on the list that wasn't already in the "
            "initial T2_4 build, append its columns to abt_v12_external.parquet "
            "(preserving the leakage cutoff). For sources NOT on the survivor "
            "list (including priority-1 ones if they fail audit — rare but "
            "honest), drop their columns from the ABT. Re-run leakage test."
        ),
        design=(
            "Idempotent rebuild. Snapshot pre-merge ABT to "
            "output/abt_v12_external.pre_merge.parquet for diff. Diff committed "
            "to output/audits/ext_merge_diff.md."
        ),
        acceptance=(
            "abt_v12_external.parquet survivor-merged; "
            "tests/test_v12_abt_no_leakage.py passes; "
            "output/audits/ext_merge_diff.md committed; columns count "
            "≥ baseline + 8."
        ),
    ),
    Ticket(
        key="EXT_LOG", parent="P_EXT", deps=["EXT_SURVIVOR_MERGE"],
        title="parent: LOG.md Phase EXT retrospective",
        type="task", priority=3, estimate=20,
        labels=["phase-ext", "subagent-parent", "log"],
        description=(
            "Append a Phase EXT entry to LOG.md: how many sources attempted, "
            "how many survived A/B, total ABT column count delta, what dropped "
            "and why. Commit + push."
        ),
        acceptance="LOG.md updated; commit pushed.",
    ),
]


# Ordering matters at create time: T2_4 (in P1_TASKS) depends on the
# priority-1 EXT_* ingest tickets, so EXT_TASKS must be created BEFORE
# P1_TASKS. Similarly, T4_3 depends on EXT_SURVIVOR_MERGE, which is also
# in EXT_TASKS — same ordering constraint.
ALL: list[Ticket] = (
    EPICS + P0_TASKS + EXT_TASKS + P1_TASKS + P2_TASKS + P3_TASKS + P4_TASKS
)

# ---------------------------------------------------------------- HELPERS

def _validate(tickets: list[Ticket]) -> None:
    keys = {t.key for t in tickets}
    if len(keys) != len(tickets):
        raise SystemExit("Duplicate ticket keys")
    for t in tickets:
        for d in t.deps:
            if d not in keys:
                raise SystemExit(f"{t.key} depends on missing {d}")
        if t.parent and t.parent not in keys:
            raise SystemExit(f"{t.key} parent missing: {t.parent}")
    indeg = {k: 0 for k in keys}
    for t in tickets:
        for d in t.deps:
            indeg[t.key] += 1
    queue = [k for k, v in indeg.items() if v == 0]
    visited = 0
    by_key = {t.key: t for t in tickets}
    blockers_of: dict[str, list[str]] = defaultdict(list)
    for t in tickets:
        for d in t.deps:
            blockers_of[d].append(t.key)
    while queue:
        k = queue.pop(0); visited += 1
        for child in blockers_of[k]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    if visited != len(tickets):
        raise SystemExit(f"DEPENDENCY CYCLE detected ({visited}/{len(tickets)})")
    print(f"✅ Validated {len(tickets)} tickets, no cycles, "
          f"{sum(len(t.deps) for t in tickets)} dependencies")


def _waves(tickets: list[Ticket]) -> dict[int, list[Ticket]]:
    """Compute parallel waves: wave i = tickets whose deps are all in wave < i."""
    by_key = {t.key: t for t in tickets}
    wave_of: dict[str, int] = {}
    remaining = set(by_key.keys())
    wave = 0
    while remaining:
        ready = [k for k in remaining
                 if all(d in wave_of for d in by_key[k].deps)]
        if not ready:
            raise SystemExit("Stuck — cycle?")
        for k in ready:
            wave_of[k] = wave
        remaining -= set(ready)
        wave += 1
    waves: dict[int, list[Ticket]] = defaultdict(list)
    for k, w in wave_of.items():
        waves[w].append(by_key[k])
    return waves


def _print_waves(waves: dict[int, list[Ticket]]) -> None:
    print("\n=== PARALLEL EXECUTION WAVES ===")
    print("(tickets in same wave have no dependency between them, can run concurrently)\n")
    for w in sorted(waves):
        ts = waves[w]
        humans = [t for t in ts if "human-required" in t.labels]
        agents = [t for t in ts if t.type != "epic" and "human-required" not in t.labels]
        epics = [t for t in ts if t.type == "epic"]
        print(f"--- Wave {w}: {len(ts)} tickets "
              f"(epics={len(epics)} agent={len(agents)} human={len(humans)}) ---")
        for t in ts:
            mark = "👤" if "human-required" in t.labels else ("📦" if t.type == "epic" else "🤖")
            print(f"  {mark} {t.key:<24} {t.title[:80]}")


def _print_human_actions(tickets: list[Ticket]) -> None:
    print("\n=== HUMAN-REQUIRED TICKETS (every click the user has to do) ===\n")
    humans = [t for t in tickets if "human-required" in t.labels]
    for t in humans:
        print(f"  {t.key:<28} ({t.estimate}m)  {t.title}")
    print(f"\nTotal user clicks: {len(humans)}  "
          f"≈ {sum(t.estimate for t in humans)} min")


def _bd_create(t: Ticket, key_to_id: dict[str, str]) -> str:
    cmd = ["bd", "create",
           "--silent",
           "--title", t.title,
           "--type", t.type,
           "--priority", str(t.priority),
           "--description", t.description or t.title,
           "--estimate", str(t.estimate),
           ]
    if t.parent:
        cmd += ["--parent", key_to_id[t.parent]]
    if t.design:
        cmd += ["--design", t.design]
    if t.acceptance:
        cmd += ["--acceptance", t.acceptance]
    if t.labels:
        cmd += ["--labels", ",".join(t.labels)]
    if t.defer:
        cmd += ["--defer", t.defer]
    if t.deps:
        # Pass deps inline at create time (avoid 175 separate `bd dep add` calls
        # which would each spin up a fresh bd process and slow down as the
        # graph grows). Format: 'blocks:bd-XX,blocks:bd-YY'. We use 'blocks'
        # because beads' --deps semantic is "this issue depends on X" via
        # default 'blocks' relation.
        deps_str = ",".join(key_to_id[d] for d in t.deps)
        cmd += ["--deps", deps_str]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true",
                    help="Actually create the tickets in beads")
    ap.add_argument("--validate", action="store_true",
                    help="Only validate the local graph")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview without creating (default)")
    ap.add_argument("--print-waves", action="store_true",
                    help="Print computed parallel waves")
    ap.add_argument("--print-humans", action="store_true",
                    help="Print every human-required ticket")
    args = ap.parse_args()

    _validate(ALL)
    waves = _waves(ALL)
    print(f"Tickets: {len(ALL)}  ({sum(1 for t in ALL if t.type=='epic')} epics)")
    print(f"Total deps: {sum(len(t.deps) for t in ALL)}")
    print(f"Parallel waves: {len(waves)}")
    print(f"Max wave fan-out: {max(len(ts) for ts in waves.values())}")

    if args.print_waves:
        _print_waves(waves)
    if args.print_humans:
        _print_human_actions(ALL)
    if args.validate or args.dry_run:
        return 0
    if not args.create:
        print("\n(Pass --create to actually populate beads)")
        return 0

    print("\n=== Creating tickets in beads (deps inline) ===", flush=True)
    key_to_id: dict[str, str] = {}
    keymap_path = ".beads/v12_v14_keymap.json"
    for i, t in enumerate(ALL, 1):
        if t.parent and t.parent not in key_to_id:
            raise SystemExit(f"Parent of {t.key} ({t.parent}) not yet created")
        for d in t.deps:
            if d not in key_to_id:
                raise SystemExit(
                    f"Dep order error: {t.key} depends on {d}, which has not "
                    f"been created yet. Reorder ALL list."
                )
        bd_id = _bd_create(t, key_to_id)
        key_to_id[t.key] = bd_id
        print(f"  [{i:>3}/{len(ALL)}] + {t.key:<28} → {bd_id}", flush=True)
        # Persist keymap incrementally so a kill mid-run can be recovered.
        with open(keymap_path, "w") as f:
            json.dump(key_to_id, f, indent=2)

    print(f"\n✅ Done. {sum(len(t.deps) for t in ALL)} dependencies wired inline.",
          flush=True)
    print("   Run `bd ready` to see immediately-actionable tickets.")
    print(f"   Keymap saved to {keymap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
