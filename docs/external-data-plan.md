# External Data Integration Plan — Beads Tracker Map

This document maps the Beads issue tree for the external-data work. Source research lives in `docs/external-data-sources.md`; this file is the execution plan.

**Root epic:** `business-process-modeling-demo-d9h`
**Total:** 31 issues (1 root + 5 sub-epics + 25 tasks)

## Dependency topology

```
ROOT: External data signal enrichment (d9h)
│
├── E_INFRA: Infrastructure (d9h.1)   [must start first]
│   ├── I1: BaseSignalLoader ABC (d9h.1.1)            — no blockers
│   ├── I2: Parquet cache + lineage (d9h.1.2)          ← I1
│   ├── I3: Leakage guard utility (d9h.1.3)            ← I1
│   └── I4: ABT join extension (d9h.1.4)               ← I1, I3
│
├── E_EVAL: Evaluation harness (d9h.4)   [parallel w/ sources after I4]
│   ├── EV1: Ablation harness add-one (d9h.4.1)        ← I4
│   ├── EV2: Leave-one-out harness (d9h.4.2)           ← EV1
│   ├── EV3: Feature importance tracker (d9h.4.3)      ← EV1
│   ├── EV4: Per-segment error decomposition (d9h.4.4) ← EV1
│   ├── EV5: Leakage validation tests (d9h.4.5)        ← I1, I3
│   └── EV6: Decision gate + report gen (d9h.4.6)      ← EV1 + ALL A/B tasks
│
├── E_TIER_A: Fast wins (d9h.2)   [cheapest-first]
│   ├── A1: NBU FX + policy rate (d9h.2.1)             ← EV1
│   ├── A2: Ukrainian public holidays (d9h.2.2)        ← EV1
│   ├── A3: School calendar (d9h.2.3)                  ← EV1
│   ├── A4: Google Trends (d9h.2.4)                    ← EV1, I3
│   └── A5: Open-Meteo weather (d9h.2.5)               ← EV1
│
├── E_TIER_B: Advanced signals (d9h.3)   [after Tier A ablation results]
│   ├── B1: ACLED conflict events (d9h.3.1)            ← EV1, I3
│   ├── B2: Air raid alerts (d9h.3.2)                  ← EV1, I3
│   ├── B3: World Bank annual macro (d9h.3.3)          ← EV1
│   ├── B4: IMF monthly CPI (d9h.3.4)                  ← EV1, I3
│   └── B5: TMDB movie releases (d9h.3.5)              ← EV1
│
└── E_V5: V5 integration (d9h.5)   [after decision report]
    ├── V5.1: Build V5 feature set (d9h.5.1)           ← EV6
    ├── V5.2: Retune ensemble weights (d9h.5.2)        ← V5.1
    ├── V5.3: ADR-003 writeup (d9h.5.3)                ← V5.2
    ├── V5.4: Update README/pipeline (d9h.5.4)         ← V5.3
    └── V5.5: Tag release, LFS, push (d9h.5.5)         ← V5.4
```

## Execution order (suggested)

### Phase 1 — Foundation (est. 2.5 h)
1. `I1` Scaffold BaseSignalLoader (45 min)
2. `I2` Parquet cache + lineage (30 min)
3. `I3` Leakage guard (45 min)
4. `I4` ABT join extension (30 min)

Phase 1 unblocks: evaluation harness, every source loader, leakage tests.

### Phase 2 — Evaluation harness (est. 3 h)
1. `EV1` Ablation harness add-one-source (90 min) — **critical path**
2. `EV5` Leakage test suite (45 min, parallel with EV1)

Phase 2 gates every downstream source task. EV1 must be operational before any source is meaningful.

### Phase 3 — Tier A sources (est. 9 h, cheapest-first)
1. `A2` Holidays (30 min) — **smallest cost, highest upside**
2. `A1` NBU FX (60 min)
3. `A5` Open-Meteo weather (90 min)
4. `A3` School calendar (60 min, incl. CSV curation)
5. `A4` Google Trends (180 min, biggest risk)

Each task runs end-to-end: fetch → cache → join → run ablation → record delta.

### Phase 4 — Eval infrastructure (est. 2.5 h, parallel with late Phase 3)
1. `EV2` LOO harness (45 min)
2. `EV3` Feature importance tracker (30 min)
3. `EV4` Per-segment decomposition (60 min)

Enables Phase 5 decisions.

### Phase 5 — Tier B sources (est. 13 h, conditional on Tier A results)
Only fully execute Tier B if Tier A ablation reveals ≥3% WAPE headroom remaining.
Priority order if proceeding:
1. `B1` ACLED conflict events (180 min) — expected highest Tier B impact, plus unlocks B2 via partner-oblast map
2. `B5` TMDB movies (120 min) — narrow signal for licensed SKUs
3. `B3` World Bank macro (60 min) — cheap, slow-moving baseline
4. `B4` IMF monthly CPI (180 min) — SDMX complexity
5. `B2` Air raid alerts (240 min) — API-key wait time; likely redundant with B1 (evaluate LOO first)

### Phase 6 — V5 release (est. 4.5 h)
1. `EV6` Generate decision report (30 min) — synthesises all ablation results
2. `V5.1` Build V5 feature set (120 min)
3. `V5.2` Retune ensemble + baselines (45 min)
4. `V5.3` ADR-003 writeup (45 min)
5. `V5.4` Update README + pipeline (30 min)
6. `V5.5` Tag, LFS, push (30 min)

## Evaluation contract (what every source task owes)

Every A/B task's acceptance criteria includes running the ablation harness. A source is **PROMOTED** to V5 if any ONE of these holds (per `EV6` decision gate):

| Criterion | Threshold |
|-----------|-----------|
| Aggregate val WAPE delta | ≤ −0.003 (0.3 pp improvement) |
| Aggregate test WAPE delta | ≤ −0.002 |
| LOO degradation when removed | ≥ +0.002 |
| Segment-specific WAPE gain (e.g. one channel, one brand, one oblast) | ≥ 0.02 in a segment covering ≥10% of volume |
| Leakage guard test | PASS (non-negotiable) |

A source is **DROPPED** if none of the above hold. Decisions are documented per-source in `docs/external-signal-decisions.md` (auto-generated by `EV6`).

## Data hygiene rules (enforced by `I3`/`EV5`)

For every external signal, the module must declare `publication_lag_days`. When joining to ABT for month `t`:

- NBU FX, holidays, school, weather: lag = 0 (observable at month start)
- TMDB release calendar: lag = 0 (announced months ahead)
- IMF monthly CPI: lag = 45 (mid-next-month publication)
- ACLED: lag = 7 (weekly release cycle)
- Google Trends: lag = 14 (finalises 2 weeks after month end)
- Air raid alerts: lag = 2–30 (live but recommend 30-day lag for training consistency)
- World Bank annual: lag = 365 (conservative)

Tests in `EV5` MUST verify that a zero-lag version produces different predictions than the declared-lag version (proving the lag is not a no-op), and that no future-dated rows leak into training.

## API keys and credentials (documented per task)

| Source | Credential | Where |
|--------|-----------|-------|
| NBU | None | — |
| python-holidays | None | — |
| School calendar | None (hand-curated) | — |
| Google Trends | None (scraping) | — |
| Open-Meteo | None | — |
| ACLED | myACLED token | `.env.ACLED_TOKEN` |
| Air raid alerts | API key (request form) | `.env.AIRRAID_API_KEY` |
| World Bank | None | — |
| IMF | None | — |
| TMDB | API key (free registration) | `.env.TMDB_API_KEY` |

Create `.env.example` as part of `V5.4`.

## Attribution obligations (tracked in `V5.3`/`V5.4`)

- Open-Meteo: "Weather data by Open-Meteo.com" (CC BY 4.0)
- World Bank: attribution required (CC BY 4.0)
- ACLED: "Armed Conflict Location & Event Data (ACLED), acleddata.com"
- TMDB: "Powered by TMDB" logo/text in README
- NBU: recommended attribution to NBU Open Data

## How to start

```bash
bd ready -n 20               # see actionable work
bd show d9h.1.1              # start with Phase 1 task I1
bd update d9h.1.1 --status in_progress
# ...do the work...
bd update d9h.1.1 --status closed
```

## Time budget

| Phase | Est. hours | Cumulative |
|-------|-----------:|-----------:|
| 1 Foundation | 2.5 | 2.5 |
| 2 Eval harness | 3 | 5.5 |
| 3 Tier A sources | 9 | 14.5 |
| 4 Eval extensions | 2.5 | 17 |
| 5 Tier B (if pursued) | 13 | 30 |
| 6 V5 release | 4.5 | 34.5 |

**Minimum viable path** (Phases 1, 2, 3, 6 only, skip Tier B if Tier A wins are sufficient): ~19 hours.
