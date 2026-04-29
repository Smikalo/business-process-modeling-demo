# V12-V14 Extended Open-Data Expansion — Phase EXT

**Status:** approved 2026-04-29, additive to docs/v12_v14_roadmap.md (Draft 3)

This document expands the V12-V14 plan with **31 additional free open-data
sources** that we have not yet exploited, organised into a new parallel
sub-epic `P_EXT`. All sources are:

* **Genuinely free** — no paid tiers, no card-on-file, no quota that we'd
  realistically blow through in one campaign.
* **Honest about leakage** — every source declares a publication-lag in days;
  the leakage guard in `src/external_data.py` blocks any feature for month M
  that was sourced from data with date ≥ first day of M.
* **UA-grain compatible** — either national-monthly, oblast-monthly, or
  pair-static (geo lookups). No daily-only sources are merged into the monthly
  ABT directly; they are aggregated.
* **Realistic** — combined expected lift is **+3-7 p.p. test SIMSCORE**, not
  +30. The point is to honestly close the gap between V11 (63%) and the
  irreducible-noise ceiling (~70% with the data we can legally and freely
  obtain) — see `docs/limitations-and-next-steps.md` for why 90%+ requires
  POS-level data we cannot get for free.

---

## Source list — 31 new ingest tickets

Group       | Key                       | Source                                                        | Grain          | Lag (d) | Expected lift | Pri
------------|---------------------------|---------------------------------------------------------------|----------------|---------|----------------|----
A — UA macro    | `EXT_UKRSTAT_RTI`         | Ukrstat Retail Trade Index (індекс роздрібного товарообігу)   | nat-month      | 30      | +0.5–1.0       | 1
A — UA macro    | `EXT_UKRSTAT_BIRTHS`      | Ukrstat births by oblast × month                              | obl-month      | 60      | +0.4–0.7       | 1
A — UA macro    | `EXT_UKRSTAT_INDPROD`     | Ukrstat Industrial Production Index                           | nat-month      | 30      | +0.3–0.5       | 1
A — UA macro    | `EXT_UKRSTAT_WAGES`       | Ukrstat average wages by industry/region                      | nat-month      | 60      | +0.2–0.4       | 2
A — UA macro    | `EXT_NBU_CCI`             | NBU Consumer Confidence Index + Inflation Expectations        | nat-month      | 7       | +0.3–0.5       | 1
A — UA macro    | `EXT_NBU_RATES`           | NBU discount + lending rates                                  | nat-day→month  | 1       | +0.1–0.3       | 2
A — UA macro    | `EXT_CUSTOMS_UA`          | UA State Customs Service: detailed HS-95 imports              | nat-month      | 30      | +0.2–0.4       | 2
A — UA macro    | `EXT_FAO_FOOD`            | FAO Global Food Price Index                                   | nat-month      | 7       | +0.1–0.3       | 3
B — War/region  | `EXT_AIRRAID_OBLAST`      | alerts.in.ua: oblast × day air-raid alarms                    | obl-day→month  | 1       | +0.3–0.5       | 1
B — War/region  | `EXT_BLACKOUT_DTEK`       | DTEK / Ukrenergo blackout schedule (hours/day by oblast)      | obl-day→month  | 7       | +0.5–1.0       | 1
B — War/region  | `EXT_IOM_IDP`             | IOM Displacement Tracking Matrix: IDP flows by destination    | obl-month      | 30      | +0.5–1.0       | 1
B — War/region  | `EXT_UNHCR`               | UNHCR refugee outflows by destination country                 | nat-month      | 30      | +0.2–0.4       | 2
B — War/region  | `EXT_DEEPSTATE`           | DeepStateMap frontline shift (km², proxy for territorial control change) | nat-month | 7 | +0.1–0.3 | 3
C — Attention   | `EXT_WIKI_PV`             | Wikipedia Pageviews (en+uk+ru) for top-30 brand/franchise pages | nat-month   | 1       | +0.3–0.7       | 1
C — Attention   | `EXT_YOUTUBE`             | YouTube Data API v3: views on top kids unboxing channels      | nat-month      | 1       | +0.3–0.5       | 2
C — Attention   | `EXT_REDDIT`              | Reddit (Pushshift archive): brand mention counts              | en-month       | 1       | +0.1–0.2       | 3
C — Attention   | `EXT_TMDB_KIDS`           | TMDb extended: G/PG family release calendar w/ revenue        | global-month   | 1       | +0.2–0.4       | 2
C — Attention   | `EXT_BOX_OFFICE`          | Box Office Mojo top opening weekends (kids movies)            | global-month   | 7       | +0.1–0.3       | 3
C — Attention   | `EXT_STEAM`               | Steam Charts CCU on top kid/family games                      | global-month   | 1       | +0.1–0.2       | 3
C — Attention   | `EXT_ROBLOX`              | Roblox monthly active users (substitute toy)                  | global-month   | 30      | +0.1–0.2       | 3
D — Climate/cal | `EXT_OPENMETEO`           | Open-Meteo regional weather aggregates (oblast capitals)      | obl-month      | 1       | +0.2–0.4       | 2
D — Climate/cal | `EXT_DAYLIGHT`            | Daylight hours by latitude × month (computed, deterministic)  | obl-month      | 0       | +0.1–0.2       | 3
D — Climate/cal | `EXT_ORTHODOX_CAL`        | Orthodox religious calendar (Easter, St. Nicholas, Christmas) | nat-month      | 0       | +0.2–0.4       | 1
D — Climate/cal | `EXT_LUNAR`               | Lunar phases per month                                        | nat-month      | 0       | +0.0–0.1       | 4
D — Climate/cal | `EXT_AQI`                 | OpenAQ air quality index (oblast capitals)                    | obl-month      | 1       | +0.1–0.2       | 3
E — Logistics   | `EXT_GOOGLE_MOB`          | Google Mobility Reports (UA, retail/recreation)               | obl-month      | 7       | +0.2–0.5       | 2
E — Logistics   | `EXT_OSM`                 | OpenStreetMap: competing toy/baby store density per partner   | pair-static    | n/a     | +0.2–0.4       | 2
E — Logistics   | `EXT_BRENT`               | Brent crude oil + USDC commodities                            | nat-day→month  | 1       | +0.1–0.3       | 3
E — Logistics   | `EXT_BALTIC_DRY`          | Baltic Dry Index + container shipping rates                   | global-day→month | 1     | +0.1–0.2       | 3
E — Logistics   | `EXT_MARINE`              | Marine Traffic Black Sea: Odesa/Chornomorsk port congestion   | port-day→month | 1       | +0.0–0.2       | 4
E — Logistics   | `EXT_BTC`                 | BTC + ETH + USDT prices (UA crypto adoption proxy)            | global-day→month | 1     | +0.0–0.2       | 4

**Combined optimistic lift:** +3.7–7.4 p.p. (sum of upper estimates).
**Combined realistic lift after diminishing returns:** +2-4 p.p.
**Combined pessimistic (if half are vetoed by A/B audit):** +1-2 p.p.

---

## Implementation pattern (reuse `src/external_data.py`)

Each EXT loader follows the existing `BaseSignalLoader` ABC:

* `name = "ukrstat_rti"` (unique short identifier)
* `signal_cols = ["retail_trade_idx", "retail_trade_idx_yoy", ...]`
* `join_keys = ["Период"]` for national, `["Период", "Регіон"]` for oblast
* `publication_lag_days` — declared explicitly so the leakage guard works
* `fetch_raw()` — pure HTTP / file I/O, no transformation
* `transform(raw)` — yields a monthly DataFrame matching the contract
* registered via `@register_loader` decorator

Cache lives at `output/external/<name>.parquet` + `<name>.meta.json`. TTL
defaults to 7 days. Failures fall back to stale cache with a warning.

---

## Dependency wiring & schedule

Phase EXT is a **new parallel sub-epic under ROOT** (siblings with P0–P4).
It runs entirely on **CPU**, so it competes with no GPU resources.

**Wave structure:**

```
Wave A (deps T0_10):  9 priority-1 ingest tickets fan out, all in parallel
Wave B (deps T0_10): 12 priority-2/3 ingest tickets fan out, all in parallel
Wave C (deps each ingest finishing): EXT_AB_AUDIT
Wave D (deps EXT_AB_AUDIT):           EXT_SURVIVOR_MERGE
                                       └─→ rebuilds abt_v12_external in place
Wave E (deps EXT_SURVIVOR_MERGE):     T2_4 (existing) gets the survivors merged
```

**Dependency surgery applied to existing graph:**

1. `T2_4` (Build `abt_v12_external.parquet`) gains new deps:
   `EXT_UKRSTAT_RTI, EXT_UKRSTAT_BIRTHS, EXT_UKRSTAT_INDPROD, EXT_NBU_CCI,
    EXT_AIRRAID_OBLAST, EXT_BLACKOUT_DTEK, EXT_IOM_IDP, EXT_WIKI_PV,
    EXT_ORTHODOX_CAL` (the 9 priority-1 tickets).
   Lower-priority sources are **not** blocking T2_4; they enter via the
   survivor-merge path after the per-source A/B audit.

2. `T3_3` (existing external A/B test) is reframed as the **shell** for the
   per-source ablation done in `EXT_AB_AUDIT`. The existing ticket title
   stays; the implementation runs N+M per-source ablations instead of just one.

3. `T4_3` (V12.5 LAD search) gains a `EXT_SURVIVOR_MERGE` dep so that
   any priority-2/3 sources that survive the A/B gate are **already merged
   into the ABT used by Day 4 LAD**.

---

## Per-source A/B audit — `EXT_AB_AUDIT`

For every EXT_* source, we run two LightGBM trainings:
1. Baseline = current ABT (no new source)
2. Treatment = baseline + the source's columns only

We compute `Δ OOF SIMSCORE val = treatment - baseline` over 5 OOF folds.
**Decision rule per source:**

* `Δ ≥ +0.10 %` and bias not worsened by > 0.5 % → **keep**
* otherwise → **drop**

Survivor list lands in `output/audits/ext_survivors.json`. The merge
ticket reads this and rebuilds the ABT keeping only winners. Audit log
in `output/audits/ext_ab_audit.md`.

This is critical: we don't want to add 31 noisy columns and overfit. The
A/B gate is the discipline.

---

## Subagent ownership map

* `subagent-forager-A` — Group A (UA macro, 8 tickets)
* `subagent-forager-B` — Group B (war/region, 5 tickets)
* `subagent-forager-C` — Group C (attention, 7 tickets)
* `subagent-forager-D` — Group D (climate/cal, 5 tickets)
* `subagent-forager-E` — Group E (logistics, 6 tickets)
* `subagent-auditor` — `EXT_AB_AUDIT`
* `subagent-trainer` — `EXT_SURVIVOR_MERGE` (rebuilds ABT)
* `subagent-parent` — `EXT_LOG` daily diary

Five forager subagents fan out fully in parallel after T0_10. Each
ticket's estimate is 60–180 minutes; total subagent-wall time ≈ 4–6 hr
(longest path, not sum).

---

## Acceptance criteria for Phase EXT (rolls into V12.5 acceptance)

* [ ] All 31 ingest tickets completed (or marked `wontfix` with documented reason in audit)
* [ ] `output/audits/ext_ab_audit.md` produced with per-source verdict
* [ ] `output/audits/ext_survivors.json` lists ≥ 8 survivors (sanity floor)
* [ ] `abt_v12_external.parquet` rebuilt with survivors merged
* [ ] Leakage test `tests/test_v12_abt_no_leakage.py` still passes
* [ ] V12.5 LAD search shows ≥ +0.5 % OOF val SIMSCORE improvement vs V11.5
      simulation re-run on the same V11 baseline (ablation is in
      `output/audits/v12.5_ext_attribution.md`)

---

## Risks & mitigations

* **Risk:** A flaky source returns 500s for a week → blocks T2_4.
  **Mitigation:** Each loader has stale-cache fallback in `BaseSignalLoader`;
  worst case the ticket marks itself `wontfix` and the survivor list
  proceeds with the available 8/9 priority-1 sources.
* **Risk:** Adding 31 columns → overfit → val ↑ test ↓.
  **Mitigation:** A/B gate per-source (above) drops anything that doesn't
  earn its keep. Bias ladder in `T4_3` already constrains aggregate bias.
* **Risk:** Unicode / encoding issues with Ukrainian sources (Cyrillic).
  **Mitigation:** Existing loaders already handle UTF-8 + Cyrillic
  (e.g., `holidays_ua.py`); reuse those patterns.
* **Risk:** Web-scraping fragility for sites without official APIs
  (Ukrenergo schedule, Box Office Mojo).
  **Mitigation:** Each scraper has a manually-snapshotted CSV checked in
  as `data/external/manual_snapshots/<source>_2026-04.csv` so worst case
  we have a static fallback.
* **Risk:** Forager subagents step on each other's git locks.
  **Mitigation:** Each agent commits to `data/external/<unique_name>` only
  and pushes via beads-coordinated git hook; no merge conflicts possible.

---

## What this phase deliberately does NOT do

* No paid sources. No card-on-file trials.
* No POS / transaction-level data. That requires partner buy-in (12-18 mo
  business project, see `docs/limitations-and-next-steps.md`).
* No competitor-pricing scrapers (juridicially-grey, also paid alternatives
  excluded).
* No customer-PII data of any kind.

These remain documented as the path to 80–90% accuracy in the limitations doc,
but they are out-of-scope for this campaign by user mandate (zero-budget).
