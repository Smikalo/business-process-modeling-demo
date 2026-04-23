# External Data Sources for Demand Signal Enrichment

This document catalogues external data sources researched for augmenting the V4 demand forecasting model. All sources listed are **free, regularly updated, and operationally feasible** within the zero-cost architecture. Ranking reflects expected predictive value × integration ease; later steps will empirically validate which ones move the needle.

**Business context:** Ukrainian toy distributor (Djeco, CubicFun, Infantino), B2B to retail chains / online stores / specialty shops. Monthly SKU-level forecasts. Consumer-facing categories: board games, puzzles, baby toys, construction sets.

---

## Tier A — High priority (recommend implementing first)

These sources have strong a-priori business relevance, reliable free APIs, and straightforward Python integration. Each has a realistic hypothesis for *why* it should improve the forecast.

### A1. NBU exchange rates and key policy rate (UAH/USD, UAH/EUR)

- **Source:** National Bank of Ukraine Developer API ([bank.gov.ua/en/open-data/api-dev](https://bank.gov.ua/en/open-data/api-dev))
- **Access:** Public, no authentication, no rate limits documented
- **Endpoints:**
  - Daily rates (historical): `https://bank.gov.ua/NBU_Exchange/exchange_site?start=YYYYMMDD&end=YYYYMMDD&valcode=USD&json`
  - Key policy rate: `https://bank.gov.ua/NBUStatService/v1/statdirectory/key?date=YYYYMMDD&json`
- **Update frequency:** Daily (official rates); policy rate on each NBU board decision
- **History:** Exchange rates since 1996; policy rate since 1992
- **Format:** JSON (also XML)
- **Integration effort:** ~30 min — simple HTTP + parse + resample to month-end / month-mean
- **Business hypothesis:** Toys are 100% imported goods (Djeco = France, CubicFun = China, Infantino = USA). Retail prices track USD/EUR quotes with 1–2 month lag because inventory is priced at purchase-time FX. Sharp UAH devaluation → demand elasticity kicks in → sales drop for premium SKUs. Policy rate signals future credit conditions for retailers.
- **Expected signal strength:** **High**. Documented 30%+ UAH devaluation in Feb 2022 and Oct 2022 should have clear signatures in the training data.

### A2. Ukrainian public holidays

- **Source:** `python-holidays` library (PyPI) — wraps official `УКАЗ ПРЕЗИДЕНТА` calendar
- **Access:** `pip install holidays`
- **Integration effort:** ~15 min — already a pandas-compatible lookup
- **Business hypothesis:** Ukraine has strong toy-demand seasonal peaks that generic `month` features miss:
  - **St. Nicholas Day (Dec 19)** — major children's gift-giving date in Ukraine, arguably bigger than Christmas for toys
  - **Christmas (Dec 25 since 2023, previously Jan 7)** — calendar itself changed in 2023
  - **International Children's Day (June 1)** — triggers back-to-school and summer toy purchases
  - **Easter (floating date)** — chocolate/small-toy gifts
  - **New Year (Jan 1)** — Grandfather Frost tradition
  - **School year start (Sep 1)** — educational toys, puzzles
- **Features to engineer:**
  - `days_to_next_holiday`, `days_from_last_holiday`
  - `pre_holiday_window_7/14/30` binary flags
  - `gift_giving_season` (mid-Nov through early-Jan)
  - `school_start_window` (mid-Aug through mid-Sep)
  - `calendar_regime` flag (pre-2023 Jan-7 Christmas vs post-2023 Dec-25)
- **Expected signal strength:** **Very high**. Current model has only `month`, `quarter`, `year` + cyclical encodings — no awareness of these specific event dates.

### A3. Google Trends — Ukrainian toy search interest

- **Source:** [pytrends-modern](https://github.com/yiromo/pytrends-modern) (community successor to archived `pytrends`)
- **Access:** Free via scraping, requires rate limiting / proxy rotation at scale
- **Resolution:** Monthly data retrievable via `timeframe='today 5-y'` (aligns perfectly with our monthly cadence)
- **Geo:** Use `geo='UA'` for Ukraine-wide, or `UA-30` (Kyiv city), `UA-32` (Kyiv oblast), etc. for regional
- **Integration effort:** ~2 hours — need to pre-collect keyword list, batch requests (5 keywords per call), handle 429 errors with backoff
- **Business hypothesis:** Search interest is a leading indicator of purchase intent by 1–4 weeks. Specifically useful for:
  - Brand-level demand: `"Djeco"`, `"CubicFun"`, `"Infantino"`
  - Category searches: `"настільні ігри"` (board games), `"пазли"` (puzzles), `"дитячі іграшки"` (children's toys)
  - Movie tie-in spikes: `"Disney іграшки"`, specific franchise names
  - Seasonal search: `"подарунок дитині"` (gift for a child) surges pre-St-Nicholas
- **Keyword list to curate:** ~20–40 Ukrainian + Russian language toy-related terms, validated with domain analyst
- **Expected signal strength:** **High for medium/large volume SKUs**, weak for long tail (insufficient search volume per SKU).

### A4. Open-Meteo historical weather (free, no API key)

- **Source:** [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Access:** Public, no key, 10 000 calls/day free tier
- **Endpoint:** `https://archive-api.open-meteo.com/v1/archive?latitude=...&longitude=...&start_date=...&end_date=...&daily=temperature_2m_mean,temperature_2m_max,precipitation_sum,snowfall_sum`
- **Data source:** ERA5 reanalysis (1940–present), 10 km resolution
- **Resolution:** Hourly/daily, aggregate to monthly in-code
- **License:** CC BY 4.0 (attribution required)
- **Cities to cover** (align with `Канал` × regional partners):
  - Kyiv (50.45, 30.52)
  - Lviv (49.84, 24.03)
  - Odesa (46.48, 30.72)
  - Kharkiv (49.99, 36.23)
  - Dnipro (48.47, 35.04)
- **Integration effort:** ~1 hour — 5 cities × 6 years monthly = 360 rows, one-time fetch + cache
- **Business hypothesis:**
  - **Cold, long winters** → indoor toys (board games, puzzles, construction sets) sell more; outdoor toys flatline
  - **Heat waves** → construction-set and creative-toy sales slump (parents take kids outside)
  - **Rainy summers** → indoor-toy demand lifts above seasonal baseline
  - **First-snow dates** influence when "winter gift shopping" starts
- **Features to engineer:** `monthly_mean_temp`, `monthly_precip_days`, `heating_degree_days`, `cooling_degree_days`, `first_snow_day_of_month`, weighted national average via partner-location weights
- **Expected signal strength:** **Moderate** — expect 2–4% WAPE reduction for outdoor-category SKUs specifically.

### A5. Ukrainian school calendar (hand-curated)

- **Source:** No API, but dates are published annually in the Ministry of Education decree (укази МОН)
- **Access:** Manually curated CSV with ~3 dates per year: `school_start`, `winter_break_start`, `winter_break_end`, `spring_break_start`, `spring_break_end`, `school_end`
- **Integration effort:** ~1 hour for 6 years of historical data, trivial to maintain going forward
- **Business hypothesis:** Educational toy and puzzle demand peaks 2 weeks before `school_start`; construction sets and board games peak during winter break
- **Features:** `days_to_school_start`, `in_school_break`, `pre_school_shopping_window_14d`
- **Expected signal strength:** **Moderate-high**, complementary to holiday features.

---

## Tier B — Medium priority (implement after Tier A is validated)

### B1. ACLED conflict events

- **Source:** [ACLED API](https://acleddata.com/acled-api-documentation) + free-access research tier via [myACLED](https://acleddata.com/user/register) registration
- **Data also available** on [HDX](https://data.humdata.org/dataset/ukraine-acled-conflict-data) without registration (aggregated monthly files)
- **Access:** Requires free registration; OAuth token for programmatic
- **Resolution:** Event-level → aggregate to (oblast × month)
- **Update frequency:** Weekly
- **History:** 2018-01-01 onwards for Ukraine
- **Integration effort:** ~3 hours — data model is rich (event_type, sub_event_type, actors, fatalities); need careful aggregation
- **Business hypothesis:** Sustained high-intensity warfare in a region depresses discretionary spending. Kharkiv / Kherson / Odesa / Zaporizhzhia have fundamentally different demand dynamics than western oblasts.
- **Features:** per oblast × month:
  - `conflict_event_count`
  - `civilian_targeting_events`
  - `fatalities`
  - `relative_intensity` (vs. national average)
- **Integration path:** join on partner oblast (need to enrich partner reference with oblast — currently has only `Направление` / `Направление.Группа`)
- **Expected signal strength:** **High for eastern/southern oblasts, low for western**. Potentially explains part of the ИМ (online) channel's poor WAPE — online buyers skew toward conflict-affected regions.

### B2. Air raid alerts history

- **Sources:**
  - Official: [api.ukrainealarm.com](https://api.ukrainealarm.com) (API key via request form, free)
  - Volunteer: [alerts-in-ua Python client](https://pypi.org/project/alerts-in-ua/)
  - Aggregate statistics: [air-alarms.in.ua](https://air-alarms.in.ua/en)
- **Resolution:** Second-level events → aggregate to oblast × month (total hours under alert, number of alerts, longest alert duration)
- **History:** Feb 2022 onwards
- **Integration effort:** ~4 hours including API key acquisition
- **Business hypothesis:** Complements ACLED. Alert frequency correlates with local logistics disruptions (courier delays, shop closures). High-alert months → shift of retail purchases online or to later months.
- **Expected signal strength:** **Moderate**. Likely overlap with ACLED; keep if ACLED alone doesn't capture regional disruption well.

### B3. World Bank macro indicators (annual)

- **Source:** [World Bank API](https://api.worldbank.org/v2/country/UKR/indicator/{ID}?format=json)
- **Access:** Public, no key
- **Key indicators:**
  - `SP.POP.0014.TO` — population ages 0–14 (target demographic for toys)
  - `SP.DYN.TFRT.IN` — fertility rate (leading indicator for 3–5 year toy demand cycle)
  - `NY.GDP.PCAP.CD` — GDP per capita
  - `FP.CPI.TOTL.ZG` — annual CPI inflation
- **Resolution:** **Annual only** — broadcast to monthly via forward-fill, treat as slow-moving baseline
- **Integration effort:** ~30 min
- **Business hypothesis:** Annual target-demographic size caps total market potential. A 28% birth-rate decline since 2022 (per Opendatabot) foreshadows a structural demand shift in 2-5 years.
- **Expected signal strength:** **Low for month-over-month forecasting, high for trend**. Best as a slowly varying baseline covariate.

### B4. IMF International Financial Statistics (monthly CPI, retail sales)

- **Source:** [IMF Data API](https://data.imf.org/en/Resource-Pages/IMF-API) (SDMX 2.1 / 3.0)
- **Access:** Public, no key
- **Python:** Libraries `sdmx1`, `imfdatapy`
- **Key series for Ukraine:**
  - `CPI` (monthly)
  - `Retail Trade Volume Index` (monthly, when published)
  - `Industrial Production Index` (monthly)
- **Integration effort:** ~2 hours (SDMX learning curve)
- **Business hypothesis:** Monthly CPI gives real-vs-nominal purchasing power adjustment. Retail Trade Index is a direct competitor-sector leading indicator.
- **Expected signal strength:** **Moderate-high**. More useful than annual World Bank data.

### B5. TMDB movie/TV release calendar (toy tie-in signal)

- **Source:** [TMDB API](https://developer.themoviedb.org/) (free, registration required)
- **Endpoint:** `/3/discover/movie?primary_release_date.gte=...&with_genres=16,10751` (Animation + Family)
- **Filter for Ukraine:** `region=UA` + release-type filter on `/3/movie/{id}/release_dates`
- **Integration effort:** ~2 hours — filter for major franchises with known toy licenses (Disney, Pixar, DreamWorks, LEGO-branded, Marvel, Star Wars, Pokémon)
- **Business hypothesis:** A new Pixar or LEGO-branded movie releasing in Ukraine drives a 20–40% demand spike in related SKUs (observed industry-wide). CubicFun has Disney license tie-ins, directly applicable.
- **Features:** `major_animation_release_this_month`, `major_animation_release_next_month`, `franchise_active_flag` (e.g., Frozen merchandise active 2023–2025)
- **Expected signal strength:** **High for licensed SKUs, none for generic toys**. Coverage is narrow but deep.

---

## Tier C — Low priority / speculative

### C1. Electricity outage schedules (Ukrenergo / DSOs)

- **Source:** Fragmented — each regional DSO publishes on its website/Telegram; no unified API
- **Aggregation:** Community projects like [chernivtsi-outages](https://github.com/denysdovhan/chernivtsi-outages) cover single cities only
- **Verdict:** **Do not implement for PoC.** Fragmentation cost too high relative to signal. Partial overlap with air-raid alerts already captures infrastructure disruption.
- **Revisit if:** Client specifically requests outage-aware forecasting, or a national-scale aggregator emerges.

### C2. Marketplace competitor pricing (Rozetka, Prom.ua, Wildberries)

- **Source:** No official public APIs for product browsing (Prom has seller-only API; Rozetka none)
- **Verdict:** **Legal grey area + brittle scrapers + maintenance burden.** Rejected for current PoC. Would require a dedicated scraping infrastructure, anti-bot evasion, and ongoing TOS risk.
- **Alternative if needed:** Periodic manual price sampling by the analyst (once/quarter) and adding a qualitative "price-position vs competitors" enum feature.

### C3. Opendatabot / Ukrstat demographic feeds

- **Verdict:** Covered by World Bank (Tier B3) at lower integration cost and more stable API. Add only if regional breakdown needed.

### C4. Social media trends (TikTok, YouTube)

- **Verdict:** No reliable free API for historical trend volume; TikTok Research API is gated for academics only. Signal would duplicate Google Trends. Rejected.

### C5. Weather forecast (prospective, for future-looking features)

- Open-Meteo has a forecast endpoint too, but since we forecast monthly 1–6 months ahead, weather forecasts don't extend that far reliably. Use historical only.

---

## Ranking summary

| Rank | Source | Expected WAPE gain | Integration effort | Operational risk |
|-----:|--------|:--------:|:--------:|:--------:|
| 1 | Ukrainian holidays (A2) | 2–4% | 15 min | None |
| 2 | NBU FX + policy rate (A1) | 2–3% | 30 min | None |
| 3 | School calendar (A5) | 1–2% | 1 hour | None (manual curation) |
| 4 | Google Trends (A3) | 1–3% | 2 hours | Rate limiting |
| 5 | Open-Meteo weather (A4) | 1–2% (category-specific) | 1 hour | None |
| 6 | ACLED conflict (B1) | 1–3% (eastern regions) | 3 hours | Need partner→oblast mapping |
| 7 | Air raid alerts (B2) | 0.5–1.5% | 4 hours | API key approval wait |
| 8 | IMF monthly CPI (B4) | 1–2% | 2 hours | SDMX complexity |
| 9 | TMDB movie releases (B5) | 2–5% for licensed SKUs | 2 hours | Low coverage |
| 10 | World Bank annual (B3) | <1% | 30 min | None (trend only) |

**Total expected gain if all Tier A implemented well:** 5–10% WAPE reduction — would move us from 0.49 → ~0.44.

**Tier A+B combined, best case:** 10–15% — would move us from 0.49 → ~0.42.

These estimates are optimistic ceilings; reality will be less due to feature redundancy. Empirical testing in the next step is essential.

---

## Data hygiene checklist (to avoid leakage)

All external signals must respect the forecast horizon:

- **Lag the signal appropriately.** When forecasting month `t`, we can use:
  - Macro / weather / holidays for month `t` and all prior months (if signal is observable in real-time or at month-start)
  - Google Trends for month `t−1` and earlier (Trends finalize with ~2-week lag)
  - ACLED for month `t−1` and earlier (weekly release cycle)
  - Movie release calendar for month `t` (releases announced 6+ months ahead)
- **Never use end-of-month aggregates of month `t`** in features for predicting month `t`.
- **Temporal split must apply to external data too.** Validation and test signals must respect the train/val/test date boundaries.

---

## Recommended implementation order (next session)

1. **Add a new module `src/external_data.py`** with one fetcher per source, each returning `pd.DataFrame(Period, feature_cols...)` cached to parquet in `output/external/`
2. **Implement Tier A1 + A2 first** (30 min + 15 min = cheapest first wins)
3. **Join to ABT, retrain V4 ensemble, measure WAPE delta.** Keep features only if they improve val WAPE by ≥0.005 absolute.
4. Repeat for A3, A4, A5 in priority order.
5. **Move to Tier B** only if Tier A yields measurable wins; otherwise budget goes into problem reformulation (coarser granularity, decision-metric training) per `docs/v4-creative-approaches.md`.
6. Document each addition in a new ADR (e.g., `adr-003-external-signals.md`) with measured WAPE impact.

---

## Licensing & attribution summary

| Source | License | Attribution required |
|--------|---------|:---:|
| NBU | Open Data (free reuse) | Recommended |
| python-holidays | MIT | No |
| Google Trends | Unofficial (TOS restricted) | N/A |
| Open-Meteo | CC BY 4.0 | **Yes** |
| ACLED | Custom (registered users) | **Yes** (attribution policy) |
| Ukrainealarm API | Custom | **Yes** |
| World Bank | CC BY 4.0 | **Yes** |
| IMF Data | Open Data Policy | Recommended |
| TMDB | Free (free API tier) | **Yes** ("Powered by TMDB") |

All attributions will be added to README.md when sources are integrated.
