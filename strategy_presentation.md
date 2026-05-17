# AI Demand Forecasting: Scaling Strategy, Cost Estimation, and Operational Framework

This document outlines the strategic roadmap for scaling the demand forecasting AI model, addressing current operational challenges, estimating computational infrastructure costs, and providing a rigorous return on investment (ROI) and budget analysis based on current market realities. Furthermore, it defines the strict legal and financial framework governing the execution of this cross-border B2B contract.

---

## 1. Technical Development and Scaling Plan

Based on the empirical results of the pilot project (encompassing 3 brands) and the confirmed availability of data, the following plan is constructed on strictly realistic premises.

### 1.1 Data Integration
* **Elimination of Manual Data Exchange:** We will establish direct integration with the client's ERP system (`bas`), which houses the comprehensive database from 2020 onwards. This will enable the model to automatically extract relevant data (shipments, inventory, master data) and output forecasts without human intervention.
* **Assortment Scaling:** The model's scope will be expanded from the 3 pilot brands to encompass the 8 priority brands (~3,000 SKUs) identified by management.

### 1.2 Targeted Model Improvements (Addressing Management Feedback)

Based on feedback from management, we have developed a precise mitigation plan:

**A) "SKU-level error is high; we want to minimize it"**
*   **Data Ceiling Reality:** The current accuracy at the most granular level (Client × Item × Month) is approximately 63%. It is critical to understand that in open global retail forecasting competitions (e.g., M5 Walmart, Rossmann), the absolute winners hit a ceiling of 62-67% on similar datasets. The remaining variance is irreducible noise (random demand fluctuations, micro-logistical disruptions).
*   **Solution — Probabilistic Forecasting (Confidence Intervals):** Instead of attempting to predict an exact scalar value (which inherently carries a margin of error), we will activate a probabilistic interval mechanism. The procurement manager will not merely see an instruction to "order 50 units," but a safe operational corridor: "40 to 65 units with 90% confidence." This enables informed decision-making regarding overstock/shortage risks.
*   **Intermittent Demand:** For items with highly unstable, intermittent sales (numerous zero-demand months), standard forecasting algorithms fail because they attempt to average out the zeros. Instead, we configure our LightGBM architecture to optimize for the Tweedie objective function. This is a specialized statistical distribution (a modern evolution of Poisson/Gamma models) specifically designed to handle zero-inflated data. It effectively learns when an item will have zero sales versus when it will spike, empirically outperforming standard third-party approaches for sparse retail data.

**B) "Some brands only have 1-2 years of historical data" (Short History)**
*   **Cross-Learning:** Unlike rudimentary spreadsheet formulas, our ML model trains on **all** brands simultaneously. The model learns seasonal patterns (e.g., pre-holiday spikes) from brands with 5 years of history (Djeco, Cubic Fun) and successfully applies this learned behavior to new brands with only 1 year of data.
*   **Foundation Models:** We continually invest in R&D and have tested state-of-the-art AI models (including Amazon Chronos). However, empirical testing confirms that our proprietary Cross-Learning mechanism via LightGBM currently yields the highest accuracy for new items; therefore, we will retain it as the primary engine.

**C) "Predicting an entirely new item based on an existing one" (Cold Start)**
*   **Analog Mapping:** This is resolved during the data engineering phase. If a column such as `Item-Analog` (e.g., a direct replacement or a different color variant) is added to the master data, the system will automatically map the historical data of the legacy item to the new release.
*   **Feature-Based Forecasting:** Even in the absence of a specified analog, the LightGBM model relies on the static characteristics of the new item (Brand, Category, Price Segment, ABC Classification). The system will automatically forecast the launch sales based on the historical performance of an "average item" with identical characteristics within that category.

**D) "The January Anomaly" (Over-forecasting in January) and External Factors**
Currently, the model overestimates January demand due to historical market distortions (COVID-19, war).
*   **What we WILL use (Promotions):** The client has confirmed the availability of **comprehensive promotional plans and budgets**. Integrating these forward-looking calendars will allow the model to preemptively adjust forecasts based on scheduled marketing activities.
*   **External Data (Open Sources):** Unlike classical systems, our current model already automatically fetches and integrates external macroeconomic shocks: Ukrainian macro-indicators, Ukrenergo blackout statistics, and air-raid alert intensity. We will continue leveraging this open data to account for force majeure events without requiring manual input.
*   **What we CANNOT use:** POS transactional data (receipts) and partner website analytics are classified as trade secrets and are unavailable. Therefore, we must rely exclusively on external macro-data enrichment.

---

## 2. Computational Costs (Infrastructure)

The current model architecture was engineered under a strict **Zero-Cost Architecture (ADR-001)** constraint. We utilize ensemble models based on the LightGBM algorithm, which executes exclusively on Central Processing Units (CPUs).

* **Current Expenditures:** The complete compilation of the production model from scratch (including 65 trained sub-models) takes **approximately 15 minutes** on a standard laptop.
* **Conclusion:** The project **does not require** expensive cloud computing, Graphics Processing Units (GPUs), or complex server clusters. The model can be deployed on any existing corporate office server or a basic cloud instance, keeping computational overhead close to zero (within $20-$50 per month).

---

## 3. Economic Impact (ROI and Business Profit)

We translate mathematical accuracy into tangible liberated capital (UAH). Expert planning errors lead to either frozen capital in the warehouse (holding costs ~1.83% per month / 22% annually) or lost sales due to inventory shortages (lost margin ~14%, accounting for partial customer return rates).

* **Pilot Savings (3 Brands):** With an annual turnover of 31.5 million UAH, transitioning from expert accuracy (~87%) to model accuracy (92%) reduces the error margin by 5%. We estimate this saves the business approximately **1.5 million UAH annually**.
* **Scaling to 8 Brands:** Expanding the model to the full set of 8 priority brands will increase the predictive volume by approximately **2.6x**. Extrapolating the pilot metrics, deploying the model across these brands will yield savings of **4.0 to 5.0 million UAH annually**.

---

## 4. Project Timeline

Total project duration "turnkey": **4 Months**.

1. **Month 1 (Data Engineering & MLOps):** Developing connectors to the `bas` database, automating data pipelines, and cleansing historical data for the additional 5 brands.
2. **Months 2-3 (Modeling & Scaling):** Scaling algorithms to all 8 brands. Integrating promotional plans and partner budgets to resolve seasonality issues (including the January anomaly).
3. **Month 4 (Deployment & UI):** Exporting results back to `bas` or Power BI via accessible "Fact vs. Plan vs. AI Forecast" dashboards. Conducting comprehensive system training for the client's team.

---

## 5. Development Budget and Market Benchmarking

Executing this project requires a dedicated team of 2 Machine Learning Engineers. An analysis of current IT market rates (Ukraine / Eastern Europe, 2025-2026) for comparable consulting projects yields the following:

* **Market Value (Outsourcing/Consulting):** The standard market rate for a dedicated Senior ML Engineer on a project basis (B2B/Outsource) ranges from \$7,500 to \$11,000 per month. A two-person team would cost the company **\$15,000 to \$22,000 per month**. A standard 4-month project on the open market is valued between **\$60,000 and \$88,000**.
* **Our Proposal:** Given our profound existing immersion in the company's data specifics and the proprietary architecture already developed (65 model iterations), we are offering a highly competitive rate: **\$11,000 per month for the entire team**.
* **Total Project Budget (4 Months): \$44,000 (approx. 1.75 million UAH).**

---

## 6. ROI Summary

* **Capital Investment:** ~1.75 million UAH.
* **Projected Annual Savings:** ~4.0 - 5.0 million UAH / year (approx. 330 - 420 thousand UAH per month).
* **Payback Period:** The project fully recoups its initial cost within **4 to 5 months** of full-scale operation across the 8 brands. Subsequently, the system generates pure operational profit by optimizing inventory holding costs and minimizing stockouts.

---

## 7. Internal Operational Framework (Legal, Tax & Banking Setup - Germany to Ukraine B2B, 2026)

*This section dictates the mandatory legal and financial compliance required for the German contracting entity executing this project.*

* **Legal Setup (GbR Designation):** Because two students are collaboratively executing a single $44,000 USD consulting contract, German civil law automatically classifies this entity as a *Gesellschaft bürgerlichen Rechts* (GbR). It is mandatory to jointly register this GbR with the local *Finanzamt* using the *Fragebogen zur steuerlichen Erfassung*. The partners operate under joint and several liability (*gesamtschuldnerische Haftung*).
* **VAT (Umsatzsteuer) & Invoicing Compliance:** The export of B2B IT services to a non-EU client (Ukraine) is not taxable in Germany, as the place of performance is deemed the recipient's location pursuant to § 3a Abs. 2 UStG. All invoices must state the net amount with "0% VAT" and must explicitly feature the following mandatory legal clause to ensure audit compliance:
  > *"Nicht im Inland steuerbare Leistung gemäß § 3a Abs. 2 UStG. Steuerschuldnerschaft des Leistungsempfängers (Reverse Charge)."*
* **Revenues must be reported in the *Umsatzsteuer-Voranmeldung* under Field 45 (*Übrige nicht steuerbare Umsätze*).**
* **Income Tax (Einkommensteuer 2026) Projection:** The German tax-free allowance (*Grundfreibetrag*) has increased to **12,348 EUR** for the 2026 fiscal year. Assuming an exchange rate of ~1.08 USD/EUR, a 50% individual share of the \$44,000 contract equates to \$22,000 USD, or approximately **20,370 EUR**. Deducting the 12,348 EUR allowance results in a taxable income of ~8,022 EUR. This falls into the initial progressive tax bracket (starting at 14%), resulting in an estimated income tax liability of 1,500 - 1,700 EUR, leaving a projected net profit of approximately **18,670 EUR** per student.
* **Health Insurance Compliance (*Werkstudentenprivileg* Trap):** To maintain the subsidized student health insurance rate (approx. 143-148 EUR/month in 2026), it is strictly mandatory that self-employed work does not exceed **20 hours per week** during the lecture period. If this threshold is breached, or if the *Krankenkasse* reclassifies the work as the primary occupation (*hauptberuflich selbstständig*) due to the high revenue volume, the partners will be forcibly transitioned to voluntary public insurance (*freiwillig gesetzlich versichert*). The 2026 rates for this are ~16.9% (Health) and 3.6% - 4.2% (Care) applied to the actual profit (minimum assessment base: 1,318.33 EUR/month), which would incur costs of over 1,000 EUR per month during the earning period. Strict time-tracking and proactive communication with the insurer are mandatory.
* **Banking and FX Optimization:** Utilizing traditional German banks (e.g., Sparkasse, Commerzbank) to receive USD will trigger SWIFT intermediary fees (€15-30) and severe foreign exchange markups (1.5% - 3.0%). To preserve project margins, the GbR must utilize a modern financial infrastructure provider (such as **Wise Business** or **Airwallex**). These platforms issue local US banking credentials (ACH/Routing numbers), allowing the Ukrainian client to remit USD domestically. The funds can then be converted to EUR at the mid-market rate (approx. 0.4% fee) and routed to German domestic accounts via free SEPA transfers.
