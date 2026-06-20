# Implementation Plan: Tech Firm Classification Pipeline

## Where the dataset stands now

`tech_potentials_with_financials.csv` is self-contained: one row per company with `CompanyName`, four SIC fields, address, age, account category, financials and `fin_source`. That is a big step up. You can now filter, size and run a first-pass classification with no external join. `target_union.csv` was only the parser's key list and drops out of the pipeline here.

But the core constraint from before still holds: **the two text signals you have, company name and SIC, are not enough to classify reliably.** SIC is too coarse (an MSP, a dev shop and a Dynamics partner all sit in the same 62xxx code) and a name is a thin, often misleading hint. The decisive signal, a company's own description of what it does, lives on its **website**, and that is the one thing still missing. So the enrichment stage shrinks to exactly two jobs: discover each firm's URL (from name plus the town you derive from the address) and scrape it. No Companies House round-trip is needed, because name, SIC, address, age and account category are already in the file.

---

## Proposed repo structure

```
/data
  tech_potentials_with_financials.csv   # EXISTING, self-contained, one row per company
  /cache
    scrapes/                            # [NEW] cached website text
/schema
  taxonomy.py                    # [NEW] frozen label enums
  screening_metrics.py           # [NEW] derived financial metric definitions
/src
  01_load_filter.py              # [NEW] load CSV, derive town, filter junk
  02_discover_urls.py            # [NEW] resolve company website from name + town
  03_scrape.py                   # [NEW] scrape + cache homepage/about/services
  04_rules.py                    # [NEW] SIC priors + keyword classification
  05_classify_llm.py             # [NEW] structured-output LLM classification
  06_cluster.py                  # [NEW] embeddings + clustering cross-check
  07_score_goldset.py            # [NEW] precision/recall vs hand-labelled gold set
  08_rollup_niches.py            # [NEW] aggregate labels + financials per niche
/gold
  gold_set.csv                   # [NEW] ~150 hand-labelled firms
/output
  classified_firms.parquet       # [NEW] every firm, every label, confidence, rationale
  niche_summary.csv              # [NEW] one row per niche, sized and scored
```

---

## Step 1 — Freeze the label schema

You freeze **two** schemas, because your data splits into two jobs.

**(a) Classification labels** (require text, frozen in `taxonomy.py` as closed enums so every output is joinable):

- `stack_layer` (single): `hardware_infra | software | services | connectivity_hosting | data_info_services`
- `function` (multi): `cyber | cloud_devops | data_analytics_ai | networking | erp_crm | app_dev | testing_qa | msp_infrastructure | other`
- `business_model` (single, the rollability driver): `recurring_managed | project_oneoff | resale_distribution | staffing`
- `vertical` (multi): `healthcare | legal | finserv | govt | education | property | construction | manufacturing | retail | horizontal`
- `primary_niche` (derived): the best-fit intersection, e.g. `managed_cyber__legal`
- `confidence` (0 to 1), `rationale` (one line), `needs_review` (bool)

**(b) Screening metrics** (derived from your 16 columns, frozen in `screening_metrics.py`):

- `size_band` (by turnover where present, else employees, else net_assets)
- `ebitda_proxy` = `operating_profit` (you have no separate D&A line, so operating profit is the closest available proxy; true EBITDA is not computable from this data)
- `operating_margin` = `operating_profit / turnover`
- `asset_intensity` = `fixed_assets / turnover`
- `working_capital` and `current_ratio` from `current_assets`, `creditors`, `debtors`
- `headcount_band` from `employees`

Nothing gets classified until both schemas are frozen and committed.

---

## Step 2 — Assemble evidence per firm

Most of this is already in the file, so the stage is short:

1. **Load** `tech_potentials_with_financials.csv`. One row per company, no joins.
2. **Derive town** from the address field, needed for URL discovery and the geographic rollup.
3. **Filter junk** before spending anything on scraping or the LLM: use account category and age to drop dormant and non-trading entities, and drop pure holding companies by SIC (64200 / 70100). If the file does not carry company status, that is the one field worth checking, so you never scrape a dissolved company.
4. **Discover the website**: query `name + town`, take the best match, cache it. This is the first genuinely missing piece.
5. **Scrape** homepage plus `/about` and `/services`, strip to clean text, cache by `CompanyName` slug. This is the decisive classification signal.

Output per firm: an evidence bundle `{name, sic[], town, scraped_text}` with the financial row riding alongside. The text bundle is what the classifier reads; the financials are carried for screening only.

Two columns earn their keep here. **Account category** tells you upfront which rows are full-accounts filers (so will have turnover) versus micro/small filers (so will not), letting you branch the size logic cleanly in Step 5 instead of inferring it from nulls. **Age** is both a junk filter and a succession signal: older incorporations are likelier to be founder-owned and near exit, which is exactly the roll-up fuel.

Cache every scrape so reruns are cheap and idempotent.

---

## Step 3 — Classify with a layered hybrid

Run three layers in order, each catching what the others can't:

1. **Rules first (`05_rules.py`)**: map SIC to a coarse `stack_layer` prior; run keyword dictionaries over the scraped text (e.g. "SOC", "penetration testing", "endpoint" to `cyber`; "Dynamics 365", "NetSuite", "SAP" to `erp_crm`). Cheap, transparent, and it cuts the volume going to the model.
2. **LLM on the remainder (`06_classify_llm.py`)**: feed `{name, sic, scraped_text}` plus the frozen enum schema, and demand strict JSON back with the labels, a confidence and a one-line rationale. Constrain output to the enum and forbid free text, or nothing joins. Batch the calls; cache by `CompanyNumber` plus prompt version.
3. **Clustering as a cross-check (`07_cluster.py`)**: embed the scraped text, cluster, and inspect. This surfaces niches your taxonomy missed and confirms your labels map to real groupings. It is for discovery, not assignment.

**Financials as a weak secondary signal only.** Asset-heavy plus low margin nudges toward `resale_distribution`; asset-light plus high margin nudges toward `software`/`services`. Use this as a tiebreaker, never as the primary signal. Note clearly: **recurring revenue, the single most important rollability driver, is invisible in financials.** It comes only from the text-based `business_model` classification or from deeper diligence.

---

## Step 4 — Make it auditable

This is what turns the output from a black box into something defensible.

- **Gold set**: hand-label ~150 firms across the niches you actually care about, then `08_score_goldset.py` reports precision and recall **per niche**. Without this you cannot claim the classifier is accurate.
- **Traceability**: store per firm the label, confidence, rationale, the evidence text that drove it, and the taxonomy plus prompt version.
- **Review queue**: anything below a confidence threshold goes to `needs_review`; route your highest-priority niches to manual review regardless of confidence.
- **Version control**: commit `taxonomy.py`, the prompts and `gold_set.csv`. Combined with the cache, the whole run is reproducible.
- **Data-quality guards**: CH bulk financials are messy from inconsistent iXBRL tagging, so range-check values and flag outliers (a "micro" firm with turnover in the billions is a mis-scaled tag). Dedup multiple `accounts_date` rows per company to the latest period, or keep the series if you want growth.

---

## Step 5 — Roll labels up into niches

Now the financials become the payload. Group by `primary_niche` and compute, per niche:

- firm count (the fragmentation read)
- size distribution: turnover bands, `operating_profit` (EBITDA-proxy) distribution, employee bands
- count of sub-threshold targets (the actual roll-up fuel)
- margin and asset-intensity profile (the cash-generation read)
- geographic spread (from town)

Output `niche_summary.csv`: one row per niche, sized and characterised. This feeds straight into the rollability scoring from the earlier work, now grounded in your real population rather than first principles.

**The caveat that genuinely bites here**: `turnover`, `gross_profit` and the P&L items are present **only for full-accounts filers**. Micro and small filers, which are exactly your sub-threshold roll-up targets, file abbreviated accounts and will have **null turnover and profit**. So for the smallest and most rollable firms you size by `employees` and `net_assets`, not turnover. Build that size-proxy fallback explicitly. The firms you most want are the ones with the thinnest numbers.

---

## What you still need to source

Almost everything is now in the file. Only two things remain:

- **Website URLs** (discovery step)
- **Scraped website text**, the decisive classification signal (scrape step)

And one thing no financials file ever contains: **recurring-revenue share**, the top rollability driver, which comes only from the text classification or deeper diligence. Company status is present (`CompanyStatus`, `Status_Label`), so junk filtering uses the real field, not a proxy.

---

## Working column set (71 raw to ~26 working)

Do this as the first transform inside `01_load_filter.py`, reading the immutable 71-column raw file and emitting the working set. Never hand-edit the raw file: keeping it intact is what makes the run reproducible and auditable.

**Keep:**
- Identity: `CompanyNumber`, `CompanyName`, `Status_Label`
- Filtering: `CompanyStatus`, `CompanyCategory`, `Accounts.AccountCategory`, `IncorporationDate`, `DissolutionDate`
- Geography (URL discovery + rollup): `RegAddress.AddressLine1`, `RegAddress.PostTown`, `RegAddress.County`, `RegAddress.PostCode`, `RegAddress.Country`
- Classification signal: `SICCode.SicText_1` through `_4`
- Accounts recency: `Accounts.LastMadeUpDate`
- Screening: `accounts_date`, `turnover`, `gross_profit`, `operating_profit`, `profit_before_tax`, `profit`, `net_assets`, `equity`, `cash`, `creditors`, `debtors`, `fixed_assets`, `current_assets`, `employees`
- Provenance: `fin_source`

**Drop:**
- Admin filing dates: `Accounts.AccountRefDay`/`Month`, `Accounts.NextDueDate`, `Returns.*`, `ConfStmt*`
- Address noise: `RegAddress.CareOf`, `RegAddress.POBox`, `RegAddress.AddressLine2`
- `CountryOfOrigin`, `URI`, partnership columns, all 20 previous-name fields (35 to 54, near-empty legacy baggage)
- Mortgages (24 to 27): drop unless you want `NumMortOutstanding` as a weak leverage hint

Two flags for the build:
- `RegAddress.PostTown` already gives a clean town, so "derive town" in Step 2 is a straight read, no parsing.
- `fin_source` blank means no financials matched, so those rows are size-blind. Don't drop them: they still classify on website text, they just can't be sized until you backfill financials. Bucket them separately in the rollup.
