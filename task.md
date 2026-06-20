# Task Checklist: Tech Firm Classification Pipeline

Execution order follows dependency graph: schemas first, then src scripts 01→08.

---

## PHASE 1 — Schema files (no dependencies; can be written in parallel)

### schema/taxonomy.py [NEW]
- [ ] Create `schema/taxonomy.py`
- [ ] Define `StackLayer` enum: `hardware_infra | software | services | connectivity_hosting | data_info_services`
- [ ] Define `Function` enum (multi-label): `cyber | cloud_devops | data_analytics_ai | networking | erp_crm | app_dev | testing_qa | msp_infrastructure | other`
- [ ] Define `BusinessModel` enum: `recurring_managed | project_oneoff | resale_distribution | staffing`
- [ ] Define `Vertical` enum (multi-label): `healthcare | legal | finserv | govt | education | property | construction | manufacturing | retail | horizontal`
- [ ] Define `PrimaryNiche` as a derived string convention (e.g. `managed_cyber__legal`) with docstring
- [ ] Add `confidence` (float 0–1), `rationale` (str), `needs_review` (bool) field specs as a `ClassificationOutput` dataclass

### schema/screening_metrics.py [NEW]
- [ ] Create `schema/screening_metrics.py`
- [ ] Define `SizeBand` enum (turnover → employees → net_assets fallback tiers)
- [ ] Define `compute_screening_metrics(row)` function signature and docstring (no implementation yet — signature freeze only)
- [ ] Document: `ebitda_proxy = operating_profit` (no D&A line available; true EBITDA not computable)
- [ ] Document: `operating_margin = operating_profit / turnover`
- [ ] Document: `asset_intensity = fixed_assets / turnover`
- [ ] Document: `working_capital`, `current_ratio` from `current_assets`, `creditors`, `debtors`
- [ ] Document: `headcount_band` from `employees`
- [ ] Add explicit note: micro/small filers have null turnover — size falls back to `employees` then `net_assets`

---

## PHASE 2 — Source scripts (strict dependency order)

### src/01_load_filter.py [NEW]  ← depends on: schema/screening_metrics.py (SizeBand)
- [ ] Create `src/01_load_filter.py`
- [ ] Load `data/tech_potentials_with_financials.csv` (71-column raw file — read-only, never mutate)
- [ ] Select working column set (~26 cols per plan): identity, filtering, geography, SIC x4, accounts recency, 14 screening cols, `fin_source`
- [ ] Drop: admin filing dates, address noise (`CareOf`, `POBox`, `AddressLine2`), `CountryOfOrigin`, `URI`, partnership cols, 20 previous-name fields, mortgages block
- [ ] Derive `town` from `RegAddress.PostTown` (straight read, no parsing needed)
- [ ] Filter junk: drop dissolved (`CompanyStatus` != active), dormant/non-trading (`Accounts.AccountCategory`), holding-co SICs (64200, 70100)
- [ ] Flag `fin_source`-blank rows as `size_blind = True` (keep them; classify on text, bucket separately in rollup)
- [ ] Emit cleaned working DataFrame to `data/cache/filtered_firms.parquet`

### src/02_discover_urls.py [NEW]  ← depends on: 01 output
- [ ] Create `src/02_discover_urls.py`
- [ ] Accept `data/cache/filtered_firms.parquet` as input
- [ ] For each firm: query `CompanyName + town` to discover website URL
- [ ] Cache results by `CompanyNumber` in `data/cache/urls.json` (idempotent — skip if already resolved)
- [ ] Output column: `website_url` (None if not found)

### src/03_scrape.py [NEW]  ← depends on: 02 output
- [ ] Create `src/03_scrape.py`
- [ ] Accept URL list from `data/cache/urls.json`
- [ ] For each firm: scrape homepage + `/about` + `/services`; strip to clean text
- [ ] Cache scraped text by `CompanyNumber` slug under `data/cache/scrapes/` (idempotent)
- [ ] Attach `scraped_text` field back to working DataFrame; emit `data/cache/firms_with_text.parquet`

### src/04_rules.py [NEW]  ← depends on: schema/taxonomy.py, 03 output
- [ ] Create `src/04_rules.py`
- [ ] Map SIC codes to coarse `stack_layer` prior using taxonomy enums
- [ ] Define keyword dictionaries keyed to `Function` enum values (e.g. `{"SOC", "penetration testing", "endpoint"} → cyber`; `{"Dynamics 365", "NetSuite", "SAP"} → erp_crm`)
- [ ] Run keyword match over `scraped_text`; assign labels where confidence is high
- [ ] Mark rule-classified firms as `source = "rules"`; pass remainder through unclassified
- [ ] Emit `data/cache/firms_rules_classified.parquet`

### src/05_classify_llm.py [NEW]  ← depends on: schema/taxonomy.py, 04 output
- [ ] Create `src/05_classify_llm.py`
- [ ] Accept unclassified rows from `data/cache/firms_rules_classified.parquet`
- [ ] Build prompt: `{name, sic[], scraped_text}` + frozen enum schema; demand strict JSON output
- [ ] Constrain output to taxonomy enums; forbid free-text labels (or nothing joins)
- [ ] Batch API calls; cache responses by `CompanyNumber` + prompt version
- [ ] Parse response into `ClassificationOutput` dataclass (from `schema/taxonomy.py`)
- [ ] Use financials as tiebreaker only: asset-heavy + low-margin → `resale_distribution`; asset-light + high-margin → `software`/`services`
- [ ] Merge rule-classified + LLM-classified into `data/cache/firms_classified.parquet`

### src/06_cluster.py [NEW]  ← depends on: 03 output (scraped text), 05 output (labels)
- [ ] Create `src/06_cluster.py`
- [ ] Embed `scraped_text` per firm
- [ ] Cluster embeddings (discovery only — not label assignment)
- [ ] Cross-check cluster membership against `primary_niche` labels from step 05
- [ ] Surface niches present in clusters but absent from taxonomy (output to stdout / log)
- [ ] Emit `data/cache/cluster_assignments.parquet` with `cluster_id` per firm

### src/07_score_goldset.py [NEW]  ← depends on: schema/taxonomy.py, 05 output, gold/gold_set.csv
- [ ] Create `src/07_score_goldset.py`
- [ ] Load `gold/gold_set.csv` (~150 hand-labelled firms)
- [ ] Join on `CompanyNumber` to predicted labels in `data/cache/firms_classified.parquet`
- [ ] Compute precision and recall **per niche** (not aggregate only)
- [ ] Flag niches below acceptable threshold
- [ ] Print report; optionally write `output/goldset_scores.csv`

### src/08_rollup_niches.py [NEW]  ← depends on: schema/screening_metrics.py, 05 output
- [ ] Create `src/08_rollup_niches.py`
- [ ] Load `data/cache/firms_classified.parquet`
- [ ] Call `compute_screening_metrics(row)` per firm
- [ ] Group by `primary_niche`; compute per niche:
  - [ ] Firm count
  - [ ] Turnover bands, `operating_profit` (EBITDA-proxy) distribution, employee bands
  - [ ] Count of sub-threshold targets (roll-up fuel)
  - [ ] Margin and asset-intensity profile
  - [ ] Geographic spread (from `town`)
- [ ] Handle micro/small filers explicitly: size by `employees` → `net_assets` when `turnover` is null
- [ ] Bucket `size_blind` firms separately in output
- [ ] Write `output/classified_firms.parquet` (every firm, every label, confidence, rationale, evidence text, taxonomy + prompt version)
- [ ] Write `output/niche_summary.csv` (one row per niche, sized and characterised)

---

## Completion gate

- [ ] `schema/taxonomy.py` committed
- [ ] `schema/screening_metrics.py` committed
- [ ] `gold/gold_set.csv` exists (hand-labelling prerequisite for step 07)
- [ ] `output/classified_firms.parquet` present
- [ ] `output/niche_summary.csv` present
- [ ] `07_score_goldset.py` run with acceptable precision/recall per target niche
