# =============================================================================
# Companies House tech roll-up screening pipeline (Stage 11 orchestration)
#
# Caching model: every stage is a FILE target whose prerequisites are its inputs
# (upstream outputs + the script itself). Make rebuilds a stage only when an
# input is newer than the output, so `make all` on an up-to-date tree does
# nothing. Re-runs stay cheap because the heavy scripts also keep their own
# on-disk caches (URL cache, scrape cache, per-firm LLM cache) — a triggered
# re-run only does genuinely new work.
#
#   make all            # load -> rules -> classify -> score -> rollup -> score-niches
#   make classify       # production per-firm LLM classification (resumes from cache)
#   make cluster        # exploratory cluster-then-classify (NOT part of `all`)
#   make help           # list targets
# =============================================================================

PY ?= python3

# ---- artefacts (timestamps drive the caching) ------------------------------
WORKING    = output/working.parquet
URLS       = data/cache/urls.json
SCRAPED    = data/cache/scrape_status.json
RULES      = output/rules_labels.parquet
CLASSIFIED = output/classified_firms.parquet
GOLD_IN    = gold/gold_set_labeled.csv
GOLD_LLM   = gold/gold_set_llm.csv
SUMMARY    = output/niche_summary.csv
SCORES     = output/niche_scores.csv

.PHONY: all load urls scrape rules classify score cluster rollup score-niches help clean

# Dependency order. cluster is deliberately excluded — it is exploratory only.
all: load rules classify score rollup score-niches  ## full pipeline in order

# ---- phony aliases -> file targets -----------------------------------------
load:         $(WORKING)     ## 01 load + filter      -> working.parquet
rules:        $(RULES)       ## 04 deterministic rules -> rules_labels.parquet
classify:     $(CLASSIFIED)  ## production LLM classification (per-firm cached)
score:        $(GOLD_LLM)    ## 05 --gold: gold-set accuracy eval
rollup:       $(SUMMARY)     ## 08 per-niche rollup    -> niche_summary.csv
score-niches: $(SCORES)      ## 09 weighted niche scoring -> niche_scores.csv

# ---- main pipeline (file targets, timestamp-cached) ------------------------
$(WORKING): src/01_load_filter.py
	@echo ">>> [load] 01_load_filter.py -> $(WORKING)"
	$(PY) src/01_load_filter.py

$(RULES): $(WORKING) src/04_rules.py
	@echo ">>> [rules] 04_rules.py -> $(RULES)"
	$(PY) src/04_rules.py

# Production classification. Scrape cache is an order-only prereq (|): required
# to exist, but never forces a rebuild and never re-runs inside `all`.
$(CLASSIFIED): $(WORKING) $(RULES) src/05_classify_llm.py scripts/run_production.py | $(SCRAPED)
	@echo ">>> [classify] scripts/run_production.py (production path) -> $(CLASSIFIED)"
	$(PY) scripts/run_production.py

$(GOLD_LLM): $(GOLD_IN) src/05_classify_llm.py
	@echo ">>> [score] 05_classify_llm.py --gold (gold-set eval) -> $(GOLD_LLM)"
	$(PY) src/05_classify_llm.py --gold

$(SUMMARY): $(CLASSIFIED) src/08_rollup_niches.py
	@echo ">>> [rollup] 08_rollup_niches.py -> $(SUMMARY)"
	$(PY) src/08_rollup_niches.py

$(SCORES): $(SUMMARY) $(WORKING) $(CLASSIFIED) src/09_score_niches.py
	@echo ">>> [score-niches] 09_score_niches.py -> $(SCORES)"
	$(PY) src/09_score_niches.py

# ---- already-complete, cached stages (NOT in `all`; won't re-run) ----------
# These are EXISTENCE-cached (no timestamp prerequisites): both outputs already
# exist, so Make never re-runs them — not during `make all` (they aren't in the
# chain) and not on an explicit `make urls`/`make scrape`. They run only if the
# output is deleted. To genuinely re-discover/re-scrape, run the script directly.
urls:   $(URLS)     ## 02 discover company URLs (cached; not in `all`)
$(URLS):
	@echo ">>> [urls] 02_discover_urls.py -> $(URLS)"
	$(PY) src/02_discover_urls.py

scrape: $(SCRAPED)  ## 03 scrape websites (cached; not in `all`)
$(SCRAPED):
	@echo ">>> [scrape] 03_scrape.py -> $(SCRAPED)"
	$(PY) src/03_scrape.py

# ---- exploratory (standalone, NOT in `all`) --------------------------------
cluster:  ## 06 cluster-then-classify (exploratory; cheaper, lower accuracy)
	@echo ">>> [cluster] 06_cluster_classify.py (exploratory, not in all)"
	$(PY) src/06_cluster_classify.py

# ---- utility ---------------------------------------------------------------
help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

clean:  ## remove derived outputs (keeps on-disk caches)
	rm -f $(CLASSIFIED) $(SUMMARY) $(SCORES) $(GOLD_LLM)
