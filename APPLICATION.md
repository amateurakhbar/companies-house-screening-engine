# Claude credits application

**Project:** UK Companies House Tech-Services Screening Engine
**GitHub:** github.com/amateurakhbar/companies-house-screening-engine · **License:** MIT (open source)

## What it does

An open-source pipeline that screens the entire UK Companies House register (~5 million companies) down to a ranked, defensible shortlist of IT/tech-services firms. It discovers and scrapes company websites, classifies each firm across a multi-axis taxonomy (stack layer × function × business model × vertical), then enriches the survivors with iXBRL financials and a defensibility score. The pipeline and a human-labeled gold set are included, so results are reproducible by re-running the pipeline (the bulk data lake is regenerated from the register, and the gold set ships with the repo for evaluation).

Beyond the IT-services use case, it is a reusable pattern: register-scale LLM classification governed by a frozen, closed-enum taxonomy and measured against a human gold set. The same harness applies to any sector screen.

## How it uses Claude

Claude powers the core classification step (`src/05_classify_llm.py`) via the Anthropic API. For each firm, Claude reads the company name, SIC code, and scraped website text and returns a strict-JSON label set validated against a frozen, closed-enum taxonomy. Labels outside the vocabulary are repaired or routed to `needs_review`, never silently accepted. The default model is `claude-opus-4-8`, with `claude-haiku-4-5` available as a cost lever for high-volume runs.

The integration is built to keep cost and reliability sane at register scale:

- **Prompt caching** on the large classification rubric (the system prompt), so repeated calls pay a fraction of the input cost.
- **Smart routing:** firms with no usable website text are sent straight to an `insufficient_evidence` result without spending an API call.
- **Safe caching:** only successful API calls are cached, so a rate-limited or credit-exhausted call is retried later rather than frozen as a wrong result.
- **Spend guards:** full-register runs are hard-disabled in the published script, and the supported scope is the gold set, to prevent accidental large API spend.

## How it's evaluated

Classification quality is measured against a gold set of about 149 human-labeled firms, reporting per-axis agreement. To keep the headline honest, agreement is reported separately for the 84 firms Claude actually classified (those with website text) versus the full set. On those 84, classifying blind from only company name, SIC code, and scraped website text, `claude-opus-4-8` agrees with the human labels 86% on average across axes: stack-layer 79%, function 93%, business-model 86%, vertical 93%, and derived niche 82%.

## Why support would help

This is an independent, open-source research project (MIT). By design the classifier currently caps at the gold set, because the full-register pass is deliberately disabled to avoid accidental spend. Claude credits would fund enabling and running the full classification pass (roughly 27,000 firms with website text, drawn from a tech-filtered universe of about 181,000), and tuning the taxonomy and prompts against the human benchmark. The spend is bounded and predictable: with the rubric prompt-cached, a complete `claude-opus-4-8` pass over the ~27,000 classifiable firms costs roughly $190 (about $40 on `claude-haiku-4-5`), and the engine clusters and classifies representatives rather than every firm, which lowers it further. A grant of around $500 in credits would cover several full passes plus gold-set-driven prompt and taxonomy iteration. All outputs, including the engine, the taxonomy, and the evaluation harness, remain open source.

> Scope note: Claude does the core classification (`src/05`). Some downstream enrichment steps under `scripts/` use a separate LLM provider, and the README states this explicitly.
