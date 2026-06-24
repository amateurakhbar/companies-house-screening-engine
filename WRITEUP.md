# Screening 5M UK companies into a buy-and-build shortlist — with Claude doing the classification

This engine starts from the **entire UK Companies House register (~5M companies, 400+ columns)** and
funnels it down to a ranked, defensible shortlist of IT/tech-services firms that fit a given roll-up
thesis — including the ones no keyword search would surface. Here's how it works.

## The funnel

```
5M UK companies  →  tech/IT SIC filter  →  discover + scrape each website
              →  rules + Claude classify (4-axis taxonomy)  →  niche rollup
              →  Companies House iXBRL financials + revenue/EBITDA triangulation
              →  defensibility scoring  →  ranked, deduped target universe
```

Two pipelines over one local data lake: **classification** (`src/`) and **enrichment & financials**
(`scripts/`).

## Why a classifier at all

SIC codes are too coarse — "IT consultancy" covers a defensible managed-security firm and a
box-shifting reseller alike. The signal that matters lives in what a firm *says it does*, on its
website. So the pipeline scrapes each site and classifies it on four axes:
**stack-layer × function × business-model × vertical**. A rules layer handles the unambiguous cases
cheaply; everything it can't resolve goes to the LLM.

## The classification step runs on Claude

The LLM layer ([`src/05_classify_llm.py`](src/05_classify_llm.py)) calls **Claude via the Anthropic
API** (`client.messages.create`, default `claude-opus-4-8`). Three things make its output trustworthy
enough to feed downstream joins:

1. **A frozen label space.** The model must return one of a closed set of enums
   ([`schema/taxonomy.py`](schema/taxonomy.py)). Off-vocabulary labels are repaired or flagged
   `needs_review` — never silently coerced — so rollups and gold-set scoring rely on a finite,
   fixed label space.
2. **A gold set.** A hand-labelled sample lets every prompt/model change be measured, not vibed:
   bump the prompt version, re-run the gold set, compare.
3. **Boring engineering for cost and trust.** The rubric is a **prompt-cached** system prompt
   (~0.1× after the first call); every result is **cached on disk** keyed by model + prompt version
   (re-runs free, crashes resume for nothing); firms with no website text are routed to
   `insufficient_evidence` *without* an API call rather than guessed at.

The model gives an opinion; the schema and the gold set keep it honest.

## The hard part wasn't the model — it was the missing P&Ls

Most small UK companies file under the **small-company P&L exemption: no public income statement.**
So EBITDA is never quoted — it's **triangulated** from balance sheets, deferred income and
headcount, pulled from Companies House iXBRL accounts. Every estimate the engine produces is an
*opening frame for a Quality-of-Earnings review*, not a number to wire against.

## Run it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python3 src/05_classify_llm.py --gold      # classify the gold set
```

Code, taxonomy, and gold set are in the repo. MIT licensed.
