# Sector Screener: Runbook

A data-driven first cut at *which* fragmented sector to roll up, built on free
Companies House data. It replaces a hand-picked shortlist with an empirical
ranking, which is also a much stronger story on the call: you are demonstrating
the engine doing the exact thing you are pitching.

## What it measures

From the free Companies House snapshot, for each candidate sector it computes:

- **active_companies** - the runway of potential targets
- **pct_small** - share that are small / micro entities (fragmentation and owner-managed supply)
- **pct_mature_15y** - share incorporated 15+ years ago (a weak succession proxy)
- **mean_age_years** - average company age
- **distinct_counties / top_county_share** - geographic dispersion
- **score** - a transparent 0 to 100 composite (weights are editable at the top of the script)

## How to run it

1. **Get the data (no API key, no signup).** Download the "Free Company Data
   Product" from:
   https://download.companieshouse.gov.uk/en_output.html
   Take either the single large file or the multi-part files, and unzip them
   into a folder (for example `./chdata`). It is roughly 5 million companies and
   refreshes monthly.

2. **Install the one dependency:**
   ```
   pip install pandas
   ```

3. **Run the screen:**
   ```
   python sector_screen.py --input ./chdata
   ```
   You can also pass a single file or a glob:
   ```
   python sector_screen.py --input BasicCompanyData-2026-06-01-part1.csv
   python sector_screen.py --input "./chdata/*.csv"
   ```

4. **Read the output.** It prints a ranked table and writes
   `sector_screen_results.csv`. The top-ranked sectors are your data-derived
   shortlist. Drop the real numbers straight into the brief's scoring matrix.

## Editing the sectors

The `SECTOR_SIC` dictionary at the top maps each candidate sector to SIC 2007
codes. Edit it freely. Verify codes against the ONS SIC 2007 condensed list. A
company is counted toward a sector if any of its (up to four) SIC codes match.

## Honest limits (say these on the call, they signal rigour)

- **Counts are indicative.** SIC codes are broad and self-reported. Fire
  compliance in particular is split across several codes and mixed with
  non-compliance work, so raw counts over- and under-count. The language-model
  classification step in the full engine refines this by reading company
  websites; this script is the empirical first cut.
- **No revenue or EBITDA.** The free data has no turnover for small firms, so
  there is no true revenue-based concentration (HHI) here. That is a paid-data
  step (FAME / Bureau van Dijk, Grata, Sourcescrub, or S&P Capital IQ).
- **Owner ages are not in the bulk file.** Verified succession signals come from
  the Companies House officers API, run per company on the shortlist. That is
  the next script to build once a sector is chosen.

## Where this sits in the pipeline

1. **This screen** -> ranks sectors on fragmentation (free data).            [you are here]
2. **Target pull + LLM classification** -> within the chosen sector, find and
   qualify the sub-1m owner-managed firms (free data + websites + LLM).
3. **Enrichment** -> officer ages, accreditations, geography (officers API).
4. **Scorecards + heat map** -> the ranked target list for the platform and bolt-ons.
