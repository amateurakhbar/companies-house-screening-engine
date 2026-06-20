#!/usr/bin/env python3
"""
sector_screen.py
================
Data-driven fragmentation screen for buy-and-build sector selection.

Reads the Companies House "Free Company Data Product" (the free monthly bulk
CSV snapshot of all live UK companies) and ranks candidate sectors by how
suitable they look for a roll-up, using only signals that are actually present
in the free data:

  - number of active companies in the sector (runway of targets)
  - share that are small / micro entities (fragmentation + owner-managed supply)
  - share that are mature (>15y old) (a weak succession proxy)
  - geographic dispersion across counties (regional roll-up potential)

It deliberately does NOT claim a revenue-based concentration (HHI) measure,
because the free data contains no turnover or EBITDA for small firms. That is
a paid-data / enrichment step (see README).

USAGE
-----
1. Download the snapshot (no key needed) from:
   https://download.companieshouse.gov.uk/en_output.html
   Either the single large file or the multi-part files. Unzip them.
2. Run:
   python sector_screen.py --input ./chdata            # a folder of CSVs
   python sector_screen.py --input BasicCompanyData.csv # a single CSV
   python sector_screen.py --input "./chdata/*.csv"     # a glob

OUTPUT
------
   - sector_screen_results.csv : full ranked table
   - prints the ranked summary to the console

NOTE
----
The SIC -> sector mapping below is a curated, editable judgment input. Raw SIC
codes are broad and noisy (fire compliance in particular is split across
several codes and mixed with non-compliance work), so counts are indicative.
The language-model classification step in the full engine refines this by
reading company websites; this script gives the empirical first cut.
"""

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict, Counter

try:
    import pandas as pd
except ImportError:
    sys.exit("Please install pandas:  pip install pandas")

# ----------------------------------------------------------------------------
# CANDIDATE SECTORS  ->  SIC 2007 codes  (edit freely)
# Verify codes against the ONS SIC 2007 condensed list before relying on them.
# A company is counted toward a sector if ANY of its (up to 4) SIC codes match.
# ----------------------------------------------------------------------------
SECTOR_SIC = {
    "Fire & building-safety compliance": ["80200", "80100", "71200", "43210"],
    "SME accountancy practices":         ["69201", "69202", "69203"],
    "IT managed services (MSPs)":        ["62020", "62090", "62030", "63110"],
    "Commercial HVAC / building svcs":   ["43220", "43290", "33120"],
    # extra candidates so this is a real screen, not just the shortlist:
    "Commercial cleaning":               ["81210", "81221", "81222", "81229"],
    "Pest control":                      ["81291"],
    "Insurance brokers":                 ["66220"],
    "Veterinary practices":              ["75000"],
    "Dental practices":                  ["86230"],
    "Funeral services":                  ["96030"],
    "Waste collection & treatment":      ["38110", "38120", "38210"],
    "Electrical contractors":            ["43210"],
}

# Account categories that indicate a small / owner-managed firm.
SMALL_ACCOUNTS = {
    "MICRO ENTITY", "SMALL", "TOTAL EXEMPTION SMALL", "TOTAL EXEMPTION FULL",
    "UNAUDITED ABRIDGED", "AUDITED ABRIDGED", "DORMANT", "NO ACCOUNTS FILED",
}

NEEDED = {
    "CompanyStatus", "IncorporationDate", "RegAddress.County",
    "Accounts.AccountCategory",
    "SICCode.SicText_1", "SICCode.SicText_2",
    "SICCode.SicText_3", "SICCode.SicText_4",
}

MATURE_YEARS = 15      # incorporated at least this long ago => succession proxy
CURRENT_YEAR = 2026

# weights for the composite roll-up suitability score (edit freely)
W_COUNT, W_SMALL, W_MATURE, W_GEO = 0.30, 0.30, 0.20, 0.20

code_re = re.compile(r"(\d{4,5})")


def build_lookup(sector_sic):
    """sic code -> list of sectors it belongs to"""
    lut = defaultdict(list)
    for sector, codes in sector_sic.items():
        for c in codes:
            lut[c].append(sector)
    return lut


def extract_codes(row):
    out = set()
    for col in ("SICCode.SicText_1", "SICCode.SicText_2",
                "SICCode.SicText_3", "SICCode.SicText_4"):
        val = row.get(col)
        if isinstance(val, str):
            m = code_re.match(val.strip())
            if m:
                out.add(m.group(1))
    return out


def inc_year(val):
    if isinstance(val, str) and "/" in val:
        try:
            return int(val.strip().split("/")[-1])
        except ValueError:
            return None
    return None


def iter_csv_paths(arg):
    if os.path.isdir(arg):
        paths = sorted(glob.glob(os.path.join(arg, "*.csv")))
    elif any(ch in arg for ch in "*?["):
        paths = sorted(glob.glob(arg))
    else:
        paths = [arg]
    if not paths:
        sys.exit(f"No CSV files found for: {arg}")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Companies House sector fragmentation screen")
    ap.add_argument("--input", required=True, help="CSV file, folder, or glob of the CH free data product")
    ap.add_argument("--output", default="sector_screen_results.csv")
    ap.add_argument("--chunksize", type=int, default=100_000)
    args = ap.parse_args()

    lut = build_lookup(SECTOR_SIC)

    # per-sector accumulators
    n_active = Counter()
    n_small = Counter()
    n_mature = Counter()
    age_sum = Counter()
    age_n = Counter()
    counties = defaultdict(Counter)
    acct_mix = defaultdict(Counter)

    usecols = lambda c: c.strip() in NEEDED

    for path in iter_csv_paths(args.input):
        print(f"reading {path} ...", file=sys.stderr)
        reader = pd.read_csv(path, usecols=usecols, dtype=str,
                             chunksize=args.chunksize, on_bad_lines="skip",
                             low_memory=False)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            for row in chunk.to_dict("records"):
                status = (row.get("CompanyStatus") or "").strip().lower()
                if status != "active":
                    continue
                codes = extract_codes(row)
                if not codes:
                    continue
                matched = set()
                for c in codes:
                    matched.update(lut.get(c, ()))
                if not matched:
                    continue

                acct = (row.get("Accounts.AccountCategory") or "").strip().upper()
                is_small = acct in SMALL_ACCOUNTS
                yr = inc_year(row.get("IncorporationDate"))
                county = (row.get("RegAddress.County") or "").strip().title()

                for s in matched:
                    n_active[s] += 1
                    if is_small:
                        n_small[s] += 1
                    if yr:
                        age = CURRENT_YEAR - yr
                        age_sum[s] += age
                        age_n[s] += 1
                        if age >= MATURE_YEARS:
                            n_mature[s] += 1
                    if county:
                        counties[s][county] += 1
                    if acct:
                        acct_mix[s][acct] += 1

    if not n_active:
        sys.exit("No matching active companies found. Check the input path and column names.")

    # assemble raw metrics
    rows = []
    for s in SECTOR_SIC:
        n = n_active[s]
        if n == 0:
            continue
        pct_small = n_small[s] / n
        pct_mature = (n_mature[s] / age_n[s]) if age_n[s] else 0.0
        mean_age = (age_sum[s] / age_n[s]) if age_n[s] else 0.0
        distinct_counties = len([c for c in counties[s] if c])
        top_county_share = (counties[s].most_common(1)[0][1] / n) if counties[s] else 0.0
        rows.append({
            "sector": s,
            "active_companies": n,
            "pct_small": round(pct_small, 3),
            "pct_mature_15y": round(pct_mature, 3),
            "mean_age_years": round(mean_age, 1),
            "distinct_counties": distinct_counties,
            "top_county_share": round(top_county_share, 3),
        })

    df = pd.DataFrame(rows)

    # normalise (min-max) for the composite score; count is log-scaled
    def norm(series):
        lo, hi = series.min(), series.max()
        if hi == lo:
            return series * 0 + 0.5
        return (series - lo) / (hi - lo)

    df["_count_log"] = df["active_companies"].apply(lambda x: math.log10(x + 1))
    df["score"] = (
        W_COUNT  * norm(df["_count_log"]) +
        W_SMALL  * norm(df["pct_small"]) +
        W_MATURE * norm(df["pct_mature_15y"]) +
        W_GEO    * norm(df["distinct_counties"])
    ) * 100
    df["score"] = df["score"].round(1)
    df = df.drop(columns="_count_log").sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    df.to_csv(args.output, index=False)

    # console summary
    pd.set_option("display.max_columns", None, "display.width", 160)
    print("\n=== Sector fragmentation screen (Companies House free data) ===\n")
    print(df.to_string(index=False))
    print(f"\nFull table written to {args.output}")
    print("\nReminder: counts are indicative (SIC codes are broad). Revenue-based")
    print("concentration and verified owner ages are enrichment / paid-data steps.")


if __name__ == "__main__":
    main()
