"""01 — Load the immutable raw CSV, project to the working column set,
derive a clean town, and filter junk before any scraping or LLM spend.

Reads the read-only 71-column raw file; never mutates it. Emits the working
set to output/working.parquet.

Junk filtering (per implementation_plan.md Step 2 / column-set notes):
  - drop dissolved companies (CompanyStatus not active, or DissolutionDate set)
  - drop dormant / non-trading filers (Accounts.AccountCategory)
  - drop holding companies by SIC (64200, 70100)
Uses the real CompanyStatus field, not a proxy.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "potentials" / "tech_potentials_operating.csv"
OUT = ROOT / "output" / "working.parquet"
# Stage-02 URL discovery cache: {CompanyNumber: {"match": {"url": ...}, ...}}.
# Folded into the working set so the resolved website propagates to every
# downstream per-firm artefact (classified_firms.parquet, exports).
URLS = ROOT / "data" / "cache" / "urls.json"

# --- Working column set (keep) -------------------------------------------
IDENTITY = ["CompanyNumber", "CompanyName", "Status_Label"]
FILTERING = [
    "CompanyStatus",
    "CompanyCategory",
    "Accounts.AccountCategory",
    "IncorporationDate",
    "DissolutionDate",
]
GEOGRAPHY = [
    "RegAddress.AddressLine1",
    "RegAddress.PostTown",
    "RegAddress.County",
    "RegAddress.PostCode",
    "RegAddress.Country",
]
SIC = [f"SICCode.SicText_{i}" for i in range(1, 5)]
ACCOUNTS_RECENCY = ["Accounts.LastMadeUpDate"]
SCREENING = [
    "accounts_date",
    "turnover",
    "gross_profit",
    "operating_profit",
    "profit_before_tax",
    "profit",
    "net_assets",
    "equity",
    "cash",
    "creditors",
    "debtors",
    "fixed_assets",
    "current_assets",
    "employees",
]
PROVENANCE = ["fin_source"]

KEEP = (
    IDENTITY + FILTERING + GEOGRAPHY + SIC + ACCOUNTS_RECENCY + SCREENING + PROVENANCE
)

# --- Junk-filter constants ------------------------------------------------
HOLDING_SICS = {"64200", "70100"}
# AccountCategory values that signal a non-trading / dormant entity.
DORMANT_CATEGORIES = {"DORMANT", "NO ACCOUNTS FILED"}


def _norm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def _is_active_status(status: pd.Series) -> pd.Series:
    s = _norm(status).str.lower()
    # Active = blank/active/active-proposal-to-strike-off counts as still live;
    # treat explicit dissolved / liquidation / receivership as junk.
    dead_markers = ("dissolved", "liquidation", "receiver", "in administration", "closed")
    return ~s.apply(lambda v: any(m in v for m in dead_markers))


def _load_websites() -> dict[str, str]:
    """Map CompanyNumber -> resolved website URL from the stage-02 cache.

    Keys in the cache are 8-char zero-padded numbers (matching the raw file);
    entries whose discovery found no site have a null url and are skipped.
    """
    if not URLS.exists():
        return {}
    cache = json.loads(URLS.read_text())
    out: dict[str, str] = {}
    for cn, rec in cache.items():
        match = (rec or {}).get("match") or {}
        url = match.get("url")
        # Skip low-quality top-hits parked for a later Serper pass: they are
        # stored in the cache but must not populate the website column yet.
        if url and not match.get("low_quality"):
            out[str(cn)] = url
    return out


def _is_holding(df: pd.DataFrame) -> pd.Series:
    """True if any SIC text starts with a holding-company code."""
    mask = pd.Series(False, index=df.index)
    for col in SIC:
        codes = _norm(df[col]).str.split(" ").str[0]
        mask = mask | codes.isin(HOLDING_SICS)
    return mask


def main() -> None:
    df = pd.read_csv(RAW, dtype=str, low_memory=False)
    n_before = len(df)

    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        raise SystemExit(f"raw file missing expected columns: {missing}")

    df = df[KEEP].copy()

    # Derive clean town: PostTown is already clean, just normalise casing/space.
    df["town"] = _norm(df["RegAddress.PostTown"]).str.title()

    # Fold in the discovered website (blank when discovery found no site).
    websites = _load_websites()
    df["website"] = df["CompanyNumber"].astype(str).map(websites).fillna("")

    # --- Junk filters ---
    active = _is_active_status(df["CompanyStatus"])
    not_dissolved = _norm(df["DissolutionDate"]) == ""
    acct = _norm(df["Accounts.AccountCategory"]).str.upper()
    not_dormant = ~acct.isin(DORMANT_CATEGORIES)
    not_holding = ~_is_holding(df)

    keep_mask = active & not_dissolved & not_dormant & not_holding
    df = df[keep_mask].copy()
    n_after = len(df)

    n_fin_blank = int((_norm(df["fin_source"]) == "").sum())
    n_website = int((df["website"] != "").sum())

    # --- Employee-bucket triage ---
    # Micro/small filers often lack employee counts; bucket by headcount with a
    # profitability gate for the smallest firms (per size-proxy fallback notes).
    emp = pd.to_numeric(df["employees"], errors="coerce")
    profit = pd.to_numeric(df["profit"], errors="coerce")
    n_confirmed_ge3 = int((emp >= 3).sum())
    n_unknown = int(emp.isna().sum())
    n_profitable_0_2 = int((emp.between(0, 2) & (profit > 0)).sum())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"rows_before_filter   : {n_before}")
    print(f"rows_after_filter    : {n_after}")
    print(f"fin_source_blank     : {n_fin_blank}")
    print(f"website_resolved     : {n_website}")
    print("employee buckets:")
    print(f"  confirmed (>=3)    : {n_confirmed_ge3}")
    print(f"  unknown (blank)    : {n_unknown}")
    print(f"  profitable (0-2)   : {n_profitable_0_2}")


if __name__ == "__main__":
    main()
