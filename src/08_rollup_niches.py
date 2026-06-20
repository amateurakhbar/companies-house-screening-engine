"""Stage 9 — niche rollup.

Joins output/classified_firms.parquet (labels) with output/working.parquet
(financials, geography, data-quality) and data/cache/scrape_status.json
(verified-URL flag), then aggregates per primary_niche into
output/niche_summary.csv.

Per niche: high-confidence vs total firm count, size-band counts, operating-
profit distribution + sub-threshold target count, median operating margin and
asset intensity, recurring-managed share (rollability proxy), top regions, and
data-quality flags (verified-URL share, fin_source-blank count).

Run:  python3 src/08_rollup_niches.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLASSIFIED = ROOT / "output" / "classified_firms.parquet"
WORKING = ROOT / "output" / "working.parquet"
SCRAPE_STATUS = ROOT / "data" / "cache" / "scrape_status.json"
OUT = ROOT / "output" / "niche_summary.csv"

HIGH_CONF = 0.70
SUBTHRESHOLD_OP_PROFIT = 100_000   # firms below this are sub-scale roll-up targets


def size_band(turnover, employees, net_assets):
    """Waterfall: turnover -> employees -> net_assets -> unknown."""
    if pd.notna(turnover):
        t = turnover
        return ("micro" if t < 1e6 else "small" if t < 5e6
                else "medium" if t < 25e6 else "large")
    if pd.notna(employees):
        e = employees
        return ("micro" if e < 10 else "small" if e < 50
                else "medium" if e < 250 else "large")
    if pd.notna(net_assets):
        n = net_assets
        return ("micro" if n < 250e3 else "small" if n < 2e6
                else "medium" if n < 10e6 else "large")
    return "unknown"


def main():
    lab = pd.read_parquet(CLASSIFIED)
    lab["CompanyNumber"] = lab["CompanyNumber"].astype(str)
    wcols = ["CompanyNumber", "turnover", "operating_profit", "net_assets",
             "fixed_assets", "current_assets", "employees", "fin_source",
             "RegAddress.PostTown", "RegAddress.County"]
    w = pd.read_parquet(WORKING)[wcols].copy()
    w["CompanyNumber"] = w["CompanyNumber"].astype(str)
    df = lab.merge(w, on="CompanyNumber", how="left")

    # verified-URL flag from scrape_status.json
    ss = json.loads(SCRAPE_STATUS.read_text())
    df["verified_url"] = df["CompanyNumber"].map(
        lambda cn: bool(ss.get(cn, {}).get("verified", False)))

    # numeric coercion
    for c in ["turnover", "operating_profit", "net_assets", "fixed_assets",
              "current_assets", "employees"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["_band"] = df.apply(
        lambda r: size_band(r["turnover"], r["employees"], r["net_assets"]), axis=1)
    df["_margin"] = np.where(df["turnover"] > 0,
                             df["operating_profit"] / df["turnover"], np.nan)
    tot_assets = df["fixed_assets"].fillna(0) + df["current_assets"].fillna(0)
    df["_assetint"] = np.where(df["turnover"] > 0, tot_assets / df["turnover"], np.nan)
    df["_finblank"] = df["fin_source"].isna() | (df["fin_source"].astype(str).str.strip() == "")

    rows = []
    for niche, g in df.groupby("primary_niche"):
        bands = g["_band"].value_counts()
        op = g["operating_profit"].dropna()
        regions = g["RegAddress.PostTown"].dropna().replace("", np.nan).dropna()
        top_regions = "; ".join(f"{r}({n})" for r, n in regions.value_counts().head(3).items())
        rows.append({
            "primary_niche": niche,
            "total": len(g),
            "high_conf": int((g["confidence"] >= HIGH_CONF).sum()),
            "high_conf_pct": round((g["confidence"] >= HIGH_CONF).mean() * 100, 1),
            "recurring_managed_share": round(
                (g["business_model"] == "recurring_managed").mean() * 100, 1),
            "band_micro": int(bands.get("micro", 0)),
            "band_small": int(bands.get("small", 0)),
            "band_medium": int(bands.get("medium", 0)),
            "band_large": int(bands.get("large", 0)),
            "band_unknown": int(bands.get("unknown", 0)),
            "op_profit_with_data": int(op.shape[0]),
            "op_profit_median": round(float(op.median()), 0) if len(op) else np.nan,
            "subthreshold_targets": int((op < SUBTHRESHOLD_OP_PROFIT).sum()),
            "op_margin_median": round(float(g["_margin"].median()), 3)
            if g["_margin"].notna().any() else np.nan,
            "asset_intensity_median": round(float(g["_assetint"].median()), 3)
            if g["_assetint"].notna().any() else np.nan,
            "verified_url_share": round(g["verified_url"].mean() * 100, 1),
            "fin_source_blank": int(g["_finblank"].sum()),
            "top_regions": top_regions,
        })

    summary = pd.DataFrame(rows).sort_values("high_conf", ascending=False)
    summary.to_csv(OUT, index=False)
    print(f"wrote {len(summary)} niches -> {OUT.relative_to(ROOT)}")
    print(f"(high-conf threshold >= {HIGH_CONF}; sub-threshold op_profit < "
          f"£{SUBTHRESHOLD_OP_PROFIT:,})\n")

    show = summary.head(20)[
        ["primary_niche", "high_conf", "total", "high_conf_pct",
         "recurring_managed_share", "subthreshold_targets", "verified_url_share"]]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print("TOP 20 niches by high-confidence firm count "
              "(count vs quality):\n")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
