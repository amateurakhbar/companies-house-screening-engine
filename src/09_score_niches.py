"""Stage 9b — weighted niche scoring for roll-up attractiveness.

Reads output/niche_summary.csv (per-niche aggregates) and output/working.parquet
(joined to firms via output/classified_firms.parquet for incorporation age and
region spread). Scores each niche on five 0-1 normalised axes, applies weights,
and ranks.

  1. Deal supply        (25%): high-conf firm count + sub-threshold op-profit count
  2. Recurring revenue  (35%): recurring_managed_share  [INFERRED PROXY, not verified]
  3. Succession signal  (15%): median incorporation age (older = more founder exits)
  4. Cash generation    (15%): median operating margin, penalised by asset intensity
  5. Scale potential    (10%): geographic spread (distinct regions)

Output: output/niche_scores.csv.  Run:  python3 src/09_score_niches.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "output" / "niche_summary.csv"
WORKING = ROOT / "output" / "working.parquet"
CLASSIFIED = ROOT / "output" / "classified_firms.parquet"
OUT = ROOT / "output" / "niche_scores.csv"

TODAY = pd.Timestamp("2026-06-08")
MIN_HIGH_CONF = 10        # floor: skip thin niches so min-max isn't noise-driven
WEIGHTS = {"deal_supply": 0.25, "recurring": 0.35, "succession": 0.15,
           "cash_generation": 0.15, "scale_potential": 0.10}


def norm(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


def main():
    summ = pd.read_csv(SUMMARY)
    summ = summ[summ["primary_niche"] != "insufficient_evidence"]
    summ = summ[summ["high_conf"] >= MIN_HIGH_CONF].copy()

    # per-firm join for incorporation age + region spread
    lab = pd.read_parquet(CLASSIFIED)[["CompanyNumber", "primary_niche"]].copy()
    lab["CompanyNumber"] = lab["CompanyNumber"].astype(str)
    w = pd.read_parquet(WORKING)[
        ["CompanyNumber", "IncorporationDate", "RegAddress.County"]].copy()
    w["CompanyNumber"] = w["CompanyNumber"].astype(str)
    firms = lab.merge(w, on="CompanyNumber", how="left")
    firms["_age"] = (TODAY - pd.to_datetime(
        firms["IncorporationDate"], errors="coerce")).dt.days / 365.25

    age = firms.groupby("primary_niche")["_age"].median().rename("median_age_yrs")
    spread = firms.groupby("primary_niche")["RegAddress.County"].nunique().rename(
        "distinct_regions")
    summ = summ.merge(age, on="primary_niche", how="left").merge(
        spread, on="primary_niche", how="left")

    # ---- raw axis signals ----
    summ["_deal_raw"] = (norm(summ["high_conf"]) + norm(summ["subthreshold_targets"])) / 2
    summ["_rec_raw"] = summ["recurring_managed_share"]
    summ["_succ_raw"] = summ["median_age_yrs"]
    # cash: operating margin per unit of asset intensity (penalises heavy assets)
    ai = summ["asset_intensity_median"].fillna(summ["asset_intensity_median"].median())
    summ["_cash_raw"] = summ["op_margin_median"].fillna(0) / (1.0 + ai.clip(lower=0))
    summ["_scale_raw"] = summ["distinct_regions"]

    # ---- normalise to 0-1 and weight ----
    summ["deal_supply"] = summ["_deal_raw"]            # already 0-1 (mean of norms)
    summ["recurring"] = norm(summ["_rec_raw"])
    summ["succession"] = norm(summ["_succ_raw"])
    summ["cash_generation"] = norm(summ["_cash_raw"])
    summ["scale_potential"] = norm(summ["_scale_raw"])

    summ["composite"] = sum(summ[a] * wt for a, wt in WEIGHTS.items())
    summ = summ.sort_values("composite", ascending=False).reset_index(drop=True)
    summ["rank"] = summ.index + 1

    cols = ["rank", "primary_niche", "high_conf", "total", "recurring_managed_share",
            "median_age_yrs", "distinct_regions",
            "deal_supply", "recurring", "succession", "cash_generation",
            "scale_potential", "composite"]
    summ[cols].round(3).to_csv(OUT, index=False)
    print(f"scored {len(summ)} niches (high_conf >= {MIN_HIGH_CONF}, "
          f"excl insufficient_evidence) -> {OUT.relative_to(ROOT)}")
    print("NOTE: recurring_managed_share is an INFERRED proxy from the LLM "
          "classification, NOT verified revenue data.\n")

    show = summ.head(15)[
        ["rank", "primary_niche", "high_conf", "recurring_managed_share",
         "deal_supply", "recurring", "succession", "cash_generation",
         "scale_potential", "composite"]].round(3)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("TOP 15 niches by composite roll-up score:\n")
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
