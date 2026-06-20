"""20 — Platform screener re-run on the FINAL inputs.

Reproduces the earlier platform-candidate screen against the now-final
defensibility + master financial layer.

INPUTS
  output/msp_defensible_final.csv          (final defensibility universe)
  output/financials_all_margin_valued.csv  (master financial layer, "financial_margin_all")

QUALIFICATION GATE (as in the original platform_candidates.csv)
  defensible firm  AND  has_real_revenue == True  AND  revenue_best_m > 5.0
  -> qualifies_via = "revenue>5m"

The master layer's revenue_best_m coalesces Pitchbook revenue + Companies-House
turnover; has_real_revenue distinguishes those from rev/head estimates.

OUTPUT
  output/platform_candidates_rerun.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEF = ROOT / "output" / "msp_defensible_final.csv"
FIN = ROOT / "output" / "financials_all_margin_valued.csv"
OUT = ROOT / "output" / "platform_candidates_rerun.csv"

REV_FLOOR_M = 5.0
REV_CEIL_M = 25.0  # platform-size cap; drop firms above this

d = pd.read_csv(DEF, dtype={"CompanyNumber": str})
f = pd.read_csv(FIN, dtype={"CompanyNumber": str})

fin_cols = [
    "CompanyNumber", "revenue_best_m", "has_real_revenue", "revenue_source",
    "ebitda_margin_actual_pct", "ebitda_normalized_m", "ev_normalized_m",
    "data_tier", "pb_ownership_status", "pb_pe_verdict",
]
m = d.merge(f[fin_cols], on="CompanyNumber", how="left", suffixes=("", "_fin"))

qual = m[(m["has_real_revenue"] == True)
         & (m["revenue_best_m"] > REV_FLOOR_M)
         & (m["revenue_best_m"] <= REV_CEIL_M)].copy()
qual["qualifies_via"] = f"revenue {REV_FLOOR_M:g}-{REV_CEIL_M:g}m"
qual = qual.sort_values("revenue_best_m", ascending=False)

show = [
    "CompanyName", "CompanyNumber", "revenue_best_m", "revenue_source",
    "ebitda_margin_actual_pct", "ebitda_normalized_m", "employees", "age_yrs",
    "defensibility", "primary_niche", "vertical", "town", "RegAddress.County",
    "ownership_type", "psc_count", "data_tier", "qualifies_via",
]
show = [c for c in show if c in qual.columns]
qual[show].to_csv(OUT, index=False)

print(f"defensible universe: {len(d)} | financial layer: {len(f)} | matched: {m['revenue_best_m'].notna().sum()}")
print(f"QUALIFY (defensible + real revenue > £{REV_FLOOR_M:g}m): {len(qual)}")
print(f"wrote {OUT.relative_to(ROOT)}")
print(qual[["CompanyName", "revenue_best_m", "revenue_source", "defensibility"]].to_string(index=False))
