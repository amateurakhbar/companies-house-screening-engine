"""
Filter Companies House bulk data for tech/IT SIC codes.
Output: potentials/tech_potentials.csv (active companies only)
"""

import pandas as pd

INPUT = "chdata/BasicCompanyDataAsOneFile.csv.gz"
OUTPUT = "potentials/tech_potentials.csv"

TARGET_CODES = {
    # Software development
    "62011", "62012",
    # IT consultancy
    "62020",
    # Computer facilities management
    "62030",
    # Other IT services
    "62090",
    # Data processing, hosting, web portals, other info services
    "63110", "63120", "63990",
    # Software publishing (games, other)
    "58210", "58290",
    # Telecommunications (all 61xxx)
    "61100", "61200", "61300", "61900",
    # Wholesale of computers & peripherals
    "46510",
    # Retail of computers
    "47410",
    # Computer repair
    "95110",
}

SIC_COLS = ["SICCode.SicText_1", "SICCode.SicText_2", "SICCode.SicText_3", "SICCode.SicText_4"]

print(f"Loading dataset...")
df = pd.read_csv(INPUT, dtype=str, low_memory=False)
df.columns = df.columns.str.strip()
print(f"Total companies: {len(df):,}")

# Match any of the 4 SIC code columns (format: "62011 - Description")
mask = pd.Series(False, index=df.index)
for col in SIC_COLS:
    if col in df.columns:
        codes = df[col].fillna("").str.strip().str[:5]
        mask |= codes.isin(TARGET_CODES)

filtered = df[mask].copy()
print(f"Matched (all statuses): {len(filtered):,}")

# Active companies only
filtered = filtered[filtered["CompanyStatus"].str.strip().str.lower() == "active"]
print(f"Active only: {len(filtered):,}")

filtered.to_csv(OUTPUT, index=False)
print(f"\nSaved to: {OUTPUT}")

print("\nBreakdown by primary SIC code:")
print(filtered["SICCode.SicText_1"].value_counts().to_string())
