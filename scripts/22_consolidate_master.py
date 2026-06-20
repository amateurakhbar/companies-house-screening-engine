#!/usr/bin/env python3
"""Consolidate the financial-layering chain into a single un-stale master:
output/financials_all.csv.

Joins, on CompanyNumber:
  base            financials_all.csv               (32 cols, incl rev_per_head_basis_k)
  + valuation     financials_all_margin_valued.csv (band, margins, ebitda, ev, risk...)
  + triangulation financials_all_revtriangulated_v2.csv (rev_from_*, revenue_est_m,
                  revenue_method, revenue_confidence, method_agreement_ratio,
                  deferred_flag, debtors_dropped)

Applies the Step-0 mis-tag fixes and re-derives the type-driven chain so the
retagged rows are internally consistent; drops NSSE. Backs up the original
financials_all.csv to financials_all.prebackfill.bak before writing in place.

Excluded: financial_layer.csv — a 44-row side artifact whose columns are sparse
and mostly renamed duplicates of base columns. Not folded in (would add ~90%
empty columns). Call out separately if its segments/account_type are wanted.
"""
import shutil
import pandas as pd

OUT = "output/financials_all.csv"
BASE = "output/financials_all.csv"
MV = "output/financials_all_margin_valued.csv"
V2 = "output/financials_all_revtriangulated_v2.csv"
ASSUMP = "output/financial_assumptions_reference.csv"

RETAGS = {
    "ADVANCED IT SERVICES NOTTINGHAM": "Managed IT services (MSP)",
    "GATHER TECHNOLOGY": "Cybersecurity specialist",
}
DROP = "NSSE"

key = "CompanyNumber"
read = lambda f: pd.read_csv(f, dtype={key: str})

base = read(BASE)
mv = read(MV)
v2 = read(V2)

# ----- assemble superset column-wise (base is the spine, preserves all base cols)
val_cols = [c for c in mv.columns if c not in base.columns]
tri_cols = [c for c in v2.columns if c not in base.columns and c not in val_cols]

df = base.merge(mv[[key] + val_cols], on=key, how="left")
df = df.merge(v2[[key] + tri_cols], on=key, how="left")
print(f"consolidated: {len(df)} rows x {df.shape[1]} cols "
      f"(base {base.shape[1]} + valuation {len(val_cols)} + triangulation {len(tri_cols)})")

# ----- assumptions
a = pd.read_csv(ASSUMP).set_index("company_type")
REV_HEAD = a["rev_per_head_k"].to_dict()
M_REP = a["margin_reported_pct"].to_dict()
M_NORM = a["margin_normalized_pct"].to_dict()

# ----- apply mis-tag fixes + re-derive type-driven chain
for substr, new_type in RETAGS.items():
    mask = df["CompanyName"].str.contains(substr, case=False, na=False)
    for i in df[mask].index:
        old = df.at[i, "company_type"]
        df.at[i, "company_type"] = new_type
        df.at[i, "type_rev_per_head_k"] = REV_HEAD[new_type]
        df.at[i, "margin_reported_pct"] = M_REP[new_type]
        df.at[i, "margin_normalized_pct"] = M_NORM[new_type]

        est = str(df.at[i, "revenue_source"]) == "estimate(rev/head)"
        emp = pd.to_numeric(df.at[i, "employees"], errors="coerce")
        if est and pd.notna(emp):
            df.at[i, "rev_per_head_basis_k"] = REV_HEAD[new_type]
            df.at[i, "revenue_best_m"] = round(emp * REV_HEAD[new_type] / 1000.0, 3)
        rev = pd.to_numeric(df.at[i, "revenue_best_m"], errors="coerce")

        if pd.notna(rev):
            df.at[i, "ebitda_reported_m"] = round(rev * M_REP[new_type] / 100.0, 3)
            ebn = round(rev * M_NORM[new_type] / 100.0, 3)
            df.at[i, "ebitda_normalized_m"] = ebn
            mult = pd.to_numeric(df.at[i, "ev_ebitda_multiple"], errors="coerce")
            if pd.notna(mult):
                df.at[i, "ev_normalized_m"] = round(ebn * mult, 3)
        print(f"  retag {df.at[i,'CompanyName']:<38} {old} -> {new_type} "
              f"| revenue_best_m={df.at[i,'revenue_best_m']} ev_norm={df.at[i,'ev_normalized_m']}")

# ----- drop NSSE
dmask = df["CompanyName"].str.contains(DROP, case=False, na=False)
for nm in df[dmask]["CompanyName"]:
    print(f"  drop  {nm}")
df = df[~dmask].reset_index(drop=True)

# ----- back up + write in place
shutil.copyfile(BASE, "output/financials_all.prebackfill.bak")
df.to_csv(OUT, index=False)
print(f"\nbackup -> output/financials_all.prebackfill.bak")
print(f"wrote  -> {OUT}  ({len(df)} rows x {df.shape[1]} cols)")
print(f"\nNote: ev_ebitda_multiple / risk_score / band / *_quality kept as-is for "
      f"retags (need upstream valuation pipeline, not in repo).")
