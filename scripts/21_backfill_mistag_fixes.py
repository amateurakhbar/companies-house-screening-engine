#!/usr/bin/env python3
"""Backfill the Step-0 mis-tag corrections into the master financial CSVs so every
downstream re-run inherits them.

Corrections:
  Advanced IT Services Nottingham -> 'Managed IT services (MSP)'
  Gather Technology               -> 'Cybersecurity specialist'
  NSSE                            -> dropped (power-equipment supplier, non-core)

For the two retags, re-derives the type-driven chain from the assumptions table:
  type_rev_per_head_k, rev_per_head_basis_k (base only), revenue_best_m (estimate
  firms), margin_reported_pct, margin_normalized_pct, ebitda_reported_m,
  ebitda_normalized_m, ev_normalized_m (= ebitda_normalized_m * existing multiple).

KEPT as-is (firm/size/risk-driven, not a pure function of niche; would need the
upstream valuation pipeline that is not in this repo):
  ev_ebitda_multiple, risk_score, band, profit_quality, margin_quality.

Backs up each file to <name>.prebackfill.bak before writing in place.
"""
import shutil
import pandas as pd

ASSUMP = "output/financial_assumptions_reference.csv"
FILES = ["output/financials_all.csv", "output/financials_all_margin_valued.csv"]

RETAGS = {
    "ADVANCED IT SERVICES NOTTINGHAM": "Managed IT services (MSP)",
    "GATHER TECHNOLOGY": "Cybersecurity specialist",
}
DROP = "NSSE"

# per-type assumptions (rev/head, reported & normalized margin)
a = pd.read_csv(ASSUMP).set_index("company_type")
REV_HEAD = a["rev_per_head_k"].to_dict()
M_REP = a["margin_reported_pct"].to_dict()
M_NORM = a["margin_normalized_pct"].to_dict()


def patch(path):
    df = pd.read_csv(path)
    cols = set(df.columns)
    shutil.copyfile(path, path.replace(".csv", ".prebackfill.bak"))
    n0 = len(df)
    changes = []

    for substr, new_type in RETAGS.items():
        mask = df["CompanyName"].str.contains(substr, case=False, na=False)
        for i in df[mask].index:
            old = df.at[i, "company_type"]
            df.at[i, "company_type"] = new_type
            if "type_rev_per_head_k" in cols:
                df.at[i, "type_rev_per_head_k"] = REV_HEAD[new_type]
            # only update the rev/head basis if it was driving an estimate
            est = str(df.at[i, "revenue_source"]) == "estimate(rev/head)" if "revenue_source" in cols else False
            if "rev_per_head_basis_k" in cols and est:
                df.at[i, "rev_per_head_basis_k"] = REV_HEAD[new_type]
            if "margin_reported_pct" in cols:
                df.at[i, "margin_reported_pct"] = M_REP[new_type]
            if "margin_normalized_pct" in cols:
                df.at[i, "margin_normalized_pct"] = M_NORM[new_type]

            emp = pd.to_numeric(df.at[i, "employees"], errors="coerce")
            if est and "revenue_best_m" in cols and pd.notna(emp):
                rev = round(emp * REV_HEAD[new_type] / 1000.0, 3)
                df.at[i, "revenue_best_m"] = rev
            else:
                rev = pd.to_numeric(df.at[i, "revenue_best_m"], errors="coerce") if "revenue_best_m" in cols else None

            # re-derive EBITDA + EV chain where those columns exist
            if rev is not None and pd.notna(rev):
                if "ebitda_reported_m" in cols:
                    df.at[i, "ebitda_reported_m"] = round(rev * M_REP[new_type] / 100.0, 3)
                if "ebitda_normalized_m" in cols:
                    ebn = round(rev * M_NORM[new_type] / 100.0, 3)
                    df.at[i, "ebitda_normalized_m"] = ebn
                    if "ev_normalized_m" in cols and "ev_ebitda_multiple" in cols:
                        mult = pd.to_numeric(df.at[i, "ev_ebitda_multiple"], errors="coerce")
                        if pd.notna(mult):
                            df.at[i, "ev_normalized_m"] = round(ebn * mult, 3)

            changes.append((df.at[i, "CompanyName"], old, new_type,
                            df.at[i, "revenue_best_m"] if "revenue_best_m" in cols else None))

    drop_mask = df["CompanyName"].str.contains(DROP, case=False, na=False)
    dropped = list(df[drop_mask]["CompanyName"])
    df = df[~drop_mask].reset_index(drop=True)

    df.to_csv(path, index=False)
    print(f"\n=== {path}")
    print(f"  rows: {n0} -> {len(df)}  (backup: {path.replace('.csv','.prebackfill.bak')})")
    for nm, old, new, rev in changes:
        print(f"  retag: {nm:<38} {old} -> {new}  (revenue_best_m={rev})")
    for nm in dropped:
        print(f"  drop:  {nm}")


for f in FILES:
    patch(f)

print("\nDone. ev_ebitda_multiple / risk_score / band / *_quality kept as-is "
      "(not re-derived — need upstream valuation pipeline).")
