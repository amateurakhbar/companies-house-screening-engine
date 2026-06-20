#!/usr/bin/env python3
"""Estimate revenue for firms missing it via two niche-aware methods + triangulation.

Reads output/financials_all_margin_valued.csv (never overwritten),
writes output/financials_all_revtriangulated.csv.
"""
import pandas as pd
import numpy as np

IN = "output/financials_all_margin_valued.csv"
OUT = "output/financials_all_revtriangulated.csv"

# Benchmark debtors->revenue multipliers, used when data-derived n<3
BENCHMARK = {
    "Managed IT services (MSP)": 8,
    "Computer repair / break-fix": 9,
    "Telecoms / connectivity / ISP": 8,
    "Software / app dev": 7,
    "Cybersecurity specialist": 6,
    "Data / analytics / AI": 6,
    "VAR / hardware reseller": 6,
    "ERP / CRM implementation": 5,
    "AV / integration": 5,
    "Data centre / hosting / cloud": 4,
}

df = pd.read_csv(IN)

ch_turn = pd.to_numeric(df["ch_turnover_m"], errors="coerce")
pb_rev = pd.to_numeric(df["pb_revenue_m"], errors="coerce")
debtors = pd.to_numeric(df["ch_debtors_m"], errors="coerce")
employees = pd.to_numeric(df["employees"], errors="coerce")
rev_per_head = pd.to_numeric(df["type_rev_per_head_k"], errors="coerce")

# real revenue = CH turnover preferred, else pitchbook revenue
real_rev = ch_turn.where(ch_turn.notna(), pb_rev)
has_real = real_rev.notna()

# ---------------- STEP 1: calibrate debtors->revenue multiplier per type ----------
calib = df[has_real & (debtors > 0) & (real_rev <= 30)].copy()
calib["_rev"] = real_rev[calib.index]
calib["_deb"] = debtors[calib.index]
calib["_ratio"] = calib["_rev"] / calib["_deb"]
calib = calib[(calib["_ratio"] >= 1.5) & (calib["_ratio"] <= 30)]

mult = {}
mult_source = {}
all_types = sorted(set(df["company_type"].dropna()) | set(BENCHMARK))
for t in all_types:
    sub = calib[calib["company_type"] == t]
    n = len(sub)
    if n >= 3:
        mult[t] = float(np.median(sub["_ratio"]))
        mult_source[t] = f"data (n={n}, median)"
    else:
        mult[t] = float(BENCHMARK.get(t, 6))  # 6 = generic fallback
        mult_source[t] = f"benchmark (n={n})"

print("=" * 70)
print("STEP 1 — debtors->revenue multiplier per company_type")
print("=" * 70)
for t in all_types:
    print(f"  {t:<34} mult={mult[t]:>5.2f}   [{mult_source[t]}]")

# ---------------- STEP 2: two estimates -------------------------------------------
rev_from_debtors = debtors.where(debtors > 0).map(lambda x: x) * df["company_type"].map(mult)
rev_from_debtors = np.where(debtors > 0, debtors * df["company_type"].map(mult), np.nan)
rev_from_debtors = pd.Series(rev_from_debtors, index=df.index)

rev_from_heads = np.where(employees > 0, employees * rev_per_head / 1000.0, np.nan)
rev_from_heads = pd.Series(rev_from_heads, index=df.index)

# ---------------- STEP 4 prep: deferred-income flag -------------------------------
vert = df["vertical"].fillna("").str.lower()
deferred_flag = vert.str.contains("education") | (
    df["company_type"] == "Data centre / hosting / cloud"
)

# ---------------- STEP 3: triangulate (only for firms missing real revenue) -------
revenue_est = pd.Series(np.nan, index=df.index, dtype=float)
revenue_method = pd.Series("", index=df.index, dtype=object)
revenue_conf = pd.Series("", index=df.index, dtype=object)
agreement = pd.Series(np.nan, index=df.index, dtype=float)

for i in df.index:
    if has_real[i]:
        # keep the real figure, mark as reported; not part of estimation task
        revenue_est[i] = real_rev[i]
        revenue_method[i] = "reported"
        revenue_conf[i] = "REPORTED"
        continue

    d = rev_from_debtors[i]
    h = rev_from_heads[i]
    has_d = pd.notna(d)
    has_h = pd.notna(h)
    note = ""

    if has_d and has_h:
        hi, lo = max(d, h), min(d, h)
        r = hi / lo if lo > 0 else np.inf
        agreement[i] = r
        if r <= 1.6:
            conf, est, meth = "HIGH", (d + h) / 2, "mean(debtors,heads)"
        elif r <= 2.5:
            conf, est, meth = "MEDIUM", (d + h) / 2, "mean(debtors,heads)"
        else:
            conf, est, meth = "LOW", lo, "min(debtors,heads)"
            note = "methods disagree"
        # STEP 4 deferred-income adjustment
        if deferred_flag[i] and h > d:
            est = h
            meth = "heads (deferred adj)"
            note = (note + "; " if note else "") + "debtors likely understates (prepay)"
        revenue_est[i] = est
        revenue_method[i] = meth + (f" | {note}" if note else "")
        revenue_conf[i] = conf
    elif has_d != has_h:
        est = d if has_d else h
        meth = "debtors only" if has_d else "heads only"
        # deferred adj only meaningful when both exist; single-method stays as-is
        revenue_est[i] = est
        revenue_method[i] = meth
        revenue_conf[i] = "MEDIUM"
    else:
        revenue_est[i] = np.nan
        revenue_method[i] = "no basis"
        revenue_conf[i] = "NONE"

# ---------------- assemble output -------------------------------------------------
df["rev_from_debtors_m"] = rev_from_debtors.round(3)
df["rev_from_heads_m"] = rev_from_heads.round(3)
df["revenue_est_m"] = revenue_est.round(3)
df["revenue_method"] = revenue_method
df["revenue_confidence"] = revenue_conf
df["method_agreement_ratio"] = agreement.round(2)
df["deferred_flag"] = deferred_flag

df.to_csv(OUT, index=False)

# ---------------- reporting -------------------------------------------------------
est_only = df[~has_real]  # firms the task asked us to estimate
print("\n" + "=" * 70)
print(f"STEP 2-4 — estimated revenue for {len(est_only)} firms missing real revenue")
print("=" * 70)
print("\nCount by confidence tier (estimated firms only):")
print(est_only["revenue_confidence"].value_counts().to_string())

disagree = est_only[est_only["revenue_method"].str.contains("methods disagree")]
print(f"\nFirms flagged 'methods disagree': {len(disagree)}")
no_basis = est_only[est_only["revenue_method"] == "no basis"]
print(f"Firms still with no basis: {len(no_basis)}")

print("\n" + "=" * 70)
print("15 largest method disagreements (highest agreement ratio) — check these")
print("=" * 70)
cols = ["CompanyName", "company_type", "rev_from_debtors_m", "rev_from_heads_m",
        "revenue_est_m", "method_agreement_ratio", "revenue_confidence"]
top = est_only.sort_values("method_agreement_ratio", ascending=False).head(15)
with pd.option_context("display.max_colwidth", 34, "display.width", 200):
    print(top[cols].to_string(index=False))

print(f"\nWrote {OUT}")
