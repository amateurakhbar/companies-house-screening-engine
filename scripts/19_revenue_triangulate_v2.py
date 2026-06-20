#!/usr/bin/env python3
"""Revenue triangulation v2: fix mis-tags, add debtors plausibility gate,
replace MIN logic with a genuine-conflict rule.

Reads output/financials_all_margin_valued.csv (never overwritten),
writes output/financials_all_revtriangulated_v2.csv.
"""
import pandas as pd
import numpy as np

IN = "output/financials_all_margin_valued.csv"
OUT = "output/financials_all_revtriangulated_v2.csv"

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

# clean per-type rev/head lookup (derived from the corrected type, not the stale cell)
REV_HEAD = (df.dropna(subset=["type_rev_per_head_k"])
              .groupby("company_type")["type_rev_per_head_k"].first().to_dict())

# ---------------- STEP 0: fix known mis-tags --------------------------------------
print("=" * 70)
print("STEP 0 — fixing known mis-tags")
print("=" * 70)

def retag(name_substr, new_type):
    mask = df["CompanyName"].str.contains(name_substr, case=False, na=False)
    for idx in df[mask].index:
        old = df.at[idx, "company_type"]
        df.at[idx, "company_type"] = new_type
        df.at[idx, "type_rev_per_head_k"] = REV_HEAD[new_type]
        print(f"  retag: {df.at[idx,'CompanyName']:<40} {old} -> {new_type} "
              f"(rev/head {REV_HEAD[new_type]:.0f}k)")

retag("ADVANCED IT SERVICES NOTTINGHAM", "Managed IT services (MSP)")
retag("GATHER TECHNOLOGY", "Cybersecurity specialist")

drop_mask = df["CompanyName"].str.contains("NSSE", case=False, na=False)
for nm in df[drop_mask]["CompanyName"]:
    print(f"  drop:  {nm:<40} (power-equipment supplier, non-core)")
df = df[~drop_mask].reset_index(drop=True)

# re-derive series after corrections
ch_turn = pd.to_numeric(df["ch_turnover_m"], errors="coerce")
pb_rev = pd.to_numeric(df["pb_revenue_m"], errors="coerce")
debtors = pd.to_numeric(df["ch_debtors_m"], errors="coerce")
employees = pd.to_numeric(df["employees"], errors="coerce")
rev_per_head = pd.to_numeric(df["type_rev_per_head_k"], errors="coerce")
real_rev = ch_turn.where(ch_turn.notna(), pb_rev)
has_real = real_rev.notna()

# ---------------- STEP 1a: re-derive multiplier per corrected type ----------------
calib = df[has_real & (debtors > 0) & (real_rev <= 30)].copy()
calib["_ratio"] = real_rev[calib.index] / debtors[calib.index]
calib = calib[(calib["_ratio"] >= 1.5) & (calib["_ratio"] <= 30)]

mult, mult_source = {}, {}
for t in sorted(set(df["company_type"].dropna()) | set(BENCHMARK)):
    sub = calib[calib["company_type"] == t]
    n = len(sub)
    if n >= 3:
        mult[t] = float(np.median(sub["_ratio"]))
        mult_source[t] = f"data (n={n}, median)"
    else:
        mult[t] = float(BENCHMARK.get(t, 6))
        mult_source[t] = f"benchmark (n={n})"

print("\n" + "=" * 70)
print("STEP 1 — debtors->revenue multiplier per company_type (post-retag)")
print("=" * 70)
for t in sorted(mult):
    print(f"  {t:<34} mult={mult[t]:>5.2f}   [{mult_source[t]}]")

mult_series = df["company_type"].map(mult)

# ---------------- STEP 1b: debtors plausibility gate ------------------------------
# raw two estimates (pre-gate) for the gate test
raw_rev_deb = np.where(debtors > 0, debtors * mult_series, np.nan)
raw_rev_deb = pd.Series(raw_rev_deb, index=df.index)
rev_from_heads = np.where(employees > 0, employees * rev_per_head / 1000.0, np.nan)
rev_from_heads = pd.Series(rev_from_heads, index=df.index)

cond_tiny = debtors < 0.02                                  # under £20k -> truncated
cond_artifact = ((debtors > 0) & (employees >= 5) &
                 (raw_rev_deb < 0.30 * rev_from_heads))     # implausibly low vs heads
debtors_dropped = (cond_tiny | cond_artifact) & debtors.notna()

# gated debtors estimate: drop the artifact debtors entirely
debtors_gated = debtors.where(~debtors_dropped)
rev_from_debtors = np.where(debtors_gated > 0, debtors_gated * mult_series, np.nan)
rev_from_debtors = pd.Series(rev_from_debtors, index=df.index)

# ---------------- deferred-income flag --------------------------------------------
vert = df["vertical"].fillna("").str.lower()
deferred_flag = vert.str.contains("education") | (
    df["company_type"] == "Data centre / hosting / cloud")

# ---------------- STEP 2/3: triangulate (new disagreement rule) -------------------
revenue_est = pd.Series(np.nan, index=df.index, dtype=float)
revenue_method = pd.Series("", index=df.index, dtype=object)
revenue_conf = pd.Series("", index=df.index, dtype=object)
agreement = pd.Series(np.nan, index=df.index, dtype=float)

for i in df.index:
    if has_real[i]:
        revenue_est[i] = real_rev[i]
        revenue_method[i] = "reported"
        revenue_conf[i] = "REPORTED"
        continue

    d = rev_from_debtors[i]
    h = rev_from_heads[i]
    has_d, has_h = pd.notna(d), pd.notna(h)
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
            # genuine conflict: debtors survived the gate, so trust headcount
            conf, est, meth = "MEDIUM-VERIFY", h, "heads (genuine conflict)"
            note = "genuine method conflict"
        # deferred-income rule still applies (heads floor for prepay verticals)
        if deferred_flag[i] and h > d:
            est = h
            meth = "heads (deferred adj)"
            note = (note + "; " if note else "") + "debtors likely understates (prepay)"
        revenue_est[i] = est
        revenue_method[i] = meth + (f" | {note}" if note else "")
        revenue_conf[i] = conf
    elif has_h and not has_d:
        revenue_est[i] = h
        if debtors_dropped[i]:
            revenue_method[i] = "heads (debtors unreliable)"
        else:
            revenue_method[i] = "heads only"
        revenue_conf[i] = "MEDIUM"
    elif has_d and not has_h:
        revenue_est[i] = d
        revenue_method[i] = "debtors only"
        revenue_conf[i] = "MEDIUM"
    else:
        revenue_est[i] = np.nan
        revenue_method[i] = "no basis"
        revenue_conf[i] = "NONE"

# ---------------- assemble & write ------------------------------------------------
df["rev_from_debtors_m"] = rev_from_debtors.round(3)
df["rev_from_heads_m"] = rev_from_heads.round(3)
df["revenue_est_m"] = revenue_est.round(3)
df["revenue_method"] = revenue_method
df["revenue_confidence"] = revenue_conf
df["method_agreement_ratio"] = agreement.round(2)
df["deferred_flag"] = deferred_flag
df["debtors_dropped"] = debtors_dropped
df.to_csv(OUT, index=False)

# ---------------- reporting -------------------------------------------------------
est = df[~has_real]
print("\n" + "=" * 70)
print(f"RESULTS — {len(est)} firms estimated  (NSSE dropped; {len(df)} rows total)")
print("=" * 70)
print("\nConfidence tier counts (estimated firms only):")
print(est["revenue_confidence"].value_counts().to_string())

print(f"\nDebtors dropped as artifacts (all rows): {int(debtors_dropped.sum())}")
print(f"  - tiny (<£20k):                {int((cond_tiny & debtors.notna()).sum())}")
print(f"  - implausible vs headcount:    {int(cond_artifact.sum())}")

genuine = est[est["revenue_method"].str.contains("genuine method conflict", na=False)]
print(f"\nGENUINE conflicts (both estimates real, still >2.5x apart): {len(genuine)}")
print("  -> this is the true verify list:")
cols = ["CompanyName", "company_type", "rev_from_debtors_m", "rev_from_heads_m",
        "revenue_est_m", "method_agreement_ratio"]
with pd.option_context("display.max_colwidth", 36, "display.width", 200):
    print(genuine.sort_values("method_agreement_ratio", ascending=False)[cols]
          .to_string(index=False))

print(f"\nNo basis: {(est['revenue_method']=='no basis').sum()}")
print(f"\nWrote {OUT}")
