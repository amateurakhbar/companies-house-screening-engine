#!/usr/bin/env python3
"""Balance-sheet-quality score for the education long-list, from real CH iXBRL.

Joins /tmp/edu_ixbrl_bs.csv (two-year facts) with output/financials_all.csv
(region, employees, revenue estimate, current creditors) and scores each company
0-100 on solvency, equity accretion (profit proxy), liquidity and low gearing.

Writes output/edu_bs_quality_scored.csv
"""
import pandas as pd, numpy as np, re

ix = pd.read_csv("/tmp/edu_ixbrl_bs.csv", dtype={"CompanyNumber":str})
m  = pd.read_csv("output/financials_all.csv", dtype={"CompanyNumber":str})

keep = ["CompanyNumber","hub" if "hub" in m.columns else "vertical","employees",
        "age_yrs","revenue_best_m","ch_creditors_m","ch_net_assets_m"]
keep = [c for c in keep if c in m.columns]
mm = m[keep].copy()
df = ix.merge(mm, on="CompanyNumber", how="left")

# region from the original map
emap = pd.read_csv("data/region_map.csv")
emap = emap[emap["CompanyName"].notna()][["CompanyName","hub","region","town"]]
def norm(s):
    s=str(s).upper(); s=re.sub(r"\b(LIMITED|LTD|LTD\.|GROUP|UK|\(UK\))\b","",s); return re.sub(r"[^A-Z0-9]","",s)
emap["k"]=emap["CompanyName"].map(norm); df["k"]=df["CompanyName"].map(norm)
df = df.merge(emap[["k","hub","region","town"]].drop_duplicates("k"), on="k", how="left", suffixes=("","_map"))

n = lambda s: pd.to_numeric(df[s], errors="coerce")
na_cur=n("net_assets_cur"); na_pri=n("net_assets_pri")
cash=n("cash_cur"); debt=n("debtors_cur"); ncurr=n("net_curr_cur"); ftang=n("fixed_tang_cur")
cred_cur = n("ch_creditors_m")*1e6   # authoritative current creditors (<1yr)

df["equity_move"]   = na_cur - na_pri
# LT obligations (LT creditors + provisions) ~= fixed + net current - net assets
df["lt_obligations"]= (ftang.fillna(0) + ncurr - na_cur)
df["cash_to_cred"]  = cash / cred_cur.replace(0,np.nan)
df["equity_move_pct"]= df["equity_move"] / na_cur.abs().replace(0,np.nan)

def pct_rank(s):  # 0..1, NaN->0.5
    r = s.rank(pct=True); return r.fillna(0.5)

# ---- components (each 0..1) ----
solvency   = (na_cur > 0).astype(float) * (0.5 + 0.5*pct_rank(na_cur.clip(lower=0)))
accretion  = pct_rank(df["equity_move"])                 # profit proxy
accretion_pct = pct_rank(df["equity_move_pct"].clip(-1,1))
liquidity  = (ncurr > 0).astype(float)*0.5 + 0.5*pct_rank(df["cash_to_cred"])
low_gearing= 1 - pct_rank(df["lt_obligations"].clip(lower=0))

# weights
df["score"] = (100*(
      0.28*solvency
    + 0.24*(0.6*accretion + 0.4*accretion_pct)
    + 0.26*liquidity
    + 0.22*low_gearing
)).round(1)

# tier label
def tier(r):
    if r["net_assets_cur"]<=0 or pd.isna(r["net_assets_cur"]): return "D - weak/neg equity"
    if r["score"]>=70: return "A - strong"
    if r["score"]>=55: return "B - solid"
    if r["score"]>=40: return "C - thin/geared"
    return "D - weak/neg equity"
df["bs_tier"] = df.apply(tier, axis=1)

cols=["CompanyName","hub","region","town","employees","age_yrs","revenue_best_m",
      "net_assets_cur","net_assets_pri","equity_move","cash_cur","debtors_cur",
      "lt_obligations","cash_to_cred","turnover_cur","ixbrl_ok","score","bs_tier"]
cols=[c for c in cols if c in df.columns]
out=df[cols].sort_values("score",ascending=False)
out.to_csv("output/edu_bs_quality_scored.csv", index=False)

pd.set_option("display.max_columns",None,"display.width",260)
print(f"scored {len(out)}  | iXBRL ok: {int(df['ixbrl_ok'].sum())}/{len(df)}")
print("\nTIER COUNTS:\n", out["bs_tier"].value_counts().to_string())
print("\nTOP 20:")
print(out.head(20)[["CompanyName","region","employees","net_assets_cur","equity_move","cash_cur","lt_obligations","score","bs_tier"]].to_string(index=False))
print("\nBOTTOM 10:")
print(out.tail(10)[["CompanyName","region","employees","net_assets_cur","equity_move","score","bs_tier"]].to_string(index=False))
