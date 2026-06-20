"""15 — Estimate turnover for firms that filed headcount but no P&L.

Most small UK firms file filleted/micro accounts (no turnover). This derives a
revenue-per-employee (RPE) benchmark FROM OUR OWN DATASET — the firms that filed
both turnover and employees — split by business_model, then estimates turnover
for firms that have employees but no filed turnover.

THIS PRODUCES ESTIMATES, NOT FILED FIGURES. Output columns:
  turnover_est            estimated turnover (employees * median RPE for model)
  turnover_est_basis      human-readable basis, e.g. "223 emp x £85,677/emp"
  turnover_is_estimated   True where turnover_est was used (no filed turnover)

Assumptions / caveats:
  - RPE = dataset median per business_model; small samples for resale/project.
  - Companies House headcount can be approximate.
  - Assumes the firm performs at the dataset median for its model.

Run:  python3 scripts/15_estimate_turnover.py [--in <csv>]
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENR = ROOT / "output" / "msp_all_companies_enriched.csv"
MIN_SAMPLE = 3  # need >=3 firms to trust a per-model RPE


def build_rpe() -> tuple[dict, float]:
    enr = pd.read_csv(ENR, dtype={"CompanyNumber": str})
    for c in ("turnover", "employees"):
        enr[c] = pd.to_numeric(enr[c], errors="coerce")
    b = enr[(enr["turnover"] > 0) & (enr["employees"] > 0)].copy()
    b["rpe"] = b["turnover"] / b["employees"]
    default = float(b["rpe"].median())
    per_model = {bm: float(g["rpe"].median())
                 for bm, g in b.groupby("business_model") if len(g) >= MIN_SAMPLE}
    print(f"RPE benchmark from {len(b)} firms (filed turnover + employees):")
    print(f"  default median: £{default:,.0f}")
    for k, v in per_model.items():
        print(f"  {k:20} £{v:,.0f}")
    return per_model, default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "output" / "platform_candidates_top50.csv"))
    args = ap.parse_args()
    inp = pathlib.Path(args.inp)

    rpe, default = build_rpe()
    df = pd.read_csv(inp)
    for c in ("turnover", "employees"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    if "business_model" not in df.columns:
        enr = pd.read_csv(ENR, dtype={"CompanyNumber": str})
        bm = dict(zip(enr["CompanyName"].str.upper(), enr["business_model"]))
        df["business_model"] = df["CompanyName"].str.upper().map(bm)

    def est(r):
        if pd.notna(r["turnover"]) or pd.isna(r["employees"]):
            return (np.nan, "")
        rate = rpe.get(r["business_model"], default)
        return (round(r["employees"] * rate),
                f"est: {int(r['employees'])} emp x £{rate:,.0f}/emp "
                f"({r['business_model'] or 'default'})")

    res = df.apply(est, axis=1, result_type="expand")
    df["turnover_est"], df["turnover_est_basis"] = res[0], res[1]
    df["turnover_is_estimated"] = df["turnover"].isna() & df["turnover_est"].notna()
    df.to_csv(inp, index=False)
    print(f"\nestimated turnover for {int(df['turnover_is_estimated'].sum())} firms "
          f"-> {inp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
