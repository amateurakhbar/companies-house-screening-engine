"""16 (STEP 4a, stage 2) — Extract strict-schema financials from cached iXBRL.

Reads the documents cached by stage 1 (output/education_accounts_raw/) and the
manifest, parses inline-XBRL facts, maps them to the strict schema (one object
per company-year), classifies account_type, and writes output/education_finps.csv.

Resumable: rebuilds purely from the cache; never touches inputs or the network.
"""
from __future__ import annotations

import json
import pathlib
import re

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCDIR = ROOT / "output" / "education_accounts_raw"
MANIFEST = DOCDIR / "_manifest.json"
OUT = ROOT / "output" / "education_finps.csv"

SCHEMA = ["turnover", "cost_of_sales", "gross_profit", "admin_expenses",
          "operating_profit", "depreciation_amortisation", "interest",
          "profit_before_tax", "tax", "profit", "director_remuneration",
          "avg_employees", "tangible_fixed_assets", "cash", "debtors",
          "creditors_within_1yr", "creditors_after_1yr", "net_assets",
          "equity", "retained_earnings"]

# field -> candidate iXBRL concept localnames (FRS 101/102/105 taxonomies)
TAGS = {
    "turnover": ["TurnoverRevenue", "Turnover", "Revenue"],
    "cost_of_sales": ["CostSales", "CostOfSales"],
    "gross_profit": ["GrossProfitLoss"],
    "admin_expenses": ["AdministrativeExpenses", "AdministrationExpenses"],
    "operating_profit": ["OperatingProfitLoss"],
    "depreciation_amortisation": [
        "DepreciationAmortisationExpense",
        "DepreciationAmortisationImpairmentExpense",
        "DepreciationExpensePropertyPlantEquipment",
        "DepreciationOfPropertyPlantEquipment",
        "AmortisationExpenseIntangibleAssets",
        "DepreciationAmortisationImpairment"],
    "interest": ["InterestPayableSimilarChargesFinanceCosts",
                 "InterestExpenseOnDebtAndBorrowings", "FinanceCosts",
                 "InterestIncomeExpenseNet", "InterestPayableExpense"],
    "profit_before_tax": ["ProfitLossOnOrdinaryActivitiesBeforeTax",
                          "ProfitLossBeforeTax"],
    "tax": ["TaxTaxCreditOnProfitOrLossOnOrdinaryActivities", "TaxOnProfitOrLoss",
            "TaxExpenseCredit"],
    "profit": ["ProfitLoss", "ProfitLossForPeriod"],
    "director_remuneration": ["DirectorRemuneration",
                              "RemunerationDirectors",
                              "DirectorsRemunerationIncludingPensionContributions",
                              "KeyManagementPersonnelCompensationTotal"],
    "avg_employees": ["AverageNumberEmployeesDuringPeriod",
                      "AverageNumberEmployees"],
    "tangible_fixed_assets": ["PropertyPlantEquipment", "TangibleFixedAssets"],
    "cash": ["CashBankOnHand", "CashBankInHand"],
    "debtors": ["Debtors", "TradeOtherReceivablesDueWithinOneYear"],
    "net_assets": ["NetAssetsLiabilities",
                   "NetAssetsLiabilitiesIncludingPensionAssetLiability"],
    "equity": ["Equity", "ShareholdersFunds", "TotalEquity"],
    "retained_earnings": ["RetainedEarningsAccumulatedLosses"],
}
CREDITOR_TAGS = ["Creditors"]


def parse_contexts(doc: str) -> dict:
    """id -> {'end': date, 'dims': set(member-localnames)}"""
    ctx = {}
    for m in re.finditer(r'<xbrli:context[^>]*id="([^"]+)"(.*?)</xbrli:context>',
                         doc, re.S):
        cid, body = m.group(1), m.group(2)
        end = re.search(r'<xbrli:(?:endDate|instant)>([\d-]+)</xbrli:', body)
        dims = set(re.findall(r'<xbrldi:explicitMember[^>]*>[^:<]*:?([^:<]+)</xbrldi:explicitMember>', body))
        ctx[cid] = {"end": end.group(1) if end else None, "dims": dims}
    return ctx


def facts(doc: str):
    """Yield (localname, contextRef, value, raw_dims_via_ctx)."""
    for m in re.finditer(r'<ix:nonFraction\b([^>]*)>(.*?)</ix:nonFraction>', doc, re.S):
        attrs, inner = m.group(1), m.group(2)
        nm = re.search(r'name="([^"]+)"', attrs)
        cref = re.search(r'contextRef="([^"]+)"', attrs)
        if not nm or not cref:
            continue
        local = nm.group(1).split(":")[-1]
        raw = re.sub(r"<[^>]+>", "", inner).replace(",", "").replace("\xa0", "").strip()
        raw = raw.replace("(", "-").replace(")", "")
        if not re.match(r"^-?\d+(\.\d+)?$", raw):
            continue
        val = float(raw)
        sc = re.search(r'scale="(-?\d+)"', attrs)
        if sc:
            val *= 10 ** int(sc.group(1))
        if re.search(r'sign="-"', attrs):
            val = -val
        yield local, cref.group(1), val


def account_type(desc: str) -> str:
    d = (desc or "").lower()
    if "micro" in d:
        return "micro"
    if "full" in d:
        return "full"
    if "small" in d or "abridged" in d or "abbreviated" in d:
        return "small"
    return "unknown"


def extract_doc(doc: str, made_up: str, desc: str) -> dict:
    ctx = parse_contexts(doc)
    out = {k: None for k in SCHEMA}
    out["account_type"] = account_type(desc)
    # collect facts for the reporting period whose context end == made_up_date.
    # Prefer the PRIMARY-statement value (no context dimensions) over note
    # sub-components; tie-break on tag priority. coll: field -> (dims_empty, prio, val)
    coll = {}
    cred_w, cred_a, cred_none = None, None, None
    retained = None
    for local, cref, val in facts(doc):
        c = ctx.get(cref, {})
        end = c.get("end")
        dims = c.get("dims", set())
        if not ((end == made_up) or (end is None)):
            continue
        dimstr = " ".join(dims).lower()
        # creditors split by maturity dimension member. Filers use either the
        # MaturitiesOrExpirationPeriods dim (WithinOneYear/AfterOneYear) or the
        # FinancialInstrumentsCurrentNon-current dim (Current/Non-current). Test
        # the "non-current"/"after" cases first (they substring-contain "current").
        if local in CREDITOR_TAGS:
            if "afteroneyear" in dimstr or "morethanoneyear" in dimstr or "non-currentfinancialinstruments" in dimstr:
                cred_a = val if cred_a is None else cred_a
            elif "withinoneyear" in dimstr or "currentfinancialinstruments" in dimstr:
                cred_w = val if cred_w is None else cred_w
            elif not dims:
                cred_none = val if cred_none is None else cred_none
            continue
        # retained earnings is an Equity fact carrying a retained-earnings dim
        if local in ("Equity", "ShareholdersFunds", "TotalEquity") and "retainedearnings" in dimstr:
            retained = val if retained is None else retained
        for field, tags in TAGS.items():
            if local not in tags:
                continue
            cand = (len(dims) == 0, -tags.index(local), val)  # prefer no-dims, then earlier tag
            if field not in coll or cand[:2] > coll[field][:2]:
                coll[field] = cand
    for f, (_, _, v) in coll.items():
        out[f] = v
    out["creditors_within_1yr"] = cred_w if cred_w is not None else cred_none
    out["creditors_after_1yr"] = cred_a
    out["retained_earnings"] = retained
    # avg_employees should be an integer count, not scaled to thousands
    if out["avg_employees"] is not None:
        out["avg_employees"] = round(out["avg_employees"])
    return out


def main():
    manifest = json.loads(MANIFEST.read_text())
    rows = []
    for cn, entry in manifest.items():
        name = entry.get("company_name")
        for f in entry.get("filings", []):
            if not f.get("doc_fetched"):
                continue
            p = ROOT / f["doc_path"]
            if not p.exists():
                continue
            doc = p.read_text("utf-8", "ignore")
            rec = extract_doc(doc, f.get("made_up_date"), f.get("description"))
            row = {"CompanyNumber": cn, "CompanyName": name,
                   "made_up_date": f.get("made_up_date"),
                   "filing_description": f.get("description"),
                   "account_type": rec.pop("account_type")}
            row.update(rec)
            rows.append(row)

    cols = ["CompanyNumber", "CompanyName", "made_up_date", "filing_description",
            "account_type"] + SCHEMA
    df = pd.DataFrame(rows)[cols].sort_values(["CompanyName", "made_up_date"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} company-year rows "
          f"({df['CompanyNumber'].nunique()} companies)\n")

    # coverage on the latest year per company
    latest = df.sort_values("made_up_date").groupby("CompanyNumber").tail(1)
    n = len(latest)
    print(f"Coverage on latest filed year ({n} companies):")
    for fld in ["turnover", "operating_profit", "depreciation_amortisation",
                "director_remuneration"]:
        c = latest[fld].notna().sum()
        print(f"  {fld:28} {c}/{n}")
    print("\nAccount-type mix (all rows):")
    print(df["account_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
