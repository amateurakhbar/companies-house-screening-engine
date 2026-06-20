"""14 — Recover missing financials for top-50 firms from filed accounts.

For each top-50 candidate missing turnover/operating_profit, pulls the latest
'accounts' filing from Companies House, downloads the iXBRL document, parses the
inline-XBRL facts, and extracts the standard P&L/balance-sheet concepts for the
most recent reporting period. Writes the recovered values back into the input
CSV in place (originals untouched; recovered values in _rec-suffixed columns
plus a fin_recovered provenance note), keeping it the single source of truth.

Reality check: most small UK firms file filleted/micro accounts (balance sheet
only), so turnover/operating profit frequently DO NOT EXIST in the filing. This
recovers what is actually tagged; blanks mean not filed.

Run:  COMPANIES_HOUSE_API_KEY=... python3 scripts/14_recover_financials.py
"""
from __future__ import annotations

import base64
import os
import pathlib
import re
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV = ROOT / "output" / "platform_candidates_top50.csv"
ENR = ROOT / "output" / "msp_all_companies_enriched.csv"
DOCDIR = ROOT / "data" / "cache" / "ch_accounts"
API = "https://api.company-information.service.gov.uk"

# iXBRL concept -> our column. Multiple tag spellings map to one concept.
CONCEPTS = {
    "turnover": ["TurnoverRevenue", "Turnover", "Revenue"],
    "gross_profit": ["GrossProfitLoss"],
    "operating_profit": ["OperatingProfitLoss"],
    "profit_before_tax": ["ProfitLossOnOrdinaryActivitiesBeforeTax",
                          "ProfitLossBeforeTax"],
    "profit": ["ProfitLoss"],
    "net_assets": ["NetAssetsLiabilities", "NetAssetsLiabilitiesIncludingPensionAssetLiability"],
    "cash": ["CashBankOnHand", "CashBankInHand"],
}


def _auth(key: str) -> str:
    return "Basic " + base64.b64encode(f"{key}:".encode()).decode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # surface the 302 so we can refetch without auth


_NOREDIR = urllib.request.build_opener(_NoRedirect)


def _get(url: str, key: str, accept: str = "application/json", binary=False):
    for attempt in range(4):
        req = urllib.request.Request(url, headers={"Authorization": _auth(key),
                                                   "Accept": accept})
        try:
            with _NOREDIR.open(req, timeout=40) as r:
                return r.read() if binary else r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            # Document content 302-redirects to S3, which rejects the CH auth
            # header. Follow the Location WITHOUT auth (mirrors `curl -L`).
            if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
                try:
                    with urllib.request.urlopen(e.headers["Location"], timeout=60) as r2:
                        return r2.read() if binary else r2.read().decode("utf-8", "ignore")
                except Exception:
                    return None
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", 5)) + 1); continue
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


# ---- iXBRL parsing --------------------------------------------------------
def _parse_contexts(doc: str) -> dict[str, str]:
    """contextRef id -> period end date (YYYY-MM-DD) for the period."""
    ctx = {}
    for m in re.finditer(r'<xbrli:context[^>]*id="([^"]+)"(.*?)</xbrli:context>',
                         doc, re.S):
        cid, body = m.group(1), m.group(2)
        end = re.search(r'<xbrli:(?:endDate|instant)>([\d-]+)</xbrli:', body)
        if end:
            ctx[cid] = end.group(1)
    return ctx


def _facts(doc: str):
    """Yield (concept_localname, contextRef, numeric_value)."""
    for m in re.finditer(r'<ix:nonFraction\b([^>]*)>(.*?)</ix:nonFraction>', doc, re.S):
        attrs, inner = m.group(1), m.group(2)
        name = re.search(r'name="([^"]+)"', attrs)
        cref = re.search(r'contextRef="([^"]+)"', attrs)
        if not name or not cref:
            continue
        local = name.group(1).split(":")[-1]
        raw = re.sub(r"<[^>]+>", "", inner)
        raw = raw.replace(",", "").replace("\xa0", "").strip()
        if not re.match(r"^-?\d+(\.\d+)?$", raw):
            continue
        val = float(raw)
        scale = re.search(r'scale="(-?\d+)"', attrs)
        if scale:
            val *= 10 ** int(scale.group(1))
        if re.search(r'sign="-"', attrs):
            val = -val
        yield local, cref.group(1), val


def extract(doc: str) -> dict:
    ctx = _parse_contexts(doc)
    latest = max(ctx.values()) if ctx else None
    found: dict[str, float] = {}
    for local, cref, val in _facts(doc):
        end = ctx.get(cref)
        for col, tags in CONCEPTS.items():
            if local in tags and col not in found and (end == latest or end is None):
                found[col] = val
    return found


def latest_accounts_doc(cn: str, key: str) -> str | None:
    fh = _get(f"{API}/company/{cn}/filing-history?category=accounts&items_per_page=5", key)
    if not fh:
        return None
    import json as _j
    items = _j.loads(fh).get("items", [])
    for it in items:  # newest first
        meta = (it.get("links") or {}).get("document_metadata")
        if meta:
            return meta + "/content"
    return None


def main():
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit("set COMPANIES_HOUSE_API_KEY")
    DOCDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)
    enr = pd.read_csv(ENR, dtype={"CompanyNumber": str})
    num = dict(zip(enr["CompanyName"].str.upper(), enr["CompanyNumber"]))
    df["CompanyNumber"] = df["CompanyName"].str.upper().map(num)

    rec_cols = list(CONCEPTS.keys())
    for c in rec_cols:
        df[c + "_rec"] = pd.NA
    df["fin_recovered"] = ""

    for i, r in df.iterrows():
        have = pd.notna(pd.to_numeric(pd.Series([r.get("turnover"), r.get("operating_profit")]),
                                      errors="coerce")).all()
        if have:
            continue
        cn = r["CompanyNumber"]
        if not isinstance(cn, str):
            continue
        cache = DOCDIR / f"{cn}.html"
        if cache.exists():
            doc = cache.read_text("utf-8", "ignore")
        else:
            url = latest_accounts_doc(cn, key)
            doc = _get(url, key, accept="application/xhtml+xml") if url else None
            cache.write_text(doc or "")
        if not doc:
            df.at[i, "fin_recovered"] = "no document"
            continue
        vals = extract(doc)
        for c in rec_cols:
            if c in vals:
                df.at[i, c + "_rec"] = vals[c]
        df.at[i, "fin_recovered"] = ("recovered: " + ",".join(vals.keys())) if vals else "no P&L tags (filleted?)"
        print(f"  {r['CompanyName'][:34]:34} -> {df.at[i,'fin_recovered']}")

    # Write the recovered _rec columns back into the input CSV in place, so the
    # candidate file stays the single source of truth (no separate audit file).
    df.to_csv(CSV, index=False)
    got = (df["turnover_rec"].notna() | df["operating_profit_rec"].notna()).sum()
    print(f"\nwrote recovered _rec columns -> {CSV.relative_to(ROOT)}")
    print(f"firms with recovered turnover or op_profit: {got}")


if __name__ == "__main__":
    main()
