"""23 — Pull + parse CH accounts for the 4 healthcare-layer firms with real figures.

Mirrors the education process (scripts 17 pull + 16 extract) for an explicit
4-firm list, into a dedicated cache so it never touches the education outputs.

  pull   : CH filing-history (category=accounts) + iXBRL docs  -> output/healthcare_filers_accounts_raw/<cn>/
  extract: reuse script 16's iXBRL parser                       -> output/healthcare_filers_finps.csv

Resumable per company (manifest). Run:
  COMPANIES_HOUSE_API_KEY=... python3 scripts/23_pull_healthcare_filers.py
"""
from __future__ import annotations
import base64, importlib.util, json, os, pathlib, time, urllib.error, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCDIR = ROOT / "output" / "healthcare_filers_accounts_raw"
MANIFEST = DOCDIR / "_manifest.json"
OUT = ROOT / "output" / "healthcare_filers_finps.csv"
API = "https://api.company-information.service.gov.uk"
SLEEP = 0.25

FIRMS = [
    ("06155295", "INSIGHTS SOLUTIONS LIMITED"),
    ("10590426", "MW IT SERVICES (HOSPITALITY) LTD"),
    ("10289770", "UNITY GPO LTD"),
    ("03955112", "ESC DIGITAL MEDIA LIMITED"),
]

# reuse the exact iXBRL parser + schema from script 16
_spec = importlib.util.spec_from_file_location(
    "extract16", ROOT / "scripts" / "16_extract_education_finps.py")
_ex = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_ex)
extract_doc, SCHEMA = _ex.extract_doc, _ex.SCHEMA


def _auth(k): return "Basic " + base64.b64encode(f"{k}:".encode()).decode()
class _NR(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a): return None
_OP = urllib.request.build_opener(_NR)


def _get(url, key, accept="application/json"):
    delay = 2.0
    for _ in range(6):
        req = urllib.request.Request(url, headers={"Authorization": _auth(key), "Accept": accept})
        try:
            with _OP.open(req, timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
                try:
                    with urllib.request.urlopen(e.headers["Location"], timeout=90) as r2:
                        return r2.read().decode("utf-8", "ignore")
                except Exception:
                    return None
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", delay)) + 1); delay = min(delay * 2, 60); continue
            time.sleep(delay); delay = min(delay * 2, 60)
        except Exception:
            time.sleep(delay); delay = min(delay * 2, 60)
    return None


def pull(key):
    DOCDIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for cn, name in FIRMS:
        e = man.get(cn, {"company_name": name, "filings": []})
        if e.get("complete") and [f for f in e["filings"] if f.get("doc_fetched")]:
            print(f"  {name[:36]:36} ({cn}) cached"); continue
        print(f"  {name[:36]:36} ({cn})")
        fh = _get(f"{API}/company/{cn}/filing-history?category=accounts&items_per_page=100", key); time.sleep(SLEEP)
        if not fh:
            e["error"] = "no filing history"; man[cn] = e; MANIFEST.write_text(json.dumps(man, indent=2)); continue
        items = [it for it in json.loads(fh).get("items", []) if (it.get("links") or {}).get("document_metadata")][:3]
        e["filings"] = []; cdir = DOCDIR / cn; cdir.mkdir(exist_ok=True)
        for it in items:
            mu = (it.get("description_values") or {}).get("made_up_date") or it.get("date")
            meta = (it.get("links") or {}).get("document_metadata"); desc = it.get("description", "")
            fp = cdir / f"{mu or it.get('transaction_id')}.xhtml"
            rec = {"made_up_date": mu, "description": desc, "doc_path": str(fp.relative_to(ROOT)), "doc_fetched": False}
            if fp.exists() and fp.stat().st_size > 200:
                rec["doc_fetched"] = True
            else:
                doc = _get(meta + "/content", key, accept="application/xhtml+xml"); time.sleep(SLEEP)
                if doc and len(doc) > 200:
                    fp.write_text(doc, "utf-8"); rec["doc_fetched"] = True
            e["filings"].append(rec); print(f"    {mu} {'ok' if rec['doc_fetched'] else 'MISS'} {desc}")
        e["complete"] = True; man[cn] = e; MANIFEST.write_text(json.dumps(man, indent=2))
    return man


def extract(man):
    rows = []
    for cn, entry in man.items():
        name = entry.get("company_name")
        for f in entry.get("filings", []):
            if not f.get("doc_fetched"):
                continue
            p = ROOT / f["doc_path"]
            if not p.exists():
                continue
            rec = extract_doc(p.read_text("utf-8", "ignore"), f.get("made_up_date"), f.get("description"))
            row = {"CompanyNumber": cn, "CompanyName": name, "made_up_date": f.get("made_up_date"),
                   "filing_description": f.get("description"), "account_type": rec.pop("account_type")}
            row.update(rec); rows.append(row)
    cols = ["CompanyNumber", "CompanyName", "made_up_date", "filing_description", "account_type"] + SCHEMA
    df = pd.DataFrame(rows)[cols].sort_values(["CompanyName", "made_up_date"]) if rows else pd.DataFrame(columns=cols)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}: {len(df)} company-year rows ({df['CompanyNumber'].nunique()} companies)")
    if len(df):
        print(df[["CompanyName", "made_up_date", "account_type", "turnover", "gross_profit",
                  "operating_profit", "profit", "net_assets"]].to_string(index=False))


def main():
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit("set COMPANIES_HOUSE_API_KEY")
    man = pull(key)
    extract(man)


if __name__ == "__main__":
    main()
