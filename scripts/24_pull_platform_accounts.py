"""24 — Pull + parse real CH filed financials for every firm in a platform shortlist.

Generalises script 23 (which did the 4 healthcare filers) to an arbitrary input
CSV with CompanyNumber/CompanyName columns. Default input is the healthcare
platform shortlist, whose `turnover` column is a placeholder (80561 for 45/47);
this replaces it with actual filed turnover / P&L from Companies House iXBRL.

  pull   : CH filing-history (category=accounts) + iXBRL docs -> <cache>/<cn>/
  extract: reuse script 16's iXBRL parser                     -> <out>.csv

Resumable per company (manifest). Reuses _get/fetch from script 23 and the
parser from script 16, so behaviour is identical to the validated 4-firm run.

Run (defaults to healthcare platform file):
  COMPANIES_HOUSE_API_KEY=... python3 scripts/24_pull_platform_accounts.py
Or point at another shortlist:
  COMPANIES_HOUSE_API_KEY=... python3 scripts/24_pull_platform_accounts.py \
      output/msp_platform_ranked_education.csv education_platform
"""
from __future__ import annotations
import importlib.util, json, pathlib, sys, time
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

# reuse the validated fetch (script 23) and iXBRL parser (script 16)
def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
_s23 = _load("s23", "23_pull_healthcare_filers.py")
_get, API, SLEEP = _s23._get, _s23.API, _s23.SLEEP
extract_doc, SCHEMA = _s23.extract_doc, _s23.SCHEMA


def pull(firms, key, docdir, manifest_path):
    docdir.mkdir(parents=True, exist_ok=True)
    man = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for i, (cn, name) in enumerate(firms, 1):
        e = man.get(cn, {"company_name": name, "filings": []})
        if e.get("complete") and [f for f in e["filings"] if f.get("doc_fetched")]:
            continue
        print(f"  [{i}/{len(firms)}] {name[:34]:34} ({cn})")
        fh = _get(f"{API}/company/{cn}/filing-history?category=accounts&items_per_page=100", key); time.sleep(SLEEP)
        if not fh:
            e["error"] = "no filing history"; man[cn] = e; manifest_path.write_text(json.dumps(man, indent=2)); continue
        items = [it for it in json.loads(fh).get("items", []) if (it.get("links") or {}).get("document_metadata")][:3]
        e["filings"] = []; e.pop("error", None); cdir = docdir / cn; cdir.mkdir(exist_ok=True)
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
            e["filings"].append(rec)
        e["complete"] = True; man[cn] = e; manifest_path.write_text(json.dumps(man, indent=2))
    return man


def extract(man, out):
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
    df.to_csv(out, index=False)
    n_co = df["CompanyNumber"].nunique() if len(df) else 0
    print(f"\nwrote {out.relative_to(ROOT)}: {len(df)} company-year rows ({n_co} companies)")
    if len(df):
        latest = df.sort_values("made_up_date").groupby("CompanyNumber").tail(1)
        N = len(latest)
        print(f"\nCoverage on latest filed year ({N} companies):")
        for fld in ["turnover", "gross_profit", "operating_profit", "profit", "net_assets"]:
            print(f"  {fld:18} {latest[fld].notna().sum()}/{N}")
        print("\nAccount-type mix (all rows):"); print(df["account_type"].value_counts().to_string())
    return df


def main():
    import os
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit("set COMPANIES_HOUSE_API_KEY")
    inp = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output" / "msp_platform_ranked_healthcare.csv"
    tag = sys.argv[2] if len(sys.argv) > 2 else "healthcare_platform"
    docdir = ROOT / "output" / f"{tag}_accounts_raw"
    out = ROOT / "output" / f"{tag}_finps.csv"
    df = pd.read_csv(inp, dtype={"CompanyNumber": str})
    firms = [(str(r.CompanyNumber).strip(), str(r.CompanyName)) for r in df.itertuples()]
    print(f"input {inp.name}: {len(firms)} firms -> cache {docdir.name}, out {out.name}\n")
    man = pull(firms, key, docdir, docdir / "_manifest.json")
    extract(man, out)


if __name__ == "__main__":
    main()
