"""15 (STEP 4a, stage 1) — Pull real accounts for the education longlist.

INPUT : output/msp_platform_ranked_education.csv  (rows where longlist == True)
ENV   : COMPANIES_HOUSE_API_KEY

For each CompanyNumber:
  - GET /company/{num}/filing-history?category=accounts ; take the latest 3
    accounts filings that have a document; read made_up_date + document_metadata.
  - Fetch each document's iXBRL/xhtml content from the document API.
  - Checkpoint per company (manifest + cached docs on disk); resume on restart.
  - Exponential backoff on 429 (CH limit 600 req / 5 min -> we stay ~2 req/s).

This stage ONLY fetches + caches. Extraction into the strict JSON schema is done
in stage 2 by the agent reading the cached documents. Inputs are never modified.

Run:  COMPANIES_HOUSE_API_KEY=... python3 scripts/15_pull_education_accounts.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
INP = ROOT / "output" / "msp_platform_ranked_education.csv"
DOCDIR = ROOT / "output" / "education_accounts_raw"
MANIFEST = DOCDIR / "_manifest.json"
API = "https://api.company-information.service.gov.uk"
SLEEP = 0.25  # ~4 req/s ceiling -> well under 600 / 5 min


def _auth(key: str) -> str:
    return "Basic " + base64.b64encode(f"{key}:".encode()).decode()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # surface the 302 so we can refetch the S3 URL without auth


_NOREDIR = urllib.request.build_opener(_NoRedirect)


def _get(url: str, key: str, accept: str = "application/json"):
    """GET with CH basic auth, S3-redirect follow, and exponential backoff."""
    delay = 2.0
    for attempt in range(6):
        req = urllib.request.Request(
            url, headers={"Authorization": _auth(key), "Accept": accept})
        try:
            with _NOREDIR.open(req, timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
                # document content -> S3; refetch WITHOUT the CH auth header
                try:
                    with urllib.request.urlopen(e.headers["Location"], timeout=90) as r2:
                        return r2.read().decode("utf-8", "ignore")
                except Exception:
                    return None
            if e.code == 404:
                return None
            if e.code == 429:
                wait = float(e.headers.get("Retry-After", delay)) + 1
                print(f"    429 rate-limited; backoff {wait:.0f}s")
                time.sleep(wait); delay = min(delay * 2, 60); continue
            time.sleep(delay); delay = min(delay * 2, 60)
        except Exception as ex:
            print(f"    transient error: {ex}; backoff {delay:.0f}s")
            time.sleep(delay); delay = min(delay * 2, 60)
    return None


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict):
    MANIFEST.write_text(json.dumps(m, indent=2))


def main():
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit("set COMPANIES_HOUSE_API_KEY")
    DOCDIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INP, dtype={"CompanyNumber": str})
    ll = df[df["longlist"] == True].copy()  # noqa: E712
    print(f"longlist companies: {len(ll)}")

    manifest = load_manifest()

    for _, row in ll.iterrows():
        cn = str(row["CompanyNumber"]).strip()
        name = row["CompanyName"]
        entry = manifest.get(cn, {"company_name": name, "filings": []})
        # resume: skip company if we already have 3 fetched docs (or all available)
        done = [f for f in entry["filings"] if f.get("doc_fetched")]
        if entry.get("complete") and done:
            print(f"  [skip] {name[:34]:34} already complete ({len(done)} docs)")
            continue

        print(f"  {name[:34]:34} ({cn})")
        fh = _get(f"{API}/company/{cn}/filing-history?category=accounts&items_per_page=100", key)
        time.sleep(SLEEP)
        if not fh:
            entry["error"] = "no filing history"; manifest[cn] = entry; save_manifest(manifest)
            print("    ! no filing history"); continue

        items = json.loads(fh).get("items", [])
        # newest-first already; keep only those with a document, take latest 3
        with_doc = [it for it in items if (it.get("links") or {}).get("document_metadata")]
        chosen = with_doc[:3]
        entry["filings"] = []

        cdir = DOCDIR / cn
        cdir.mkdir(exist_ok=True)
        for it in chosen:
            made_up = (it.get("description_values") or {}).get("made_up_date") or it.get("date")
            txn = it.get("transaction_id", "")
            desc = it.get("description", "")
            meta = (it.get("links") or {}).get("document_metadata")
            fname = f"{made_up or txn}.xhtml"
            fpath = cdir / fname
            rec = {"made_up_date": made_up, "transaction_id": txn,
                   "description": desc, "doc_path": str(fpath.relative_to(ROOT)),
                   "doc_fetched": False}
            if fpath.exists() and fpath.stat().st_size > 200:
                rec["doc_fetched"] = True
                rec["bytes"] = fpath.stat().st_size
                entry["filings"].append(rec)
                print(f"    [cached] {made_up}  ({rec['bytes']} B)  {desc}")
                continue
            doc = _get(meta + "/content", key, accept="application/xhtml+xml")
            time.sleep(SLEEP)
            if doc and len(doc) > 200:
                fpath.write_text(doc, "utf-8")
                rec["doc_fetched"] = True
                rec["bytes"] = len(doc.encode("utf-8"))
                print(f"    [fetch ] {made_up}  ({rec['bytes']} B)  {desc}")
            else:
                rec["error"] = "no xhtml content"
                print(f"    [miss  ] {made_up}  (no iXBRL)  {desc}")
            entry["filings"].append(rec)
            manifest[cn] = entry; save_manifest(manifest)  # checkpoint per doc

        entry["complete"] = True
        manifest[cn] = entry
        save_manifest(manifest)

    # summary
    tot = sum(1 for c in manifest.values() for f in c.get("filings", []) if f.get("doc_fetched"))
    print(f"\ncompanies in manifest: {len(manifest)} | fetched docs: {tot}")
    print(f"manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
