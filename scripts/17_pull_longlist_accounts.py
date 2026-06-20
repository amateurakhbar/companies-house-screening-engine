"""17 — Pull CH accounts for ALL longlist targets (education+healthcare+sector union).

Reuses the shared cache output/education_accounts_raw (the 15 education firms are
already there); fetches the remaining ~29. Same auth/redirect/backoff as script 15.
Resumable per company. Run:
  COMPANIES_HOUSE_API_KEY=... python3 scripts/17_pull_longlist_accounts.py
"""
from __future__ import annotations
import base64, json, os, pathlib, time, urllib.error, urllib.request
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
INP = ROOT / "output" / "longlist_union.csv"
DOCDIR = ROOT / "output" / "education_accounts_raw"      # shared cache
MANIFEST = DOCDIR / "_manifest.json"
API = "https://api.company-information.service.gov.uk"
SLEEP = 0.25

def _auth(k): return "Basic " + base64.b64encode(f"{k}:".encode()).decode()
class _NR(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*a): return None
_OP = urllib.request.build_opener(_NR)
def _get(url,key,accept="application/json"):
    delay=2.0
    for _ in range(6):
        req=urllib.request.Request(url,headers={"Authorization":_auth(key),"Accept":accept})
        try:
            with _OP.open(req,timeout=60) as r: return r.read().decode("utf-8","ignore")
        except urllib.error.HTTPError as e:
            if e.code in (301,302,303,307) and e.headers.get("Location"):
                try:
                    with urllib.request.urlopen(e.headers["Location"],timeout=90) as r2:
                        return r2.read().decode("utf-8","ignore")
                except Exception: return None
            if e.code==404: return None
            if e.code==429:
                time.sleep(float(e.headers.get("Retry-After",delay))+1); delay=min(delay*2,60); continue
            time.sleep(delay); delay=min(delay*2,60)
        except Exception:
            time.sleep(delay); delay=min(delay*2,60)
    return None

def main():
    key=os.environ.get("COMPANIES_HOUSE_API_KEY") or _SystemExit()
    DOCDIR.mkdir(parents=True,exist_ok=True)
    man=json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    df=pd.read_csv(INP,dtype={'CompanyNumber':str})
    for _,row in df.iterrows():
        cn=str(row['CompanyNumber']).strip(); name=row['CompanyName']
        e=man.get(cn,{"company_name":name,"filings":[]})
        if e.get("complete") and [f for f in e["filings"] if f.get("doc_fetched")]:
            continue
        print(f"  {name[:36]:36} ({cn})")
        fh=_get(f"{API}/company/{cn}/filing-history?category=accounts&items_per_page=100",key); time.sleep(SLEEP)
        if not fh:
            e["error"]="no filing history"; man[cn]=e; MANIFEST.write_text(json.dumps(man,indent=2)); continue
        items=[it for it in json.loads(fh).get("items",[]) if (it.get("links") or {}).get("document_metadata")][:3]
        e["filings"]=[]; cdir=DOCDIR/cn; cdir.mkdir(exist_ok=True)
        for it in items:
            mu=(it.get("description_values") or {}).get("made_up_date") or it.get("date")
            meta=(it.get("links") or {}).get("document_metadata"); desc=it.get("description","")
            fp=cdir/f"{mu or it.get('transaction_id')}.xhtml"
            rec={"made_up_date":mu,"description":desc,"doc_path":str(fp.relative_to(ROOT)),"doc_fetched":False}
            if fp.exists() and fp.stat().st_size>200:
                rec["doc_fetched"]=True
            else:
                doc=_get(meta+"/content",key,accept="application/xhtml+xml"); time.sleep(SLEEP)
                if doc and len(doc)>200: fp.write_text(doc,"utf-8"); rec["doc_fetched"]=True
            e["filings"].append(rec); print(f"    {mu} {'ok' if rec['doc_fetched'] else 'MISS'} {desc}")
        e["complete"]=True; man[cn]=e; MANIFEST.write_text(json.dumps(man,indent=2))
    tot=sum(1 for c in man.values() for f in c.get("filings",[]) if f.get("doc_fetched"))
    print(f"\nmanifest companies {len(man)} | fetched docs {tot}")

def _SystemExit(): raise SystemExit("set COMPANIES_HOUSE_API_KEY")
if __name__=="__main__": main()
