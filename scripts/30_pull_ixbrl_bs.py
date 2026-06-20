#!/usr/bin/env python3
"""Pull two-year balance-sheet facts from Companies House iXBRL for the education
long-list, for a balance-sheet-quality score. No API key needed (public site).

Reads  /tmp/edu_numbers.csv  (CompanyNumber, CompanyName)
Writes /tmp/edu_ixbrl_bs.csv
"""
import requests, re, csv, time, sys

HDR = {"User-Agent": "Mozilla/5.0"}
BASE = "https://find-and-update.company-information.service.gov.uk"

# concept keyword -> regex on the iXBRL fact name local part
CONCEPTS = {
    "net_assets":  r"NetAssetsLiabilities(?:IncludingPensionAssetLiability)?",
    "cash":        r"CashBankOnHand|CashBankInHand",
    "debtors":     r"Debtors",
    "creditors":   r"Creditors",
    "fixed_tang":  r"PropertyPlantEquipment|TangibleFixedAssets",
    "net_curr":    r"NetCurrentAssetsLiabilities",
    "turnover":    r"TurnoverRevenue|^Turnover|Revenue",
    "employees":   r"AverageNumberEmployees(?:DuringPeriod)?",
}

def num(s):
    s = s.replace(",", "").strip()
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v

def facts(txt, pattern):
    # iXBRL tagged numbers: <ix:nonFraction name="...LocalName" ...>1,234</ix:nonFraction>
    out = []
    for m in re.finditer(r'name="[^"]*?(' + pattern + r')"[^>]*?>([\-\d,\.\(\)\s]+?)<', txt):
        v = num(m.group(2))
        if v is not None:
            out.append(v)
    return out

def latest_accounts_docid(numid):
    fh = requests.get(f"{BASE}/company/{numid}/filing-history?category=accounts",
                      headers=HDR, timeout=30).text
    for mt in re.finditer(r'/company/' + numid + r'/filing-history/([A-Za-z0-9_-]+)/document', fh):
        w = re.sub(r'<[^>]+>', ' ', fh[max(0, mt.start()-600):mt.start()+200])
        if re.search(r'accounts', w, re.I):
            # also grab made-up date if present
            d = re.search(r'(\d{1,2}\s+\w+\s+20\d\d)', w)
            return mt.group(1), (d.group(1) if d else "")
    return None, ""

rows = list(csv.DictReader(open("/tmp/edu_numbers.csv")))
fields = ["CompanyNumber","CompanyName","made_up","ixbrl_ok",
          "net_assets_cur","net_assets_pri","cash_cur","cash_pri",
          "debtors_cur","creditors_all","fixed_tang_cur","net_curr_cur",
          "turnover_cur","employees_cur"]
out = open("/tmp/edu_ixbrl_bs.csv","w",newline="")
w = csv.DictWriter(out, fieldnames=fields); w.writeheader()

for i, r in enumerate(rows, 1):
    numid = str(r["CompanyNumber"]).zfill(8); name = r["CompanyName"]
    rec = {"CompanyNumber":numid,"CompanyName":name,"ixbrl_ok":0}
    try:
        docid, mu = latest_accounts_docid(numid)
        rec["made_up"] = mu
        if docid:
            doc = requests.get(f"{BASE}/company/{numid}/filing-history/{docid}/document?format=xhtml",
                               headers={**HDR,"Accept":"application/xhtml+xml"}, timeout=60)
            if doc.status_code == 200 and b"xbrl" in doc.content.lower():
                t = doc.text; rec["ixbrl_ok"] = 1
                na = facts(t, CONCEPTS["net_assets"])
                ca = facts(t, CONCEPTS["cash"])
                rec["net_assets_cur"] = na[0] if len(na)>0 else ""
                rec["net_assets_pri"] = na[1] if len(na)>1 else ""
                rec["cash_cur"] = ca[0] if len(ca)>0 else ""
                rec["cash_pri"] = ca[1] if len(ca)>1 else ""
                db = facts(t, CONCEPTS["debtors"]); rec["debtors_cur"] = db[0] if db else ""
                cr = facts(t, CONCEPTS["creditors"]); rec["creditors_all"] = "|".join(str(int(x)) for x in cr[:6])
                ft = facts(t, CONCEPTS["fixed_tang"]); rec["fixed_tang_cur"] = ft[0] if ft else ""
                nc = facts(t, CONCEPTS["net_curr"]); rec["net_curr_cur"] = nc[0] if nc else ""
                tv = facts(t, CONCEPTS["turnover"]); rec["turnover_cur"] = tv[0] if tv else ""
                em = facts(t, CONCEPTS["employees"]); rec["employees_cur"] = em[0] if em else ""
    except Exception as e:
        rec["made_up"] = f"ERR:{type(e).__name__}"
    w.writerow(rec); out.flush()
    print(f"[{i:>2}/{len(rows)}] {numid} ok={rec['ixbrl_ok']} NA={rec.get('net_assets_cur','')} {name[:32]}", flush=True)
    time.sleep(0.6)

out.close()
print("DONE")
