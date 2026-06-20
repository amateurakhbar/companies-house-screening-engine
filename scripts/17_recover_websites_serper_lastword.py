"""17 — Second Serper pass over the 200 firms still lacking a verified website.

Different query rule (per request): take the company name and drop ONLY the last
word, then append the town. Example:
    "IT SUPPORT DESK LTD"  +  Weybridge  ->  "IT SUPPORT DESK Weybridge"

Stores the top non-blocked organic hit, flagged verified (strict rule) vs
low_quality, into data/cache/urls.json with method=serper_lastword. Targets only
firms that still have NO verified website. Idempotent re-write.

  SERPER_API_KEY=$(grep -v '^#' secrets/serper_paid_key.txt|grep .|head -1) \
      python3 scripts/17_recover_websites_serper_lastword.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import time
from urllib.parse import urlparse

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "urls.json"

_spec = importlib.util.spec_from_file_location("st10", ROOT / "scripts" / "10_recover_websites.py")
st10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st10)
disc = st10.disc

ENDPOINT = "https://google.serper.dev/search"


def drop_last_word(name: str) -> str:
    parts = name.split()
    return " ".join(parts[:-1]) if len(parts) > 1 else name


def serper_search(query: str, key: str) -> list[dict]:
    resp = requests.post(
        ENDPOINT,
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "gl": "uk"},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("organic", []) or []


def discover(name: str, town: str, key: str) -> dict:
    q = f"{drop_last_word(name)} {town}".strip()
    try:
        results = serper_search(q, key)
    except Exception as e:
        return {"query": q, "match": None, "error": str(e)}
    for r in results:
        url = r.get("link") or ""
        if not url:
            continue
        parsed = urlparse(url)
        if disc._CH_NUMBER_PATH.search(parsed.path or ""):
            continue
        dom = disc._registrable_domain(parsed.netloc)
        if not dom or disc._blocked(dom):
            continue
        verified = st10.rule_match(name, dom)
        return {
            "query": q,
            "match": {
                "domain": dom,
                "url": url,
                "title": r.get("title") or "",
                "method": "serper_lastword",
                "verified": verified,
                "low_quality": not verified,
            },
        }
    return {"query": q, "match": None}


def main() -> None:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise SystemExit("SERPER_API_KEY not set")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    targets = []
    for cnum, name, town in st10.load_targets():
        key8 = cnum.zfill(8)
        rec = cache.get(key8) or cache.get(cnum)
        m = (rec or {}).get("match") or {}
        if not m.get("verified"):
            targets.append((key8, name, town))

    print(f"targets: {len(targets)}", flush=True)
    verified = low = errors = nohit = 0
    newly = []
    for i, (key8, name, town) in enumerate(targets, 1):
        rec = discover(name, town, key)
        rec["CompanyName"] = name
        cache[key8] = rec
        m = rec.get("match")
        if rec.get("error"):
            errors += 1
        elif m and m["verified"]:
            verified += 1
            newly.append((name, m["url"]))
        elif m:
            low += 1
        else:
            nohit += 1
        time.sleep(0.3)
        if i % 25 == 0 or i == len(targets):
            tmp = CACHE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=2))
            tmp.replace(CACHE)
            print(f"{i}/{len(targets)}  verified={verified} low={low} "
                  f"err={errors} nohit={nohit}", flush=True)

    print(f"DONE verified={verified} low={low} err={errors} nohit={nohit}", flush=True)
    print("--- newly verified ---", flush=True)
    for n, u in newly:
        print(f"  {n[:40]:40} {u}", flush=True)


if __name__ == "__main__":
    main()
