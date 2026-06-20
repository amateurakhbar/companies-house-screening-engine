"""16 — Serper pass over firms stage-10 (ddgs) left as low_quality / no-hit.

Same short-query strategy and strict accept rule as scripts/10_recover_websites.py
(first 1-2 distinctive name tokens + town, keep the top non-blocked organic hit,
flag verified vs low_quality), but uses serper.dev (Google) which returns far
better top hits than DuckDuckGo. Reuses stage-10's helpers directly.

Targets the firms that still have NO verified website: cache match missing, or
present but low_quality. Idempotent re-write -- a serper verified hit replaces a
stale ddgs low_quality entry; otherwise the (possibly improved) low_quality top
hit is stored.

  SERPER_API_KEY=$(grep -v '^#' secrets/serper_paid_key.txt|grep .|head -1) \
      python3 scripts/16_recover_websites_serper.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import time
from urllib.parse import urlparse

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache" / "urls.json"

# Reuse stage-10 short-query helpers + the ddgs blocklist/CH-path regex.
_spec = importlib.util.spec_from_file_location("st10", ROOT / "scripts" / "10_recover_websites.py")
st10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st10)
disc = st10.disc

ENDPOINT = "https://google.serper.dev/search"


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
    q = f"{' '.join(st10.raw_tokens(name)[:2])} {town}".strip()
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
                "method": "serper_short_query",
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

    # Targets: financials + no *verified* website yet (low_quality or no-hit).
    targets = []
    for cnum, name, town in st10.load_targets():
        key8 = cnum.zfill(8)
        rec = cache.get(key8) or cache.get(cnum)
        m = (rec or {}).get("match") or {}
        if not m.get("verified"):
            targets.append((key8, name, town))

    print(f"targets needing serper: {len(targets)}", flush=True)
    verified = low = errors = nohit = 0
    for i, (key8, name, town) in enumerate(targets, 1):
        rec = discover(name, town, key)
        rec["CompanyName"] = name
        cache[key8] = rec
        m = rec.get("match")
        if rec.get("error"):
            errors += 1
        elif m and m["verified"]:
            verified += 1
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


if __name__ == "__main__":
    main()
