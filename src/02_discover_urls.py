"""02 — Discover each firm's website by querying CompanyName + town.

Uses the duckduckgo-search package (imported as ddgs) as a free resolver.
For each firm we search "<CompanyName> <town>", then pick the best-matching
domain by scoring candidate result URLs against the company-name tokens.

Caching: a single JSON map at data/cache/urls.json, keyed by CompanyNumber.
Each result is flushed to disk immediately after it is fetched, so a rerun
never re-queries a firm that is already done (idempotent, crash-safe).

Politeness: a fixed 1.5s delay between live requests, plus exponential
backoff when DuckDuckGo returns a rate-limit error.

CLI:
  python3 src/02_discover_urls.py --sample 50   # random sample only
  python3 src/02_discover_urls.py               # full run (all rows)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import time
from urllib.parse import urlparse

import pandas as pd
from ddgs import DDGS

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKING = ROOT / "output" / "working.parquet"
CACHE = ROOT / "data" / "cache" / "urls.json"

REQUEST_DELAY = 0.75         # seconds between live requests
MAX_RETRIES = 5              # backoff attempts on rate-limit
BACKOFF_BASE = 2.0           # seconds; doubled each retry
SCORE_THRESHOLD = 1.8        # minimum relevance score to accept a match

# A Companies-House number in the URL path (8 digits) marks a directory page.
_CH_NUMBER_PATH = re.compile(r"/\d{8}/")

# Domains that are never a company's own site.
BLOCKLIST = {
    # UK company directory / aggregator domains (explicit, per request).
    "efinding.co.uk", "efinding.uk", "efinder.uk", "find-open.co.uk",
    "companiesintheuk.co.uk", "datalog.co.uk", "laei.uk", "opengovuk.com",
    "companieshouse.gov.uk", "opencorporates.com", "endole.co.uk",
    "bizdb.co.uk", "companycheck.co.uk", "duedil.com", "linkedin.com",
    "facebook.com", "yell.com", "cylex.co.uk", "thomsonlocal.com",
    "yelp.co.uk", "chamberofcommerce.uk", "iptoolskit.com",
    # Directory tail surfaced by the frequency filter (incl. subdomained cylex).
    "zoominfo.com", "jars.lt", "prospeo.io", "okredo.co.uk", "rocketreach.co",
    "northdata.de", "lei-ireland.ie", "cylex-uk.co.uk",
    # Previously-known noise, retained.
    "find-and-update.company-information.service.gov.uk", "gov.uk",
    "twitter.com", "x.com", "instagram.com", "youtube.com", "wikipedia.org",
    "company-check.co.uk", "companieslist.co.uk",
    "company-information.service.gov.uk", "dnb.com", "bloomberg.com",
    "192.com", "tussell.com", "globaldatabase.com", "ukbusinessdirectory.co.uk",
    "thegazette.co.uk", "creditsafe.com", "kompass.com", "trustpilot.com",
    "indeed.com", "glassdoor.com", "crunchbase.com", "amazon.com",
    "ised-isde.canada.ca", "find-and-update.company-information.gov.uk",
    "companydatashop.com",
}

_STOPWORDS = {
    "ltd", "limited", "llp", "plc", "uk", "the", "and", "co", "company",
    "group", "holdings", "services", "solutions", "systems", "technologies",
    "technology", "consulting", "consultants", "international",
}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(name: str) -> list[str]:
    toks = _TOKEN_RE.findall(name.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


def _registrable_domain(netloc: str) -> str:
    host = netloc.lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    return host


def _blocked(domain: str) -> bool:
    return any(domain == b or domain.endswith("." + b) for b in BLOCKLIST)


def _score(name_tokens: list[str], url: str, title: str) -> float:
    parsed = urlparse(url)
    domain = _registrable_domain(parsed.netloc)
    if not domain or _blocked(domain):
        return -1.0
    # Reject directory pages that embed a CH company number in the path.
    if _CH_NUMBER_PATH.search(parsed.path or ""):
        return -1.0
    domain_core = domain.split(".")[0]
    score = 0.0
    for t in name_tokens:
        if t in domain_core:
            score += 2.0
        elif t in title.lower():
            score += 0.5
    if domain.endswith(".uk"):
        score += 0.3
    return score


def _is_ratelimit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in ("ratelimit", "rate limit", "429", "202", "too many"))


def _search_with_backoff(query: str, max_results: int = 8) -> list[dict]:
    delay = BACKOFF_BASE
    for attempt in range(MAX_RETRIES):
        try:
            return DDGS().text(query, max_results=max_results) or []
        except Exception as e:
            if _is_ratelimit(e) and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []


def discover_one(name: str, town: str) -> dict:
    query = f"{name} {town}".strip()
    name_tokens = _tokens(name)
    best = {"domain": None, "url": None, "title": None, "score": -1.0}
    try:
        results = _search_with_backoff(query)
    except Exception as e:
        return {"query": query, "match": None, "error": str(e)}

    for r in results:
        url = r.get("href") or r.get("url") or ""
        title = r.get("title") or ""
        if not url:
            continue
        s = _score(name_tokens, url, title)
        if s > best["score"]:
            best = {
                "domain": _registrable_domain(urlparse(url).netloc),
                "url": url,
                "title": title,
                "score": s,
            }
    match = best if best["score"] >= SCORE_THRESHOLD else None
    return {"query": query, "match": match}


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _flush_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(CACHE)  # atomic, crash-safe


def run(sample: int | None, seed: int = 42, batch_size: int = 500,
        batch_pause: float = 2.0) -> None:
    df = pd.read_parquet(WORKING)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)

    cache = _load_cache()
    total = len(df)
    hits = 0
    processed = 0
    queried = 0  # live requests made this run (excludes cache hits)

    rows = list(df.iterrows())
    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        for _, row in batch:
            cnum = str(row["CompanyNumber"])
            name = str(row["CompanyName"])
            town = str(row.get("town") or "")

            if cnum in cache:
                rec = cache[cnum]
            else:
                rec = discover_one(name, town)
                rec["CompanyName"] = name
                cache[cnum] = rec
                queried += 1
                time.sleep(REQUEST_DELAY)    # politeness gap per live request

            processed += 1
            if rec.get("match"):
                hits += 1

        # Persist after every batch so the run is resumable if it drops.
        _flush_cache(cache)
        rate = hits / processed if processed else 0.0
        print(
            f"progress: {processed}/{total}  hits={hits} ({rate:.0%})  "
            f"queried_this_run={queried}  cache_size={len(cache)}",
            flush=True,
        )
        if start + batch_size < total:
            time.sleep(batch_pause)

    final = hits / processed if processed else 0.0
    print(f"DONE  processed={processed}  hits={hits}  hit_rate={final:.0%}  "
          f"queried_this_run={queried}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="random sample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--batch-pause", type=float, default=2.0)
    args = ap.parse_args()
    run(sample=args.sample, seed=args.seed,
        batch_size=args.batch_size, batch_pause=args.batch_pause)
