"""10 — Recover websites for firms that have financials but no resolved site.

Strategy (validated on samples): search the first 1-2 distinctive name tokens
plus the town ("SMARTFITS Burton-On-Trent" instead of the full legal name),
then STORE THE TOP non-blocked organic hit. Every stored hit is flagged:

  verified = True   strict rule matched (two-token prefix of the domain core,
                    or a distinctive token that equals the whole domain core).
                    High precision -- treat as the firm's website.
  verified = False  top hit kept but rule did not match ("low quality").
                    These are the batch to re-run on Serper/Google later.

Writes into the SAME data/cache/urls.json that src/01_load_filter.py reads
(match.url -> website column), keyed by zero-padded CompanyNumber. Idempotent:
a firm with an existing match is skipped, so reruns only retry the misses/errors.

  python3 scripts/10_recover_websites.py            # full target set
  python3 scripts/10_recover_websites.py --limit 50 # first N (smoke test)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pathlib
import re
import time
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENRICHED = ROOT / "output" / "msp_all_companies_enriched.csv"
CACHE = ROOT / "data" / "cache" / "urls.json"

spec = importlib.util.spec_from_file_location("disc", ROOT / "src" / "02_discover_urls.py")
disc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(disc)

LEGAL = {"ltd", "limited", "llp", "plc", "uk", "co", "company", "the", "and", "t", "a"}
GENERIC = {
    "it", "ict", "tech", "technology", "technologies", "computer", "computers",
    "computing", "solutions", "solution", "services", "service", "systems",
    "system", "network", "networks", "digital", "communications", "comms",
    "telecoms", "telecom", "support", "cloud", "video", "power", "management",
    "consulting", "consultants", "one", "group", "holdings", "international",
    "data", "media", "security", "software", "web", "online", "global",
    "installations", "maintenance", "supplies",
}
TOKEN = re.compile(r"[a-z0-9]+")
REQUEST_DELAY = 0.75


def raw_tokens(name: str) -> list[str]:
    return [t for t in TOKEN.findall(name.lower()) if len(t) > 1 and t not in LEGAL]


def norm_core(domain: str) -> str:
    return re.sub(r"[^a-z0-9]", "", domain.split(".")[0])


def rule_match(name: str, domain: str) -> bool:
    """Strict, high-precision acceptance test."""
    core = norm_core(domain)
    raw = raw_tokens(name)
    if not raw or not core:
        return False
    if len(raw) >= 2 and core.startswith(raw[0] + raw[1]):
        return True
    distinctive = [t for t in raw if t not in GENERIC]
    return any(core == t for t in distinctive)


def discover(name: str, town: str) -> dict | None:
    """Return {match: {...}} or None on error/no usable hit."""
    q = f"{' '.join(raw_tokens(name)[:2])} {town}".strip()
    try:
        results = disc._search_with_backoff(q, max_results=6)
    except Exception as e:
        return {"query": q, "match": None, "error": str(e)}
    for r in results:
        url = r.get("href") or r.get("url") or ""
        if not url:
            continue
        parsed = urlparse(url)
        if disc._CH_NUMBER_PATH.search(parsed.path or ""):
            continue
        dom = disc._registrable_domain(parsed.netloc)
        if not dom or disc._blocked(dom):
            continue
        verified = rule_match(name, dom)
        return {
            "query": q,
            "match": {
                "domain": dom,
                "url": url,
                "title": r.get("title") or "",
                "method": "ddg_short_query",
                "verified": verified,
                "low_quality": not verified,
            },
        }
    return {"query": q, "match": None}  # no non-blocked hit


def load_targets() -> list[tuple[str, str, str]]:
    out = []
    for r in csv.DictReader(open(ENRICHED)):
        web = (r.get("website") or "").strip()
        fin = (r.get("fin_source") or "").strip()
        has_fin = fin != "" or any(
            (r.get(c) or "").strip() not in ("", "0")
            for c in ["turnover", "net_assets", "equity", "profit"]
        )
        if web == "" and has_fin:
            town = r.get("town") or r.get("RegAddress.PostTown") or ""
            out.append((str(r["CompanyNumber"]), r["CompanyName"], town))
    return out


def main(limit: int | None) -> None:
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    targets = load_targets()
    if limit:
        targets = targets[:limit]

    verified = low = errors = nohit = skipped = 0
    for i, (cnum, name, town) in enumerate(targets, 1):
        key = cnum.zfill(8)
        existing = cache.get(key) or cache.get(cnum)
        if existing and existing.get("match"):
            skipped += 1
            continue

        rec = discover(name, town)
        rec["CompanyName"] = name
        cache[key] = rec
        time.sleep(REQUEST_DELAY)

        m = rec.get("match")
        if rec.get("error"):
            errors += 1
        elif m and m["verified"]:
            verified += 1
        elif m:
            low += 1
        else:
            nohit += 1

        if i % 25 == 0 or i == len(targets):
            tmp = CACHE.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(cache, indent=2))
            tmp.replace(CACHE)
            print(
                f"{i}/{len(targets)}  verified={verified} low_quality={low} "
                f"errors={errors} no_hit={nohit} skipped={skipped}",
                flush=True,
            )

    print(
        f"DONE  verified={verified} low_quality={low} errors={errors} "
        f"no_hit={nohit} skipped={skipped}",
        flush=True,
    )
    print("Re-run to retry the errors (idempotent: stored matches are skipped).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
