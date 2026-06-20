"""02 (Serper backend) — resolve company websites via Serper.dev google search.

A budget-limited, higher-quality alternative to the free ddgs backend. Reuses
the exact scoring / blocklist / threshold logic from src/02_discover_urls.py so
results are directly comparable and write into the SAME data/cache/urls.json.

Targets unresolved firms first (those whose ddgs attempt errored or returned no
match), then any not yet attempted. Hard-capped at MAX_CALLS so it can never
overspend the API budget. Each accepted result is tagged source='serper'.

Key is read from the SERPER_API_KEY env var — never hard-coded.

  SERPER_API_KEY=... python3 src/02_serper.py --max-calls 449
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import time
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKING = ROOT / "output" / "working.parquet"
CACHE = ROOT / "data" / "cache" / "urls.json"

# Two supported backends, selected by SEARCH_PROVIDER env var.
#   serper  : serper.dev   (POST, JSON body, results under "organic")
#   serperx : rsecloud      (GET,  query params, results under "organicResults")
PROVIDER = os.environ.get("SEARCH_PROVIDER", "serperx").lower()
_BACKENDS = {
    "serper": {
        "endpoint": "https://google.serper.dev/search",
        "method": "POST",
        "results_key": "organic",
    },
    "serperx": {
        "endpoint": "https://serperx.rsecloud.com/api/search",
        "method": "GET",
        "results_key": "organicResults",
    },
}

# Reuse scoring + blocklist from the ddgs module (filename starts with a digit).
_spec = importlib.util.spec_from_file_location(
    "discover", ROOT / "src" / "02_discover_urls.py"
)
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)


def _load_cache() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _flush(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, indent=2))
    tmp.replace(CACHE)


def _serper_search(query: str, key: str) -> list[dict]:
    cfg = _BACKENDS[PROVIDER]
    if cfg["method"] == "POST":
        resp = requests.post(
            cfg["endpoint"],
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "gl": "uk"},
            timeout=20,
        )
    else:  # GET (serperx)
        resp = requests.get(
            cfg["endpoint"],
            headers={"X-API-KEY": key},
            params={"q": query, "page": 1},
            timeout=20,
        )
    resp.raise_for_status()
    return resp.json().get(cfg["results_key"], []) or []


def discover_one(name: str, town: str, key: str) -> dict:
    query = f"{name} {town}".strip()
    name_tokens = _d._tokens(name)
    best = {"domain": None, "url": None, "title": None, "score": -1.0}
    try:
        results = _serper_search(query, key)
    except Exception as e:
        return {"query": query, "match": None, "error": str(e), "backend": PROVIDER}

    for r in results:
        url = r.get("link") or ""
        title = r.get("title") or ""
        if not url:
            continue
        s = _d._score(name_tokens, url, title)
        if s > best["score"]:
            best = {
                "domain": _d._registrable_domain(urlparse(url).netloc),
                "url": url,
                "title": title,
                "score": s,
            }
    match = best if best["score"] >= _d.SCORE_THRESHOLD else None
    return {"query": query, "match": match, "backend": PROVIDER}


def _needs_resolution(rec: dict | None) -> bool:
    """A firm is worth a Serper call if not yet cached, or errored, or no match."""
    if rec is None:
        return True
    if rec.get("error"):
        return True
    if not rec.get("match"):
        return True
    return False


def run(max_calls: int, flush_every: int = 25) -> None:
    key = os.environ.get("SERPER_API_KEY")
    if not key:
        raise SystemExit("SERPER_API_KEY env var not set")

    df = pd.read_parquet(WORKING)
    cache = _load_cache()

    # Priority 1: firms that errored (e.g. ddgs 0x304). Priority 2: no-match.
    # Priority 3: never attempted.
    errored, nomatch, fresh = [], [], []
    for _, row in df.iterrows():
        cnum = str(row["CompanyNumber"])
        rec = cache.get(cnum)
        if rec is None:
            fresh.append(row)
        elif rec.get("error"):
            errored.append(row)
        elif not rec.get("match"):
            nomatch.append(row)
    queue = errored + nomatch + fresh

    calls = 0
    hits = 0
    for row in queue:
        if calls >= max_calls:
            break
        cnum = str(row["CompanyNumber"])
        name = str(row["CompanyName"])
        town = str(row.get("town") or "")

        rec = discover_one(name, town, key)
        rec["CompanyName"] = name
        cache[cnum] = rec
        calls += 1
        if rec.get("match"):
            hits += 1
        if calls % flush_every == 0:
            _flush(cache)
            print(f"calls={calls}/{max_calls}  hits={hits} ({hits/calls:.0%})", flush=True)
        time.sleep(0.3)  # Serper is fast; light spacing only

    _flush(cache)
    rate = hits / calls if calls else 0.0
    print(
        f"DONE  serper_calls={calls}  hits={hits}  hit_rate={rate:.0%}  "
        f"(queue: errored={len(errored)} nomatch={len(nomatch)} fresh={len(fresh)})",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-calls", type=int, default=449,
                    help="hard cap on Serper API calls (budget guard)")
    args = ap.parse_args()
    run(max_calls=args.max_calls)
