"""02 (parallel) — resolve company websites with multiple API keys at once.

serper.dev rate-limits per key, so N keys run concurrently for ~N x throughput.
Reads keys from a gitignored file (secrets/serper_keys.txt), shards the queue of
unresolved firms across them, and runs one worker thread per key. Each worker is
hard-capped (default 2500 = the per-key credit limit) so it can't overspend.

Optionally a SerperX key (secrets/serperx_key.txt) runs as one more worker on the
remaining 290 credits, via the serperx backend.

Reuses scoring/blocklist/threshold and the per-provider search from 02_serper.py.
All workers write into the SAME data/cache/urls.json under a lock; the cache is
flushed periodically so the run stays resumable.

  python3 src/02_parallel.py --per-key 2500 --serperx-calls 290
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import threading
import time
from urllib.parse import urlparse

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKING = ROOT / "output" / "working.parquet"
CACHE = ROOT / "data" / "cache" / "urls.json"
KEYS_FILE = ROOT / "secrets" / "serper_keys.txt"
SERPERX_KEY_FILE = ROOT / "secrets" / "serperx_key.txt"
PAID_KEY_FILE = ROOT / "secrets" / "serper_paid_key.txt"

# Reuse scoring/blocklist from the ddgs module.
_spec = importlib.util.spec_from_file_location("discover", ROOT / "src" / "02_discover_urls.py")
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

_BACKENDS = {
    "serper": {"endpoint": "https://google.serper.dev/search", "method": "POST", "results_key": "organic"},
    "serperx": {"endpoint": "https://serperx.rsecloud.com/api/search", "method": "GET", "results_key": "organicResults"},
}

_lock = threading.Lock()
_PLACEHOLDERS = {"KEY_1_HERE", "KEY_2_HERE", "KEY_3_HERE", "KEY_4_HERE", "KEY_5_HERE",
                 "PAID_KEY_HERE", ""}


def _read_keys(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    keys = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("#") or s in _PLACEHOLDERS:
            continue
        keys.append(s)
    return keys


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


def _search(provider: str, query: str, key: str, retries: int = 4) -> list[dict]:
    cfg = _BACKENDS[provider]
    delay = 1.0
    for attempt in range(retries + 1):
        if cfg["method"] == "POST":
            resp = requests.post(cfg["endpoint"], headers={"X-API-KEY": key, "Content-Type": "application/json"},
                                 json={"q": query, "gl": "uk"}, timeout=20)
        else:
            resp = requests.get(cfg["endpoint"], headers={"X-API-KEY": key},
                                params={"q": query, "page": 1}, timeout=20)
        # Retry on rate-limit / transient server errors so the run never stops.
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json().get(cfg["results_key"], []) or []
    return []


AUTO_BLOCK_FILE = ROOT / "data" / "cache" / "auto_blocklist.json"


def _load_auto_block() -> set:
    if AUTO_BLOCK_FILE.exists():
        try:
            return set(json.loads(AUTO_BLOCK_FILE.read_text()))
        except json.JSONDecodeError:
            return set()
    return set()


# Data-driven directory blocklist (domains resolved to many distinct companies),
# unioned with the hand-maintained BLOCKLIST in the ddgs module.
AUTO_BLOCK = _load_auto_block()


def _is_directory(url: str, domain: str) -> bool:
    p = urlparse(url)
    return (not domain) or _d._blocked(domain) or (domain in AUTO_BLOCK) \
        or bool(_d._CH_NUMBER_PATH.search(p.path or ""))


def discover_one(provider: str, name: str, town: str, key: str) -> dict:
    """Resolve a website with two precision-tiered outcomes:

      1. scorer-verified : strict scorer >= threshold (high-precision auto-accept).
      2. tentative        : no verified hit, but a non-directory candidate exists.
                           Stored with verified=False for Stage 4 to confirm by
                           checking name tokens on the fetched page.

    Directory results (hand blocklist + data-driven AUTO_BLOCK + CH-number paths)
    are excluded from the tentative fallback. Scorer thresholds are NOT changed.
    """
    query = f"{name} {town}".strip()
    name_tokens = _d._tokens(name)
    best = {"domain": None, "url": None, "title": None, "score": -1.0}
    tentative = {"domain": None, "url": None, "title": None, "score": 0.0}
    try:
        results = _search(provider, query, key)
    except Exception as e:
        return {"query": query, "match": None, "error": str(e), "backend": provider}

    for r in results:
        url = r.get("link") or ""
        title = r.get("title") or ""
        if not url:
            continue
        domain = _d._registrable_domain(urlparse(url).netloc)
        s = _d._score(name_tokens, url, title)
        if s > best["score"]:
            best = {"domain": domain, "url": url, "title": title, "score": s}
        if _is_directory(url, domain):
            continue
        # Tentative = HIGHEST-scoring non-directory candidate, gated to score > 0
        # (must have some name-token overlap). Zero-signal results are noise.
        if s > tentative["score"]:
            tentative = {"domain": domain, "url": url, "title": title, "score": s}

    if best["score"] >= _d.SCORE_THRESHOLD and not _is_directory(best["url"] or "", best["domain"] or ""):
        m = dict(best); m["verified"] = True; m["via"] = "scorer"
    elif tentative["score"] > 0 and tentative["url"]:
        m = dict(tentative); m["verified"] = False; m["via"] = "tentative"
    else:
        m = None
    return {"query": query, "match": m, "backend": provider,
            "verified": (m["verified"] if m else None)}


# Shared progress counters.
_progress = {"calls": 0, "hits": 0, "verified": 0, "unverified": 0}


def _worker(wid: int, provider: str, key: str, shard: list, cache: dict,
            max_calls: int, flush_every: int, sleep: float,
            report_every: int = 2000) -> None:
    local = 0
    for row in shard:
        if local >= max_calls:
            break
        cnum = str(row["CompanyNumber"])
        name = str(row["CompanyName"])
        town = str(row.get("town") or "")
        rec = discover_one(provider, name, town, key)
        rec["CompanyName"] = name
        with _lock:
            cache[cnum] = rec
            _progress["calls"] += 1
            if rec.get("match"):
                _progress["hits"] += 1
                if rec["match"].get("verified"):
                    _progress["verified"] += 1
                else:
                    _progress["unverified"] += 1
            c = _progress["calls"]
            if c % flush_every == 0:
                _flush(cache)
            if c % report_every == 0:
                h, v, u = _progress["hits"], _progress["verified"], _progress["unverified"]
                print(f"calls={c}  hits={h} ({h/c:.0%})  verified={v} ({v/c:.0%})  "
                      f"tentative={u} ({u/c:.0%})", flush=True)
        local += 1
        time.sleep(sleep)


def run(per_key: int, serperx_calls: int, paid_calls: int, paid_threads: int,
        paid_only: bool = False, flush_every: int = 50, sleep: float = 0.3) -> None:
    keys = [] if paid_only else _read_keys(KEYS_FILE)
    serperx_keys = [] if paid_only else _read_keys(SERPERX_KEY_FILE)
    paid_keys = _read_keys(PAID_KEY_FILE)
    if not keys and not serperx_keys and not paid_keys:
        raise SystemExit(f"no keys found in {KEYS_FILE} / {SERPERX_KEY_FILE} / {PAID_KEY_FILE}")

    df = pd.read_parquet(WORKING)
    cache = _load_cache()

    # Queue = firms without a resolved match yet: unattempted, purged, no-match,
    # or errored. Firms that already hold a match are skipped.
    queue = []
    for _, row in df.iterrows():
        rec = cache.get(str(row["CompanyNumber"]))
        if rec is None or not rec.get("match"):
            queue.append(row)
    print(f"queue (unresolved): {len(queue)}  | serper.dev keys: {len(keys)} x {per_key}  | serperx: {len(serperx_keys)} x {serperx_calls}", flush=True)

    # Assemble workers: each serper.dev key gets per_key budget; serperx gets its own.
    workers = [("serper", k, per_key) for k in keys]
    workers += [("serperx", k, serperx_calls) for k in serperx_keys]
    # Paid key: run several worker threads on the SAME key (paid tier allows
    # higher concurrency), splitting the paid budget across them for speed.
    if paid_keys:
        pk = paid_keys[0]
        each = max(1, paid_calls // paid_threads)
        workers += [("serper", pk, each) for _ in range(paid_threads)]
        print(f"paid key: {paid_threads} threads x {each} = {paid_threads*each} calls", flush=True)

    # Round-robin shard the queue across workers, capped by each worker's budget.
    shards: list[list] = [[] for _ in workers]
    budgets = [w[2] for w in workers]
    wi = 0
    for row in queue:
        # find next worker with remaining budget
        for _ in range(len(workers)):
            if len(shards[wi]) < budgets[wi]:
                shards[wi].append(row)
                wi = (wi + 1) % len(workers)
                break
            wi = (wi + 1) % len(workers)
        else:
            break  # all budgets full

    threads = []
    for i, (provider, key, budget) in enumerate(workers):
        t = threading.Thread(target=_worker, args=(i, provider, key, shards[i], cache, budget, flush_every, sleep))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    _flush(cache)
    c, h = _progress["calls"], _progress["hits"]
    v, u = _progress["verified"], _progress["unverified"]
    rate = h / c if c else 0.0
    print(f"DONE  calls={c}  hits={h}  hit_rate={rate:.0%}  "
          f"verified={v} ({v/c:.0%})  unverified={u} ({u/c:.0%})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-key", type=int, default=2500, help="cap per free serper.dev key")
    ap.add_argument("--serperx-calls", type=int, default=290, help="cap for serperx key")
    ap.add_argument("--paid-calls", type=int, default=50000, help="cap for paid serper.dev key")
    ap.add_argument("--paid-threads", type=int, default=8, help="concurrent threads on the paid key")
    ap.add_argument("--paid-only", action="store_true", help="use only the paid key (skip free + serperx)")
    args = ap.parse_args()
    run(per_key=args.per_key, serperx_calls=args.serperx_calls,
        paid_calls=args.paid_calls, paid_threads=args.paid_threads,
        paid_only=args.paid_only)
