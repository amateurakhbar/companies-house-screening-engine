"""03 — Scrape + cache website text for classification.

For EVERY firm in output/working.parquet, record a scrape_status:
    no-url   : no usable URL in urls.json (or a tentative discarded on confirm)
    scraped  : clean text fetched successfully
    failed   : a URL existed but every fetch errored / timed out
    parked   : domain-for-sale / under-construction / empty placeholder page

For firms WITH a URL (verified or tentative) we fetch homepage + /about + /services,
extract clean visible text (trafilatura strips nav/footer/cookie/boilerplate), cap at
~5,000 tokens, and cache per company slug under data/cache/scrapes/.

Tentative URLs are CONFIRMED against the page text: if any company-name token appears
the match is promoted to verified=True; if not, the URL is discarded and the firm
becomes no-url (this is the Stage-4 verification the discovery stage deferred).

Parked / placeholder pages are detected (tiny text volume or for-sale / under-
construction markers) and marked parked so they are never fed to the LLM as real text.

Concurrency: 8 threads, 1.5s delay between requests per worker, exponential backoff.

  python3 src/03_scrape.py --sample 100        # sample run
  python3 src/03_scrape.py                      # full run
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
import trafilatura

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKING = ROOT / "output" / "working.parquet"
URLS = ROOT / "data" / "cache" / "urls.json"
SCRAPES = ROOT / "data" / "cache" / "scrapes"
STATUS_OUT = ROOT / "data" / "cache" / "scrape_status.json"

# Reuse the name tokenizer from the discovery module (file starts with a digit).
_spec = importlib.util.spec_from_file_location("discover", ROOT / "src" / "02_discover_urls.py")
_d = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_d)

PATHS = ["", "/about", "/services"]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-GB,en;q=0.9",
}
REQUEST_DELAY = 0.5      # between same-domain page fetches (each firm = diff domain)
MAX_RETRIES = 2
TIMEOUT = 5              # don't let one slow site stall a thread for long
TOKEN_CAP = 5000                 # ~ approx tokens
CHAR_CAP = TOKEN_CAP * 4         # ~4 chars/token
PARKED_MIN_WORDS = 40            # below this = placeholder/empty

PARKED_MARKERS = (
    "domain for sale", "buy this domain", "this domain is for sale",
    "under construction", "site coming soon", "website coming soon",
    "coming soon", "account suspended", "parked free", "domain parking",
    "godaddy", "sedoparking", "hugedomains", "page cannot be found",
    "default web page", "future home of",
)

_lock = threading.Lock()
_status: dict[str, dict] = {}


def _slug(company_number: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(company_number))


MAX_BYTES = 2_000_000   # cap downloaded HTML to bound memory (skip huge pages)


def _fetch(url: str) -> str | None:
    """GET a URL with retry + backoff, capped at MAX_BYTES. Returns HTML or None."""
    delay = REQUEST_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            with requests.get(url, headers=HEADERS, timeout=TIMEOUT,
                              allow_redirects=True, stream=True) as r:
                ctype = r.headers.get("Content-Type", "")
                if r.status_code in (429, 500, 502, 503, 504):
                    time.sleep(delay); delay *= 2; continue
                if r.status_code != 200 or "html" not in ctype.lower():
                    return None
                chunks, total = [], 0
                for chunk in r.iter_content(8192, decode_unicode=False):
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total >= MAX_BYTES:
                        break
                raw = b"".join(chunks)
                enc = r.encoding or "utf-8"
                return raw.decode(enc, errors="replace") if raw else None
        except requests.RequestException:
            time.sleep(delay); delay *= 2
    return None


def _extract(html: str) -> str:
    """Boilerplate-free main visible text (nav/footer/cookie stripped)."""
    txt = trafilatura.extract(
        html, include_comments=False, include_tables=False,
        no_fallback=False, favor_precision=True,
    )
    return (txt or "").strip()


def _is_parked(text: str) -> bool:
    low = text.lower()
    if any(m in low for m in PARKED_MARKERS):
        return True
    if len(text.split()) < PARKED_MIN_WORDS:
        return True
    return False


def _cap_tokens(text: str) -> str:
    if len(text) <= CHAR_CAP:
        return text
    cut = text[:CHAR_CAP]
    return cut[:cut.rfind(" ")] if " " in cut else cut


def _base(url: str) -> str:
    p = urlparse(url if "://" in url else "https://" + url)
    return f"{p.scheme}://{p.netloc}"


def scrape_firm(cnum: str, name: str, rec: dict) -> dict:
    """Scrape one firm; returns its status record. Caches text on success."""
    match = rec.get("match")
    if not match or not match.get("url"):
        return {"scrape_status": "no-url"}

    base = _base(match["url"])
    parts, any_text = [], False
    for path in PATHS:
        target = base if path == "" else urljoin(base + "/", path.lstrip("/"))
        html = _fetch(target)
        time.sleep(REQUEST_DELAY)
        if not html:
            continue
        text = _extract(html)
        if not text:
            continue
        any_text = True            # a page returned extractable text
        if _is_parked(text):
            continue               # placeholder — don't keep it
        parts.append(text)

    combined = _cap_tokens("\n\n".join(parts).strip())

    # No usable text retained.
    if not combined:
        # If some page DID extract text but all of it was placeholder -> parked;
        # if nothing fetched/extracted at all -> failed.
        return {"scrape_status": "parked" if any_text else "failed"}

    # Tentative confirmation: a company-name token must appear in the text.
    verified = bool(match.get("verified"))
    if not verified:
        toks = _d._tokens(name)
        low = combined.lower()
        if not any(t in low for t in toks):
            # Could not confirm this tentative URL -> discard it.
            return {"scrape_status": "no-url", "tentative_discarded": True}
        verified = True  # confirmed by on-page name match

    slug = _slug(cnum)
    (SCRAPES / f"{slug}.txt").write_text(combined)
    return {
        "scrape_status": "scraped",
        "verified": verified,
        "url": match["url"],
        "chars": len(combined),
        "words": len(combined.split()),
        "confirmed_tentative": (not bool(match.get("verified"))) and verified,
    }


def _worker(item):
    cnum, name, rec = item
    try:
        st = scrape_firm(cnum, name, rec)
    except Exception as e:
        st = {"scrape_status": "failed", "error": str(e)[:200]}
    with _lock:
        _status[cnum] = {"CompanyName": name, **st}
    _maybe_flush()


_flush_lock = threading.Lock()
_since_flush = [0]


def _maybe_flush():
    with _flush_lock:
        _since_flush[0] += 1
        if _since_flush[0] % 100 == 0:
            STATUS_OUT.write_text(json.dumps(_status, indent=2))


def run(sample: int | None, seed: int = 7, threads: int = 8,
        resume: bool = True) -> None:
    SCRAPES.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(WORKING)
    urls = json.loads(URLS.read_text()) if URLS.exists() else {}

    # Resume: load prior statuses, AND treat any firm with a cached scrape .txt
    # as already done (so re-runs never re-fetch cached text).
    if resume and not sample:
        if STATUS_OUT.exists():
            try:
                _status.update(json.loads(STATUS_OUT.read_text()))
            except json.JSONDecodeError:
                pass
        for f in SCRAPES.glob("*.txt"):
            cnum = f.stem
            if cnum not in _status:
                _status[cnum] = {"scrape_status": "scraped", "from_cache": True}

    if sample:
        # Mix of verified + tentative URL-holders (plus whatever no-url falls in).
        have = df[df["CompanyNumber"].astype(str).isin(urls.keys())]
        def is_tent(c):
            m = urls.get(str(c), {}).get("match")
            return bool(m) and not m.get("verified")
        tent = have[have["CompanyNumber"].apply(is_tent)]
        ver = have[~have["CompanyNumber"].apply(is_tent)]
        n_t = min(len(tent), sample // 2)
        n_v = min(len(ver), sample - n_t)
        df = pd.concat([ver.sample(n_v, random_state=seed),
                        tent.sample(n_t, random_state=seed)])

    items = []
    for _, row in df.iterrows():
        cnum = str(row["CompanyNumber"])
        if cnum in _status:          # already processed (resume) -> skip
            continue
        items.append((cnum, str(row["CompanyName"]), urls.get(cnum, {})))

    print(f"to process: {len(items)}  (already done: {len(_status)})", flush=True)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(_worker, items))

    STATUS_OUT.write_text(json.dumps(_status, indent=2))
    _report()


def _report() -> None:
    from collections import Counter
    c = Counter(v["scrape_status"] for v in _status.values())
    total = len(_status)
    print("=== scrape_status breakdown ===")
    for k in ("scraped", "no-url", "failed", "parked"):
        print(f"  {k:8}: {c.get(k,0):5}  ({c.get(k,0)/total:.0%})")

    # Tentative confirmation rate.
    confirmed = sum(1 for v in _status.values() if v.get("confirmed_tentative"))
    discarded = sum(1 for v in _status.values() if v.get("tentative_discarded"))
    tent_total = confirmed + discarded
    if tent_total:
        print(f"\ntentative confirmation: {confirmed}/{tent_total} "
              f"({confirmed/tent_total:.0%}) confirmed, {discarded} discarded")

    # Three example scrapes.
    examples = [(cn, v) for cn, v in _status.items() if v["scrape_status"] == "scraped"][:3]
    print("\n=== example scrapes ===")
    for cn, v in examples:
        slug = _slug(cn)
        text = (SCRAPES / f"{slug}.txt").read_text()
        print(f"\n--- {v['CompanyName']}  ({v['url']})  [{v['words']} words] ---")
        print(text[:280].replace("\n", " ") + " ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    run(sample=args.sample, seed=args.seed, threads=args.threads)
