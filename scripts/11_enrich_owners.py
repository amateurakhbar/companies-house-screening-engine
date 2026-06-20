"""11 — Enrich msp firms with owners (PSCs) + directors from Companies House.

Pulls persons-with-significant-control and officers for every msp firm that has
a resolved website, derives owner/director name lists and an ownership-type
flag (individual vs PE/corporate vs mixed), and writes the enriched CSV.

Resumable + cached: each firm's raw PSC and officers JSON is cached on disk, so
re-runs only fetch firms not yet seen. Respects the Companies House rate limit
(600 requests / 5 min) with a token-bucket throttle and 429 back-off.

Auth: COMPANIES_HOUSE_API_KEY env var (HTTP Basic, key as username, no password).

Run:  COMPANIES_HOUSE_API_KEY=... python3 scripts/11_enrich_owners.py
"""
from __future__ import annotations

import base64
import json
import os
import pathlib
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
IN_CSV = ROOT / "output" / "msp_all_companies.csv"
OUT_CSV = ROOT / "output" / "msp_all_companies_enriched.csv"
PSC_DIR = ROOT / "data" / "cache" / "ch_psc"
OFF_DIR = ROOT / "data" / "cache" / "ch_officers"
BASE = "https://api.company-information.service.gov.uk"

MAX_WORKERS = 6
# Rate limit: 600 requests / 300 s. Stay under it with a small safety margin.
RATE_MAX = 580
RATE_WINDOW = 300.0


class RateLimiter:
    """Sliding-window throttle shared across worker threads."""

    def __init__(self, max_calls: int, window: float):
        self.max_calls = max_calls
        self.window = window
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                self.calls = [t for t in self.calls if now - t < self.window]
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    return
                sleep = self.window - (now - self.calls[0]) + 0.05
            time.sleep(max(sleep, 0.01))


def _auth_header(key: str) -> str:
    return "Basic " + base64.b64encode(f"{key}:".encode()).decode()


def _get(url: str, key: str, limiter: RateLimiter, retries: int = 4) -> dict | None:
    """GET with rate-limit + 429/5xx back-off. None on 404 (no data)."""
    for attempt in range(retries):
        limiter.acquire()
        req = urllib.request.Request(url, headers={"Authorization": _auth_header(key)})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                wait = float(e.headers.get("Retry-After", 5)) + 1
                time.sleep(wait)
                continue
            if 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    return None


def _fetch_cached(cn: str, kind: str, path: pathlib.Path, key: str,
                  limiter: RateLimiter) -> dict | None:
    f = path / f"{cn}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except json.JSONDecodeError:
            pass
    endpoint = ("persons-with-significant-control" if kind == "psc" else "officers")
    data = _get(f"{BASE}/company/{cn}/{endpoint}", key, limiter)
    # Cache an empty dict for 404s too, so we don't re-hit them on resume.
    f.write_text(json.dumps(data if data is not None else {}))
    return data


# -- derivation -------------------------------------------------------------
def _derive_psc(data: dict | None) -> dict:
    items = [i for i in (data or {}).get("items", []) if not i.get("ceased")]
    names, kinds = [], set()
    for it in items:
        nm = it.get("name")
        if nm:
            names.append(nm)
        k = it.get("kind", "")
        if "individual" in k:
            kinds.add("individual")
        elif "corporate" in k or "legal-person" in k:
            kinds.add("corporate")
    if not items:
        otype = "none/unknown"
    elif kinds == {"individual"}:
        otype = "individual"
    elif kinds == {"corporate"}:
        otype = "corporate/PE"
    elif kinds:
        otype = "mixed"
    else:
        otype = "none/unknown"
    return {"owners": "; ".join(names), "psc_count": len(items),
            "ownership_type": otype}


def _derive_officers(data: dict | None) -> dict:
    items = [i for i in (data or {}).get("items", []) if not i.get("resigned_on")]
    directors = [i for i in items
                 if "director" in (i.get("officer_role") or "").lower()]
    names = [d.get("name") for d in directors if d.get("name")]
    return {"directors": "; ".join(names), "director_count": len(directors),
            "active_officer_count": len(items)}


def main() -> None:
    key = os.environ.get("COMPANIES_HOUSE_API_KEY")
    if not key:
        raise SystemExit("set COMPANIES_HOUSE_API_KEY")
    PSC_DIR.mkdir(parents=True, exist_ok=True)
    OFF_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CSV, dtype={"CompanyNumber": str})
    # Enrich every msp firm. Website-resolved firms were cached on an earlier
    # pass; the disk cache makes this resume and only fetch the rest.
    targets = df["CompanyNumber"].tolist()
    print(f"firms to enrich (all): {len(targets)} / {len(df)}")

    limiter = RateLimiter(RATE_MAX, RATE_WINDOW)
    rows: dict[str, dict] = {}
    done = {"n": 0}
    lock = threading.Lock()

    def work(cn: str) -> None:
        psc = _fetch_cached(cn, "psc", PSC_DIR, key, limiter)
        off = _fetch_cached(cn, "off", OFF_DIR, key, limiter)
        rec = {"CompanyNumber": cn}
        rec.update(_derive_psc(psc))
        rec.update(_derive_officers(off))
        rows[cn] = rec
        with lock:
            done["n"] += 1
            if done["n"] % 200 == 0:
                print(f"  ...{done['n']}/{len(targets)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(work, targets))

    enrich = pd.DataFrame(rows.values())
    out = df.merge(enrich, on="CompanyNumber", how="left")
    for c in ["owners", "ownership_type", "directors"]:
        out[c] = out[c].fillna("")
    for c in ["psc_count", "director_count", "active_officer_count"]:
        out[c] = out[c].fillna(0).astype(int)
    out.to_csv(OUT_CSV, index=False)

    print(f"\nwrote {OUT_CSV.relative_to(ROOT)}  rows={len(out)}")
    print("ownership_type distribution (enriched firms):")
    print(enrich["ownership_type"].value_counts().to_string())
    print(f"firms with >=1 owner named : {(enrich['psc_count'] > 0).sum()}")
    print(f"firms with >=1 director    : {(enrich['director_count'] > 0).sum()}")


if __name__ == "__main__":
    main()
