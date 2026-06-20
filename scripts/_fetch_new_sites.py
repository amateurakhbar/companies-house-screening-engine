"""Fetch + cache scrape text for the newly recovered websites, so the existing
13_descriptors.py (which reads cached scrape text) can describe them."""
from __future__ import annotations

import html
import importlib.util
import json
import pathlib
import re
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

CACHE = ROOT / "data" / "cache" / "urls.json"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def fetch(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(400_000).decode("utf-8", "ignore")
    except Exception:
        return ""
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    return html.unescape(re.sub(r"\s+", " ", text)).strip()[:6000]


c = json.loads(CACHE.read_text())
targets = []
for cn, rec in c.items():
    m = rec.get("match") or {}
    if m.get("verified"):
        targets.append((cn, m["url"]))  # cn is the 8-char CompanyNumber key

clf.SCRAPES.mkdir(parents=True, exist_ok=True)
state = {"ok": 0, "empty": 0, "skip": 0}
lock = threading.Lock()


def work(t):
    cn, url = t
    f = clf.SCRAPES / f"{clf._slug(cn)}.txt"
    if f.exists() and f.stat().st_size > 200:
        with lock:
            state["skip"] += 1
        return
    text = fetch(url)
    if len(text) >= 80:
        f.write_text(text, "utf-8")
        with lock:
            state["ok"] += 1
    else:
        with lock:
            state["empty"] += 1


with ThreadPoolExecutor(max_workers=12) as ex:
    list(ex.map(work, targets))
print(f"targets={len(targets)} fetched={state['ok']} empty={state['empty']} "
      f"already_cached={state['skip']}")
