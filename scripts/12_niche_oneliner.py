"""12 — One-line niche descriptor per top-50 platform candidate.

Visits each firm's website (the manually-verified URL in the CSV), falls back to
the cached scrape text if the live fetch is thin/blocked, and asks Gemini for a
single plain-English sentence describing what the firm actually does / its niche.
Writes the descriptor back into output/platform_candidates_top50.csv as a new
`niche_oneliner` column. Cached per firm so re-runs are free.

Run:  python3 scripts/12_niche_oneliner.py
"""
from __future__ import annotations

import html
import importlib.util
import json
import pathlib
import re
import urllib.request

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

CSV = ROOT / "output" / "platform_candidates_top50.csv"
ENRICHED = ROOT / "output" / "msp_all_companies_enriched.csv"
CACHE = ROOT / "data" / "cache" / "niche_oneliner.json"
MODEL = "gemini-2.5-flash-lite"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}


def _fetch_live(url: str) -> str:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read(400_000).decode("utf-8", "ignore")
    except Exception:
        return ""
    # strip script/style then tags
    raw = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return text[:3000]


def _cached_scrape(cn: str | None) -> str:
    if not cn:
        return ""
    f = clf.SCRAPES / f"{clf._slug(cn)}.txt"
    if f.exists():
        return f.read_text("utf-8", "ignore")[:3000]
    return ""


def _oneliner(api_key: str, name: str, text: str) -> str:
    from google.genai import types
    client = clf._client(api_key)
    prompt = (
        f"Company: {name}\n\nWebsite text:\n{text}\n\n"
        "In ONE sentence (max 25 words), describe exactly what this company "
        "does and the niche it serves. Be specific (e.g. 'managed IT support "
        "and cybersecurity for UK law firms'), not generic. No company name, "
        "no marketing fluff, no preamble — just the sentence.")
    cfg = types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        resp = client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        return " ".join((resp.text or "").split()).strip().strip('"')
    except Exception as e:  # noqa: BLE001
        return f"(error: {type(e).__name__})"


def main() -> None:
    api_key = clf._load_api_key()
    df = pd.read_csv(CSV)
    enr = pd.read_csv(ENRICHED, dtype={"CompanyNumber": str})
    num_by_name = dict(zip(enr["CompanyName"].str.upper(), enr["CompanyNumber"]))

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    out = []
    for _, r in df.iterrows():
        name, url = r["CompanyName"], str(r.get("website") or "")
        key = name.upper()
        if key in cache:
            out.append(cache[key]); continue
        text = _fetch_live(url) if url.startswith("http") else ""
        src = "live"
        if len(text) < 200:
            text = _cached_scrape(num_by_name.get(key)); src = "cache"
        line = _oneliner(api_key, name, text) if len(text) >= 80 else "(no website text available)"
        cache[key] = line
        out.append(line)
        print(f"  [{src:5}] {name[:34]:34} -> {line[:80]}")
    CACHE.write_text(json.dumps(cache, indent=2))

    df["niche_oneliner"] = out
    df.to_csv(CSV, index=False)
    print(f"\nwrote niche_oneliner for {len(df)} firms -> {CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
