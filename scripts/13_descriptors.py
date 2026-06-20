"""13 — One-line niche descriptor + 4-6 word tag for msp firms (bulk, cheap).

Cost-optimised pass over an enriched CSV: reads each firm's CACHED website scrape
text (no live fetch), and in a SINGLE gemini-2.5-flash-lite call (thinking off)
returns both a one-sentence niche description and a 4-6 word lowercase tag of
what the business actually is. Results cache per firm so re-runs are free, and
work is concurrent for throughput.

Why this is cheap: ~870 input + ~40 output tokens/firm on flash-lite; only firms
with cached scrape text cost anything. Full msp set (~3.2k firms, ~2.1k with
text) lands well under $1.

Adds columns: niche_oneliner, descriptor.

Run:  python3 scripts/13_descriptors.py [--in output/msp_all_companies_enriched.csv]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

CACHE = ROOT / "data" / "cache" / "descriptors.json"
MODEL = "gemini-2.5-flash-lite"
MAX_WORKERS = 8
TEXT_CAP = 3000


def _scrape_text(cn: str) -> str:
    f = clf.SCRAPES / f"{clf._slug(cn)}.txt"
    return f.read_text("utf-8", "ignore")[:TEXT_CAP] if f.exists() else ""


def _describe(client, name: str, text: str) -> dict:
    from google.genai import types
    prompt = (
        f"Company: {name}\n\nWebsite text:\n{text}\n\n"
        "Return JSON with two keys:\n"
        '  "oneliner": one sentence (max 25 words) stating exactly what the firm '
        "does and the niche it serves — specific, no fluff, no company name.\n"
        '  "descriptor": a 4-6 word lowercase noun phrase naming what the business '
        "actually is (e.g. 'managed it for law firms', 'refurbished hardware "
        "reseller', 'cashmere knitwear retailer').")
    cfg = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        raw = client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text
        d = json.loads(raw)
        return {"oneliner": " ".join(str(d.get("oneliner", "")).split()).strip(),
                "descriptor": " ".join(str(d.get("descriptor", "")).split()).strip().rstrip(".")}
    except Exception as e:  # noqa: BLE001
        return {"oneliner": f"(error: {type(e).__name__})", "descriptor": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp",
                    default=str(ROOT / "output" / "msp_all_companies_enriched.csv"))
    args = ap.parse_args()
    inp = pathlib.Path(args.inp)

    api_key = clf._load_api_key()
    client = clf._client(api_key)
    df = pd.read_csv(inp, dtype={"CompanyNumber": str})
    if "CompanyNumber" not in df.columns:
        enr = pd.read_csv(ROOT / "output" / "msp_all_companies_enriched.csv",
                          dtype={"CompanyNumber": str})
        num = dict(zip(enr["CompanyName"].str.upper(), enr["CompanyNumber"]))
        df["CompanyNumber"] = df["CompanyName"].str.upper().map(num)

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    clock = threading.Lock()
    state = {"n": 0, "new": 0, "notext": 0}

    def work(rec):
        cn, name = rec
        if not isinstance(cn, str) or cn in cache:
            return
        text = _scrape_text(cn)
        if len(text) < 80:
            cache[cn] = {"oneliner": "(no website text available)", "descriptor": ""}
            with clock:
                state["notext"] += 1
            return
        res = _describe(client, name, text)
        with clock:
            cache[cn] = res
            state["new"] += 1
            if state["new"] % 200 == 0:
                print(f"  ...{state['new']} described")

    recs = list(zip(df["CompanyNumber"], df["CompanyName"]))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(work, recs))
    CACHE.write_text(json.dumps(cache, indent=2))

    df["niche_oneliner"] = df["CompanyNumber"].map(lambda c: (cache.get(c) or {}).get("oneliner", ""))
    df["descriptor"] = df["CompanyNumber"].map(lambda c: (cache.get(c) or {}).get("descriptor", ""))
    df.to_csv(inp, index=False)
    print(f"\nwrote niche_oneliner + descriptor -> {inp.relative_to(ROOT)}")
    print(f"  newly described: {state['new']}  | no-text: {state['notext']}  | total rows: {len(df)}")


if __name__ == "__main__":
    main()
