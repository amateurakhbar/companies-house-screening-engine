"""PRODUCTION classification run — all scraped firms, gemini-2.5-flash-lite.

Priority order: confirmed employees>=3 first, then turnover desc, then
net_assets desc (most acquirable firms get the budget first). No-text firms are
rules-only (no API). Concurrent workers for throughput; cumulative spend is
lock-protected and HARD-STOPS at $9.00. Every result caches the instant it
returns. Output: output/classified_firms.parquet.

Run:  python3 scripts/run_production.py
"""
import collections
import importlib.util
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)
from schema.taxonomy import INSUFFICIENT_EVIDENCE  # noqa: E402

# Conservative (bill-calibrated, ~2.7x list) flash-lite rates so the hard stop
# never overshoots real spend even if billing tracks list more closely.
PRICE_IN_PER_M = 0.27
PRICE_OUT_PER_M = 1.08
SPEND_CAP = 8.00  # ample to finish the remaining firms; cache hits are free
MAX_WORKERS = 8
OUT = ROOT / "output" / "classified_firms.parquet"


def cost_of(u):
    return u.get("in", 0) / 1e6 * PRICE_IN_PER_M + \
        (u.get("out", 0) + u.get("think", 0)) / 1e6 * PRICE_OUT_PER_M


def main():
    api_key = clf._load_api_key()
    df = pd.read_parquet(ROOT / "output" / "working.parquet")
    df["CompanyNumber"] = df["CompanyNumber"].astype(str)
    rules = pd.read_parquet(clf.RULES).set_index("CompanyNumber")

    # segment
    has_text = df["CompanyNumber"].map(
        lambda cn: (clf.SCRAPES / f"{clf._slug(cn)}.txt").exists())
    scraped = df[has_text].copy()
    notext = df[~has_text].copy()

    # priority order
    emp = pd.to_numeric(scraped["employees"], errors="coerce")
    scraped["_ge3"] = (emp >= 3).fillna(False)
    scraped["_turn"] = pd.to_numeric(scraped["turnover"], errors="coerce")
    scraped["_na"] = pd.to_numeric(scraped["net_assets"], errors="coerce")
    scraped = scraped.sort_values(
        ["_ge3", "_turn", "_na"], ascending=[False, False, False],
        na_position="last").reset_index(drop=True)
    total_ge3 = int(scraped["_ge3"].sum())
    print(f"scraped={len(scraped)}  no-text={len(notext)}  ge3-scraped={total_ge3}")

    # concurrent classify with lock-protected hard stop
    stop = threading.Event()
    lock = threading.Lock()
    state = {"spend": 0.0, "reached": 0}
    results = {}

    def work(rec):
        cn, name, sic = rec
        if stop.is_set():
            return None
        # Only a true cache MISS costs money; a hit returns stored usage that
        # was already paid for, so it must not be charged again.
        cache_file = clf.CACHE / f"{clf._slug(cn)}__{clf.PROMPT_VERSION}.json"
        was_cached = cache_file.exists()
        res = clf.classify_firm(api_key, cn, name, sic, "scraped")
        u = res.get("usage") or {}
        c = 0.0 if was_cached else (
            cost_of(u) if (res.get("api_ok") and u and not res.get("skipped_llm")) else 0.0)
        with lock:
            state["spend"] += c
            state["reached"] += 1
            if state["reached"] % 5000 == 0:
                print(f"  ...{state['reached']} processed  spend=${state['spend']:.4f}")
            if state["spend"] > SPEND_CAP:
                if not stop.is_set():
                    print(f"  HARD STOP: spend ${state['spend']:.4f} > ${SPEND_CAP} "
                          f"after {state['reached']} firms")
                stop.set()
        results[cn] = res
        return cn

    recs = list(zip(scraped["CompanyNumber"], scraped["CompanyName"],
                    scraped["SICCode.SicText_1"]))
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        list(ex.map(work, recs))

    # ---------------- assemble output (all firms) ----------------
    def rules_row(cn, name, source):
        rr = rules.loc[cn].to_dict() if cn in rules.index else {}
        func = rr.get("function")
        return {"CompanyNumber": cn, "CompanyName": name,
                "stack_layer": rr.get("stack_layer"),
                "function": func if isinstance(func, str) else "",
                "business_model": rr.get("business_model") or None,
                "vertical": "",
                "primary_niche": func if isinstance(func, str) and func else INSUFFICIENT_EVIDENCE,
                "confidence": float(rr.get("stack_layer_conf") or 0.0),
                "needs_review": True, "source": source}

    rows = []
    llm_ok = 0
    llm_attempted = 0
    for cn, name in zip(scraped["CompanyNumber"], scraped["CompanyName"]):
        res = results.get(cn)
        if res is None:
            rows.append(rules_row(cn, name, "budget_skipped"))
            continue
        llm_attempted += 1
        row = clf.parse_and_validate(res)
        if not row.get("schema_ok"):
            rows.append(rules_row(cn, name, "llm_invalid_fallback_rules"))
            continue
        llm_ok += 1
        rows.append({"CompanyNumber": cn, "CompanyName": name,
                     "stack_layer": row["stack_layer"],
                     "function": "|".join(row["function"]),
                     "business_model": row["business_model"],
                     "vertical": "|".join(row["vertical"]),
                     "primary_niche": row["primary_niche"],
                     "confidence": row["confidence"],
                     "needs_review": row["needs_review"], "source": "llm_flashlite"})
    for cn, name in zip(notext["CompanyNumber"], notext["CompanyName"]):
        rows.append(rules_row(cn, name, "rules_only"))

    out = pd.DataFrame(rows)
    # Carry the discovered website through from the working set so every
    # per-firm downstream consumer/export has it without re-reading the cache.
    if "website" in df.columns:
        out = out.merge(df[["CompanyNumber", "website"]], on="CompanyNumber", how="left")
        out["website"] = out["website"].fillna("")
    out.to_parquet(OUT, index=False)

    # ---------------- report ----------------
    spend = state["spend"]
    # coverage of ge3 scraped cohort
    ge3_cns = set(scraped[scraped["_ge3"]]["CompanyNumber"])
    ge3_classified = sum(1 for cn in ge3_cns
                         if results.get(cn) and clf.parse_and_validate(results[cn]).get("schema_ok"))
    print("\n" + "=" * 60)
    print(f"firms classified (LLM ok): {llm_ok}  / attempted {llm_attempted}")
    print(f"total cost (calibrated)  : ${spend:.4f}  (cap ${SPEND_CAP})")
    if llm_attempted:
        print(f"real per-firm rate       : ${spend/llm_attempted:.6f}")
    print(f">=3-emp scraped coverage : {ge3_classified}/{len(ge3_cns)} "
          f"= {ge3_classified/len(ge3_cns):.1%}" if ge3_cns else "no ge3")
    print(f"schema-conformance       : {llm_ok}/{llm_attempted} = "
          f"{llm_ok/llm_attempted:.1%}" if llm_attempted else "n/a")
    print(f"output rows              : {len(out)} -> {OUT.relative_to(ROOT)}")

    print("\nconfidence distribution by source:")
    for src, grp in out.groupby("source"):
        c = grp["confidence"]
        print(f"  {src:<26} n={len(grp):<6} mean={c.mean():.3f} min={c.min():.2f} max={c.max():.2f}")

    print("\nstack_layer distribution (all firms):")
    for k, v in out["stack_layer"].value_counts(dropna=False).items():
        print(f"  {str(k):<22} {v}")

    print(f"\nfull primary_niche distribution ({out['primary_niche'].nunique()} niches):")
    for k, v in out["primary_niche"].value_counts().items():
        print(f"  {k:<34} {v}")


if __name__ == "__main__":
    main()
