"""Recover llm_invalid_fallback_rules firms: clear their cache entries and
re-classify ONLY those firms on flash-lite (same PROMPT_VERSION, so the 22.6k
valid cache stays intact). Report how many now pass schema, and update
output/classified_firms.parquet in place for the recovered ones.
"""
import importlib.util
import pathlib
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

OUT = ROOT / "output" / "classified_firms.parquet"
PRICE_IN, PRICE_OUT = 0.27, 1.08  # calibrated flash-lite per-1M


def main():
    api_key = clf._load_api_key()
    df = pd.read_parquet(OUT)
    df["CompanyNumber"] = df["CompanyNumber"].astype(str)
    inv = df[df["source"] == "llm_invalid_fallback_rules"]["CompanyNumber"].tolist()
    print(f"invalid firms to recover: {len(inv)}")

    wk = pd.read_parquet(ROOT / "output" / "working.parquet")[
        ["CompanyNumber", "CompanyName", "SICCode.SicText_1"]]
    wk["CompanyNumber"] = wk["CompanyNumber"].astype(str)
    wk = wk.set_index("CompanyNumber")

    # clear just these cache entries
    cleared = 0
    for cn in inv:
        f = clf.CACHE / f"{clf._slug(cn)}__{clf.PROMPT_VERSION}.json"
        if f.exists():
            f.unlink(); cleared += 1
    print(f"cleared cache entries: {cleared}")

    lock = threading.Lock()
    state = {"in": 0, "out": 0}
    recovered = {}

    def work(cn):
        name = wk.loc[cn, "CompanyName"] if cn in wk.index else cn
        sic = wk.loc[cn, "SICCode.SicText_1"] if cn in wk.index else ""
        res = clf.classify_firm(api_key, cn, name, sic, "scraped")
        u = res.get("usage") or {}
        with lock:
            state["in"] += u.get("in", 0); state["out"] += u.get("out", 0) + u.get("think", 0)
        row = clf.parse_and_validate(res)
        if row.get("schema_ok"):
            recovered[cn] = row
        return cn

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, inv))

    # update parquet rows for recovered firms
    for cn, row in recovered.items():
        m = df["CompanyNumber"] == cn
        df.loc[m, "stack_layer"] = row["stack_layer"]
        df.loc[m, "function"] = "|".join(row["function"])
        df.loc[m, "business_model"] = row["business_model"]
        df.loc[m, "vertical"] = "|".join(row["vertical"])
        df.loc[m, "primary_niche"] = row["primary_niche"]
        df.loc[m, "confidence"] = row["confidence"]
        df.loc[m, "needs_review"] = row["needs_review"]
        df.loc[m, "source"] = "llm_flashlite"
    df.to_parquet(OUT, index=False)

    cost = state["in"] / 1e6 * PRICE_IN + state["out"] / 1e6 * PRICE_OUT
    print("\n" + "=" * 50)
    print(f"re-attempted     : {len(inv)}")
    print(f"now pass schema  : {len(recovered)} ({len(recovered)/len(inv):.1%})")
    print(f"still invalid    : {len(inv) - len(recovered)}")
    print(f"cost (calibrated): ${cost:.4f}")
    print(f"source counts now: {df['source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
