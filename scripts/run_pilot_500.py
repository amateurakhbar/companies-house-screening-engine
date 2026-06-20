"""500-firm direct-classification pilot (gemini-2.5-flash, thinking off).

Selects 500 random SCRAPED firms excluding the gold set and any firm already
cached under the current PROMPT_VERSION, classifies each directly (no cluster
propagation), records ACTUAL token usage, and HARD-STOPS if cumulative spend
exceeds $2.00. Every result is cached the instant it returns (credit-safe).

Run (requires non-depleted GEMINI credits):
  python3 scripts/run_pilot_500.py
"""
import collections
import importlib.util
import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("clf", ROOT / "src" / "05_classify_llm.py")
clf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(clf)

# gemini-2.5-flash pricing (USD per 1M tokens), CALIBRATED to the actual bill:
# the first 500-firm pilot (575K input + 51K output tokens) was billed $0.80,
# ~2.7x the $0.30/$2.50 list assumption. Back-calculated keeping the ~8.3:1
# output:input ratio -> ~$0.80/1M in, ~$6.67/1M out (blended ~$0.0016/firm).
PRICE_IN_PER_M = 0.80
PRICE_OUT_PER_M = 6.67
SPEND_CAP = 2.00
N_PILOT = 500
SCRAPED_TOTAL = 27348


def cost_of(usage: dict) -> float:
    return (usage.get("in", 0) / 1e6 * PRICE_IN_PER_M
            + (usage.get("out", 0) + usage.get("think", 0)) / 1e6 * PRICE_OUT_PER_M)


def main():
    api_key = clf._load_api_key()
    df = pd.read_parquet(clf.ROOT / "output" / "working.parquet")[
        ["CompanyNumber", "CompanyName", "SICCode.SicText_1"]].copy()
    df["CompanyNumber"] = df["CompanyNumber"].astype(str)

    gold = set(pd.read_csv(clf.GOLD, dtype={"CompanyNumber": str})["CompanyNumber"])

    def eligible(cn):
        if cn in gold:
            return False
        if not (clf.SCRAPES / f"{clf._slug(cn)}.txt").exists():
            return False  # scraped firms only
        if (clf.CACHE / f"{clf._slug(cn)}__{clf.PROMPT_VERSION}.json").exists():
            return False  # already cached under this model -> not "fresh"
        return True

    pool = df[df["CompanyNumber"].map(eligible)]
    sample = pool.sample(n=min(N_PILOT, len(pool)), random_state=42).reset_index(drop=True)
    print(f"eligible pool: {len(pool)}   pilot sample: {len(sample)}")

    rows, usages, spend = [], [], 0.0
    for i, g in sample.iterrows():
        res = clf.classify_firm(api_key, g["CompanyNumber"], g["CompanyName"],
                                g["SICCode.SicText_1"], "scraped")
        if not res.get("api_ok"):
            print(f"  ABORT at {i}: API call failed (rate-limit / credits). "
                  f"Cached successes are preserved.")
            break
        u = res.get("usage", {"in": 0, "out": 0, "think": 0})
        spend += cost_of(u)
        usages.append(u)
        rows.append(clf.parse_and_validate(res))
        if spend > SPEND_CAP:
            print(f"  HARD STOP at firm {i+1}: cumulative spend ${spend:.4f} > ${SPEND_CAP}")
            break
        if (i + 1) % 50 == 0:
            print(f"  ...{i+1}/{len(sample)}  spend=${spend:.4f}")

    # persist pilot output
    ok_rows = [r for r in rows if r.get("schema_ok")]
    if ok_rows:
        out = pd.DataFrame(ok_rows)
        out["function"] = out["function"].apply(lambda x: "|".join(x))
        out["vertical"] = out["vertical"].apply(lambda x: "|".join(x))
        out.to_csv(ROOT / "output" / "pilot_500.csv", index=False)

    # ---------------- report ----------------
    n = len(rows)
    schema_ok = len(ok_rows)
    ti = sum(u["in"] for u in usages); to = sum(u["out"] for u in usages)
    tt = sum(u["think"] for u in usages)
    print("\n" + "=" * 60)
    print(f"firms classified         : {n}")
    print(f"schema-conformance       : {schema_ok}/{n} = {schema_ok/n:.1%}" if n else "no firms")
    if n:
        print(f"avg input tokens/firm    : {ti/n:.0f}")
        print(f"avg output tokens/firm   : {to/n:.0f}  (thinking tokens total={tt})")
        print(f"avg cost/firm            : ${spend/n:.6f}")
        print(f"pilot total spend        : ${spend:.4f}  (cap ${SPEND_CAP})")
        print(f"\nEXTRAPOLATION to {SCRAPED_TOTAL} scraped firms:")
        print(f"  est. total cost        : ${spend/n*SCRAPED_TOTAL:.2f}")
        print(f"  est. total tokens in/out: {ti/n*SCRAPED_TOTAL/1e6:.1f}M / {to/n*SCRAPED_TOTAL/1e6:.1f}M")

    print("\nconfidence distribution by scrape_status bucket:")
    bucket = collections.defaultdict(list)
    for r in ok_rows:
        bucket[r["scrape_status"]].append(r["confidence"])
    for b, vals in bucket.items():
        print(f"  {b:<10} n={len(vals):<3} mean={sum(vals)/len(vals):.3f} "
              f"min={min(vals):.2f} max={max(vals):.2f}")

    print("\ntop 15 primary_niche labels by frequency:")
    niche = collections.Counter(r["primary_niche"] for r in ok_rows)
    for k, v in niche.most_common(15):
        print(f"  {k:<32} {v}")


if __name__ == "__main__":
    main()
