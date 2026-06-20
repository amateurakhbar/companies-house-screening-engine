"""18 — AI-defensibility re-scoring of the MSP universe (resumable batch classifier).

Re-scores every surviving MSP firm for pricing-power durability under widespread
AI adoption, using the PE buy-and-build rubric defined in RUBRIC below.

Pipeline:
  STEP 1  Pre-filter in code (no LLM spend on excluded rows):
            drop ownership_type == "corporate/PE"
            drop business_model in {"project_oneoff", "resale_distribution"}
          Survivors (~2,244) are written to a working file.
  STEP 2  Classify each survivor from ONLY CompanyName + niche_oneliner +
          descriptor (falls back to name + niche when descriptor is empty).
  STEP 3  Batched 15/call, defensive JSON parsing, one retry then needs_review,
          temperature 0.1, checkpoint every 100 rows, resume by skipping any
          CompanyNumber already in the checkpoint, exponential backoff on 429.
  STEP 4  Gold set: with --gold, classify a RANDOM 30 survivors and dump them
          (inputs + 6 output fields) to gold_set_check.csv, then stop.
  STEP 5  Full run prints a summary and writes msp_defensible_classified.csv,
          joined back onto the enriched table on CompanyNumber. The input file
          is NEVER overwritten.

Run:
  python3 scripts/18_defensibility_classify.py --gold          # 30-row eyeball
  python3 scripts/18_defensibility_classify.py --full          # all survivors
  python3 scripts/18_defensibility_classify.py --full --resume # resume a crash
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

INPUT = ROOT / "output" / "msp_all_companies_enriched.csv"
SURVIVORS = ROOT / "output" / "msp_defensibility_survivors.csv"
CHECKPOINT = ROOT / "output" / "msp_defensibility_checkpoint.csv"
GOLD_OUT = ROOT / "output" / "gold_set_check.csv"
FINAL_OUT = ROOT / "output" / "msp_defensible_classified.csv"

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 15
CHECKPOINT_EVERY = 100
TEMPERATURE = 0.1
GOLD_N = 30
GOLD_SEED = 42

OUTPUT_FIELDS = [
    "defensibility", "pricing_power_score", "ai_cost_tailwind",
    "moat", "rationale", "confidence",
]
EXTRA_FIELDS = ["needs_review_defensibility"]

VALID_DEFENSIBILITY = {"DEFENSIBLE", "BORDERLINE", "EXPOSED"}
VALID_TAILWIND = {"yes", "partial", "no"}
VALID_MOAT = {
    "compliance", "regulated_vertical", "switching_cost", "accreditation",
    "trust_sla", "physical_onsite", "none",
}

RUBRIC = """You are a private equity analyst screening UK IT/managed-service firms for a
buy-and-build roll-up. The ONE question that matters: under widespread AI
adoption, does this firm KEEP its pricing power, or does AI let its clients
self-serve or let competitors undercut it until pricing collapses?

Cost reduction from AI is a BONUS, not the test. The test is pricing-power
durability. A firm whose costs fall but whose pricing also collapses is NOT
defensible.

DEFENSIBLE (pricing holds): value comes from things AI does not erode —
 - regulatory / compliance liability the client cannot self-insure
   (managed cybersecurity, SOC, Cyber Essentials, ISO 27001, GDPR, pen testing)
 - regulated verticals with accreditation and procurement barriers
   (NHS/healthcare, legal, government/public sector, education, financial services)
 - deep integration / high switching cost (embedded ERP/CRM, full-stack managed
   infrastructure, end-to-end outsourced IT with hardware and networking lock-in)
 - trust, SLA accountability, a throat to choke during outages
 - physical/on-site dependency (cabling, networking install, hands-on support)

EXPOSED (pricing erodes): commodity work AI commoditises further —
 - generic L1 helpdesk / break-fix with no specialism
 - software / web / app development heavy (code generation is being automated,
   so dev-led firms are MORE exposed, not less — do not reward this)
 - reselling, licensing arbitrage, box-shifting, procurement-only
 - generic "IT support" / "IT solutions" with no moat language
 - SEO / digital marketing / website-build adjacent work

BORDERLINE: generic horizontal managed infrastructure with no clear moat, or
mixed signals where you cannot tell.

For EACH company return an object with these keys:
{
  "defensibility": "DEFENSIBLE" | "BORDERLINE" | "EXPOSED",
  "pricing_power_score": 1-5,        // 5 = pricing fully holds under AI
  "ai_cost_tailwind": "yes" | "partial" | "no",  // does AI cut their delivery cost
  "moat": "compliance" | "regulated_vertical" | "switching_cost" | "accreditation" | "trust_sla" | "physical_onsite" | "none",
  "rationale": "<= 20 words, quote the signal in the text",
  "confidence": 0.0-1.0
}
If the text is too thin to judge, return BORDERLINE with confidence < 0.4.
Never invent facts not in the supplied text."""

SYSTEM_INSTRUCTION = (
    RUBRIC
    + "\n\nYou will be given a numbered list of companies. Return ONLY a valid "
    "JSON array (no prose, no markdown fences) with exactly one object per "
    "company, in the SAME ORDER, each object additionally carrying an integer "
    '"idx" key matching the company\'s number in the list. Return nothing but '
    "the JSON array."
)


# --------------------------------------------------------------------------
# API key + client (mirrors src/05_classify_llm.py)
# --------------------------------------------------------------------------
def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    for candidate in (ROOT / ".env", ROOT / "secrets" / ".env",
                      ROOT / "secrets" / "gemini.env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found in env, .env, or secrets/gemini.env")


_CLIENT = None


def _client(api_key: str):
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


# --------------------------------------------------------------------------
# Pre-filter (STEP 1)
# --------------------------------------------------------------------------
def build_survivors() -> pd.DataFrame:
    df = pd.read_csv(INPUT, dtype={"CompanyNumber": str}, low_memory=False)
    n0 = len(df)
    keep = (
        (df["ownership_type"] != "corporate/PE")
        & (~df["business_model"].isin(["project_oneoff", "resale_distribution"]))
    )
    surv = df.loc[keep].copy()
    surv.to_csv(SURVIVORS, index=False)
    print(f"[step1] {n0} rows -> {len(surv)} survivors "
          f"(dropped {n0 - len(surv)}). Working file: {SURVIVORS.name}")
    return surv


# --------------------------------------------------------------------------
# Prompt text for one company
# --------------------------------------------------------------------------
def _company_block(idx: int, row: pd.Series) -> str:
    name = str(row.get("CompanyName") or "").strip()
    niche = str(row.get("niche_oneliner") or "").strip()
    desc = str(row.get("descriptor") or "").strip()
    if desc and desc.lower() != "nan":
        body = f"niche: {niche}\ndescriptor: {desc}"
    else:
        body = f"niche: {niche}\ndescriptor: (none — judge from name + niche only)"
    return f"[{idx}] CompanyName: {name}\n{body}"


def _build_user_prompt(rows: list[tuple[int, pd.Series]]) -> str:
    blocks = [_company_block(i, r) for i, r in rows]
    return (
        "Classify the following companies. Return a JSON array of "
        f"{len(rows)} objects as instructed.\n\n" + "\n\n".join(blocks)
    )


# --------------------------------------------------------------------------
# Defensive JSON parsing
# --------------------------------------------------------------------------
def _extract_json_array(raw: str):
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except Exception:
            return None
    return None


def _coerce_one(obj: dict) -> dict | None:
    """Validate/normalise a single object; None if unusable."""
    if not isinstance(obj, dict):
        return None
    out = {}
    d = str(obj.get("defensibility", "")).strip().upper()
    if d not in VALID_DEFENSIBILITY:
        return None
    out["defensibility"] = d
    try:
        sc = int(round(float(obj.get("pricing_power_score"))))
        out["pricing_power_score"] = min(5, max(1, sc))
    except Exception:
        return None
    t = str(obj.get("ai_cost_tailwind", "")).strip().lower()
    out["ai_cost_tailwind"] = t if t in VALID_TAILWIND else "partial"
    m = str(obj.get("moat", "")).strip().lower()
    out["moat"] = m if m in VALID_MOAT else "none"
    out["rationale"] = str(obj.get("rationale", "")).strip()[:300]
    try:
        c = float(obj.get("confidence"))
        out["confidence"] = min(1.0, max(0.0, c))
    except Exception:
        out["confidence"] = 0.3
    return out


def _needs_review_record(cn: str) -> dict:
    return {
        "CompanyNumber": cn,
        "defensibility": "BORDERLINE",
        "pricing_power_score": "",
        "ai_cost_tailwind": "",
        "moat": "",
        "rationale": "unparsed LLM response",
        "confidence": "",
        "needs_review_defensibility": True,
    }


# --------------------------------------------------------------------------
# One LLM call with exponential backoff on rate-limit / transient errors
# --------------------------------------------------------------------------
def _call(api_key: str, prompt: str, *, max_retries: int = 6) -> str:
    from google.genai import types
    client = _client(api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=TEMPERATURE,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    delay = 2.0
    last_exc = None
    for _ in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=config)
            return resp.text or ""
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            last_exc = e
            # Longer backoff specifically on rate-limit / quota signals.
            if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
            else:
                time.sleep(min(delay, 8.0))
                delay = min(delay * 2, 60.0)
    print(f"[warn] call failed after {max_retries} retries: {last_exc}")
    return ""


def _classify_batch(api_key: str, rows: list[tuple[int, pd.Series]]) -> dict[str, dict]:
    """Return {CompanyNumber: result_dict} for the batch. Malformed rows are
    retried once as a batch, then marked needs_review individually."""
    results: dict[str, dict] = {}

    def parse_into(raw: str) -> set[int]:
        """Fill results from raw; return set of local idx that were filled."""
        filled: set[int] = set()
        arr = _extract_json_array(raw)
        if not isinstance(arr, list):
            return filled
        # Map by idx when present, else fall back to positional.
        by_idx = {}
        positional = []
        for obj in arr:
            if isinstance(obj, dict) and "idx" in obj:
                try:
                    by_idx[int(obj["idx"])] = obj
                    continue
                except Exception:
                    pass
            positional.append(obj)
        for local_i, (gi, row) in enumerate(rows):
            cn = str(row["CompanyNumber"])
            if cn in results:
                continue
            obj = by_idx.get(gi)
            if obj is None and local_i < len(positional):
                obj = positional[local_i]
            coerced = _coerce_one(obj) if obj is not None else None
            if coerced is not None:
                rec = {"CompanyNumber": cn, **coerced,
                       "needs_review_defensibility": False}
                results[cn] = rec
                filled.add(gi)
        return filled

    prompt = _build_user_prompt(rows)
    parse_into(_call(api_key, prompt))

    missing = [(gi, r) for gi, r in rows if str(r["CompanyNumber"]) not in results]
    if missing:
        # Retry the whole batch once.
        parse_into(_call(api_key, _build_user_prompt(missing)))

    for gi, r in rows:
        cn = str(r["CompanyNumber"])
        if cn not in results:
            results[cn] = _needs_review_record(cn)
    return results


# --------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------
CHECKPOINT_COLS = ["CompanyNumber", *OUTPUT_FIELDS, *EXTRA_FIELDS]


def _load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    df = pd.read_csv(CHECKPOINT, dtype={"CompanyNumber": str})
    done = {}
    for _, r in df.iterrows():
        done[str(r["CompanyNumber"])] = r.to_dict()
    return done


def _write_checkpoint(done: dict[str, dict]) -> None:
    df = pd.DataFrame(list(done.values()))
    df = df.reindex(columns=CHECKPOINT_COLS)
    tmp = CHECKPOINT.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(CHECKPOINT)


# --------------------------------------------------------------------------
# Core run loop
# --------------------------------------------------------------------------
def classify_frame(api_key: str, surv: pd.DataFrame, *, resume: bool,
                   checkpoint: bool) -> dict[str, dict]:
    done = _load_checkpoint() if (resume and checkpoint) else {}
    if done:
        print(f"[resume] {len(done)} companies already in checkpoint — skipping them")

    todo = [r for _, r in surv.iterrows() if str(r["CompanyNumber"]) not in done]
    print(f"[run] {len(todo)} companies to classify "
          f"({len(surv) - len(todo)} skipped)")

    since_ckpt = 0
    for start in range(0, len(todo), BATCH_SIZE):
        # idx = absolute position in todo, so it is stable & unique per prompt.
        batch = [(start + i, row)
                 for i, row in enumerate(todo[start:start + BATCH_SIZE])]
        res = _classify_batch(api_key, batch)
        done.update(res)
        since_ckpt += len(batch)
        nrev = sum(1 for v in res.values() if v.get("needs_review_defensibility"))
        print(f"[batch] {start + len(batch)}/{len(todo)} done "
              f"(+{len(batch)}, needs_review={nrev})")
        if checkpoint and since_ckpt >= CHECKPOINT_EVERY:
            _write_checkpoint(done)
            since_ckpt = 0
            print(f"[checkpoint] wrote {len(done)} rows -> {CHECKPOINT.name}")

    if checkpoint:
        _write_checkpoint(done)
        print(f"[checkpoint] final write {len(done)} rows -> {CHECKPOINT.name}")
    return done


# --------------------------------------------------------------------------
# Summary (STEP 5)
# --------------------------------------------------------------------------
def print_summary(results: dict[str, dict]) -> None:
    df = pd.DataFrame(list(results.values()))
    print("\n================ DEFENSIBILITY SUMMARY ================")
    print(f"classified: {len(df)}")
    print("\nby defensibility tier:")
    print(df["defensibility"].value_counts().to_string())
    scores = pd.to_numeric(df["pricing_power_score"], errors="coerce")
    print(f"\nmean pricing_power_score: {scores.mean():.3f} "
          f"(n={scores.notna().sum()})")
    print("\nby moat type:")
    print(df["moat"].fillna("").replace("", "(blank)").value_counts().to_string())
    nrev = df["needs_review_defensibility"].fillna(False).astype(bool).sum()
    print(f"\nneeds_review count: {nrev}")
    print("======================================================\n")


# --------------------------------------------------------------------------
# Join + final write (STEP 5) — never touches the input file
# --------------------------------------------------------------------------
def write_final(results: dict[str, dict]) -> None:
    base = pd.read_csv(INPUT, dtype={"CompanyNumber": str}, low_memory=False)
    res = pd.DataFrame(list(results.values()))
    res = res.reindex(columns=CHECKPOINT_COLS)
    merged = base.merge(res, on="CompanyNumber", how="left")
    assert FINAL_OUT.name != INPUT.name, "refusing to overwrite input"
    merged.to_csv(FINAL_OUT, index=False)
    print(f"[step5] wrote {len(merged)} rows ({res['CompanyNumber'].nunique()} "
          f"classified) -> {FINAL_OUT}")


# --------------------------------------------------------------------------
# Gold set (STEP 4)
# --------------------------------------------------------------------------
def run_gold(api_key: str) -> None:
    surv = build_survivors()
    sample = surv.sample(n=min(GOLD_N, len(surv)), random_state=GOLD_SEED)
    print(f"[step4] classifying {len(sample)} random survivors for eyeball")
    results = classify_frame(api_key, sample, resume=False, checkpoint=False)

    rows = []
    for _, r in sample.iterrows():
        cn = str(r["CompanyNumber"])
        res = results.get(cn, {})
        rows.append({
            "CompanyNumber": cn,
            "CompanyName": r.get("CompanyName"),
            "niche_oneliner": r.get("niche_oneliner"),
            "descriptor": r.get("descriptor"),
            **{k: res.get(k) for k in OUTPUT_FIELDS},
        })
    pd.DataFrame(rows).to_csv(GOLD_OUT, index=False)
    print(f"[step4] wrote {GOLD_OUT}")
    print_summary(results)
    print("STOP: eyeball gold_set_check.csv, then run with --full.")


def run_full(api_key: str, resume: bool) -> None:
    surv = build_survivors()
    results = classify_frame(api_key, surv, resume=resume, checkpoint=True)
    print_summary(results)
    write_final(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--gold", action="store_true",
                   help="classify a random 30 survivors -> gold_set_check.csv, then stop")
    g.add_argument("--full", action="store_true",
                   help="classify all survivors -> msp_defensible_classified.csv")
    ap.add_argument("--resume", action="store_true",
                    help="(full only) resume from checkpoint, skipping done rows")
    args = ap.parse_args()

    random.seed(GOLD_SEED)
    api_key = _load_api_key()
    if args.gold:
        run_gold(api_key)
    else:
        run_full(api_key, resume=args.resume)


if __name__ == "__main__":
    main()
