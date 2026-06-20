"""19 — is_it_business scope gate (hybrid rules + LLM) over the WHOLE universe.

The defensibility classifier answers "is this a durable business?" well, but it
never asked "is this even an IT / managed-service firm?" — so ~32 non-IT trades
(grease traps, HVAC, HGV repair, building automation, EV chargers, marine nav)
scored DEFENSIBLE on a genuine physical/compliance moat yet do not belong in an
IT roll-up. This gate runs BEFORE defensibility and tags every company.

Hybrid decision:
  1. RULES pre-pass (free, instant):
       - strong IT anchor in text/SIC and no non-IT trade signal  -> is_it_business=true  (rule)
       - strong non-IT physical/industrial trade signal and no IT anchor -> false (rule)
  2. LLM adjudicates only the AMBIGUOUS middle (batched 15/call, temp 0.1),
     deciding whether the firm's CORE business is delivering IT / managed
     services vs a physical/industrial trade that merely uses/installs tech.

Core test (mirrors stack_layer logic in src/05_classify_llm.py): a firm that
DELIVERS IT/managed services (support, helpdesk, MSP, cyber, cloud, networks,
hosting, telecoms, software) is IT. A firm whose core trade is physical/
industrial (HVAC, drainage, vehicle/fleet, building/industrial automation,
catering equipment, marine, manufacturing) is NOT — even if it installs or
maintains technology.

Never overwrites the input. Writes output/msp_is_it_business.csv keyed on
CompanyNumber and joins the column onto output/msp_defensible_classified.csv.

Run:  python3 scripts/19_is_it_business_gate.py            # full run
      python3 scripts/19_is_it_business_gate.py --resume   # resume a crash
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT = ROOT / "output" / "msp_all_companies_enriched.csv"
OUT = ROOT / "output" / "msp_is_it_business.csv"
CHECKPOINT = ROOT / "output" / "is_it_business_checkpoint.csv"
CLASSIFIED = ROOT / "output" / "msp_defensible_classified.csv"

MODEL = "gemini-2.5-flash-lite"
BATCH_SIZE = 15
CHECKPOINT_EVERY = 100
TEMPERATURE = 0.1

# Strong "this IS IT/managed services" anchors.
IT_ANCHOR = re.compile(
    r"\b(managed (it|service)|msp\b|mssp|helpdesk|help desk|it support|it services|"
    r"cyber ?security|cybersecurity|soc\b|pen test|cloud|saas|software|app develop|"
    r"web develop|website|microsoft 365|office 365|m365|azure|aws\b|server|data ?cent|"
    r"voip|telecom|telephony|unified comms|hosting|colocation|backup|disaster recovery|"
    r"erp\b|crm\b|network support|endpoint|firewall|sql|database|it consultanc|it infrastructure|"
    r"computer repair|laptop|break-?fix|it managed|outsourced it)\b", re.I)

# Strong "this is a physical/industrial trade, NOT IT" signals.
NON_IT = re.compile(
    r"\b(grease|drain|sewer|hvac|heat pump|air conditioning|refrigerat|plumb|boiler|gas safe|"
    r"hgv|lorry|vehicle repair|fleet repair|garage|mot test|electrical testing|pat testing|"
    r"fixed wire|facilities management|cleaning services|janitorial|landscap|grounds maintenance|"
    r"pest control|scaffold|roofing|fencing|water treatment|wastewater|pipeline|catering|"
    r"commercial kitchen|locksmith|fire extinguisher|valve|metallurg|automotive|car audio|"
    r"marine navigation|bridge|streetlight|building automation|industrial automation|"
    r"process control|life safety|lift\b|elevator|escalator|crane|forklift|solar panel|"
    r"ev charg|signage manufactur|aerial install|satellite station|instrumentation|"
    r"enclosure|thermal management|explosion-proof|hazardous (area|environment))\b", re.I)

SYSTEM_INSTRUCTION = (
    "You screen UK companies for an IT / managed-service roll-up. For EACH company "
    "decide ONE thing: is the firm's CORE business delivering IT or managed services "
    "(IT support, MSP, cybersecurity, cloud, networks, hosting, telecoms, software, "
    "computer repair) — or is it a physical / industrial / trade business that merely "
    "uses, installs, or maintains technology (HVAC, drainage, vehicle/fleet, building "
    "or industrial automation, control systems, catering equipment, marine, "
    "manufacturing, electrical/mechanical contracting)? A firm that installs HVAC "
    "controls or services grease traps is NOT IT even if it mentions 'systems' or "
    "'technology'. A firm that manages servers, helpdesks, or networks IS IT. If a "
    "firm genuinely does both and IT is a real core line, answer true.\n"
    "You are given a numbered list. Return ONLY a JSON array, one object per company, "
    "same order, each carrying integer \"idx\" matching the number, plus "
    "\"is_it_business\" (true|false) and \"reason\" (<=12 words). Return nothing but "
    "the JSON array.")


def _load_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key.strip()
    for c in (ROOT / ".env", ROOT / "secrets" / ".env", ROOT / "secrets" / "gemini.env"):
        if c.exists():
            for line in c.read_text().splitlines():
                if line.strip().startswith("GEMINI_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("GEMINI_API_KEY not found in env, .env, or secrets/gemini.env")


_CLIENT = None


def _client(api_key: str):
    global _CLIENT
    if _CLIENT is None:
        from google import genai
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _text(r) -> str:
    sic = " ".join(str(r.get(c, "")) for c in
                   ("SICCode.SicText_1", "SICCode.SicText_2", "SICCode.SicText_3"))
    return f"{r.get('CompanyName','')} {r.get('niche_oneliner','')} {r.get('descriptor','')} {sic}"


def rule_decision(r) -> tuple[bool | None, str]:
    """Return (is_it_business or None-if-ambiguous, reason)."""
    t = _text(r)
    it = bool(IT_ANCHOR.search(t))
    non = bool(NON_IT.search(t))
    if it and not non:
        return True, "rule: clear IT anchor, no trade signal"
    if non and not it:
        return False, "rule: physical/industrial trade, no IT anchor"
    return None, "ambiguous -> LLM"


# ---------- LLM for the ambiguous middle ----------
def _extract_array(raw: str):
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?", "", raw.strip()).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        a, b = raw.find("["), raw.rfind("]")
        if a != -1 and b > a:
            try:
                return json.loads(raw[a:b + 1])
            except Exception:
                return None
    return None


def _call(api_key: str, prompt: str, *, retries: int = 6) -> str:
    from google.genai import types
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION, temperature=TEMPERATURE,
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0))
    delay = 2.0
    for _ in range(retries):
        try:
            return _client(api_key).models.generate_content(
                model=MODEL, contents=prompt, config=cfg).text or ""
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            time.sleep(delay)
            delay = min(delay * 2, 60.0) if any(k in msg for k in
                       ("429", "rate", "quota", "resource_exhausted")) else min(delay * 2, 12.0)
    return ""


def _llm_batch(api_key: str, rows: list[tuple[int, pd.Series]]) -> dict[str, dict]:
    blocks = []
    for gi, r in rows:
        blocks.append(f"[{gi}] {r.get('CompanyName','')} :: "
                      f"{str(r.get('niche_oneliner','') or r.get('descriptor','') or '')[:200]}")
    prompt = "Classify each company.\n\n" + "\n\n".join(blocks)
    out: dict[str, dict] = {}

    def parse(raw):
        arr = _extract_array(raw)
        if not isinstance(arr, list):
            return
        by_idx = {}
        for o in arr:
            if isinstance(o, dict) and "idx" in o:
                try:
                    by_idx[int(o["idx"])] = o
                except Exception:
                    pass
        for gi, r in rows:
            cn = str(r["CompanyNumber"])
            if cn in out:
                continue
            o = by_idx.get(gi)
            if isinstance(o, dict) and "is_it_business" in o:
                val = o["is_it_business"]
                val = (str(val).strip().lower() in ("true", "yes", "1")) if not isinstance(val, bool) else val
                out[cn] = {"CompanyNumber": cn, "is_it_business": bool(val),
                           "gate_reason": "llm: " + str(o.get("reason", ""))[:80],
                           "gate_source": "llm"}

    parse(_call(api_key, prompt))
    missing = [(gi, r) for gi, r in rows if str(r["CompanyNumber"]) not in out]
    if missing:
        parse(_call(api_key, "Classify each company.\n\n" + "\n\n".join(
            f"[{gi}] {r.get('CompanyName','')} :: {str(r.get('niche_oneliner','') or '')[:200]}"
            for gi, r in missing)))
    # default unparsed -> true (do not silently drop a firm) + needs_review
    for gi, r in rows:
        cn = str(r["CompanyNumber"])
        if cn not in out:
            out[cn] = {"CompanyNumber": cn, "is_it_business": True,
                       "gate_reason": "llm-unparsed: defaulted true (needs review)",
                       "gate_source": "llm-failed"}
    return out


COLS = ["CompanyNumber", "is_it_business", "gate_reason", "gate_source"]


def _load_ckpt() -> dict[str, dict]:
    if not CHECKPOINT.exists():
        return {}
    return {str(r["CompanyNumber"]): r.to_dict()
            for _, r in pd.read_csv(CHECKPOINT, dtype={"CompanyNumber": str}).iterrows()}


def _write_ckpt(done: dict[str, dict]):
    tmp = CHECKPOINT.with_suffix(".tmp.csv")
    pd.DataFrame(list(done.values())).reindex(columns=COLS).to_csv(tmp, index=False)
    tmp.replace(CHECKPOINT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    api_key = _load_api_key()

    df = pd.read_csv(INPUT, dtype={"CompanyNumber": str}, low_memory=False)
    df["CompanyNumber"] = df["CompanyNumber"].str.strip()
    done = _load_ckpt() if args.resume else {}

    # rules pass for everything not already done
    ambiguous = []
    rule_true = rule_false = 0
    for _, r in df.iterrows():
        cn = str(r["CompanyNumber"])
        if cn in done:
            continue
        d, reason = rule_decision(r)
        if d is True:
            done[cn] = {"CompanyNumber": cn, "is_it_business": True, "gate_reason": reason, "gate_source": "rule"}
            rule_true += 1
        elif d is False:
            done[cn] = {"CompanyNumber": cn, "is_it_business": False, "gate_reason": reason, "gate_source": "rule"}
            rule_false += 1
        else:
            ambiguous.append(r)
    print(f"[rules] {rule_true} IT, {rule_false} non-IT decided by rules; {len(ambiguous)} ambiguous -> LLM")

    since = 0
    for start in range(0, len(ambiguous), BATCH_SIZE):
        batch = [(start + i, row) for i, row in enumerate(ambiguous[start:start + BATCH_SIZE])]
        done.update(_llm_batch(api_key, batch))
        since += len(batch)
        if since >= CHECKPOINT_EVERY:
            _write_ckpt(done); since = 0
            print(f"[ckpt] {len(done)} done")
    _write_ckpt(done)

    res = pd.DataFrame(list(done.values())).reindex(columns=COLS)
    res.to_csv(OUT, index=False)
    print(f"\n[gate] wrote {OUT.name}  ({len(res)} rows)")
    print(res["is_it_business"].value_counts().to_string())
    print("by source:"); print(res["gate_source"].value_counts().to_string())

    # join onto classified file (never the input)
    if CLASSIFIED.exists():
        c = pd.read_csv(CLASSIFIED, dtype={"CompanyNumber": str})
        c["CompanyNumber"] = c["CompanyNumber"].str.strip()
        c = c.drop(columns=[x for x in ("is_it_business", "gate_reason", "gate_source") if x in c.columns])
        c = c.merge(res, on="CompanyNumber", how="left")
        c.to_csv(CLASSIFIED, index=False)
        print(f"[gate] joined is_it_business onto {CLASSIFIED.name}")


if __name__ == "__main__":
    main()
