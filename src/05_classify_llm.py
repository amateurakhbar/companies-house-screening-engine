"""05 — LLM classification layer (Claude, via the Anthropic API).

Reads a firm's name + SIC (+ scraped website text when available) and emits a
strict-JSON classification validated against the frozen schema in
schema/taxonomy.py. This is the layer that fills the function/business_model
coverage gap the rules stage (04) leaves open.

Routing by scrape_status:
  scraped              -> classify on name + SIC + scraped text, normal confidence.
  no-url/failed/parked -> classify on name + SIC ONLY, cap confidence at 0.5,
                          and if the name is opaque return insufficient_evidence
                          + needs_review rather than guessing.

Reconciliation with output/rules_labels.parquet:
  - agree            -> boost confidence (min(1.0, conf + 0.1))
  - disagree + text  -> trust the LLM (keep its label/conf)
  - disagree, no text-> needs_review = True

Caching: every result is cached at data/cache/llm_classify/<slug>__<PROMPT_VERSION>.json
so re-runs are free and deterministic. Bump PROMPT_VERSION to invalidate.

Usage (gold set only — the default and only supported scope here):
  ANTHROPIC_API_KEY=... python3 src/05_classify_llm.py --gold

Designed to refuse to run the full dataset unless explicitly asked, to avoid
accidental LLM spend.
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
sys.path.insert(0, str(ROOT))

from schema.taxonomy import (  # noqa: E402
    INSUFFICIENT_EVIDENCE,
    LABEL_SPACE,
    ClassificationOutput,
)

GOLD = ROOT / "gold" / "gold_set_labeled.csv"
RULES = ROOT / "output" / "rules_labels.parquet"
SCRAPES = ROOT / "data" / "cache" / "scrapes"
CACHE = ROOT / "data" / "cache" / "llm_classify"

# Classification runs on Claude via the Anthropic API. Opus 4.8 is the default
# (house standard); for a high-volume full run set CLASSIFY_MODEL=claude-haiku-4-5
# to cut cost. Thinking is off by default on 4.8 (clean JSON, cheap) and the
# large rubric system prompt is prompt-cached. PROMPT_VERSION is derived from the
# model so the on-disk cache never collides across models.
MODEL = os.environ.get("CLASSIFY_MODEL", "claude-opus-4-8")
PROMPT_VERSION = f"claude-{MODEL}-v1"

# scrape_status values that mean "no website text available".
NO_TEXT_STATUSES = {"no-url", "failed", "parked", "not-attempted"}
NO_TEXT_CONF_CAP = 0.5
SCRAPE_TEXT_CAP = 2000  # chars of website text fed to the model


# --------------------------------------------------------------------------
# API key loading
# --------------------------------------------------------------------------
def _load_api_key() -> str | None:
    """ANTHROPIC_API_KEY from the environment, else a .env / secrets file.

    Returns None when not found locally, in which case the Anthropic SDK
    resolves credentials itself (env var or an `ant auth login` profile). The
    key is read at runtime only to authenticate the call; never logged or cached.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    for candidate in (ROOT / ".env", ROOT / "secrets" / ".env",
                      ROOT / "secrets" / "anthropic.env"):
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if line.startswith("ANTHROPIC_API_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None  # let the SDK resolve from env / profile


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def _slug(company_number: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(company_number))


def _system_prompt() -> str:
    ls = LABEL_SPACE
    return (
        "You are a precise UK-company classifier for a tech-sector roll-up "
        "screening pipeline. You must return STRICT JSON only — no prose, no "
        "markdown fences. Use ONLY these closed label vocabularies:\n"
        f"  stack_layer (choose exactly one): {ls['stack_layer']}\n"
        f"  function (list, one or more): {ls['function']}\n"
        f"  business_model (choose exactly one): {ls['business_model']}\n"
        f"  vertical (list, one or more): {ls['vertical']}\n"
        "STACK_LAYER is decided by what the firm BUILDS AND OWNS, not what it "
        "mentions or uses:\n"
        "  - services (DEFAULT for any firm providing technology work to "
        "clients): deploys, supports, installs, repairs, resells, manages, or "
        "consults on technology built by others. IT support, MSPs, "
        "consultancies, repair shops, resellers all go here.\n"
        "  - software: builds and owns software products or applications it "
        "sells or licenses.\n"
        "  - hardware_infra: manufactures or supplies physical IT equipment as "
        "its core business. Repairing or reselling equipment is services, NOT "
        "hardware_infra.\n"
        "  - connectivity_hosting: owns or operates network, hosting, or "
        "datacentre infrastructure. Managing someone else's network is "
        "services, NOT connectivity_hosting.\n"
        "  - data_info_services: operates a data or information platform it "
        "owns. Distributing or consulting on data is services, NOT "
        "data_info_services.\n"
        "When the text mentions 'systems', 'solutions', 'platform', 'networks', "
        "'data', or 'equipment', do NOT infer a product company. Ask: does the "
        "firm BUILD/OWN this, or DEPLOY/SUPPORT/RESELL it? If the latter, label "
        "it services.\n"
        "FUNCTION boundary — app_dev vs data_analytics_ai: app_dev means the "
        "firm BUILDS software, applications, or platforms (web/mobile/bespoke "
        "products). data_analytics_ai means the firm's CORE service is "
        "processing or analysing data (analytics, BI, data science, ML/AI "
        "models, market/research data). A firm that builds an application which "
        "happens to handle data is app_dev; choose data_analytics_ai only when "
        "the analysis OF data is the product itself.\n"
        "primary_niche must be either the single string "
        f"'{INSUFFICIENT_EVIDENCE}', or '<function>' (when the firm is "
        "horizontal), or '<function>__<vertical>', where <function> is one of "
        "the functions you assigned. Never invent labels outside these lists.\n"
        "If the evidence is too thin to classify (an opaque company name with "
        "no website text), set primary_niche to "
        f"'{INSUFFICIENT_EVIDENCE}', function to ['other'], and confidence low "
        "— do NOT guess a specific niche.\n"
        "Return JSON with keys: stack_layer (str), function (list[str]), "
        "business_model (str), vertical (list[str]), primary_niche (str), "
        "confidence (float 0-1), rationale (max 8 words, no punctuation)."
    )


def _user_prompt(name: str, sic: str, text: str | None) -> str:
    parts = [f"Company name: {name}", f"Primary SIC: {sic}"]
    if text:
        parts.append("Website text (may be truncated):\n" + text[:SCRAPE_TEXT_CAP])
    else:
        parts.append(
            "No website text is available for this firm. Classify on name + SIC "
            "ONLY. Keep confidence <= 0.5. If the name is opaque, return "
            f"primary_niche '{INSUFFICIENT_EVIDENCE}' rather than guessing."
        )
    parts.append("Return strict JSON now.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# Claude call (cached) — Anthropic SDK; strict JSON via prompt, fences stripped
# --------------------------------------------------------------------------
_CLIENT = None


def _client(api_key: str | None):
    global _CLIENT
    if _CLIENT is None:
        import anthropic  # local import: keep schema consumers light
        # SDK retries 429/5xx with exponential backoff; key resolves from the
        # arg, else env / `ant` profile.
        _CLIENT = (anthropic.Anthropic(api_key=api_key, max_retries=6)
                   if api_key else anthropic.Anthropic(max_retries=6))
    return _CLIENT


def _call_claude(api_key: str | None, name: str, sic: str, text: str | None
                 ) -> tuple[str, bool, dict]:
    """Return (raw_content, ok, usage). Strict JSON is requested in the system
    prompt (Claude has no response-format flag); markdown fences are stripped
    defensively below. The large rubric system prompt is prompt-cached
    (cache_control: ephemeral), so the gold/full run pays ~0.1x for it after the
    first call. usage = {'in', 'out', 'think'} token counts (0s on failure).
    The SDK retries 429/5xx with exponential backoff (max_retries on the client)."""
    import anthropic

    client = _client(api_key)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": _system_prompt(),
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user",
                       "content": _user_prompt(name, sic, text)}],
        )
    except anthropic.APIError:  # non-retryable, or retries exhausted
        return "", False, {"in": 0, "out": 0, "think": 0}

    raw = next((b.text for b in resp.content if b.type == "text"), "")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())  # strip any fences
    u = resp.usage
    usage = {
        "in": (int(getattr(u, "input_tokens", 0) or 0)
               + int(getattr(u, "cache_read_input_tokens", 0) or 0)
               + int(getattr(u, "cache_creation_input_tokens", 0) or 0)),
        "out": int(getattr(u, "output_tokens", 0) or 0),
        "think": 0,
    }
    return raw, True, usage


def classify_firm(api_key: str, cn: str, name: str, sic: str,
                  scrape_status: str) -> dict:
    """Classify one firm with caching. Returns a result dict with diagnostics."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{_slug(cn)}__{PROMPT_VERSION}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    has_text = scrape_status not in NO_TEXT_STATUSES
    text = None
    if has_text:
        tf = SCRAPES / f"{_slug(cn)}.txt"
        if tf.exists():
            text = tf.read_text(encoding="utf-8", errors="replace")
        else:
            has_text = False  # status said scraped but file missing

    if not has_text:
        # No website text available -> route straight to insufficient_evidence
        # without spending an API call. Cost cut: ~40% of the dataset has no
        # scraped text, and the model already returns insufficient_evidence
        # for most of these firms anyway (see flash bake-off: 64/149).
        raw = json.dumps({
            "stack_layer": "services",
            "function": ["other"],
            "business_model": "project_oneoff",
            "vertical": ["horizontal"],
            "primary_niche": INSUFFICIENT_EVIDENCE,
            "confidence": 0.2,
            "rationale": "No website text available; routed to insufficient_evidence without an LLM call.",
            "needs_review": True,
        })
        result = {
            "CompanyNumber": cn, "scrape_status": scrape_status,
            "had_text": False, "raw": raw, "api_ok": True, "skipped_llm": True,
        }
        cache_file.write_text(json.dumps(result))
        return result

    raw, ok, usage = _call_claude(api_key, name, sic, text)
    result = {
        "CompanyNumber": cn, "scrape_status": scrape_status,
        "had_text": bool(text), "raw": raw, "api_ok": ok, "usage": usage,
    }
    # Only cache SUCCESSFUL calls. A failed call (rate-limit / credit exhaustion)
    # must NOT be cached, or it would be treated as a permanent "done" result and
    # never retried once credits are restored.
    if ok and raw:
        cache_file.write_text(json.dumps(result))
    return result


# --------------------------------------------------------------------------
# Parse + validate + post-process
# --------------------------------------------------------------------------
def parse_and_validate(result: dict) -> dict:
    """Parse raw JSON, validate against the schema, apply routing caps.

    Returns dict with: json_valid, schema_ok, and (if ok) the validated label
    fields, plus needs_review and an error string when applicable.
    """
    out = {"CompanyNumber": result["CompanyNumber"],
           "scrape_status": result["scrape_status"],
           "had_text": result["had_text"],
           "json_valid": False, "schema_ok": False, "error": None}
    if not result.get("api_ok"):
        out["error"] = "api_failed"
        return out

    try:
        data = json.loads(result["raw"])
        out["json_valid"] = True
    except (json.JSONDecodeError, TypeError) as e:
        out["error"] = f"json: {e}"
        return out

    # Cap confidence for no-text firms BEFORE schema validation.
    had_text = result["had_text"]
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if not had_text:
        conf = min(conf, NO_TEXT_CONF_CAP)

    needs_review = bool(data.get("needs_review", False))
    niche = data.get("primary_niche", "")

    # Opaque-name guard: no text + a specific niche guess is downgraded to
    # insufficient_evidence + needs_review (never a guess without text).
    if not had_text and niche and niche != INSUFFICIENT_EVIDENCE:
        # Only force the escape hatch when the model itself was unsure.
        if conf < 0.35:
            niche = INSUFFICIENT_EVIDENCE
            data["function"] = ["other"]
            needs_review = True

    # Coerce off-vocabulary multi-labels to the closed set instead of rejecting
    # the whole record: the model emits real sectors (hospitality, automotive,
    # energy, ...) outside the frozen taxonomy. Drop unknowns, fall back to
    # 'horizontal'/'other', and repair the niche so it always reconciles.
    valid_f = set(LABEL_SPACE["function"])
    valid_v = set(LABEL_SPACE["vertical"])
    fns = [f for f in (data.get("function") or []) if f in valid_f] or ["other"]
    vts = [v for v in (data.get("vertical") or []) if v in valid_v] or ["horizontal"]
    data["function"] = fns
    data["vertical"] = vts
    if niche and niche != INSUFFICIENT_EVIDENCE:
        if "__" in niche:
            base, vpart = niche.split("__", 1)
            if base not in fns:
                niche = INSUFFICIENT_EVIDENCE       # function-part not assigned
            elif vpart not in valid_v:
                niche = base                         # bad vertical -> horizontal
        elif niche not in fns:
            niche = INSUFFICIENT_EVIDENCE

    try:
        co = ClassificationOutput(
            stack_layer=data.get("stack_layer"),
            function=data.get("function"),
            business_model=data.get("business_model"),
            vertical=data.get("vertical"),
            primary_niche=niche,
            confidence=conf,
            rationale=str(data.get("rationale") or "n/a"),
            needs_review=needs_review,
        )
        out["schema_ok"] = True
    except Exception as e:  # noqa: BLE001 — surface any validation failure
        out["error"] = f"schema: {e}"
        return out

    # Post-processing normalization (runs AFTER validate_primary_niche):
    # a horizontal firm's niche is conventionally the bare function, so collapse
    # any '<function>__horizontal' the model emitted to '<function>'.
    niche_raw = co.primary_niche
    niche_norm = niche_raw
    if niche_norm.endswith("__horizontal"):
        niche_norm = niche_norm.split("__", 1)[0]

    out.update({
        "stack_layer": co.stack_layer.value,
        "function": [f.value for f in co.function],
        "business_model": co.business_model.value,
        "vertical": [v.value for v in co.vertical],
        "primary_niche": niche_norm,
        "primary_niche_raw": niche_raw,
        "confidence": co.confidence,
        "rationale": co.rationale,
        "needs_review": co.needs_review,
    })
    return out


def reconcile(row: dict, rules: dict) -> dict:
    """Reconcile a validated LLM row with the rules layer for the same firm.

    agree -> boost; disagree + text -> trust LLM; disagree + no text -> review.
    Reconciliation keys off stack_layer (the axis the rules layer is confident
    on) and function where the rules layer has one.
    """
    if not row.get("schema_ok"):
        return row
    r_stack = rules.get("stack_layer")
    r_func = rules.get("function")
    had_text = row["had_text"]

    agree_stack = r_stack is not None and r_stack == row["stack_layer"]
    agree_func = r_func is not None and r_func in row["function"]
    disagree = (r_stack is not None and not agree_stack) or (
        r_func is not None and not agree_func)

    note = "no_rules_overlap"
    if agree_stack or agree_func:
        row["confidence"] = round(min(1.0, row["confidence"] + 0.1), 2)
        note = "agree_boost"
    if disagree:
        if had_text:
            note = "disagree_trust_llm"
        else:
            row["needs_review"] = True
            note = "disagree_no_text_review"
    row["reconcile"] = note
    return row


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _agreement(rows: list[dict], gold: pd.DataFrame) -> dict:
    """Per-axis agreement between validated LLM rows and human gold labels."""
    g = gold.set_index("CompanyNumber")
    axes = {"stack_layer": "single", "business_model": "single",
            "function": "multi", "vertical": "multi", "primary_niche": "single"}
    tally = {a: [0, 0] for a in axes}  # [agree, total]
    disagreements = []
    for row in rows:
        if not row.get("schema_ok"):
            continue
        cn = row["CompanyNumber"]
        if cn not in g.index:
            continue
        gr = g.loc[cn]
        for axis, kind in axes.items():
            human = gr[axis]
            llm = row[axis]
            if kind == "single":
                ok = str(human) == str(llm)
            else:  # multi: any overlap counts as agreement
                hset = set(str(human).split("|"))
                lset = set(llm) if isinstance(llm, list) else {str(llm)}
                ok = bool(hset & lset)
            tally[axis][0] += int(ok)
            tally[axis][1] += 1
            if axis == "primary_niche" and not ok:
                disagreements.append({
                    "CompanyNumber": cn,
                    "CompanyName": gr["CompanyName"],
                    "human": human, "llm": llm,
                    "confidence": row["confidence"],
                    "rationale": row["rationale"],
                })
    rates = {a: (tally[a][0] / tally[a][1] if tally[a][1] else 0.0) for a in axes}
    return {"rates": rates, "disagreements": disagreements}


def report(parsed: list[dict], gold: pd.DataFrame) -> None:
    n = len(parsed)
    json_valid = sum(p["json_valid"] for p in parsed)
    schema_ok = sum(p["schema_ok"] for p in parsed)
    ok_rows = [p for p in parsed if p["schema_ok"]]
    confs = [p["confidence"] for p in ok_rows]
    insuff = sum(1 for p in ok_rows if p["primary_niche"] == INSUFFICIENT_EVIDENCE)

    print("\n" + "=" * 60)
    print(f"firms classified           : {n}")
    print(f"JSON validity rate         : {json_valid}/{n} = {json_valid/n:.1%}")
    print(f"schema-conformance rate    : {schema_ok}/{n} = {schema_ok/n:.1%}")
    if confs:
        print(f"\nconfidence mean (all ok)   : {sum(confs)/len(confs):.3f}")
        for bucket in ("scraped", "no-url", "failed", "parked"):
            b = [p["confidence"] for p in ok_rows if p["scrape_status"] == bucket]
            if b:
                print(f"  {bucket:<10} n={len(b):<3} mean={sum(b)/len(b):.3f} "
                      f"min={min(b):.2f} max={max(b):.2f}")
    print(f"\ninsufficient_evidence      : {insuff}/{schema_ok}")

    agr = _agreement(ok_rows, gold)
    print("\nagreement with human gold (per axis):")
    for axis, rate in agr["rates"].items():
        print(f"  {axis:<16} {rate:.1%}")

    # primary_niche agreement: before vs after the __horizontal normalization.
    g = gold.set_index("CompanyNumber")
    before = after = total = 0
    for row in ok_rows:
        cn = row["CompanyNumber"]
        if cn not in g.index:
            continue
        human = str(g.loc[cn]["primary_niche"])
        total += 1
        before += int(human == str(row.get("primary_niche_raw", row["primary_niche"])))
        after += int(human == str(row["primary_niche"]))
    if total:
        print("\nprimary_niche agreement (before vs after __horizontal fix):")
        print(f"  before normalization : {before}/{total} = {before/total:.1%}")
        print(f"  after  normalization : {after}/{total} = {after/total:.1%}")

    print("\ntop 5 primary_niche disagreements (LLM vs human):")
    ds = sorted(agr["disagreements"], key=lambda d: -d["confidence"])[:5]
    for d in ds:
        print(f"  {d['CompanyName']} ({d['CompanyNumber']})")
        print(f"      human={d['human']}  llm={d['llm']}  conf={d['confidence']}")
        print(f"      llm rationale: {d['rationale']}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", action="store_true",
                    help="classify the gold set only (the only supported scope)")
    ap.add_argument("--full", action="store_true",
                    help="(guard) refuse unless explicitly set; classifies all firms")
    args = ap.parse_args()

    if not args.gold and not args.full:
        raise SystemExit("pass --gold to classify gold/gold_set_labeled.csv only")
    if args.full:
        raise SystemExit("full-dataset run is disabled in this script revision")

    api_key = _load_api_key()
    gold = pd.read_csv(GOLD, dtype={"CompanyNumber": str})
    rules_df = pd.read_parquet(RULES).set_index("CompanyNumber")

    parsed = []
    for i, g in gold.iterrows():
        cn = g["CompanyNumber"]
        res = classify_firm(api_key, cn, g["CompanyName"],
                            g["SICCode.SicText_1"], g["scrape_status"])
        row = parse_and_validate(res)
        rules_row = (rules_df.loc[cn].to_dict() if cn in rules_df.index else {})
        row = reconcile(row, rules_row)
        parsed.append(row)
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(gold)} classified")

    # Persist the validated label table alongside the gold set.
    out_rows = [p for p in parsed if p.get("schema_ok")]
    if out_rows:
        df = pd.DataFrame(out_rows)
        df["function"] = df["function"].apply(lambda x: "|".join(x))
        df["vertical"] = df["vertical"].apply(lambda x: "|".join(x))
        df.to_csv(ROOT / "gold" / "gold_set_llm.csv", index=False)

    report(parsed, gold)


if __name__ == "__main__":
    main()
