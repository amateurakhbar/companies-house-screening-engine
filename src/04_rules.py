"""04 — Rules / keyword classification layer.

Cheap, transparent first pass that runs before any LLM spend:

  * stack_layer  — a coarse PRIOR mapped from the firm's SIC codes
    (SICCode.SicText_1..4), by 5-digit code ranges.
  * function     — keyword dictionaries over CompanyName.
  * business_model — keyword dictionaries over CompanyName.

NOTE on signal: at this stage scraped website text does not yet exist
(Stage 03 is still discovering URLs), so function/business_model here read
ONLY the company name. A name is a thin, often-misleading hint, so confidences
are deliberately modest and many firms will get no confident label on those
two axes — that is expected. The LLM stage (05) picks up the remainder once
scraped text is available. Every labelled row is tagged source='rules'.

Output: output/rules_labels.parquet (one row per firm, labels + confidences).
"""

from __future__ import annotations

import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from schema.taxonomy import (  # noqa: E402
    BusinessModel,
    Function,
    StackLayer,
)

WORKING = ROOT / "output" / "working.parquet"
OUT = ROOT / "output" / "rules_labels.parquet"

SIC_COLS = [f"SICCode.SicText_{i}" for i in range(1, 5)]
_CODE_RE = re.compile(r"^\s*(\d{4,5})")

# --- stack_layer prior by SIC code (5-digit) -----------------------------
# Keyed by exact code where the mapping is clean; range buckets handle the
# manufacturing/electronics and telecoms families.
_SIC_EXACT: dict[str, StackLayer] = {
    "46510": StackLayer.hardware_infra,   # wholesale of computers (resale, but HW layer)
    "47410": StackLayer.hardware_infra,   # retail of computers
    "58210": StackLayer.software,         # publishing of computer games
    "58290": StackLayer.software,         # other software publishing
    "62011": StackLayer.software,         # ready-made software
    "62012": StackLayer.software,         # bespoke software development
    "62020": StackLayer.services,         # IT consultancy
    "62030": StackLayer.services,         # computer facilities management
    "62090": StackLayer.services,         # other IT service activities
    "63110": StackLayer.connectivity_hosting,  # data processing, hosting
    "63120": StackLayer.data_info_services,    # web portals
    "63990": StackLayer.data_info_services,    # other information service
    "95110": StackLayer.services,         # repair of computers
    "77330": StackLayer.hardware_infra,   # renting office machinery incl. computers
}


def _stack_from_code(code: str) -> StackLayer | None:
    if code in _SIC_EXACT:
        return _SIC_EXACT[code]
    if not code:
        return None
    head3 = code[:3]
    head2 = code[:2]
    # Manufacture of computers / electronics / electrical equipment.
    if head2 in {"26", "27"}:
        return StackLayer.hardware_infra
    # Telecommunications (wired/wireless/satellite/other telecoms).
    if head3 in {"611", "612", "613", "619"}:
        return StackLayer.connectivity_hosting
    # Software publishing family.
    if head3 == "582":
        return StackLayer.software
    # Computer programming / consultancy family.
    if head3 == "620":
        return StackLayer.services
    # Information service activities.
    if head3 == "639":
        return StackLayer.data_info_services
    return None


def stack_layer_prior(row: pd.Series) -> tuple[str | None, float]:
    """Best stack_layer prior across the up-to-4 SIC codes.

    Returns (label, confidence). Multiple distinct hits lower confidence
    (the firm straddles layers); a single clean hit is more confident.
    """
    hits: list[StackLayer] = []
    for col in SIC_COLS:
        val = row.get(col)
        if not isinstance(val, str):
            continue
        m = _CODE_RE.match(val)
        if not m:
            continue
        code = m.group(1).zfill(5)
        layer = _stack_from_code(code)
        if layer is not None:
            hits.append(layer)
    if not hits:
        return None, 0.0
    # Pick the most common; confidence reflects agreement.
    top = max(set(hits), key=hits.count)
    agree = hits.count(top) / len(hits)
    conf = 0.45 + 0.2 * agree  # 0.55 (split) .. 0.65 (unanimous), SIC is coarse
    return top.value, round(conf, 2)


# --- keyword dictionaries over CompanyName -------------------------------
# Ordered by specificity; first strong match wins. Word-boundary matched.
_FUNCTION_KW: list[tuple[Function, list[str]]] = [
    (Function.cyber, ["cyber", "security", "secure", "infosec", "pentest", "soc"]),
    (Function.erp_crm, ["erp", "crm", "dynamics", "salesforce", "netsuite", "sap", "sage"]),
    (Function.cloud_devops, ["cloud", "devops", "kubernetes", "hosting", "saas"]),
    (Function.data_analytics_ai, ["data", "analytics", "intelligence", "insight", "\\bai\\b", "machine learning"]),
    (Function.networking, ["network", "networks", "connectivity", "telecom", "telecoms", "fibre", "wireless", "broadband"]),
    (Function.testing_qa, ["testing", "\\bqa\\b", "quality assurance"]),
    (Function.msp_infrastructure, ["managed", "\\bmsp\\b", "it support", "it services", "infrastructure"]),
    (Function.app_dev, ["software", "\\bapp\\b", "apps", "digital", "\\bweb\\b", "development", "code", "studio"]),
]

_BUSINESS_KW: list[tuple[BusinessModel, list[str]]] = [
    (BusinessModel.staffing, ["recruitment", "staffing", "resourcing", "\\btalent\\b", "personnel"]),
    (BusinessModel.resale_distribution, ["distribution", "reseller", "wholesale", "supplies", "trading", "\\bsales\\b"]),
    (BusinessModel.recurring_managed, ["managed", "\\bmsp\\b", "saas", "subscription", "\\bsupport\\b", "hosting"]),
    (BusinessModel.project_oneoff, ["consulting", "consultancy", "solutions", "studio", "agency", "development"]),
]


def _match_keywords(name: str, table) -> tuple[str | None, float]:
    low = name.lower()
    for label, kws in table:
        for kw in kws:
            pat = kw if kw.startswith("\\b") or "\\b" in kw else re.escape(kw)
            if re.search(pat, low):
                # Name-only signal: modest confidence.
                return label.value, 0.5
    return None, 0.0


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["CompanyNumber"] = df["CompanyNumber"].values
    out["CompanyName"] = df["CompanyName"].values

    sl, slc, fn, fnc, bm, bmc = [], [], [], [], [], []
    for _, row in df.iterrows():
        s_label, s_conf = stack_layer_prior(row)
        name = str(row.get("CompanyName") or "")
        f_label, f_conf = _match_keywords(name, _FUNCTION_KW)
        b_label, b_conf = _match_keywords(name, _BUSINESS_KW)
        sl.append(s_label); slc.append(s_conf)
        fn.append(f_label); fnc.append(f_conf)
        bm.append(b_label); bmc.append(b_conf)

    out["stack_layer"] = sl
    out["stack_layer_conf"] = slc
    out["function"] = fn
    out["function_conf"] = fnc
    out["business_model"] = bm
    out["business_model_conf"] = bmc
    # source tag only where at least one axis got a label.
    labelled = out[["stack_layer", "function", "business_model"]].notna().any(axis=1)
    out["source"] = labelled.map(lambda x: "rules" if x else None)
    return out


def main() -> None:
    df = pd.read_parquet(WORKING)
    n = len(df)
    out = classify(df)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)

    pct = lambda c: f"{out[c].notna().sum() / n:.1%}"
    print(f"rows                       : {n}")
    print(f"stack_layer  (SIC prior)   : {pct('stack_layer')} confident")
    print(f"function     (name kw)     : {pct('function')} confident")
    print(f"business_model (name kw)   : {pct('business_model')} confident")
    print(f"any axis labelled          : {pct('source')}")


if __name__ == "__main__":
    main()
