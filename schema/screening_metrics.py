"""Derived financial screening metrics for the tech-firm pipeline.

These are computed from the ~14 financial columns carried in the working set.
They are for *screening only* — they never drive classification (financials
are a weak tiebreaker at most). Recurring-revenue share, the top rollability
driver, is invisible here and comes only from text classification.

Key caveat baked in throughout: turnover / gross_profit / P&L items exist
**only for full-accounts filers**. Micro and small filers (exactly the
sub-threshold roll-up targets) file abbreviated accounts and have null
turnover. So size falls back: turnover -> employees -> net_assets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

METRICS_VERSION = "1.0.0"


class SizeBand(str, Enum):
    """Coarse size band, computed by the best signal available per firm.

    Determined by turnover where present, else employees, else net_assets.
    ``unsized`` is for ``fin_source``-blank rows that carry no financials at
    all — kept in the population, bucketed separately in the rollup.
    """

    micro = "micro"
    small = "small"
    medium = "medium"
    large = "large"
    unsized = "unsized"


class HeadcountBand(str, Enum):
    """Employee band, the size proxy for abbreviated-accounts filers."""

    h_1_4 = "1-4"
    h_5_9 = "5-9"
    h_10_49 = "10-49"
    h_50_249 = "50-249"
    h_250_plus = "250+"
    unknown = "unknown"


# Turnover thresholds (GBP) for full-accounts filers.
_TURNOVER_BANDS = [
    (2_000_000, SizeBand.micro),
    (10_000_000, SizeBand.small),
    (50_000_000, SizeBand.medium),
    (float("inf"), SizeBand.large),
]
# Net-assets fallback thresholds (GBP) when turnover is null.
_NET_ASSETS_BANDS = [
    (1_000_000, SizeBand.micro),
    (5_000_000, SizeBand.small),
    (25_000_000, SizeBand.medium),
    (float("inf"), SizeBand.large),
]
# Employee fallback thresholds.
_HEADCOUNT_BANDS = [
    (5, HeadcountBand.h_1_4),
    (10, HeadcountBand.h_5_9),
    (50, HeadcountBand.h_10_49),
    (250, HeadcountBand.h_50_249),
    (float("inf"), HeadcountBand.h_250_plus),
]


@dataclass
class ScreeningMetrics:
    """Derived screening metrics for one firm. None = not computable."""

    size_band: SizeBand
    headcount_band: HeadcountBand
    ebitda_proxy: float | None          # = operating_profit (no D&A line available)
    operating_margin: float | None      # = operating_profit / turnover
    asset_intensity: float | None       # = fixed_assets / turnover
    working_capital: float | None       # = current_assets - creditors
    current_ratio: float | None         # = current_assets / creditors
    size_source: str                    # which signal set the size_band


def _to_float(value) -> float | None:
    """Best-effort numeric coercion; None/blank/non-numeric -> None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _band(value: float, table) -> Enum:
    for threshold, band in table:
        if value < threshold:
            return band
    return table[-1][1]


def compute_size_band(turnover, employees, net_assets) -> tuple[SizeBand, str]:
    """Size band by the best available signal: turnover -> employees -> net_assets.

    Returns (band, source). ``unsized`` when no signal is present.
    """
    t = _to_float(turnover)
    if t is not None and t > 0:
        return _band(t, _TURNOVER_BANDS), "turnover"

    e = _to_float(employees)
    if e is not None and e > 0:
        # Map headcount onto the size scale for a rough proxy.
        if e < 10:
            return SizeBand.micro, "employees"
        if e < 50:
            return SizeBand.small, "employees"
        if e < 250:
            return SizeBand.medium, "employees"
        return SizeBand.large, "employees"

    na = _to_float(net_assets)
    if na is not None and na > 0:
        return _band(na, _NET_ASSETS_BANDS), "net_assets"

    return SizeBand.unsized, "none"


def compute_headcount_band(employees) -> HeadcountBand:
    e = _to_float(employees)
    if e is None or e <= 0:
        return HeadcountBand.unknown
    return _band(e, _HEADCOUNT_BANDS)


def compute_screening_metrics(row: dict) -> ScreeningMetrics:
    """Compute all screening metrics from a working-set row.

    ``row`` is a mapping with the financial columns: turnover, operating_profit,
    fixed_assets, current_assets, creditors, net_assets, employees, etc.
    Any metric whose inputs are null/zero-denominator returns None rather than
    raising — abbreviated-accounts filers are expected to be sparse here.
    """
    turnover = _to_float(row.get("turnover"))
    operating_profit = _to_float(row.get("operating_profit"))
    fixed_assets = _to_float(row.get("fixed_assets"))
    current_assets = _to_float(row.get("current_assets"))
    creditors = _to_float(row.get("creditors"))
    net_assets = _to_float(row.get("net_assets"))
    employees = row.get("employees")

    size_band, size_source = compute_size_band(turnover, employees, net_assets)

    return ScreeningMetrics(
        size_band=size_band,
        headcount_band=compute_headcount_band(employees),
        ebitda_proxy=operating_profit,
        operating_margin=_safe_div(operating_profit, turnover),
        asset_intensity=_safe_div(fixed_assets, turnover),
        working_capital=(
            current_assets - creditors
            if current_assets is not None and creditors is not None
            else None
        ),
        current_ratio=_safe_div(current_assets, creditors),
        size_source=size_source,
    )


def _safe_div(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def validate(metrics: ScreeningMetrics) -> None:
    """Reject any size/headcount band outside the frozen enums. Raises ValueError."""
    if not isinstance(metrics.size_band, SizeBand):
        raise ValueError(
            f"size_band {metrics.size_band!r} not a SizeBand; "
            f"allowed: {[m.value for m in SizeBand]}"
        )
    if not isinstance(metrics.headcount_band, HeadcountBand):
        raise ValueError(
            f"headcount_band {metrics.headcount_band!r} not a HeadcountBand; "
            f"allowed: {[m.value for m in HeadcountBand]}"
        )


LABEL_SPACE: dict[str, list[str]] = {
    "size_band": [m.value for m in SizeBand],
    "headcount_band": [m.value for m in HeadcountBand],
}

DERIVED_METRICS = [
    "ebitda_proxy = operating_profit (no D&A line; true EBITDA not computable)",
    "operating_margin = operating_profit / turnover",
    "asset_intensity = fixed_assets / turnover",
    "working_capital = current_assets - creditors",
    "current_ratio = current_assets / creditors",
    "size_band = turnover -> employees -> net_assets fallback",
    "headcount_band = banded employees",
]


if __name__ == "__main__":
    print(f"screening_metrics v{METRICS_VERSION}")
    for field_name, labels in LABEL_SPACE.items():
        print(f"  {field_name} ({len(labels)}): {labels}")
    print("  derived metrics:")
    for m in DERIVED_METRICS:
        print(f"    - {m}")
