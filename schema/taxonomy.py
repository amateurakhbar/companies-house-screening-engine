"""Frozen label schema for the tech-firm classification pipeline.

Every classification output must validate against these closed enums.
A label outside the enum is rejected, never silently coerced, because the
whole point of freezing the schema is that downstream joins (rollup, gold-set
scoring) can rely on a fixed, finite label space.

Do not add, rename or remove members without bumping TAXONOMY_VERSION and
re-running the gold set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

TAXONOMY_VERSION = "1.1.0"

# Escape hatch: emitted instead of a guessed niche when the firm is dormant,
# a bare shell, or otherwise has no discernible niche signal.
INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class StackLayer(str, Enum):
    """Single-valued: where the firm sits in the tech stack."""

    hardware_infra = "hardware_infra"
    software = "software"
    services = "services"
    connectivity_hosting = "connectivity_hosting"
    data_info_services = "data_info_services"


class Function(str, Enum):
    """Multi-valued: what the firm actually does."""

    cyber = "cyber"
    cloud_devops = "cloud_devops"
    data_analytics_ai = "data_analytics_ai"
    networking = "networking"
    erp_crm = "erp_crm"
    app_dev = "app_dev"
    testing_qa = "testing_qa"
    msp_infrastructure = "msp_infrastructure"
    other = "other"


class BusinessModel(str, Enum):
    """Single-valued: the rollability driver."""

    recurring_managed = "recurring_managed"
    project_oneoff = "project_oneoff"
    resale_distribution = "resale_distribution"
    staffing = "staffing"


class Vertical(str, Enum):
    """Multi-valued: sector focus."""

    healthcare = "healthcare"
    legal = "legal"
    finserv = "finserv"
    govt = "govt"
    education = "education"
    property = "property"
    construction = "construction"
    manufacturing = "manufacturing"
    retail = "retail"
    horizontal = "horizontal"


# Convenience: the full label space, by field.
LABEL_SPACE: dict[str, list[str]] = {
    "stack_layer": [m.value for m in StackLayer],
    "function": [m.value for m in Function],
    "business_model": [m.value for m in BusinessModel],
    "vertical": [m.value for m in Vertical],
}


def derive_primary_niche(
    function: list[Function] | list[str],
    vertical: list[Vertical] | list[str],
) -> str:
    """Best-fit intersection of the dominant function and vertical.

    Convention: ``<function>__<vertical>`` (e.g. ``managed_cyber__legal``).
    Takes the first (highest-priority) function and vertical. ``horizontal``
    verticals collapse to a bare function niche.
    """
    fn = function[0].value if isinstance(function[0], Function) else function[0]
    vt = vertical[0].value if isinstance(vertical[0], Vertical) else vertical[0]
    if vt == Vertical.horizontal.value:
        return fn
    return f"{fn}__{vt}"


@dataclass
class ClassificationOutput:
    """One firm's frozen classification result.

    ``primary_niche`` is derived, not free-form; ``confidence`` is 0..1;
    ``rationale`` is one line; ``needs_review`` routes low-confidence or
    priority firms to manual review.
    """

    stack_layer: StackLayer
    function: list[Function]
    business_model: BusinessModel
    vertical: list[Vertical]
    primary_niche: str
    confidence: float
    rationale: str
    needs_review: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject any label outside the frozen enums. Raises ValueError."""
        # Coerce + check single-valued enum fields.
        self.stack_layer = _coerce_enum(self.stack_layer, StackLayer, "stack_layer")
        self.business_model = _coerce_enum(
            self.business_model, BusinessModel, "business_model"
        )

        # Multi-valued enum fields.
        if not isinstance(self.function, list) or not self.function:
            raise ValueError("function must be a non-empty list")
        self.function = [
            _coerce_enum(v, Function, "function") for v in self.function
        ]
        if not isinstance(self.vertical, list) or not self.vertical:
            raise ValueError("vertical must be a non-empty list")
        self.vertical = [
            _coerce_enum(v, Vertical, "vertical") for v in self.vertical
        ]

        # Scalars.
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError(f"confidence {self.confidence!r} not in [0, 1]")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if not isinstance(self.needs_review, bool):
            raise ValueError("needs_review must be a bool")
        if not isinstance(self.primary_niche, str) or not self.primary_niche.strip():
            raise ValueError("primary_niche must be a non-empty string")
        validate_primary_niche(
            self.primary_niche, self.stack_layer, self.function, self.business_model
        )


def validate_primary_niche(
    primary_niche: str,
    stack_layer: StackLayer | str,
    function: list[Function] | list[str],
    business_model: BusinessModel | str,
) -> str:
    """Reject a ``primary_niche`` that doesn't reconcile with its labels.

    ``insufficient_evidence`` is always allowed (the escape hatch for dormant
    shells). Otherwise the niche must be ``<function>`` or ``<function>__<vertical>``
    where ``<function>`` is one of the firm's assigned functions, and a
    ``staffing``/``resale_distribution`` business model must not be paired with
    a ``software`` stack_layer niche of ``app_dev`` or ``data_analytics_ai``
    (those imply the firm builds product, which contradicts a pure
    staffing/resale model).
    """
    if primary_niche == INSUFFICIENT_EVIDENCE:
        return primary_niche

    sl = stack_layer.value if isinstance(stack_layer, StackLayer) else stack_layer
    bm = business_model.value if isinstance(business_model, BusinessModel) else business_model
    fns = {f.value if isinstance(f, Function) else f for f in function}

    head = primary_niche.split("__", 1)[0]
    if head not in fns:
        raise ValueError(
            f"primary_niche {primary_niche!r} does not reconcile: "
            f"{head!r} is not among assigned function labels {sorted(fns)}"
        )

    if (
        sl == StackLayer.software.value
        and bm in (BusinessModel.staffing.value, BusinessModel.resale_distribution.value)
        and head in (Function.app_dev.value, Function.data_analytics_ai.value)
    ):
        raise ValueError(
            f"primary_niche {primary_niche!r} does not reconcile: "
            f"stack_layer={sl!r} function={head!r} implies product-building, "
            f"which contradicts business_model={bm!r}"
        )

    return primary_niche


def _coerce_enum(value, enum_cls: type[Enum], field_name: str):
    """Return the enum member for ``value`` or raise ValueError.

    Accepts either an enum member or its string value. Anything else, or any
    string not in the enum, is rejected — no silent coercion.
    """
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError:
        allowed = [m.value for m in enum_cls]
        raise ValueError(
            f"{field_name}: {value!r} is not a valid label; allowed: {allowed}"
        ) from None


def validate_label(field_name: str, value) -> str:
    """Validate a single label against its field's enum. Returns the value."""
    enum_cls = {
        "stack_layer": StackLayer,
        "function": Function,
        "business_model": BusinessModel,
        "vertical": Vertical,
    }.get(field_name)
    if enum_cls is None:
        raise ValueError(f"unknown classification field: {field_name!r}")
    return _coerce_enum(value, enum_cls, field_name).value


if __name__ == "__main__":
    print(f"taxonomy v{TAXONOMY_VERSION}")
    for field_name, labels in LABEL_SPACE.items():
        print(f"  {field_name} ({len(labels)}): {labels}")
    print("  primary_niche: derived, e.g. 'cyber__legal' or 'cyber' (horizontal)")
    print("  confidence: float in [0, 1]")
    print("  rationale: one-line str")
    print("  needs_review: bool")
