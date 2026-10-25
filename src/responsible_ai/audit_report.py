"""Bias audit reports: slice metrics, intersections, worst-slice flagging,
and configurable pass/fail gates.

A slice is one value of one attribute (e.g. ``region=west``); an
intersection is a combination across two attributes (e.g.
``region=west x age_band=18-29``). Intersections are where aggregate
fairness hides harm, so they are first-class here. Slices below
``min_group_size`` are still computed but flagged, never silently dropped
-- dropping small slices is how the worst harm gets hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from responsible_ai.errors import ValidationError
from responsible_ai.fairness import (
    MIN_GROUP_SIZE_DEFAULT,
    compute_group_metrics,
    summarize_fairness,
)


@dataclass(frozen=True)
class SliceMetric:
    """One slice (or intersection) with its headline metrics."""

    slice_id: str
    attributes: dict[str, str]
    n: int
    accuracy: float
    positive_rate: float
    true_positive_rate: float | None
    false_positive_rate: float | None
    flagged_small: bool

    def as_dict(self) -> dict:
        return {
            "slice_id": self.slice_id,
            "attributes": self.attributes,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "positive_rate": round(self.positive_rate, 4),
            "tpr": None if self.true_positive_rate is None else round(self.true_positive_rate, 4),
            "fpr": None if self.false_positive_rate is None else round(self.false_positive_rate, 4),
            "flagged_small": self.flagged_small,
        }


@dataclass
class SliceReport:
    """Full audit: overall fairness, single-attribute slices, intersections."""

    overall: dict
    slices: list[SliceMetric] = field(default_factory=list)
    intersections: list[SliceMetric] = field(default_factory=list)
    min_group_size: int = MIN_GROUP_SIZE_DEFAULT

    def as_dict(self) -> dict:
        return {
            "overall": self.overall,
            "slices": [s.as_dict() for s in self.slices],
            "intersections": [s.as_dict() for s in self.intersections],
            "min_group_size": self.min_group_size,
        }


def _slice_metrics(
    y_true: list[int],
    y_pred: list[int],
    labels: list[str],
    min_group_size: int,
) -> list[SliceMetric]:
    per_group = compute_group_metrics(y_true, y_pred, labels, min_group_size)
    out: list[SliceMetric] = []
    for label, m in per_group.items():
        attrs: dict[str, str] = {}
        for part in label.split(" x "):
            if "=" in part:
                key, value = part.split("=", 1)
                attrs[key] = value
            else:
                attrs["group"] = part
        out.append(
            SliceMetric(
                slice_id=label,
                attributes=attrs,
                n=m.n,
                accuracy=m.accuracy,
                positive_rate=m.positive_rate,
                true_positive_rate=m.true_positive_rate,
                false_positive_rate=m.false_positive_rate,
                flagged_small=m.n < min_group_size,
            )
        )
    return out


def build_slice_report(
    y_true: list[int],
    y_pred: list[int],
    attributes: dict[str, list[str]],
    intersect: tuple[str, str] | None = None,
    min_group_size: int = MIN_GROUP_SIZE_DEFAULT,
) -> SliceReport:
    """Build a slice report over each attribute, plus one intersection pair.

    ``attributes`` maps attribute name -> parallel list of values. The
    overall fairness summary is computed over the first attribute (or a
    single "all" group when no attributes are given).
    """
    if not attributes:
        overall = summarize_fairness(
            y_true, y_pred, ["all"] * len(y_true), min_group_size=min_group_size
        ).as_dict()
        return SliceReport(overall=overall, min_group_size=min_group_size)
    for name, values in attributes.items():
        if len(values) != len(y_true):
            raise ValidationError(
                f"Attribute {name!r} has {len(values)} values for {len(y_true)} rows"
            )
    first = next(iter(attributes))
    overall = summarize_fairness(
        y_true, y_pred, attributes[first], min_group_size=min_group_size
    ).as_dict()
    slices: list[SliceMetric] = []
    for name, values in attributes.items():
        labels = [f"{name}={v}" for v in values]
        slices.extend(_slice_metrics(y_true, y_pred, labels, min_group_size))
    intersections: list[SliceMetric] = []
    if intersect is not None:
        a, b = intersect
        if a not in attributes or b not in attributes:
            raise ValidationError(f"Intersection attributes {intersect} not in attributes")
        labels = [
            f"{a}={va} x {b}={vb}"
            for va, vb in zip(attributes[a], attributes[b])
        ]
        intersections = _slice_metrics(y_true, y_pred, labels, min_group_size)
    return SliceReport(
        overall=overall,
        slices=slices,
        intersections=intersections,
        min_group_size=min_group_size,
    )


def worst_slice(
    slices: list[SliceMetric],
    metric: str = "true_positive_rate",
) -> SliceMetric | None:
    """The slice furthest from the (defined) mean of ``metric``.

    Small slices are included -- a flagged small slice winning this
    ranking is itself a finding, and the flag travels with it.
    """
    candidates = [s for s in slices if getattr(s, metric) is not None]
    if not candidates:
        return None
    mean = sum(getattr(s, metric) for s in candidates) / len(candidates)  # type: ignore[arg-type]
    return max(candidates, key=lambda s: abs(getattr(s, metric) - mean))  # type: ignore[arg-type]


@dataclass(frozen=True)
class Gate:
    """A pass/fail gate: ``metric operator threshold`` (e.g. DPD <= 0.10)."""

    name: str
    metric: str  # key in FairnessSummary.as_dict(), or "disparate_impact"
    operator: str  # "<=", ">=", "<", ">"
    threshold: float
    severity: str = "blocker"  # or "warning"

    def check(self, value: float | None) -> bool | None:
        """True = pass, False = fail, None = metric undefined (cannot pass)."""
        if value is None:
            return None
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<":
            return value < self.threshold
        if self.operator == ">":
            return value > self.threshold
        raise ValidationError(f"Unknown gate operator {self.operator!r}")


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    value: float | None
    passed: bool | None
    message: str


def evaluate_gates(
    summary_dict: dict,
    gates: list[Gate],
) -> list[GateResult]:
    """Evaluate gates against a ``FairnessSummary.as_dict()`` payload."""
    results: list[GateResult] = []
    for gate in gates:
        value = summary_dict.get(gate.metric)
        passed = gate.check(value)
        if passed is None:
            message = f"{gate.name}: UNDEFINED ({gate.metric} unavailable) -- cannot pass"
        elif passed:
            message = f"{gate.name}: PASS ({gate.metric}={value} {gate.operator} {gate.threshold})"
        else:
            message = f"{gate.name}: FAIL ({gate.metric}={value} {gate.operator} {gate.threshold})"
        results.append(GateResult(gate=gate, value=value, passed=passed, message=message))
    return results


DEFAULT_GATES: list[Gate] = [
    Gate("demographic-parity", "demographic_parity_difference", "<=", 0.10),
    Gate("disparate-impact", "disparate_impact", ">=", 0.80),
    Gate("equal-opportunity", "equal_opportunity_difference", "<=", 0.10),
    Gate("predictive-parity", "predictive_parity_difference", "<=", 0.10),
]
