"""Drift and data-quality screens with a fairness lens.

Fairness degrades silently when the production population stops looking
like the audited reference: a group that shrinks in representation, or
whose label rate shifts, changes every fairness metric even if the model
never changes. These screens catch that *before* re-running a full audit.

* Representation drift: per-group share change + total variation distance
  between reference and production group distributions.
* Label-rate shift: per-slice positive-label rate change with a flag
  threshold (default 0.05 absolute).
* Data-quality screen: missing-rate per field, plus out-of-range counts
  for caller-supplied numeric bounds. A field that goes missing for one
  group is a fairness event, so missing rates are reported per slice too.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from responsible_ai.errors import ValidationError


@dataclass(frozen=True)
class DriftReport:
    reference_n: int
    production_n: int
    per_group_share_change: dict[str, float]
    total_variation_distance: float
    max_abs_share_change: float
    flagged: bool

    def as_dict(self) -> dict:
        return {
            "reference_n": self.reference_n,
            "production_n": self.production_n,
            "per_group_share_change": {
                k: round(v, 4) for k, v in self.per_group_share_change.items()
            },
            "total_variation_distance": round(self.total_variation_distance, 4),
            "max_abs_share_change": round(self.max_abs_share_change, 4),
            "flagged": self.flagged,
        }


def representation_drift(
    reference_groups: list[str],
    production_groups: list[str],
    flag_threshold: float = 0.05,
) -> DriftReport:
    """Compare group distributions between reference and production data."""
    if not reference_groups or not production_groups:
        raise ValidationError("Both reference and production groups are required")
    ref_counts = Counter(str(g) for g in reference_groups)
    prod_counts = Counter(str(g) for g in production_groups)
    all_groups = sorted(set(ref_counts) | set(prod_counts))
    changes: dict[str, float] = {}
    tvd = 0.0
    for group in all_groups:
        ref_share = ref_counts.get(group, 0) / len(reference_groups)
        prod_share = prod_counts.get(group, 0) / len(production_groups)
        changes[group] = prod_share - ref_share
        tvd += abs(prod_share - ref_share)
    tvd /= 2.0
    max_change = max((abs(v) for v in changes.values()), default=0.0)
    return DriftReport(
        reference_n=len(reference_groups),
        production_n=len(production_groups),
        per_group_share_change=changes,
        total_variation_distance=tvd,
        max_abs_share_change=max_change,
        flagged=max_change >= flag_threshold,
    )


@dataclass(frozen=True)
class LabelShift:
    group: str
    reference_rate: float | None
    production_rate: float | None
    change: float | None
    flagged: bool


def label_rate_shift(
    reference_labels: list[int],
    reference_groups: list[str],
    production_labels: list[int],
    production_groups: list[str],
    flag_threshold: float = 0.05,
) -> list[LabelShift]:
    """Per-group positive-label rate change between reference and production."""
    if len(reference_labels) != len(reference_groups) or len(production_labels) != len(production_groups):
        raise ValidationError("Labels and groups length mismatch")

    def _rates(labels: list[int], groups: list[str]) -> dict[str, float]:
        totals: Counter = Counter()
        positives: Counter = Counter()
        for label, group in zip(labels, groups):
            totals[str(group)] += 1
            if int(label) == 1:
                positives[str(group)] += 1
        return {g: positives[g] / n for g, n in totals.items()}

    ref_rates = _rates(reference_labels, reference_groups)
    prod_rates = _rates(production_labels, production_groups)
    out: list[LabelShift] = []
    for group in sorted(set(ref_rates) | set(prod_rates)):
        ref = ref_rates.get(group)
        prod = prod_rates.get(group)
        change = None if ref is None or prod is None else prod - ref
        out.append(
            LabelShift(
                group=group,
                reference_rate=ref,
                production_rate=prod,
                change=change,
                flagged=change is not None and abs(change) >= flag_threshold,
            )
        )
    return out


@dataclass
class DataQualityReport:
    n_rows: int
    missing_rate_by_field: dict[str, float] = field(default_factory=dict)
    missing_rate_by_field_and_group: dict[str, dict[str, float]] = field(default_factory=dict)
    out_of_range_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "missing_rate_by_field": {k: round(v, 4) for k, v in self.missing_rate_by_field.items()},
            "missing_rate_by_field_and_group": {
                f: {g: round(v, 4) for g, v in d.items()}
                for f, d in self.missing_rate_by_field_and_group.items()
            },
            "out_of_range_counts": dict(self.out_of_range_counts),
        }


def data_quality_screen(
    rows: list[dict],
    groups: list[str] | None = None,
    numeric_bounds: dict[str, tuple[float, float]] | None = None,
) -> DataQualityReport:
    """Missing-rate (overall and per group) and out-of-range screens."""
    if not rows:
        raise ValidationError("Empty input to data-quality screen")
    if groups is not None and len(groups) != len(rows):
        raise ValidationError("Groups length mismatch with rows")
    fields = sorted({key for row in rows for key in row.keys()})
    missing: dict[str, float] = {}
    for field_name in fields:
        missing[field_name] = sum(1 for row in rows if row.get(field_name) is None) / len(rows)
    per_group: dict[str, dict[str, float]] = {}
    if groups is not None:
        for field_name in fields:
            buckets: dict[str, list[int]] = {}
            for row, group in zip(rows, groups):
                buckets.setdefault(str(group), []).append(1 if row.get(field_name) is None else 0)
            per_group[field_name] = {
                g: sum(values) / len(values) for g, values in buckets.items()
            }
    out_of_range: dict[str, int] = {}
    for field_name, (low, high) in (numeric_bounds or {}).items():
        count = 0
        for row in rows:
            value = row.get(field_name)
            if value is not None and not (low <= float(value) <= high):
                count += 1
        out_of_range[field_name] = count
    return DataQualityReport(
        n_rows=len(rows),
        missing_rate_by_field=missing,
        missing_rate_by_field_and_group=per_group,
        out_of_range_counts=out_of_range,
    )
