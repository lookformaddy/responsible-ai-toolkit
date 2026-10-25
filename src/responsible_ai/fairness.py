"""Fairness metrics computed from predictions + labels, per protected group.

All metrics are pure standard library. Rates that are undefined for a
group (e.g. true positive rate when the group has no positive labels)
are ``None`` rather than silently imputed as 0, so a report can never
claim a group is fair on a metric that does not exist for it.

Conventions
-----------
* ``y_true`` / ``y_pred`` are binary (0/1) sequences of equal length.
* ``groups`` is a parallel sequence of group labels (strings).
* The *reference group* defaults to the largest group; every difference
  metric is ``max - min`` across groups unless a reference is given, in
  which case it is ``max |group - reference|``. Ratio metrics use
  ``min / max`` (the disparate-impact convention).
* The four-fifths rule flags disparate impact below 0.8 (EEOC guidance);
  it is a screening flag, not a legal finding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from responsible_ai.errors import ValidationError

MIN_GROUP_SIZE_DEFAULT = 30


def _check_lengths(y_true, y_pred, groups) -> None:
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValidationError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"groups={len(groups)}"
        )
    if not y_true:
        raise ValidationError("Empty input: no rows to measure")
    for name, seq in (("y_true", y_true), ("y_pred", y_pred)):
        for value in seq:
            if value not in (0, 1, True, False):
                raise ValidationError(f"{name} must be binary (0/1), got {value!r}")


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def rate_confidence_interval(
    successes: int, n: int, z: float = 1.96
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial rate (95% by default).

    Returns ``None`` for an empty sample. The Wilson interval is used
    instead of the normal approximation because it behaves sensibly for
    small groups and rates near 0 or 1 -- exactly the cases fairness
    work cares about.
    """
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class GroupMetrics:
    """Confusion-matrix counts and rates for one group."""

    group: str
    n: int
    positives: int
    predicted_positives: int
    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float
    true_positive_rate: float | None  # recall / equal opportunity
    false_positive_rate: float | None
    positive_predictive_value: float | None  # precision / predictive parity
    negative_predictive_value: float | None
    positive_rate: float  # selection rate
    base_rate: float  # prevalence
    warnings: tuple[str, ...] = ()

    @property
    def is_small(self) -> bool:
        return any(w.startswith("small-group") for w in self.warnings)


def _group_metrics(
    group: str,
    y_true: list[int],
    y_pred: list[int],
    min_group_size: int,
) -> GroupMetrics:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    n = len(y_true)
    warnings: list[str] = []
    if n < min_group_size:
        warnings.append(
            f"small-group: n={n} below min_group_size={min_group_size}; "
            "rates are noisy -- treat as a screen, not a verdict"
        )
    if sum(y_true) == 0:
        warnings.append("no-positive-labels: TPR undefined for this group")
    if sum(y_true) == n:
        warnings.append("no-negative-labels: FPR undefined for this group")
    if sum(y_pred) == 0:
        warnings.append("no-positive-predictions: PPV undefined for this group")
    return GroupMetrics(
        group=group,
        n=n,
        positives=sum(y_true),
        predicted_positives=sum(y_pred),
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        accuracy=(tp + tn) / n if n else 0.0,
        true_positive_rate=_rate(tp, tp + fn),
        false_positive_rate=_rate(fp, fp + tn),
        positive_predictive_value=_rate(tp, tp + fp),
        negative_predictive_value=_rate(tn, tn + fn),
        positive_rate=sum(y_pred) / n if n else 0.0,
        base_rate=sum(y_true) / n if n else 0.0,
        warnings=tuple(warnings),
    )


def compute_group_metrics(
    y_true: list[int],
    y_pred: list[int],
    groups: list[str],
    min_group_size: int = MIN_GROUP_SIZE_DEFAULT,
) -> dict[str, GroupMetrics]:
    """Per-group confusion metrics. Groups are returned in first-seen order."""
    _check_lengths(y_true, y_pred, groups)
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    buckets: dict[str, tuple[list[int], list[int]]] = {}
    for y, p, g in zip(y_true, y_pred, groups):
        buckets.setdefault(str(g), ([], []))
        buckets[str(g)][0].append(y)
        buckets[str(g)][1].append(p)
    return {
        g: _group_metrics(g, ys, ps, min_group_size)
        for g, (ys, ps) in buckets.items()
    }


def _spread(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    if len(defined) < 2:
        return None
    return max(defined) - min(defined)


def _ratio(values: list[float | None]) -> float | None:
    defined = [v for v in values if v is not None]
    if len(defined) < 2 or max(defined) == 0:
        return None
    return min(defined) / max(defined)


@dataclass(frozen=True)
class FairnessSummary:
    """Cross-group fairness summary for one evaluation."""

    per_group: dict[str, GroupMetrics]
    reference_group: str
    demographic_parity_difference: float
    demographic_parity_ratio: float | None
    disparate_impact: float | None
    four_fifths_rule_violation: bool | None
    equal_opportunity_difference: float | None
    equalized_odds_difference: float | None
    predictive_parity_difference: float | None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        def _fmt(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "reference_group": self.reference_group,
            "demographic_parity_difference": _fmt(self.demographic_parity_difference),
            "demographic_parity_ratio": _fmt(self.demographic_parity_ratio),
            "disparate_impact": _fmt(self.disparate_impact),
            "four_fifths_rule_violation": self.four_fifths_rule_violation,
            "equal_opportunity_difference": _fmt(self.equal_opportunity_difference),
            "equalized_odds_difference": _fmt(self.equalized_odds_difference),
            "predictive_parity_difference": _fmt(self.predictive_parity_difference),
            "per_group": {
                g: {
                    "n": m.n,
                    "accuracy": round(m.accuracy, 4),
                    "tpr": _fmt(m.true_positive_rate),
                    "fpr": _fmt(m.false_positive_rate),
                    "ppv": _fmt(m.positive_predictive_value),
                    "positive_rate": round(m.positive_rate, 4),
                    "base_rate": round(m.base_rate, 4),
                    "warnings": list(m.warnings),
                }
                for g, m in self.per_group.items()
            },
            "warnings": list(self.warnings),
        }


def summarize_fairness(
    y_true: list[int],
    y_pred: list[int],
    groups: list[str],
    reference_group: str | None = None,
    min_group_size: int = MIN_GROUP_SIZE_DEFAULT,
) -> FairnessSummary:
    """Compute the standard fairness metric set across groups.

    * Demographic parity difference: max - min selection (positive) rate.
    * Demographic parity ratio / disparate impact: min / max selection rate.
    * Equal opportunity difference: max - min true positive rate.
    * Equalized odds difference: max of the TPR spread and the FPR spread.
    * Predictive parity difference: max - min positive predictive value.
    """
    per_group = compute_group_metrics(y_true, y_pred, groups, min_group_size)
    if reference_group is None:
        reference_group = max(per_group.values(), key=lambda m: m.n).group
    elif reference_group not in per_group:
        raise ValidationError(f"reference_group {reference_group!r} not present")
    tprs = [m.true_positive_rate for m in per_group.values()]
    fprs = [m.false_positive_rate for m in per_group.values()]
    ppvs = [m.positive_predictive_value for m in per_group.values()]
    rates = [m.positive_rate for m in per_group.values()]
    di = _ratio(rates)
    eod_tpr = _spread(tprs)
    eod_fpr = _spread(fprs)
    equalized = None
    if eod_tpr is not None and eod_fpr is not None:
        equalized = max(eod_tpr, eod_fpr)
    elif eod_tpr is not None:
        equalized = eod_tpr
    elif eod_fpr is not None:
        equalized = eod_fpr
    warnings: list[str] = []
    for m in per_group.values():
        warnings.extend(f"{m.group}: {w}" for w in m.warnings)
    if di is not None and di < 0.8:
        warnings.append(
            f"four-fifths-rule: disparate impact {di:.3f} < 0.8 -- screen flagged, "
            "investigate before concluding discrimination"
        )
    return FairnessSummary(
        per_group=per_group,
        reference_group=reference_group,
        demographic_parity_difference=max(rates) - min(rates),
        demographic_parity_ratio=di,
        disparate_impact=di,
        four_fifths_rule_violation=None if di is None else di < 0.8,
        equal_opportunity_difference=eod_tpr,
        equalized_odds_difference=equalized,
        predictive_parity_difference=_spread(ppvs),
        warnings=tuple(warnings),
    )
