"""Transparent post-processing mitigation aids -- never silent fixes.

These helpers are decision support, not auto-remediation:

* Threshold adjustment: pick a per-group decision threshold (from the
  group's own score distribution) that brings each group's true positive
  rate closest to a target -- by default the reference group's TPR at
  the base threshold -- staying inside caller-set bounds. The result
  reports before/after TPRs and whether the bound bound, so a reviewer
  can see exactly what was traded: equalizing opportunity usually moves
  overall accuracy and false positive rates, and this module does not
  hide that.
* Reweighting suggestions: per (group, label) weights that would make
  each group's label distribution match the overall one, capped so no
  example dominates training. Suggestions only -- applying them is a
  retraining decision for a human to make.

Neither helper mutates a model. Both return data structures a human
reviews and approves before anything changes in production.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from responsible_ai.errors import ValidationError


@dataclass(frozen=True)
class ThresholdAdjustment:
    group: str
    threshold_before: float
    threshold_after: float
    tpr_before: float | None
    tpr_after: float | None
    target_tpr: float | None
    hit_bound: bool

    def as_dict(self) -> dict:
        def _fmt(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "group": self.group,
            "threshold_before": self.threshold_before,
            "threshold_after": round(self.threshold_after, 4),
            "tpr_before": _fmt(self.tpr_before),
            "tpr_after": _fmt(self.tpr_after),
            "target_tpr": _fmt(self.target_tpr),
            "hit_bound": self.hit_bound,
        }


def _tpr_at_threshold(scores: list[float], labels: list[int], threshold: float) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    if not positives:
        return None
    return sum(1 for s in positives if s >= threshold) / len(positives)


def adjust_thresholds_per_group(
    scores: list[float],
    labels: list[int],
    groups: list[str],
    base_threshold: float = 0.5,
    reference_group: str | None = None,
    lower_bound: float = 0.1,
    upper_bound: float = 0.9,
) -> list[ThresholdAdjustment]:
    """Per-group thresholds equalizing opportunity toward the reference TPR.

    Candidate thresholds are the group's own observed scores (plus the
    bounds), so the chosen threshold is always attainable. Ties break
    toward the threshold closest to the base threshold -- the smallest
    intervention that achieves the target.
    """
    if not (len(scores) == len(labels) == len(groups)):
        raise ValidationError("Scores, labels, and groups length mismatch")
    if not scores:
        raise ValidationError("Empty input to threshold adjustment")
    if not (0.0 <= lower_bound < upper_bound <= 1.0):
        raise ValidationError("Bounds must satisfy 0 <= lower < upper <= 1")
    buckets: dict[str, tuple[list[float], list[int]]] = {}
    for score, label, group in zip(scores, labels, groups):
        buckets.setdefault(str(group), ([], []))
        buckets[str(group)][0].append(float(score))
        buckets[str(group)][1].append(int(label))
    if reference_group is None:
        reference_group = max(buckets, key=lambda g: len(buckets[g][0]))
    if reference_group not in buckets:
        raise ValidationError(f"reference_group {reference_group!r} not present")
    ref_scores, ref_labels = buckets[reference_group]
    target = _tpr_at_threshold(ref_scores, ref_labels, base_threshold)
    results: list[ThresholdAdjustment] = []
    for group, (g_scores, g_labels) in buckets.items():
        tpr_before = _tpr_at_threshold(g_scores, g_labels, base_threshold)
        candidates = sorted({lower_bound, upper_bound, base_threshold} | set(g_scores))
        candidates = [c for c in candidates if lower_bound <= c <= upper_bound]
        best: float | None = None
        best_gap = float("inf")
        for candidate in candidates:
            tpr = _tpr_at_threshold(g_scores, g_labels, candidate)
            if tpr is None or target is None:
                continue
            gap = abs(tpr - target)
            # tie-break: threshold closest to base threshold
            if gap < best_gap - 1e-12 or (
                abs(gap - best_gap) <= 1e-12
                and best is not None
                and abs(candidate - base_threshold) < abs(best - base_threshold)
            ):
                best, best_gap = candidate, gap
        chosen = base_threshold if best is None else best
        results.append(
            ThresholdAdjustment(
                group=group,
                threshold_before=base_threshold,
                threshold_after=chosen,
                tpr_before=tpr_before,
                tpr_after=_tpr_at_threshold(g_scores, g_labels, chosen),
                target_tpr=target,
                hit_bound=chosen in (lower_bound, upper_bound),
            )
        )
    return results


@dataclass(frozen=True)
class ReweightingSuggestion:
    group: str
    label: int
    n: int
    weight: float
    capped: bool

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "label": self.label,
            "n": self.n,
            "weight": round(self.weight, 4),
            "capped": self.capped,
        }


def reweighting_suggestions(
    labels: list[int],
    groups: list[str],
    max_weight: float = 5.0,
) -> list[ReweightingSuggestion]:
    """Weights making each group's label mix match the overall mix.

    Weight for (group, label) = overall label rate / group label rate,
    capped at ``max_weight``. A weight above 1 means that cell is
    under-represented relative to the overall population.
    """
    if len(labels) != len(groups):
        raise ValidationError("Labels and groups length mismatch")
    if not labels:
        raise ValidationError("Empty input to reweighting suggestions")
    overall = Counter(int(y) for y in labels)
    total = len(labels)
    overall_rate = {label: count / total for label, count in overall.items()}
    group_totals: Counter = Counter(str(g) for g in groups)
    cell_counts: Counter = Counter((str(g), int(y)) for g, y in zip(groups, labels))
    out: list[ReweightingSuggestion] = []
    for (group, label), count in sorted(cell_counts.items()):
        group_rate = count / group_totals[group]
        raw = overall_rate[label] / group_rate if group_rate > 0 else max_weight
        capped = raw > max_weight
        out.append(
            ReweightingSuggestion(
                group=group,
                label=label,
                n=count,
                weight=min(raw, max_weight),
                capped=capped,
            )
        )
    return out
