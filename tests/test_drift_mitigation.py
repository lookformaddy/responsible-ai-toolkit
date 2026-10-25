"""Drift screens (hand-computed) and transparent mitigation aids."""

import pytest

from responsible_ai.drift import (
    data_quality_screen,
    label_rate_shift,
    representation_drift,
)
from responsible_ai.mitigation import (
    adjust_thresholds_per_group,
    reweighting_suggestions,
)


def test_representation_drift_hand_computed():
    # ref: a=2/3, b=1/3; prod: a=1/3, b=2/3
    # changes: a=-1/3, b=+1/3; TVD=(1/3+1/3)/2=1/3
    report = representation_drift(["a", "a", "b"], ["a", "b", "b"])
    assert report.per_group_share_change["a"] == pytest.approx(-1 / 3)
    assert report.per_group_share_change["b"] == pytest.approx(1 / 3)
    assert report.total_variation_distance == pytest.approx(1 / 3)
    assert report.flagged is True


def test_representation_drift_no_change_not_flagged():
    report = representation_drift(["a", "b"], ["a", "b"])
    assert report.total_variation_distance == 0.0
    assert report.flagged is False


def test_label_rate_shift_flags_large_moves_only():
    shifts = label_rate_shift(
        [1, 1, 0, 0], ["a", "a", "a", "a"],
        [1, 0, 0, 0], ["a", "a", "a", "a"],
    )
    assert len(shifts) == 1
    assert shifts[0].reference_rate == pytest.approx(0.5)
    assert shifts[0].production_rate == pytest.approx(0.25)
    assert shifts[0].change == pytest.approx(-0.25)
    assert shifts[0].flagged is True


def test_data_quality_screen_missing_and_range():
    rows = [
        {"income": 50.0, "note": None},
        {"income": 200.0, "note": "x"},
        {"income": None, "note": "y"},
    ]
    report = data_quality_screen(
        rows, groups=["a", "a", "b"], numeric_bounds={"income": (0.0, 100.0)}
    )
    assert report.missing_rate_by_field["income"] == pytest.approx(1 / 3)
    assert report.missing_rate_by_field["note"] == pytest.approx(1 / 3)
    assert report.out_of_range_counts["income"] == 1
    assert report.missing_rate_by_field_and_group["income"]["b"] == pytest.approx(1.0)


def test_threshold_adjustment_equalizes_opportunity():
    # Group A positives score high; group B positives score low.
    scores = [0.9, 0.8, 0.2, 0.4, 0.3, 0.1]
    labels = [1, 1, 0, 1, 1, 0]
    groups = ["A", "A", "A", "B", "B", "B"]
    adjustments = adjust_thresholds_per_group(scores, labels, groups)
    by_group = {a.group: a for a in adjustments}
    # A: TPR at 0.5 is 1.0 (target). B: TPR at 0.5 is 0.0; adjustment must
    # lower B's threshold to 0.3 so both positives clear it.
    assert by_group["B"].tpr_before == pytest.approx(0.0)
    assert by_group["B"].tpr_after == pytest.approx(1.0)
    assert by_group["B"].threshold_after == pytest.approx(0.3)
    assert by_group["B"].hit_bound is False
    assert by_group["A"].threshold_after == pytest.approx(0.5)


def test_threshold_bounds_are_respected():
    scores = [0.05, 0.04, 0.9, 0.8]
    labels = [1, 1, 1, 1]
    groups = ["A", "A", "B", "B"]
    # Reference A at 0.5: TPR 0.0. B at 0.5: TPR 1.0; to match 0.0, B's
    # threshold would need to exceed its scores, but bounds cap at 0.9.
    adjustments = adjust_thresholds_per_group(
        scores, labels, groups, lower_bound=0.1, upper_bound=0.9
    )
    by_group = {a.group: a for a in adjustments}
    assert by_group["B"].threshold_after <= 0.9
    assert by_group["B"].hit_bound is True


def test_reweighting_suggestions_direction():
    # Group A: 3 positives / 1 negative; group B: 1 positive / 3 negatives.
    # Overall positive rate 0.5. B positives are under-represented -> weight > 1.
    labels = [1, 1, 1, 0, 1, 0, 0, 0]
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    weights = {(s.group, s.label): s.weight for s in reweighting_suggestions(labels, groups)}
    assert weights[("B", 1)] > 1.0
    assert weights[("A", 1)] < 1.0
    assert weights[("B", 1)] == pytest.approx(0.5 / 0.25)
