"""Slice reports, intersections, worst-slice flagging, and gates."""

import pytest

from responsible_ai.audit_report import (
    DEFAULT_GATES,
    Gate,
    build_slice_report,
    evaluate_gates,
    worst_slice,
)
from responsible_ai.errors import ValidationError
from responsible_ai.fairness import summarize_fairness

Y_TRUE = [1, 1, 0, 0, 1, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 0]
GROUPS = ["A", "A", "A", "A", "B", "B", "B", "B"]
BANDS = ["young", "old", "young", "old", "young", "old", "young", "old"]


def test_slice_and_intersection_counts():
    report = build_slice_report(
        Y_TRUE, Y_PRED,
        {"group": GROUPS, "age_band": BANDS},
        intersect=("group", "age_band"),
        min_group_size=1,
    )
    assert len(report.slices) == 4  # 2 groups + 2 bands
    assert len(report.intersections) == 4  # 2 x 2
    assert report.overall["demographic_parity_difference"] == pytest.approx(0.75)


def test_small_slices_flagged_not_dropped():
    report = build_slice_report(
        Y_TRUE, Y_PRED, {"group": GROUPS}, min_group_size=30
    )
    assert all(s.flagged_small for s in report.slices)
    assert len(report.slices) == 2


def test_worst_slice_picks_furthest_tpr():
    report = build_slice_report(
        Y_TRUE, Y_PRED, {"group": GROUPS}, min_group_size=1
    )
    worst = worst_slice(report.slices, "true_positive_rate")
    assert worst is not None
    # TPRs are 1.0 (A) and 0.0 (B); mean 0.5 -- both tie; either is valid,
    # but the winner must be one of the two real slices.
    assert worst.slice_id in ("group=A", "group=B")


def test_gates_pass_fail_and_undefined():
    summary = summarize_fairness(Y_TRUE, Y_PRED, GROUPS, min_group_size=1).as_dict()
    results = evaluate_gates(summary, DEFAULT_GATES)
    by_name = {r.gate.name: r for r in results}
    assert by_name["demographic-parity"].passed is False  # 0.75 > 0.10
    assert by_name["disparate-impact"].passed is False  # 0.0 < 0.80
    assert by_name["equal-opportunity"].passed is False  # 1.0 > 0.10
    # Predictive parity is undefined (only one defined PPV) -> cannot pass.
    assert by_name["predictive-parity"].passed is None
    assert "UNDEFINED" in by_name["predictive-parity"].message


def test_custom_gate_operator():
    summary = {"disparate_impact": 0.9}
    result = evaluate_gates(summary, [Gate("di-floor", "disparate_impact", ">=", 0.8)])[0]
    assert result.passed is True
    with pytest.raises(ValidationError):
        Gate("bad", "disparate_impact", "==", 0.5).check(0.9)


def test_attribute_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        build_slice_report(Y_TRUE, Y_PRED, {"group": ["A", "B"]})
