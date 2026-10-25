"""Hand-computed fairness cases (verified by hand, not by the code itself).

Group A: y=[1,1,0,0], p=[1,1,1,0] -> tp=2, fp=1, tn=1, fn=0
  accuracy=0.75, TPR=1.0, FPR=0.5, PPV=2/3, positive_rate=0.75
Group B: y=[1,0,0,0], p=[0,0,0,0] -> tp=0, fp=0, tn=3, fn=1
  accuracy=0.75, TPR=0.0, FPR=0.0, PPV=None (no positive predictions),
  positive_rate=0.0
Summary: DPD=0.75, disparate impact=0.0 (four-fifths violation),
  equal opportunity difference=1.0, equalized odds=1.0,
  predictive parity difference=None (only one defined PPV).
"""

import pytest

from responsible_ai.errors import ValidationError
from responsible_ai.fairness import (
    compute_group_metrics,
    rate_confidence_interval,
    summarize_fairness,
)

Y_TRUE = [1, 1, 0, 0, 1, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 0, 0, 0]
GROUPS = ["A", "A", "A", "A", "B", "B", "B", "B"]


def test_hand_computed_group_metrics():
    metrics = compute_group_metrics(Y_TRUE, Y_PRED, GROUPS, min_group_size=30)
    a, b = metrics["A"], metrics["B"]
    assert (a.tp, a.fp, a.tn, a.fn) == (2, 1, 1, 0)
    assert a.accuracy == pytest.approx(0.75)
    assert a.true_positive_rate == pytest.approx(1.0)
    assert a.false_positive_rate == pytest.approx(0.5)
    assert a.positive_predictive_value == pytest.approx(2 / 3)
    assert a.positive_rate == pytest.approx(0.75)
    assert (b.tp, b.fp, b.tn, b.fn) == (0, 0, 3, 1)
    assert b.true_positive_rate == pytest.approx(0.0)
    assert b.false_positive_rate == pytest.approx(0.0)
    assert b.positive_predictive_value is None
    assert b.positive_rate == pytest.approx(0.0)


def test_hand_computed_summary():
    summary = summarize_fairness(Y_TRUE, Y_PRED, GROUPS, min_group_size=30)
    assert summary.demographic_parity_difference == pytest.approx(0.75)
    assert summary.disparate_impact == pytest.approx(0.0)
    assert summary.four_fifths_rule_violation is True
    assert summary.equal_opportunity_difference == pytest.approx(1.0)
    assert summary.equalized_odds_difference == pytest.approx(1.0)
    assert summary.predictive_parity_difference is None
    # Both groups are small (n=4 < 30): warnings must surface, not hide.
    assert any("small-group" in w for w in summary.warnings)


def test_perfectly_fair_case_has_zero_differences():
    y = [1, 0, 1, 0, 1, 0, 1, 0]
    p = [1, 0, 1, 0, 1, 0, 1, 0]
    g = ["A", "A", "A", "A", "B", "B", "B", "B"]
    summary = summarize_fairness(y, p, g, min_group_size=1)
    assert summary.demographic_parity_difference == 0.0
    assert summary.disparate_impact == pytest.approx(1.0)
    assert summary.four_fifths_rule_violation is False
    assert summary.equal_opportunity_difference == 0.0
    assert summary.equalized_odds_difference == 0.0


def test_reference_group_validation():
    with pytest.raises(ValidationError):
        summarize_fairness(Y_TRUE, Y_PRED, GROUPS, reference_group="missing")


def test_length_mismatch_rejected():
    with pytest.raises(ValidationError):
        compute_group_metrics([1, 0], [1], ["A", "B"])


def test_non_binary_rejected():
    with pytest.raises(ValidationError):
        compute_group_metrics([1, 2], [1, 0], ["A", "B"])


def test_wilson_interval_contains_rate_and_stays_in_bounds():
    low, high = rate_confidence_interval(8, 10)
    assert 0.0 <= low <= 0.8 <= high <= 1.0
    assert rate_confidence_interval(0, 0) is None
    # Small samples get wide intervals; large samples get narrow ones.
    small = rate_confidence_interval(1, 2)
    large = rate_confidence_interval(500, 1000)
    assert (small[1] - small[0]) > (large[1] - large[0])
