"""Explainability: informative features rank above noise; PD is monotone
for a monotone model; results are deterministic across runs."""

from responsible_ai.explainability import partial_dependence, permutation_importance


def _predict(rows):
    # Strong signal on x1, none on x2.
    return [2.0 * row["x1"] + 0.0 * row["x2"] for row in rows]


X = [
    {"x1": 0.0, "x2": 1.0},
    {"x1": 1.0, "x2": 0.0},
    {"x1": 0.5, "x2": 0.5},
    {"x1": 1.0, "x2": 1.0},
    {"x1": 0.0, "x2": 0.0},
    {"x1": 0.25, "x2": 0.75},
]
Y = [0, 1, 1, 1, 0, 0]


def test_permutation_importance_ranks_signal_above_noise():
    importances = permutation_importance(_predict, X, Y, n_repeats=5, seed=42)
    assert importances["x1"] > importances["x2"]
    assert importances["x1"] > 0


def test_permutation_importance_is_deterministic():
    first = permutation_importance(_predict, X, Y, seed=123)
    second = permutation_importance(_predict, X, Y, seed=123)
    assert first == second
    different = permutation_importance(_predict, X, Y, seed=999)
    # Different seeds may still agree on this toy data; determinism of
    # the same seed is the contract under test, so only assert equality.
    assert different["x1"] >= 0


def test_partial_dependence_monotone_for_monotone_model():
    pd = partial_dependence(_predict, X, "x1", [0.0, 0.5, 1.0])
    values = [mean for _, mean in pd]
    assert values == sorted(values)
    assert values[0] == 0.0 and values[-1] == 2.0


def test_partial_dependence_flat_for_irrelevant_feature():
    pd = partial_dependence(_predict, X, "x2", [0.0, 1.0])
    assert pd[0][1] == pd[1][1]
