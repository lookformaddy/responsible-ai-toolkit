"""Model-agnostic explainability helpers (pure stdlib, deterministic).

* Permutation importance: shuffle one feature at a time (seeded RNG, so
  results are reproducible), measure the drop in a caller-supplied score.
  A negative importance means the model does *better* without the
  feature -- a real signal, not noise to clamp away.
* Partial dependence: for each grid value, set the feature to that value
  for every row and average predictions. This is the textbook PD
  definition, simplified to numeric features; it describes the model's
  average response, not a causal effect, and correlated features make
  the averaged rows unrealistic -- the docstring and README say so.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from responsible_ai.errors import ValidationError

Row = dict[str, float]
PredictFn = Callable[[list[Row]], list[float]]
ScoreFn = Callable[[list[float], list[float]], float]


def _default_score(predictions: list[float], y_true: list[float]) -> float:
    """Accuracy at a 0.5 threshold (binary predictions)."""
    if not y_true:
        raise ValidationError("Empty input to score function")
    correct = sum(1 for p, y in zip(predictions, y_true) if (1 if p >= 0.5 else 0) == int(y))
    return correct / len(y_true)


def permutation_importance(
    predict_fn: PredictFn,
    X: Sequence[Row],
    y_true: Sequence[float],
    features: Sequence[str] | None = None,
    n_repeats: int = 5,
    seed: int = 42,
    score_fn: ScoreFn | None = None,
) -> dict[str, float]:
    """Mean score drop per feature when that feature is shuffled.

    Deterministic: a single seeded ``random.Random`` drives every shuffle,
    so the same inputs always produce the same importances.
    """
    if not X:
        raise ValidationError("Empty input to permutation importance")
    if len(X) != len(y_true):
        raise ValidationError("X and y_true length mismatch")
    score_fn = score_fn or _default_score
    features = list(features) if features is not None else sorted(X[0].keys())
    for feature in features:
        if any(feature not in row for row in X):
            raise ValidationError(f"Feature {feature!r} missing from some rows")
    baseline = score_fn(predict_fn([dict(row) for row in X]), list(y_true))
    rng = random.Random(seed)
    importances: dict[str, float] = {}
    for feature in features:
        column = [row[feature] for row in X]
        drops: list[float] = []
        for _ in range(n_repeats):
            shuffled = list(column)
            rng.shuffle(shuffled)
            perturbed = [dict(row) for row in X]
            for row, value in zip(perturbed, shuffled):
                row[feature] = value
            drops.append(baseline - score_fn(predict_fn(perturbed), list(y_true)))
        importances[feature] = sum(drops) / len(drops)
    return importances


def partial_dependence(
    predict_fn: PredictFn,
    X: Sequence[Row],
    feature: str,
    grid: Sequence[float],
) -> list[tuple[float, float]]:
    """Average model output as ``feature`` sweeps each grid value."""
    if not X:
        raise ValidationError("Empty input to partial dependence")
    if any(feature not in row for row in X):
        raise ValidationError(f"Feature {feature!r} missing from some rows")
    out: list[tuple[float, float]] = []
    for value in grid:
        perturbed = [dict(row) for row in X]
        for row in perturbed:
            row[feature] = value
        predictions = predict_fn(perturbed)
        out.append((value, sum(predictions) / len(predictions)))
    return out
