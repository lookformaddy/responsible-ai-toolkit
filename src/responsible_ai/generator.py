"""Deterministic synthetic data for demos and tests.

Generic domains only (a fictional loan-eligibility screen and a
fictional content-moderation queue). Groups are synthetic labels
("group_a" / "group_b" / "group_c") with a deliberately skewed model so
the fairness metrics have something real to show: the demo model
under-predicts positives for one group. Nothing here resembles any real
person, customer, or employer dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class SyntheticDataset:
    y_true: list[int]
    y_pred: list[int]
    scores: list[float]
    groups: list[str]
    age_bands: list[str]
    features: list[dict[str, float]]


def generate_synthetic(
    n: int = 600,
    seed: int = 7,
) -> SyntheticDataset:
    """Generate a synthetic evaluation set with a known group skew."""
    rng = random.Random(seed)
    groups: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    scores: list[float] = []
    age_bands: list[str] = []
    features: list[dict[str, float]] = []
    for i in range(n):
        group = rng.choice(["group_a", "group_a", "group_b", "group_c"])
        age_band = rng.choice(["18-29", "30-49", "50+"])
        income = rng.uniform(20.0, 120.0)
        tenure = rng.uniform(0.0, 20.0)
        # Base rate differs slightly by group (population difference, not bias).
        base = {"group_a": 0.45, "group_b": 0.40, "group_c": 0.38}[group]
        label = 1 if rng.random() < base + 0.15 * (income > 70) else 0
        score = 0.35 + 0.4 * (income / 120.0) + 0.1 * (tenure / 20.0)
        score += rng.uniform(-0.15, 0.15)
        # Deliberate model skew: group_c scores are deflated, so its TPR lags.
        if group == "group_c":
            score -= 0.18
        score = max(0.0, min(1.0, score))
        groups.append(group)
        age_bands.append(age_band)
        y_true.append(label)
        scores.append(round(score, 4))
        y_pred.append(1 if score >= 0.5 else 0)
        features.append({"income": round(income, 2), "tenure": round(tenure, 2), "score": round(score, 4)})
        _ = i
    return SyntheticDataset(
        y_true=y_true,
        y_pred=y_pred,
        scores=scores,
        groups=groups,
        age_bands=age_bands,
        features=features,
    )
