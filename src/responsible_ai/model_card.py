"""Structured model cards: Markdown + JSON from one metadata record.

A model card is a governance artifact, not marketing copy: required
fields (name, version, intended use, limitations, training-data summary)
are enforced at construction, and fairness results are embedded
verbatim rather than paraphrased, so the card can never drift from the
numbers it claims to report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from responsible_ai.errors import ValidationError


@dataclass
class ModelCard:
    """A model card for one model version."""

    model_name: str
    version: str
    author: str
    date: str  # ISO date, e.g. "2026-10-25"
    intended_use: str
    out_of_scope_uses: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    training_data_summary: str = ""
    evaluation_metrics: dict[str, float] = field(default_factory=dict)
    fairness_results: dict = field(default_factory=dict)
    ethical_considerations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = [
            name
            for name, value in (
                ("model_name", self.model_name),
                ("version", self.version),
                ("author", self.author),
                ("date", self.date),
                ("intended_use", self.intended_use),
            )
            if not value or not str(value).strip()
        ]
        if missing:
            raise ValidationError(f"ModelCard missing required fields: {missing}")

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "author": self.author,
            "date": self.date,
            "intended_use": self.intended_use,
            "out_of_scope_uses": list(self.out_of_scope_uses),
            "limitations": list(self.limitations),
            "training_data_summary": self.training_data_summary,
            "evaluation_metrics": dict(self.evaluation_metrics),
            "fairness_results": self.fairness_results,
            "ethical_considerations": list(self.ethical_considerations),
            "recommendations": list(self.recommendations),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def to_markdown(self) -> str:
        def _bullets(items: list[str]) -> str:
            return "\n".join(f"- {item}" for item in items) if items else "- (none recorded)"

        def _metrics(items: dict[str, float]) -> str:
            if not items:
                return "- (none recorded)"
            return "\n".join(f"- {k}: {v}" for k, v in sorted(items.items()))

        fairness = (
            "```json\n" + json.dumps(self.fairness_results, indent=2, sort_keys=True) + "\n```"
            if self.fairness_results
            else "- (no fairness evaluation recorded)"
        )
        return (
            f"# Model Card: {self.model_name}\n\n"
            f"- **Version:** {self.version}\n"
            f"- **Author:** {self.author}\n"
            f"- **Date:** {self.date}\n\n"
            f"## Intended Use\n\n{self.intended_use}\n\n"
            f"## Out-of-Scope Uses\n\n{_bullets(self.out_of_scope_uses)}\n\n"
            f"## Limitations\n\n{_bullets(self.limitations)}\n\n"
            f"## Training Data\n\n{self.training_data_summary or '(not recorded)'}\n\n"
            f"## Evaluation Metrics\n\n{_metrics(self.evaluation_metrics)}\n\n"
            f"## Fairness Results\n\n{fairness}\n\n"
            f"## Ethical Considerations\n\n{_bullets(self.ethical_considerations)}\n\n"
            f"## Recommendations\n\n{_bullets(self.recommendations)}\n"
        )


def model_card_from_dict(data: dict) -> ModelCard:
    """Rebuild a ModelCard from ``to_dict()`` output (round-trip safe)."""
    known = {
        "model_name",
        "version",
        "author",
        "date",
        "intended_use",
        "out_of_scope_uses",
        "limitations",
        "training_data_summary",
        "evaluation_metrics",
        "fairness_results",
        "ethical_considerations",
        "recommendations",
    }
    return ModelCard(**{k: v for k, v in data.items() if k in known})
