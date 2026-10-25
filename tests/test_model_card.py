"""Model cards: required-field enforcement, content, JSON round-trip."""

import json

import pytest

from responsible_ai.errors import ValidationError
from responsible_ai.model_card import ModelCard, model_card_from_dict


def _card():
    return ModelCard(
        model_name="demo-screen",
        version="1.0.0",
        author="M. Example",
        date="2026-10-25",
        intended_use="Screening synthetic demo applications.",
        limitations=["Not validated on real populations."],
        training_data_summary="600 synthetic rows, generated 2026-10-25.",
        evaluation_metrics={"accuracy": 0.82},
        fairness_results={"demographic_parity_difference": 0.25},
    )


def test_markdown_contains_key_sections_and_numbers():
    md = _card().to_markdown()
    assert "# Model Card: demo-screen" in md
    assert "1.0.0" in md and "M. Example" in md
    assert "Not validated on real populations." in md
    assert "0.25" in md  # fairness numbers embedded verbatim, not paraphrased
    assert "Screening synthetic demo applications." in md


def test_json_round_trip():
    card = _card()
    restored = model_card_from_dict(json.loads(card.to_json()))
    assert restored.to_dict() == card.to_dict()


def test_missing_required_fields_rejected():
    with pytest.raises(ValidationError):
        ModelCard(
            model_name="",
            version="1.0.0",
            author="M. Example",
            date="2026-10-25",
            intended_use="",
        )


def test_empty_card_still_renders_placeholders():
    card = ModelCard(
        model_name="x", version="0.1", author="a",
        date="2026-10-25", intended_use="demo",
    )
    assert "(none recorded)" in card.to_markdown()
    assert "(no fairness evaluation recorded)" in card.to_markdown()
