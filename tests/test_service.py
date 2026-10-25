"""API suite: auto-skips without FastAPI; full suite in a venv with it."""

import pytest

from responsible_ai.service import FASTAPI_AVAILABLE, create_app

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE, reason="FastAPI not installed on system python"
)


@pytest.fixture()
def client():
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:  # httpx missing, etc.
        pytest.skip(f"TestClient unavailable: {exc}")
    return TestClient(create_app())


PAYLOAD = {
    "y_true": [1, 1, 0, 0, 1, 0, 0, 0],
    "y_pred": [1, 1, 1, 0, 0, 0, 0, 0],
    "groups": ["A", "A", "A", "A", "B", "B", "B", "B"],
    "min_group_size": 1,
}


def test_health_and_fairness(client):
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/fairness", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json()["demographic_parity_difference"] == pytest.approx(0.75)


def test_report_includes_gates(client):
    response = client.post("/report", json=PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "report" in body and "gates" in body
    assert any("FAIL" in g for g in body["gates"])


def test_model_card_and_audit_verify(client):
    response = client.post(
        "/model-card",
        json={
            "model_name": "demo",
            "version": "1.0",
            "author": "a",
            "date": "2026-10-25",
            "intended_use": "demo",
        },
    )
    assert response.status_code == 200
    assert "Model Card: demo" in response.json()["markdown"]
    verify = client.get("/audit/verify").json()
    assert verify["ok"] is True
    assert verify["entries"] >= 1


def test_drift_endpoint(client):
    response = client.post(
        "/drift",
        json={"reference_groups": ["a", "a", "b"], "production_groups": ["a", "b", "b"]},
    )
    assert response.status_code == 200
    # API rounds rates to 4 decimals; allow for that rounding.
    assert response.json()["total_variation_distance"] == pytest.approx(1 / 3, abs=1e-3)
