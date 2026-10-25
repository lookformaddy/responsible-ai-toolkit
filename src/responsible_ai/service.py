"""Optional FastAPI surface. Import-guarded: ``create_app()`` returns None
when FastAPI is not installed, so the library, CLI, and non-API tests
work without it. No authentication is built in -- this is a demo surface;
production deployment adds auth and a real store in front of it."""

from __future__ import annotations

from responsible_ai.audit_log import AuditLog
from responsible_ai.audit_report import (
    DEFAULT_GATES,
    build_slice_report,
    evaluate_gates,
)
from responsible_ai.drift import representation_drift
from responsible_ai.fairness import summarize_fairness
from responsible_ai.model_card import ModelCard

try:
    from fastapi import FastAPI
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without FastAPI
    FastAPI = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    _FASTAPI_AVAILABLE = False

FASTAPI_AVAILABLE = _FASTAPI_AVAILABLE


if _FASTAPI_AVAILABLE:

    class EvaluationRequest(BaseModel):
        y_true: list[int]
        y_pred: list[int]
        groups: list[str]
        attributes: dict[str, list[str]] | None = None
        min_group_size: int = 30

    class ModelCardRequest(BaseModel):
        model_name: str
        version: str
        author: str
        date: str
        intended_use: str
        training_data_summary: str = ""
        evaluation_metrics: dict[str, float] = {}
        fairness_results: dict = {}

    class DriftRequest(BaseModel):
        reference_groups: list[str]
        production_groups: list[str]
        flag_threshold: float = 0.05


def create_app():
    """Return a FastAPI app, or None when FastAPI is not installed."""
    if not _FASTAPI_AVAILABLE:
        return None
    app = FastAPI(title="responsible-ai-toolkit")
    log = AuditLog()

    @app.get("/health")
    def health():
        return {"status": "ok", "entries": len(log)}

    @app.post("/fairness")
    def fairness(payload: EvaluationRequest):
        summary = summarize_fairness(
            payload.y_true, payload.y_pred, payload.groups,
            min_group_size=payload.min_group_size,
        )
        result = summary.as_dict()
        log.append("fairness_evaluation", {"n": len(payload.y_true)})
        return result

    @app.post("/report")
    def report(payload: EvaluationRequest):
        attributes = payload.attributes or {"group": payload.groups}
        result = build_slice_report(
            payload.y_true, payload.y_pred, attributes,
            min_group_size=payload.min_group_size,
        ).as_dict()
        gates = evaluate_gates(result["overall"], DEFAULT_GATES)
        log.append("bias_audit_report", {"slices": len(result["slices"])})
        return {"report": result, "gates": [g.message for g in gates]}

    @app.post("/model-card")
    def model_card(payload: ModelCardRequest):
        card = ModelCard(**payload.model_dump())
        log.append("model_card_generated", {"model": card.model_name, "version": card.version})
        return {"markdown": card.to_markdown(), "json": card.to_dict()}

    @app.post("/drift")
    def drift(payload: DriftRequest):
        result = representation_drift(
            payload.reference_groups, payload.production_groups, payload.flag_threshold
        )
        log.append("drift_check", {"flagged": result.flagged})
        return result.as_dict()

    @app.get("/audit/verify")
    def audit_verify():
        ok, first_bad = log.verify()
        return {"ok": ok, "first_bad_index": first_bad, "entries": len(log)}

    return app
