"""Responsible AI toolkit: fairness metrics, bias audits, explainability,
model cards, tamper-evident audit logging, drift screens, and transparent
post-processing mitigation aids. Pure standard library core; FastAPI is
optional (see ``service.py``)."""

from responsible_ai.fairness import (
    GroupMetrics,
    FairnessSummary,
    compute_group_metrics,
    rate_confidence_interval,
    summarize_fairness,
)
from responsible_ai.audit_report import (
    Gate,
    GateResult,
    SliceMetric,
    SliceReport,
    build_slice_report,
    evaluate_gates,
    worst_slice,
)
from responsible_ai.explainability import (
    partial_dependence,
    permutation_importance,
)
from responsible_ai.model_card import ModelCard, model_card_from_dict
from responsible_ai.audit_log import AuditEntry, AuditLog, hash_entry
from responsible_ai.drift import (
    DataQualityReport,
    DriftReport,
    data_quality_screen,
    label_rate_shift,
    representation_drift,
)
from responsible_ai.mitigation import (
    ThresholdAdjustment,
    adjust_thresholds_per_group,
    reweighting_suggestions,
)

__all__ = [
    "GroupMetrics",
    "FairnessSummary",
    "compute_group_metrics",
    "rate_confidence_interval",
    "summarize_fairness",
    "Gate",
    "GateResult",
    "SliceMetric",
    "SliceReport",
    "build_slice_report",
    "evaluate_gates",
    "worst_slice",
    "partial_dependence",
    "permutation_importance",
    "ModelCard",
    "model_card_from_dict",
    "AuditEntry",
    "AuditLog",
    "hash_entry",
    "DataQualityReport",
    "DriftReport",
    "data_quality_screen",
    "label_rate_shift",
    "representation_drift",
    "ThresholdAdjustment",
    "adjust_thresholds_per_group",
    "reweighting_suggestions",
]

__version__ = "0.1.0"
