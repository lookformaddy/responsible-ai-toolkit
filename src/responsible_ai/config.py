"""Environment-driven configuration with validated defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

from responsible_ai.errors import ValidationError


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValidationError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValidationError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Runtime settings; see ``.env.example`` for every variable."""

    min_group_size: int = 30
    drift_flag_threshold: float = 0.05
    label_shift_flag_threshold: float = 0.05
    host: str = "127.0.0.1"
    port: int = 8000
    audit_log_path: str = "data/audit.jsonl"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            min_group_size=_env_int("RAI_MIN_GROUP_SIZE", 30),
            drift_flag_threshold=_env_float("RAI_DRIFT_FLAG_THRESHOLD", 0.05),
            label_shift_flag_threshold=_env_float("RAI_LABEL_SHIFT_FLAG_THRESHOLD", 0.05),
            host=os.environ.get("RAI_HOST", "127.0.0.1"),
            port=_env_int("RAI_PORT", 8000),
            audit_log_path=os.environ.get("RAI_AUDIT_LOG_PATH", "data/audit.jsonl"),
        )
