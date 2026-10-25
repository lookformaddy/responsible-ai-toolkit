"""Minimal structured logging: one JSON object per line, no PII in fields
unless the caller puts it there (callers must pass group labels and
counts, never raw personal data)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone


def get_logger(name: str = "responsible_ai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_event(logger: logging.Logger, event: str, **fields: object) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **{k: v for k, v in fields.items()},
    }
    logger.info(json.dumps(payload, sort_keys=True))
