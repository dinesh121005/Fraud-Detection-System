"""FinPulse R6.1 — Structured JSON Logging & Sensitive Data Redaction.

Provides standardized, operational audit logging with:
- Correlation context auto-injection (transaction_id, customer_id, event_id)
- Frozen model and feature schema metadata
- Strict sensitive data protection and redaction (passwords, tokens, PANs, CVVs)
- JSON format output for log aggregators (ELK, Loki, CloudWatch)
"""

import json
import logging
import datetime
from typing import Dict, Any, Optional

from .correlation import get_correlation_context

# Sensitive field keys that must be scrubbed / redacted
SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "authorization",
    "auth_token",
    "access_token",
    "card_number",
    "pan",
    "cvv",
    "cvc",
    "pin",
    "ssn",
    "credit_card",
}


def sanitize_log_payload(payload: Any) -> Any:
    """
    Recursively sanitize dictionaries/iterables to redact sensitive keys.
    Values of sensitive keys are replaced with '[REDACTED]'.
    """
    if isinstance(payload, dict):
        cleaned = {}
        for k, v in payload.items():
            k_lower = str(k).lower()
            if any(sensitive in k_lower for sensitive in SENSITIVE_KEYS):
                cleaned[k] = "[REDACTED]"
            elif isinstance(v, (dict, list)):
                cleaned[k] = sanitize_log_payload(v)
            else:
                cleaned[k] = v
        return cleaned
    elif isinstance(payload, list):
        return [sanitize_log_payload(item) for item in payload]
    return payload


class StructuredJsonFormatter(logging.Formatter):
    """Custom logging formatter that serializes records into structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        ctx = get_correlation_context()

        log_data: Dict[str, Any] = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject correlation context if available
        if ctx is not None:
            log_data["transaction_id"] = ctx.transaction_id
            log_data["customer_id"] = ctx.customer_id
            if ctx.event_id:
                log_data["event_id"] = ctx.event_id
            log_data["model_version"] = ctx.model_version
            log_data["feature_schema_version"] = ctx.feature_schema_version

        # Inject extra fields if passed via extra dict
        if hasattr(record, "structured_data") and isinstance(record.structured_data, dict):
            sanitized_extra = sanitize_log_payload(record.structured_data)
            log_data.update(sanitized_extra)

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class StructuredLogger:
    """Wrapper around standard python logger for structured telemetry events."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_event(
        self,
        level: int,
        message: str,
        **kwargs: Any
    ):
        """Emit structured log event with correlation context."""
        cleaned_extra = sanitize_log_payload(kwargs)
        self.logger.log(level, message, extra={"structured_data": cleaned_extra})

    def info(self, message: str, **kwargs: Any):
        self.log_event(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any):
        self.log_event(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any):
        self.log_event(logging.ERROR, message, **kwargs)

    def debug(self, message: str, **kwargs: Any):
        self.log_event(logging.DEBUG, message, **kwargs)


def get_logger(name: str = "FinPulse") -> StructuredLogger:
    """Factory helper to obtain a StructuredLogger instance."""
    return StructuredLogger(name)
