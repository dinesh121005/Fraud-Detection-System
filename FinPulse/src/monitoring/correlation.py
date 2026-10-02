"""FinPulse R6.1 — Transaction Correlation Context.

Propagates request and transaction correlation metadata across all stages of
the FinPulse execution pipeline:
- transaction_id: Business idempotency key
- customer_id: Entity identifier
- event_id: Canonical deterministic event identifier
- model_version: Frozen ML model release identifier ("finpulse-v3")
- feature_schema_version: Frozen feature schema release identifier ("2.0")
"""

import contextvars
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any

_CORRELATION_VAR: contextvars.ContextVar[Optional["CorrelationContext"]] = contextvars.ContextVar(
    "finpulse_correlation_context", default=None
)


@dataclass(frozen=True)
class CorrelationContext:
    """Immutable transaction correlation context container."""
    transaction_id: str
    customer_id: str
    event_id: str = ""
    model_version: str = "finpulse-v3"
    feature_schema_version: str = "2.0"
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to standard dictionary representation."""
        data = {
            "transaction_id": self.transaction_id,
            "customer_id": self.customer_id,
            "event_id": self.event_id,
            "model_version": self.model_version,
            "feature_schema_version": self.feature_schema_version,
        }
        if self.extra_metadata:
            data["metadata"] = self.extra_metadata
        return data


def set_correlation_context(
    transaction_id: str,
    customer_id: str,
    event_id: str = "",
    model_version: str = "finpulse-v3",
    feature_schema_version: str = "2.0",
    **extra: Any
) -> CorrelationContext:
    """Set current execution correlation context in ContextVar."""
    ctx = CorrelationContext(
        transaction_id=str(transaction_id),
        customer_id=str(customer_id),
        event_id=str(event_id),
        model_version=str(model_version),
        feature_schema_version=str(feature_schema_version),
        extra_metadata=extra
    )
    _CORRELATION_VAR.set(ctx)
    return ctx


def get_correlation_context() -> Optional[CorrelationContext]:
    """Retrieve current execution correlation context."""
    return _CORRELATION_VAR.get()


def clear_correlation_context() -> None:
    """Reset current execution correlation context."""
    _CORRELATION_VAR.set(None)
