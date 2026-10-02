"""FinPulse R6.1 — Prometheus Metrics & Instrumentation Foundation.

Exposes low-cardinality, production-grade Prometheus metrics for:
- Transaction processing rates and throughput
- Decision outcomes (APPROVE, REVIEW, BLOCK)
- Categorical risk-level distribution (LOW, MEDIUM, HIGH)
- Fraud alert publication routing
- End-to-end and stage-specific processing latency histograms
- Component-level failure and error classification
- Replay / duplicate detection
- Dead-letter queue (DLQ) events
- Multi-signal compounding triggers and rule categories

Cardinality Guardrails:
Strictly forbids transaction_id, customer_id, event_id, timestamps, raw amounts,
free-form strings, or stack traces from appearing in metric labels.
"""

import time
from typing import Optional, List, Dict, Any, Union
from contextlib import contextmanager

from prometheus_client import (
    Counter,
    Histogram,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# -----------------------------------------------------------------------------
# Canonical Bounded Label Domains
# -----------------------------------------------------------------------------
VALID_DECISIONS = {"APPROVE", "REVIEW", "BLOCK"}
VALID_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}
VALID_STATUSES = {"accepted", "rejected", "processed", "error"}
VALID_COMPONENTS = {
    "kafka",
    "redis",
    "feature_engine",
    "model",
    "risk_engine",
    "rule_engine",
    "serialization",
    "publisher",
    "api",
}
VALID_ERROR_TYPES = {
    "connection_error",
    "timeout",
    "validation_error",
    "schema_error",
    "internal_error",
    "not_found",
    "runtime_error",
    "circuit_open",
    "deserialization_error",
}
VALID_RULE_CATEGORIES = {
    "velocity",
    "amount",
    "location_device",
    "authentication",
    "transaction_context",
    "hard_block",
}
VALID_TRIGGERS = {"ml", "velocity", "behavioral", "rules", "anomaly"}
VALID_STAGES = {
    "kafka_consume",
    "redis",
    "feature_engine",
    "model",
    "risk_engine",
    "rule_engine",
    "serialization",
    "kafka_publish",
    "e2e",
}

# Standard latency histogram buckets in milliseconds
LATENCY_BUCKETS_MS = (
    0.25, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0
)
FAST_STAGE_BUCKETS_MS = (
    0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0
)


class FinPulseMetrics:
    """
    Encapsulated Prometheus metrics registry and instruments for FinPulse.
    Supports either the default global REGISTRY or an isolated custom CollectorRegistry.
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry if registry is not None else REGISTRY

        # Helper to safely register or reuse existing metric in the registry
        def _get_or_create(cls, name: str, doc: str, *args, **kwargs):
            try:
                return cls(name, doc, *args, registry=self.registry, **kwargs)
            except ValueError:
                # Metric already registered in this registry
                return self.registry._names_to_collectors.get(name)

        # ---------------------------------------------------------------------
        # 1. Transaction & Decision Counters
        # ---------------------------------------------------------------------
        self.transactions_total = _get_or_create(
            Counter,
            "finpulse_transactions_total",
            "Total transactions accepted for processing by the FinPulse pipeline",
            ["status"]
        )

        self.decisions_total = _get_or_create(
            Counter,
            "finpulse_decisions_total",
            "Total fraud decisions completed by FinPulse, partitioned by decision outcome",
            ["decision"]
        )

        self.fraud_alerts_total = _get_or_create(
            Counter,
            "finpulse_fraud_alerts_total",
            "Total fraud alerts routed to the high-priority fraud-alerts Kafka topic"
        )

        self.risk_levels_total = _get_or_create(
            Counter,
            "finpulse_risk_levels_total",
            "Total fraud decisions classified by categorical risk level",
            ["risk_level"]
        )

        self.hard_blocks_total = _get_or_create(
            Counter,
            "finpulse_hard_blocks_total",
            "Total hard-block overrides triggered by critical safety rules"
        )

        self.replays_total = _get_or_create(
            Counter,
            "finpulse_replays_total",
            "Total replayed or duplicate transactions detected by idempotency layer"
        )

        self.dlq_total = _get_or_create(
            Counter,
            "finpulse_dlq_total",
            "Total dead-letter queue events routed for failed or unprocessable messages"
        )

        self.risk_triggers_total = _get_or_create(
            Counter,
            "finpulse_risk_triggers_total",
            "Total active multi-signal compounding triggers observed",
            ["trigger"]
        )

        self.matched_rules_total = _get_or_create(
            Counter,
            "finpulse_matched_rules_total",
            "Total business rule evaluations matched, partitioned by rule category",
            ["category"]
        )

        # ---------------------------------------------------------------------
        # 2. Error Counters (strictly bounded component & error_type)
        # ---------------------------------------------------------------------
        self.errors_total = _get_or_create(
            Counter,
            "finpulse_errors_total",
            "Total system and component errors encountered in the FinPulse pipeline",
            ["component", "error_type"]
        )

        # ---------------------------------------------------------------------
        # 3. Latency Histograms
        # ---------------------------------------------------------------------
        self.processing_latency_ms = _get_or_create(
            Histogram,
            "finpulse_processing_latency_ms",
            "Total end-to-end transaction processing latency in milliseconds",
            buckets=LATENCY_BUCKETS_MS
        )

        self.model_latency_ms = _get_or_create(
            Histogram,
            "finpulse_model_latency_ms",
            "Model inference, calibration, anomaly, and SHAP explanation latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.redis_latency_ms = _get_or_create(
            Histogram,
            "finpulse_redis_latency_ms",
            "Redis state store query and sliding-window update latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.kafka_publish_latency_ms = _get_or_create(
            Histogram,
            "finpulse_kafka_publish_latency_ms",
            "Kafka message publishing and acknowledgement latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        # Stage-specific latency breakdown
        self.feature_engine_latency_ms = _get_or_create(
            Histogram,
            "finpulse_feature_engine_latency_ms",
            "32-feature extraction pipeline latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.risk_engine_latency_ms = _get_or_create(
            Histogram,
            "finpulse_risk_engine_latency_ms",
            "Hybrid risk engine weighted fusion and escalation latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.rule_engine_latency_ms = _get_or_create(
            Histogram,
            "finpulse_rule_engine_latency_ms",
            "Deterministic business rule evaluation latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.serialization_latency_ms = _get_or_create(
            Histogram,
            "finpulse_serialization_latency_ms",
            "DecisionEvent canonical UTF-8 JSON serialization latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

        self.kafka_consume_latency_ms = _get_or_create(
            Histogram,
            "finpulse_kafka_consume_latency_ms",
            "Kafka transaction consumer ingress and deserialization latency in milliseconds",
            buckets=FAST_STAGE_BUCKETS_MS
        )

    # -------------------------------------------------------------------------
    # Safe Recording Helpers with Bounded Cardinality Validation
    # -------------------------------------------------------------------------
    def record_transaction(self, status: str = "accepted") -> None:
        """Record accepted/processed transaction counter."""
        safe_status = status.lower() if status.lower() in VALID_STATUSES else "accepted"
        self.transactions_total.labels(status=safe_status).inc()

    def record_decision(
        self,
        decision: str,
        risk_level: Optional[str] = None,
        hard_block: bool = False,
        active_triggers: Optional[List[str]] = None,
        matched_rules: Optional[List[Union[str, Dict[str, Any]]]] = None
    ) -> None:
        """
        Record decision outcome and associated bounded categorical signals.
        Strictly enforces low-cardinality labels.
        """
        dec = decision.upper() if decision and decision.upper() in VALID_DECISIONS else "REVIEW"
        self.decisions_total.labels(decision=dec).inc()

        if risk_level:
            rl = risk_level.upper() if risk_level.upper() in VALID_RISK_LEVELS else "MEDIUM"
            self.risk_levels_total.labels(risk_level=rl).inc()

        if hard_block:
            self.hard_blocks_total.inc()

        if active_triggers:
            for trig in active_triggers:
                t_clean = str(trig).lower().strip()
                # Map complex trigger names to canonical bounded labels
                matched_label = None
                for candidate in VALID_TRIGGERS:
                    if candidate in t_clean:
                        matched_label = candidate
                        break
                if matched_label:
                    self.risk_triggers_total.labels(trigger=matched_label).inc()

        if matched_rules:
            for rule in matched_rules:
                cat = "velocity"
                if isinstance(rule, dict):
                    raw_cat = rule.get("category", "")
                    if str(raw_cat).lower() in VALID_RULE_CATEGORIES:
                        cat = str(raw_cat).lower()
                self.matched_rules_total.labels(category=cat).inc()

    def record_fraud_alert(self) -> None:
        """Record a fraud alert emitted to 'fraud-alerts' topic."""
        self.fraud_alerts_total.inc()

    def record_error(self, component: str, error_type: str = "runtime_error") -> None:
        """
        Record an error event with strictly bounded component and error_type labels.
        Never allows exception traces or unbounded strings in labels.
        """
        comp = component.lower() if component.lower() in VALID_COMPONENTS else "internal_error"
        err = error_type.lower() if error_type.lower() in VALID_ERROR_TYPES else "runtime_error"
        self.errors_total.labels(component=comp, error_type=err).inc()

    def record_replay(self) -> None:
        """Record duplicate transaction replay detected by idempotency layer."""
        self.replays_total.inc()

    def record_dlq(self) -> None:
        """Record dead-letter queue event."""
        self.dlq_total.inc()

    def record_stage_latency(self, stage: str, latency_ms: float) -> None:
        """Record latency observation for a specific processing stage."""
        val = max(0.0, float(latency_ms))
        if stage == "e2e" or stage == "processing":
            self.processing_latency_ms.observe(val)
        elif stage == "model":
            self.model_latency_ms.observe(val)
        elif stage == "redis":
            self.redis_latency_ms.observe(val)
        elif stage == "kafka_publish" or stage == "publisher":
            self.kafka_publish_latency_ms.observe(val)
        elif stage == "feature_engine":
            self.feature_engine_latency_ms.observe(val)
        elif stage == "risk_engine":
            self.risk_engine_latency_ms.observe(val)
        elif stage == "rule_engine":
            self.rule_engine_latency_ms.observe(val)
        elif stage == "serialization":
            self.serialization_latency_ms.observe(val)
        elif stage == "kafka_consume":
            self.kafka_consume_latency_ms.observe(val)

    def generate_exposition(self) -> bytes:
        """Return canonical Prometheus text exposition format."""
        return generate_latest(self.registry)


# -----------------------------------------------------------------------------
# Global Singleton Instance & Helpers
# -----------------------------------------------------------------------------
_GLOBAL_METRICS: Optional[FinPulseMetrics] = None

def get_metrics() -> FinPulseMetrics:
    """Access global FinPulseMetrics singleton."""
    global _GLOBAL_METRICS
    if _GLOBAL_METRICS is None:
        _GLOBAL_METRICS = FinPulseMetrics()
    return _GLOBAL_METRICS


def record_transaction(status: str = "accepted") -> None:
    get_metrics().record_transaction(status=status)


def record_decision(
    decision: str,
    risk_level: Optional[str] = None,
    hard_block: bool = False,
    active_triggers: Optional[List[str]] = None,
    matched_rules: Optional[List[Union[str, Dict[str, Any]]]] = None
) -> None:
    get_metrics().record_decision(
        decision=decision,
        risk_level=risk_level,
        hard_block=hard_block,
        active_triggers=active_triggers,
        matched_rules=matched_rules
    )


def record_fraud_alert() -> None:
    get_metrics().record_fraud_alert()


def record_error(component: str, error_type: str = "runtime_error") -> None:
    get_metrics().record_error(component=component, error_type=error_type)


def record_replay() -> None:
    get_metrics().record_replay()


def record_dlq() -> None:
    get_metrics().record_dlq()


def record_stage_latency(stage: str, latency_ms: float) -> None:
    get_metrics().record_stage_latency(stage=stage, latency_ms=latency_ms)


@contextmanager
def StageTimer(stage: str, metrics_inst: Optional[FinPulseMetrics] = None):
    """Context manager to measure and observe stage execution latency."""
    t0 = time.perf_counter()
    m = metrics_inst or get_metrics()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        m.record_stage_latency(stage, elapsed_ms)
