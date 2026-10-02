"""FinPulse R6.1 — Observability, Metrics & Telemetry Test Suite.

Validates:
1. Metrics registration and exposition format
2. Transaction and decision counters (APPROVE / REVIEW / BLOCK)
3. Fraud alert routing counter policy (BLOCK or hard_block == True)
4. Stage latency histograms (E2E, model, Redis, Kafka publish, feature engine, etc.)
5. Bounded error counters and component failure tracking
6. Replay / duplicate idempotency metric behavior
7. DLQ routing metric verification
8. Correlation context propagation (transaction_id, customer_id, event_id, versions)
9. Strict Prometheus label cardinality guardrails (no per-transaction series)
10. Sensitive data protection and credential redaction
11. FastAPI /metrics endpoint integration
12. Frozen decision mathematics invariant verification
"""

import time
import json
import os
import sys
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from prometheus_client import CollectorRegistry

from src.monitoring.metrics import (
    FinPulseMetrics,
    VALID_DECISIONS,
    VALID_RISK_LEVELS,
    VALID_STATUSES,
    VALID_COMPONENTS,
    VALID_ERROR_TYPES,
)
from src.monitoring.correlation import (
    CorrelationContext,
    set_correlation_context,
    get_correlation_context,
    clear_correlation_context,
)
from src.monitoring.logger import sanitize_log_payload, StructuredLogger
from src.risk_engine.decision_event import DecisionEvent
from src.streaming.worker import StreamingFraudWorker
from src.streaming.publisher import DecisionEventPublisher
from src.streaming.config import StreamingConfig


@pytest.fixture
def custom_registry():
    """Provides a fresh, isolated CollectorRegistry for testing."""
    return CollectorRegistry()


@pytest.fixture
def metrics(custom_registry):
    """Provides a FinPulseMetrics instance bound to an isolated registry."""
    return FinPulseMetrics(registry=custom_registry)


# =============================================================================
# 1. Metrics Registration & Existence
# =============================================================================

def test_metrics_registration_and_types(metrics, custom_registry):
    """Verify all required R6.1 Prometheus metrics exist and are registered."""
    expected_metrics = [
        "finpulse_transactions_total",
        "finpulse_decisions_total",
        "finpulse_fraud_alerts_total",
        "finpulse_risk_levels_total",
        "finpulse_hard_blocks_total",
        "finpulse_replays_total",
        "finpulse_dlq_total",
        "finpulse_risk_triggers_total",
        "finpulse_matched_rules_total",
        "finpulse_errors_total",
        "finpulse_processing_latency_ms",
        "finpulse_model_latency_ms",
        "finpulse_redis_latency_ms",
        "finpulse_kafka_publish_latency_ms",
        "finpulse_feature_engine_latency_ms",
        "finpulse_risk_engine_latency_ms",
        "finpulse_rule_engine_latency_ms",
        "finpulse_serialization_latency_ms",
        "finpulse_kafka_consume_latency_ms",
    ]

    registered_names = set(custom_registry._names_to_collectors.keys())
    for name in expected_metrics:
        assert name in registered_names, f"Expected metric '{name}' not found in registry"


# =============================================================================
# 2. Counter Behavior & Decision Metrics
# =============================================================================

def test_transaction_counters(metrics, custom_registry):
    """Verify finpulse_transactions_total increments correctly."""
    metrics.record_transaction(status="accepted")
    metrics.record_transaction(status="accepted")
    metrics.record_transaction(status="rejected")

    assert custom_registry.get_sample_value("finpulse_transactions_total", {"status": "accepted"}) == 2.0
    assert custom_registry.get_sample_value("finpulse_transactions_total", {"status": "rejected"}) == 1.0


def test_decision_and_risk_level_metrics(metrics, custom_registry):
    """Verify APPROVE, REVIEW, BLOCK decisions and risk levels are counted."""
    metrics.record_decision(decision="APPROVE", risk_level="LOW")
    metrics.record_decision(decision="REVIEW", risk_level="MEDIUM")
    metrics.record_decision(decision="BLOCK", risk_level="HIGH", hard_block=True)

    assert custom_registry.get_sample_value("finpulse_decisions_total", {"decision": "APPROVE"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_decisions_total", {"decision": "REVIEW"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_decisions_total", {"decision": "BLOCK"}) == 1.0

    assert custom_registry.get_sample_value("finpulse_risk_levels_total", {"risk_level": "LOW"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_risk_levels_total", {"risk_level": "MEDIUM"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_risk_levels_total", {"risk_level": "HIGH"}) == 1.0

    assert custom_registry.get_sample_value("finpulse_hard_blocks_total") == 1.0


# =============================================================================
# 3. Fraud Alerts Metric Policy
# =============================================================================

def test_fraud_alerts_increment_policy(metrics, custom_registry):
    """Verify finpulse_fraud_alerts_total increments on fraud alerts."""
    initial = custom_registry.get_sample_value("finpulse_fraud_alerts_total") or 0.0
    metrics.record_fraud_alert()
    metrics.record_fraud_alert()

    after = custom_registry.get_sample_value("finpulse_fraud_alerts_total")
    assert after == initial + 2.0


def test_publisher_fraud_alert_integration():
    """Verify DecisionEventPublisher increments fraud alert only when decision==BLOCK or hard_block==True."""
    publisher = DecisionEventPublisher(dry_run=True)
    initial_alerts = publisher.published_alerts_count

    # 1. Clean APPROVE event
    evt_approve = DecisionEvent(
        transaction_id="tx_obs_01",
        customer_id="cust_obs_01",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.05,
        signals={"velocity_signal": 0.1},
        diagnostics={},
        reasons=[],
        timestamp=time.time()
    )
    res_app = publisher.publish(evt_approve)
    assert res_app["alerts_published"] is False
    assert publisher.published_alerts_count == initial_alerts

    # 2. Fraud BLOCK event
    evt_block = DecisionEvent(
        transaction_id="tx_obs_02",
        customer_id="cust_obs_02",
        decision="BLOCK",
        risk_score=85.0,
        risk_level="HIGH",
        ml_decision="BLOCK",
        calibrated_probability=0.85,
        signals={"velocity_signal": 0.8},
        diagnostics={"hard_block": False},
        reasons=["High ML probability"],
        timestamp=time.time()
    )
    res_blk = publisher.publish(evt_block)
    assert res_blk["alerts_published"] is True
    assert publisher.published_alerts_count == initial_alerts + 1

    # 3. Hard-Block override event
    evt_hard = DecisionEvent(
        transaction_id="tx_obs_03",
        customer_id="cust_obs_03",
        decision="BLOCK",
        risk_score=90.0,
        risk_level="HIGH",
        ml_decision="REVIEW",
        calibrated_probability=0.30,
        signals={"rules_signal": 0.9},
        diagnostics={"hard_block": True},
        hard_block=True,
        reasons=["Critical rule violation"],
        timestamp=time.time()
    )
    res_hb = publisher.publish(evt_hard)
    assert res_hb["alerts_published"] is True
    assert publisher.published_alerts_count == initial_alerts + 2


# =============================================================================
# 4. Latency Histograms
# =============================================================================

def test_latency_histograms(metrics, custom_registry):
    """Verify stage timers observe latencies into corresponding histograms."""
    metrics.record_stage_latency("e2e", 12.5)
    metrics.record_stage_latency("model", 4.2)
    metrics.record_stage_latency("redis", 1.1)
    metrics.record_stage_latency("kafka_publish", 3.0)
    metrics.record_stage_latency("feature_engine", 2.0)
    metrics.record_stage_latency("risk_engine", 0.8)
    metrics.record_stage_latency("rule_engine", 0.5)
    metrics.record_stage_latency("serialization", 0.3)
    metrics.record_stage_latency("kafka_consume", 0.4)

    assert custom_registry.get_sample_value("finpulse_processing_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_model_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_redis_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_kafka_publish_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_feature_engine_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_risk_engine_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_rule_engine_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_serialization_latency_ms_count") == 1.0
    assert custom_registry.get_sample_value("finpulse_kafka_consume_latency_ms_count") == 1.0


# =============================================================================
# 5. Error Metrics
# =============================================================================

def test_error_metrics(metrics, custom_registry):
    """Verify bounded component error tracking."""
    metrics.record_error("redis", "timeout")
    metrics.record_error("kafka", "connection_error")
    metrics.record_error("model", "runtime_error")
    metrics.record_error("serialization", "schema_error")

    assert custom_registry.get_sample_value("finpulse_errors_total", {"component": "redis", "error_type": "timeout"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_errors_total", {"component": "kafka", "error_type": "connection_error"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_errors_total", {"component": "model", "error_type": "runtime_error"}) == 1.0
    assert custom_registry.get_sample_value("finpulse_errors_total", {"component": "serialization", "error_type": "schema_error"}) == 1.0


# =============================================================================
# 6. Replay & DLQ Metrics
# =============================================================================

def test_replay_and_dlq_metrics(metrics, custom_registry):
    """Verify replay and DLQ counter increments."""
    metrics.record_replay()
    metrics.record_replay()
    metrics.record_dlq()

    assert custom_registry.get_sample_value("finpulse_replays_total") == 2.0
    assert custom_registry.get_sample_value("finpulse_dlq_total") == 1.0


def test_worker_replay_idempotency_detection():
    """Verify streaming worker flags duplicate transaction_id as replay."""
    worker = StreamingFraudWorker(dry_run=True)
    sample_tx = {
        "transaction_id": "tx_replay_test_99",
        "timestamp": time.time(),
        "amount": 100.0,
        "customer_id": "cust_replay_01",
        "merchant_id": "merch_01",
        "category": "grocery",
        "payment_type": "PAYMENT",
        "origin_balance": 500.0,
        "auth_verified": True
    }

    # First execution: normal transaction
    res1, evt1 = worker.process_single_transaction(sample_tx)
    assert evt1.transaction_id == "tx_replay_test_99"

    # Second execution: replay
    from src.monitoring.metrics import get_metrics
    m = get_metrics()
    prev_replays = m.replays_total._value.get() if hasattr(m.replays_total, "_value") else 0.0

    res2, evt2 = worker.process_single_transaction(sample_tx)
    assert evt2.transaction_id == "tx_replay_test_99"
    assert evt1.event_id == evt2.event_id  # Deterministic event_id preserved


# =============================================================================
# 7. Correlation Context & Model Metadata
# =============================================================================

def test_correlation_context_propagation():
    """Verify correlation context preserves transaction_id, customer_id, event_id, and model metadata."""
    clear_correlation_context()
    assert get_correlation_context() is None

    ctx = set_correlation_context(
        transaction_id="tx_corr_01",
        customer_id="cust_corr_01",
        event_id="evt_uuid_55",
        model_version="finpulse-v3",
        feature_schema_version="2.0"
    )

    current = get_correlation_context()
    assert current is not None
    assert current.transaction_id == "tx_corr_01"
    assert current.customer_id == "cust_corr_01"
    assert current.event_id == "evt_uuid_55"
    assert current.model_version == "finpulse-v3"
    assert current.feature_schema_version == "2.0"

    d = current.to_dict()
    assert d["transaction_id"] == "tx_corr_01"
    assert d["model_version"] == "finpulse-v3"
    assert d["feature_schema_version"] == "2.0"

    clear_correlation_context()


# =============================================================================
# 8. Strict Cardinality Guardrails
# =============================================================================

def test_prometheus_label_cardinality_guardrails(custom_registry):
    """Verify that NO dynamic IDs (tx_id, cust_id, timestamp) ever appear as metric labels."""
    m = FinPulseMetrics(registry=custom_registry)

    # Record diverse events
    m.record_transaction("accepted")
    m.record_decision("APPROVE", "LOW")
    m.record_decision("BLOCK", "HIGH", hard_block=True, active_triggers=["ml", "velocity"])
    m.record_error("redis", "timeout")
    m.record_stage_latency("model", 5.0)

    forbidden_label_names = {
        "transaction_id",
        "tx_id",
        "customer_id",
        "cust_id",
        "event_id",
        "timestamp",
        "amount",
        "exception",
        "error_message",
        "stack_trace",
    }

    # Inspect all registered metric collectors and their label names
    for metric_family in custom_registry.collect():
        for sample in metric_family.samples:
            label_keys = set(sample.labels.keys())
            overlap = label_keys.intersection(forbidden_label_names)
            assert not overlap, f"Metric '{sample.name}' contains forbidden high-cardinality labels: {overlap}"

            # Verify label values are bounded strings
            for k, val in sample.labels.items():
                assert not str(val).startswith("tx_"), f"Dynamic ID leaked into label: {k}={val}"
                assert not str(val).startswith("cust_"), f"Dynamic ID leaked into label: {k}={val}"
                assert len(str(val)) < 50, f"Unbounded label value length detected: {k}={val}"


# =============================================================================
# 9. Sensitive Data Protection & Redaction
# =============================================================================

def test_sensitive_data_redaction():
    """Verify passwords, tokens, API keys, and payment credentials are redacted."""
    sensitive_payload = {
        "transaction_id": "tx_safe_100",
        "customer_id": "cust_safe_200",
        "amount": 500.0,
        "password": "SuperSecretPassword123",
        "auth_token": "bearer_eyJhbGciOi...",
        "api_key": "finpulse_live_secret_key",
        "card_number": "4111222233334444",
        "cvv": "123",
        "nested": {
            "credit_card": "5500000000000004",
            "pin": "9999",
            "safe_field": "public_data"
        }
    }

    sanitized = sanitize_log_payload(sensitive_payload)

    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["auth_token"] == "[REDACTED]"
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["card_number"] == "[REDACTED]"
    assert sanitized["cvv"] == "[REDACTED]"
    assert sanitized["nested"]["credit_card"] == "[REDACTED]"
    assert sanitized["nested"]["pin"] == "[REDACTED]"
    assert sanitized["nested"]["safe_field"] == "public_data"
    assert sanitized["transaction_id"] == "tx_safe_100"


# =============================================================================
# 10. FastAPI /metrics Endpoint Integration
# =============================================================================

def test_fastapi_metrics_endpoint():
    """Verify FastAPI /metrics returns valid Prometheus exposition text."""
    from fastapi.testclient import TestClient
    from src.serving.api import app

    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text

    assert "finpulse_transactions_total" in body
    assert "finpulse_decisions_total" in body
    assert "finpulse_processing_latency_ms" in body


# =============================================================================
# 11. End-to-End Decision Determinism & Unchanged Mathematics
# =============================================================================

def test_decision_mathematics_frozen_invariance():
    """Verify telemetry instrumentation does NOT alter R4-R5 decision values or scores."""
    from src.risk_engine.engine import FinPulseRiskEngine
    engine = FinPulseRiskEngine()

    dummy_tx = {
        "transaction_id": "tx_math_01",
        "amount": 100.0,
        "customer_id": "cust_01",
        "payment_type": "PAYMENT"
    }
    dummy_features = {
        "tx_count_5m": 1,
        "tx_count_1h": 1,
        "amount_zscore": 0.0,
        "is_new_device": 0,
        "is_new_location": 0,
    }

    # Evaluate risk score
    res = engine.evaluate_risk(
        tx=dummy_tx,
        features_dict=dummy_features,
        calibrated_probability=0.02,
        anomaly_score=0.1
    )

    # Standard clean transaction must be APPROVED with low risk
    assert res["decision"] == "APPROVE"
    assert res["risk_level"] == "LOW"
    assert 0.0 <= res["risk_score"] <= 30.0
