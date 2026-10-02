"""R5.6 — End-to-End Evaluation & Validation Test Suite.

Validates:
1. Complete End-to-End Pipeline Trace:
   Transaction -> Kafka Consumer -> Redis State -> 32 Features -> CatBoost + Calibration ->
   R5 Hybrid Risk Engine -> R5.2 Diagnostics -> R5.3 Business Rules -> DecisionEvent v1.0 ->
   Kafka Publisher (predictions & fraud-alerts) -> Offset Commit.
2. Decision Consistency & Determinism: 100% deterministic outputs for identical state/inputs.
3. ML Decision vs R5 Final Decision Policy Independence.
4. Redis State Read-Before-Write Invariant and Historical State Isolation.
5. Kafka Topic Routing (predictions for all, fraud-alerts for BLOCK / hard-block only).
6. Offset Safety: Broker failure prevents offset commit.
7. Replay Idempotency & Deterministic event_id Stability.
8. Controlled Failure Modes (Malformed inputs, Redis down, Kafka down, Model missing, Invalid features).
9. Decision Matrix Validation across all 8 standard operational profiles.
10. Frozen Model & Artifact Checksum Manifest Integrity.
"""
import os
import sys
import json
import time
import copy
import hashlib
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.decision_event import DecisionEvent
from src.streaming.config import StreamingConfig, KafkaTopicsConfig
from src.streaming.publisher import DecisionEventPublisher, KafkaDeliveryError
from src.streaming.worker import StreamingFraudWorker
from src.state.manager import RedisStateManager, CustomerHistoricalContext
from src.state.redis_client import InMemoryRedisMock
from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.consumer import TransactionConsumer
from src.models.inference import (
    ProductionModelService,
    ModelBundleIntegrityError,
    FeatureContractError
)
from src.features.schema import validate_feature_vector
from src.risk_engine.hybrid import HybridRiskEngine, NormalizedRiskSignals
from src.risk_engine.rules import BusinessRuleEngine


@pytest.fixture
def clean_streaming_config():
    """Provides an isolated test streaming configuration."""
    cfg = StreamingConfig()
    cfg.bootstrap_servers = "mock-kafka:9092"
    cfg.topics = KafkaTopicsConfig(
        inbound_transactions="test-transactions",
        outbound_predictions="test-predictions",
        alerts="test-fraud-alerts",
        dead_letter="test-dlq",
    )
    return cfg


# ==============================================================================
# 1. Complete End-to-End Pipeline Trace
# ==============================================================================

def test_e2e_complete_pipeline_trace(clean_streaming_config):
    """
    R5.6.1: Exercise full pipeline from raw transaction to offset commitment.
    Verifies every stage executes and produces a valid DecisionEvent v1.0.
    """
    worker = StreamingFraudWorker(config=clean_streaming_config, dry_run=True)
    
    sample_tx = {
        "transaction_id": "tx_e2e_trace_001",
        "timestamp": 1710000000.0,
        "amount": 150.0,
        "customer_id": "cust_trace_01",
        "merchant_id": "merch_dept_store",
        "category": "shopping_pos",
        "payment_type": "TRANSFER",
        "origin_balance": 10000.0,
        "auth_verified": True,
        "latitude": 37.7749,
        "longitude": -122.4194,
    }

    # Step 1: Ingest, compute features, inference, R5 hybrid, rules, and construct event
    result, event = worker.process_single_transaction(sample_tx)

    # Verification of intermediate result
    assert result["transaction_id"] == "tx_e2e_trace_001"
    assert "calibrated_probability" in result
    assert "risk_score" in result
    assert "decision" in result
    assert "signals" in result
    assert "diagnostics" in result

    # Verification of DecisionEvent v1.0 contract
    assert isinstance(event, DecisionEvent)
    assert event.schema_version == "1.0"
    assert event.event_type == "finpulse.fraud_decision"
    assert event.transaction_id == "tx_e2e_trace_001"
    assert event.customer_id == "cust_trace_01"
    assert 0.0 <= event.calibrated_probability <= 1.0
    assert 0.0 <= event.risk_score <= 100.0
    assert event.decision in ["APPROVE", "REVIEW", "BLOCK"]
    assert event.ml_decision in ["APPROVE", "REVIEW", "BLOCK"]
    assert len(event.event_id) in [32, 36]
    assert event.latency_ms is not None

    # Step 2: Publish to Kafka
    pub_result = worker.publisher.publish(event)
    assert pub_result["predictions_published"] is True
    assert "test-predictions" in pub_result["topics"]
    assert worker.publisher.published_predictions_count == 1

    # Step 3: Consumer Offset Commitment Verification
    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = [
        MagicMock(value=sample_tx)
    ]
    with patch("kafka.KafkaConsumer", return_value=mock_consumer):
        worker.run_consumer_loop(in_topic="test-transactions", out_topic="test-predictions")
        assert mock_consumer.commit.call_count == 1


# ==============================================================================
# 2. Decision Consistency & Determinism
# ==============================================================================

def test_decision_consistency_determinism(clean_streaming_config):
    """
    R5.6.2: Verify 100% deterministic decisions across repeated evaluations with identical inputs.
    """
    worker = StreamingFraudWorker(config=clean_streaming_config, dry_run=True)
    
    sample_tx = {
        "transaction_id": "tx_consistent_001",
        "timestamp": 1710005000.0,
        "amount": 350.0,
        "customer_id": "cust_consistent_01",
        "merchant_id": "merch_online_01",
        "category": "shopping_net",
        "payment_type": "TRANSFER",
        "origin_balance": 4500.0,
        "auth_verified": True,
    }

    results = []
    events = []
    for _ in range(5):
        res, ev = worker.process_single_transaction(sample_tx)
        results.append(res)
        events.append(ev)

    for i in range(1, 5):
        assert results[i]["risk_score"] == results[0]["risk_score"]
        assert results[i]["decision"] == results[0]["decision"]
        assert results[i]["calibrated_probability"] == results[0]["calibrated_probability"]
        assert results[i]["ml_decision"] == results[0]["ml_decision"]
        assert events[i].event_id == events[0].event_id
        assert events[i].decision == events[0].decision
        assert events[i].risk_score == events[0].risk_score


# ==============================================================================
# 3. Policy Independence: ML Decision vs R5 Final Decision
# ==============================================================================

def test_ml_decision_vs_r5_decision_independence():
    """
    R5.6.3: Verify R4 ML decision and R5 final decision are distinct and independent policies.
    Case A: ML says BLOCK (prob=0.85) -> R5 with clean signals results in REVIEW (score=48.25)
    Case B: ML says APPROVE (prob=0.05) -> Hard block rule forces R5 to BLOCK.
    """
    engine = HybridRiskEngine()

    # Case A: High ML probability, all other risk signals zero
    res_a = engine.evaluate(
        signals=NormalizedRiskSignals(
            calibrated_probability=0.85,
            velocity_signal=0.0,
            behavioral_signal=0.0,
            rules_signal=0.0,
            anomaly_signal=0.0,
        ),
        transaction_id="tx_indep_a",
        ml_decision="BLOCK"
    )
    assert res_a.ml_decision == "BLOCK"
    assert res_a.decision == "REVIEW"  # 0.85 * 45 = 38.25 + 10 = 48.25 -> REVIEW
    assert res_a.ml_decision != res_a.decision

    # Case B: Low ML probability, hard-block triggered
    res_b = engine.evaluate(
        signals=NormalizedRiskSignals(
            calibrated_probability=0.05,
            velocity_signal=0.0,
            behavioral_signal=0.0,
            rules_signal=0.0,
            anomaly_signal=0.0,
        ),
        transaction_id="tx_indep_b",
        ml_decision="APPROVE",
        hard_block=True
    )
    assert res_b.ml_decision == "APPROVE"
    assert res_b.decision == "BLOCK"
    assert res_b.hard_block is True


# ==============================================================================
# 4. Redis State Read-Before-Write Ordering & Isolation
# ==============================================================================

def test_redis_state_read_before_write_ordering():
    """
    R5.6.4: Validate that historical state query strictly precedes state updates.
    T1 features must NOT see T1.
    T2 features MUST see T1.
    """
    mock_redis = InMemoryRedisMock()
    state_mgr = RedisStateManager(redis_client=mock_redis)
    received_contexts = []

    def handler(event: TransactionEvent, context: CustomerHistoricalContext):
        received_contexts.append((event.transaction_id, context.to_dict()))

    consumer = TransactionConsumer(
        config=StreamingConfig(),
        handler=handler,
        state_manager=state_mgr
    )

    tx1 = create_sample_transaction(
        transaction_id="tx_order_1",
        customer_id="cust_order_x",
        amount=100.0,
        timestamp=1000.0
    )
    tx2 = create_sample_transaction(
        transaction_id="tx_order_2",
        customer_id="cust_order_x",
        amount=250.0,
        timestamp=1030.0
    )

    consumer.process_event_with_state(tx1)
    consumer.process_event_with_state(tx2)

    assert len(received_contexts) == 2
    
    # Tx1 saw 0 prior transactions
    tx1_id, ctx1 = received_contexts[0]
    assert tx1_id == "tx_order_1"
    assert ctx1["velocity"]["tx_count_1h"] == 0
    assert ctx1["behavior"]["count"] == 0

    # Tx2 saw exactly Tx1
    tx2_id, ctx2 = received_contexts[1]
    assert tx2_id == "tx_order_2"
    assert ctx2["velocity"]["tx_count_1h"] == 1
    assert ctx2["velocity"]["amount_sum_1h"] == 100.0
    assert ctx2["behavior"]["count"] == 1


# ==============================================================================
# 5. Kafka Topic Routing (predictions vs fraud-alerts)
# ==============================================================================

def test_kafka_predictions_routing_all_decisions(clean_streaming_config):
    """
    R5.6.5: Verify 100% of decisions are routed to 'predictions'.
    """
    pub = DecisionEventPublisher(config=clean_streaming_config, dry_run=True)
    
    for decision in ["APPROVE", "REVIEW", "BLOCK"]:
        event = DecisionEvent(
            transaction_id=f"tx_route_{decision}",
            customer_id="cust_test",
            decision=decision,
            risk_score=15.0 if decision == "APPROVE" else 50.0 if decision == "REVIEW" else 85.0,
            risk_level="LOW" if decision == "APPROVE" else "MEDIUM" if decision == "REVIEW" else "HIGH",
            ml_decision=decision,
            calibrated_probability=0.1,
            signals={"velocity": 0.1},
            diagnostics={"contributions": {}},
            reasons=["Standard evaluation verified."],
        )
        res = pub.publish(event)
        assert res["predictions_published"] is True
        assert "test-predictions" in res["topics"]

    assert pub.published_predictions_count == 3


def test_kafka_fraud_alerts_routing_policy(clean_streaming_config):
    """
    R5.6.5b: Verify only BLOCK or hard-block=True decisions are routed to 'fraud-alerts'.
    """
    pub = DecisionEventPublisher(config=clean_streaming_config, dry_run=True)
    
    # 1. Clean APPROVE -> predictions only
    app_ev = DecisionEvent(
        transaction_id="tx_app_clean",
        customer_id="cust_1",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.05,
        signals={},
        diagnostics={},
        reasons=["Clean transaction"],
        hard_block=False
    )
    res_app = pub.publish(app_ev)
    assert res_app["alerts_published"] is False
    assert "test-fraud-alerts" not in res_app["topics"]

    # 2. High-Risk BLOCK -> predictions AND fraud-alerts
    block_ev = DecisionEvent(
        transaction_id="tx_block_high",
        customer_id="cust_2",
        decision="BLOCK",
        risk_score=90.0,
        risk_level="HIGH",
        ml_decision="BLOCK",
        calibrated_probability=0.85,
        signals={},
        diagnostics={},
        reasons=["High calibrated probability"],
        hard_block=False
    )
    res_block = pub.publish(block_ev)
    assert res_block["alerts_published"] is True
    assert "test-fraud-alerts" in res_block["topics"]

    # 3. Hard-Block override on REVIEW -> predictions AND fraud-alerts
    hb_ev = DecisionEvent(
        transaction_id="tx_hb_override",
        customer_id="cust_3",
        decision="REVIEW",
        risk_score=55.0,
        risk_level="MEDIUM",
        ml_decision="REVIEW",
        calibrated_probability=0.45,
        signals={},
        diagnostics={},
        reasons=["Hard block trigger"],
        hard_block=True
    )
    res_hb = pub.publish(hb_ev)
    assert res_hb["alerts_published"] is True
    assert "test-fraud-alerts" in res_hb["topics"]

    assert pub.published_alerts_count == 2


# ==============================================================================
# 6. Offset Safety: Failure Prevents Offset Commit
# ==============================================================================

def test_offset_safety_on_publishing_failure(clean_streaming_config):
    """
    R5.6.6: Verify offset is NOT committed when publishing fails.
    """
    worker = StreamingFraudWorker(config=clean_streaming_config, dry_run=False)
    worker.publisher.is_connected = True
    worker.publisher.producer = MagicMock()
    
    # Simulate delivery error
    mock_future = MagicMock()
    mock_future.get.side_effect = Exception("Kafka broker connection timeout (simulated)")
    worker.publisher.producer.send.return_value = mock_future

    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = [
        MagicMock(value={
            "transaction_id": "tx_fail_offset",
            "timestamp": 1710000000.0,
            "amount": 100.0,
            "customer_id": "cust_fail",
        })
    ]

    with patch("kafka.KafkaConsumer", return_value=mock_consumer):
        worker.run_consumer_loop(in_topic="test-transactions", out_topic="test-predictions")
        # Offset must NOT be committed on failure
        assert mock_consumer.commit.call_count == 0


# ==============================================================================
# 7. Replay Idempotency & Event ID Stability
# ==============================================================================

def test_replay_idempotency_and_event_id_stability(clean_streaming_config):
    """
    R5.6.7: Verify replaying a transaction preserves deterministic event_id and idempotency key.
    """
    event1 = DecisionEvent(
        transaction_id="tx_replay_100",
        customer_id="cust_replay",
        decision="APPROVE",
        risk_score=20.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.08,
        signals={"velocity_signal": 0.05},
        diagnostics={},
        reasons=["Clean transaction"],
        timestamp=1710000000.0
    )
    event2 = DecisionEvent(
        transaction_id="tx_replay_100",
        customer_id="cust_replay",
        decision="APPROVE",
        risk_score=20.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.08,
        signals={"velocity_signal": 0.05},
        diagnostics={},
        reasons=["Clean transaction"],
        timestamp=1710000000.0
    )

    assert event1.event_id == event2.event_id
    assert event1.transaction_id == event2.transaction_id
    assert event1.to_bytes() == event2.to_bytes()


# ==============================================================================
# 8. Failure Modes Validation
# ==============================================================================

def test_failure_mode_malformed_transaction():
    """R5.6.8a: Reject malformed transactions missing required transaction_id or amount."""
    with pytest.raises(Exception):
        create_sample_transaction(transaction_id="", customer_id="c1", amount=10.0)


def test_failure_mode_invalid_feature_vector_dimension():
    """R5.6.8b: Feature validation rejects non-32 dimensional vectors."""
    with pytest.raises(ValueError, match=r"expected \(32,\)"):
        validate_feature_vector([1.0] * 31, expected_dim=32)
    with pytest.raises(ValueError, match=r"expected \(32,\)"):
        validate_feature_vector([1.0] * 33, expected_dim=32)


def test_failure_mode_invalid_feature_vector_nans_infs():
    """R5.6.8c: Feature validation strictly rejects NaN and Inf values."""
    vec_nan = [1.0] * 32
    vec_nan[5] = float("nan")
    with pytest.raises(ValueError, match="contains NaN"):
        validate_feature_vector(vec_nan, expected_dim=32)

    vec_inf = [1.0] * 32
    vec_inf[12] = float("inf")
    with pytest.raises(ValueError, match="contains Inf"):
        validate_feature_vector(vec_inf, expected_dim=32)


def test_failure_mode_model_artifact_missing():
    """R5.6.8d: ProductionModelService safely rejects missing bundle path."""
    with pytest.raises(ModelBundleIntegrityError):
        ProductionModelService(bundle_dir="/non/existent/path/to/model")


def test_failure_mode_publisher_unserializable_payload(clean_streaming_config):
    """R5.6.8e: Publisher strictly rejects invalid payload objects."""
    pub = DecisionEventPublisher(clean_streaming_config, dry_run=True)
    with pytest.raises(TypeError, match="Expected DecisionEvent instance"):
        pub.publish({"raw": "dict_is_not_allowed"})


# ==============================================================================
# 9. Decision Matrix Validation (8 Operational Profiles)
# ==============================================================================

def test_decision_matrix_all_profiles():
    """
    R5.6.9: Verify all 8 core operational profiles produce canonical expected results.
    """
    engine = HybridRiskEngine(enable_compounding=True)
    rule_engine = BusinessRuleEngine()

    # Profile 1: Clean Low-Risk (APPROVE)
    p1 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.02, velocity_signal=0.0, behavioral_signal=0.0, rules_signal=0.0, anomaly_signal=0.0),
        transaction_id="tx_p1", ml_decision="APPROVE"
    )
    assert p1.decision == "APPROVE"
    assert p1.risk_level == "LOW"

    # Profile 2: Medium-Risk (REVIEW)
    p2 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.60, velocity_signal=0.4, behavioral_signal=0.3, rules_signal=0.2, anomaly_signal=0.1),
        transaction_id="tx_p2", ml_decision="REVIEW"
    )
    assert p2.decision == "REVIEW"
    assert p2.risk_level == "MEDIUM"

    # Profile 3: High-Risk (BLOCK)
    p3 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.95, velocity_signal=0.9, behavioral_signal=0.9, rules_signal=0.8, anomaly_signal=0.7),
        transaction_id="tx_p3", ml_decision="BLOCK"
    )
    assert p3.decision == "BLOCK"
    assert p3.risk_level == "HIGH"

    # Profile 4: Hard-Block Override
    p4 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.01, velocity_signal=0.0, behavioral_signal=0.0, rules_signal=0.0, anomaly_signal=0.0),
        transaction_id="tx_p4", ml_decision="APPROVE", hard_block=True
    )
    assert p4.decision == "BLOCK"
    assert p4.hard_block is True

    # Profile 5: ML High + Hybrid Lower (ML BLOCK, Hybrid REVIEW due to clean secondary signals)
    p5 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.80, velocity_signal=0.0, behavioral_signal=0.0, rules_signal=0.0, anomaly_signal=0.0),
        transaction_id="tx_p5", ml_decision="BLOCK"
    )
    assert p5.ml_decision == "BLOCK"
    assert p5.decision == "REVIEW"

    # Profile 6: ML Low + Rule Escalation (ML APPROVE, Rules push score to REVIEW)
    p6 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.05, velocity_signal=0.7, behavioral_signal=0.6, rules_signal=0.8, anomaly_signal=0.4),
        transaction_id="tx_p6", ml_decision="APPROVE"
    )
    assert p6.ml_decision == "APPROVE"
    assert p6.decision in ["REVIEW", "BLOCK"]

    # Profile 7: Multiple Active Triggers (K >= 2 Escalation Boost)
    p7 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.8, velocity_signal=0.8, behavioral_signal=0.8, rules_signal=0.0, anomaly_signal=0.0),
        transaction_id="tx_p7"
    )
    assert p7.diagnostics["trigger_count"] >= 2
    assert p7.diagnostics["escalation_boost"] > 0.0

    # Profile 8: Zero Active Triggers (K = 0, No Escalation Boost)
    p8 = engine.evaluate(
        signals=NormalizedRiskSignals(calibrated_probability=0.1, velocity_signal=0.1, behavioral_signal=0.1, rules_signal=0.1, anomaly_signal=0.1),
        transaction_id="tx_p8"
    )
    assert p8.diagnostics["trigger_count"] == 0
    assert p8.diagnostics["escalation_boost"] == 0.0


# ==============================================================================
# 10. Model Artifact & Checksum Integrity
# ==============================================================================

def test_model_bundle_integrity_manifest():
    """
    R5.6.10: Validate that finpulse-v3 production bundle exists and passes SHA-256 verification.
    """
    svc = ProductionModelService()
    verified_hashes = svc._verify_checksums()
    assert len(verified_hashes) >= 5
    assert "model.pkl" in verified_hashes
    assert "calibrator.pkl" in verified_hashes
    assert "feature_schema.json" in verified_hashes
    assert "threshold_policy.json" in verified_hashes
    assert "model_metadata.json" in verified_hashes
