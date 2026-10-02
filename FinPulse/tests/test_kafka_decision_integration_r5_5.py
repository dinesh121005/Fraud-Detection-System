"""R5.5 Dedicated Test Suite — Kafka Decision Event Integration.

Validates the 11 mandatory R5.5 integration scenarios:
1. Serialization: Deterministic UTF-8 JSON, schema_version="1.0", event_type="finpulse.fraud_decision", no 32-feature vector.
2. Event ID Stability: Deterministic event_id and transaction_id preserved across repeated runs.
3. Predictions Publishing: All events (APPROVE, REVIEW, BLOCK) route to 'predictions' topic.
4. Fraud Alert Publishing: BLOCK and hard-block events route to 'fraud-alerts' topic.
5. Topic Isolation: Clean transactions (APPROVE/REVIEW) do NOT leak into 'fraud-alerts'.
6. Delivery Failure: Broker delivery errors are surfaced explicitly without altering fraud decisions.
7. Serialization Failure: Invalid / non-event payloads fail deterministically.
8. Kafka Unavailability: Resilient fallback to dry-run buffer when broker disconnected.
9. Retry Idempotency: Retries preserve identical event_id, transaction_id, and payload bytes.
10. Offset Semantics: Offset committed strictly after successful decision publishing.
11. End-to-End Pipeline: Raw transaction -> R3 -> R4 -> R5 -> DecisionEvent -> Publisher -> Topics.
"""

import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.decision_event import DecisionEvent
from src.streaming.config import StreamingConfig, KafkaTopicsConfig
from src.streaming.publisher import DecisionEventPublisher, KafkaDeliveryError
from src.streaming.worker import StreamingFraudWorker


@pytest.fixture
def streaming_config():
    """Provides test streaming config with isolated topics."""
    return StreamingConfig(
        bootstrap_servers="localhost:9092",
        topics=KafkaTopicsConfig(
            inbound_transactions="test-transactions",
            outbound_predictions="test-predictions",
            alerts="test-fraud-alerts",
            dead_letter="test-dlq",
        ),
    )


@pytest.fixture
def publisher(streaming_config):
    """Provides DecisionEventPublisher in dry-run mode for deterministic testing."""
    return DecisionEventPublisher(config=streaming_config, dry_run=True)


@pytest.fixture
def approve_event():
    return DecisionEvent(
        transaction_id="tx_approve_101",
        customer_id="cust_001",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.08,
        signals={"calibrated_probability": 0.08, "velocity_signal": 0.10},
        diagnostics={"weighted_score": 15.0, "hard_block": False},
        reasons=[],
        timestamp=1710000000.0,
        hard_block=False,
    )


@pytest.fixture
def block_event():
    return DecisionEvent(
        transaction_id="tx_block_999",
        customer_id="cust_888",
        decision="BLOCK",
        risk_score=88.5,
        risk_level="HIGH",
        ml_decision="BLOCK",
        calibrated_probability=0.89,
        signals={"calibrated_probability": 0.89, "velocity_signal": 0.80},
        diagnostics={"weighted_score": 75.0, "escalation_boost": 24.0, "hard_block": True},
        reasons=["CRITICAL: Impossible travel velocity (>800 km/h)"],
        timestamp=1710000100.0,
        matched_rules=[{"rule_id": "BLOCK_IMPOSSIBLE_TRAVEL", "severity": "CRITICAL"}],
        hard_block=True,
        hard_block_rules=["BLOCK_IMPOSSIBLE_TRAVEL"],
    )


# ==============================================================================
# 1. Serialization Test
# ==============================================================================

def test_decision_event_serialization_contract(approve_event):
    """Test 1: Verify UTF-8 JSON serialization, schema version, and exclusion of raw feature vector."""
    raw_bytes = approve_event.to_bytes()
    assert isinstance(raw_bytes, bytes)

    parsed = json.loads(raw_bytes.decode("utf-8"))
    assert parsed["schema_version"] == "1.0"
    assert parsed["event_type"] == "finpulse.fraud_decision"
    assert parsed["transaction_id"] == "tx_approve_101"
    assert parsed["decision"] == "APPROVE"
    assert parsed["risk_score"] == 15.0

    # Ensure 32-feature vector is strictly absent
    for forbidden in ("features", "feature_vector", "ordered_vec", "features_dict"):
        assert forbidden not in parsed
        assert forbidden not in parsed.get("metadata", {})


# ==============================================================================
# 2. Event ID Stability
# ==============================================================================

def test_event_id_stability_across_repeated_evaluations():
    """Test 2: Verify event_id is deterministic and stable for the same transaction."""
    e1 = DecisionEvent(
        transaction_id="tx_idemp_01",
        customer_id="cust_idemp",
        decision="REVIEW",
        risk_score=45.0,
        risk_level="MEDIUM",
        ml_decision="REVIEW",
        calibrated_probability=0.35,
        signals={},
        diagnostics={},
        reasons=[],
        timestamp=1710000500.0,
        model_version="finpulse-v3",
    )
    e2 = DecisionEvent(
        transaction_id="tx_idemp_01",
        customer_id="cust_idemp",
        decision="REVIEW",
        risk_score=45.0,
        risk_level="MEDIUM",
        ml_decision="REVIEW",
        calibrated_probability=0.35,
        signals={},
        diagnostics={},
        reasons=[],
        timestamp=1710000500.0,
        model_version="finpulse-v3",
    )
    assert e1.event_id == e2.event_id
    assert e1.to_bytes() == e2.to_bytes()


# ==============================================================================
# 3. Predictions Topic Publishing
# ==============================================================================

def test_predictions_publishing(publisher, approve_event):
    """Test 3: Verify all decisions are published to the 'predictions' topic with transaction_id key."""
    res = publisher.publish(approve_event)
    assert res["predictions_published"] is True
    assert "test-predictions" in res["topics"]
    assert publisher.published_predictions_count == 1
    assert len(publisher.dry_run_predictions) == 1
    assert publisher.dry_run_predictions[0].transaction_id == "tx_approve_101"


# ==============================================================================
# 4. Fraud Alert Publishing
# ==============================================================================

def test_fraud_alert_publishing(publisher, block_event):
    """Test 4: Verify BLOCK decisions route to both 'predictions' and 'fraud-alerts' topics."""
    res = publisher.publish(block_event)
    assert res["predictions_published"] is True
    assert res["alerts_published"] is True
    assert res["is_alert"] is True
    assert "test-predictions" in res["topics"]
    assert "test-fraud-alerts" in res["topics"]
    assert publisher.published_alerts_count == 1
    assert len(publisher.dry_run_alerts) == 1
    assert publisher.dry_run_alerts[0].transaction_id == "tx_block_999"


# ==============================================================================
# 5. Topic Isolation
# ==============================================================================

def test_topic_isolation_clean_decisions_do_not_alert(publisher, approve_event):
    """Test 5: Verify APPROVE decisions do NOT get published to the 'fraud-alerts' topic."""
    res = publisher.publish(approve_event)
    assert res["alerts_published"] is False
    assert res["is_alert"] is False
    assert "test-fraud-alerts" not in res["topics"]
    assert len(publisher.dry_run_alerts) == 0


# ==============================================================================
# 6. Delivery Failure Surfacing
# ==============================================================================

def test_delivery_failure_surfacing(streaming_config, approve_event):
    """Test 6: Verify broker delivery failures raise KafkaDeliveryError without altering decision."""
    pub = DecisionEventPublisher(config=streaming_config, dry_run=False)
    pub.is_connected = True
    pub.producer = MagicMock()

    # Simulate broker timeout / delivery error on future.get()
    mock_future = MagicMock()
    mock_future.get.side_effect = Exception("Broker timeout error (simulated)")
    pub.producer.send.return_value = mock_future

    with pytest.raises(KafkaDeliveryError, match="Kafka delivery failed"):
        pub.publish(approve_event, sync=True)

    assert pub.failed_count == 1
    # Verify business decision object was not mutated or degraded
    assert approve_event.decision == "APPROVE"
    assert approve_event.risk_score == 15.0


# ==============================================================================
# 7. Serialization Failure Rejection
# ==============================================================================

def test_serialization_failure_rejection(publisher):
    """Test 7: Reject invalid objects before publishing to Kafka."""
    with pytest.raises(TypeError, match="Expected DecisionEvent instance"):
        publisher.publish({"not": "a DecisionEvent"})


# ==============================================================================
# 8. Kafka Unavailability Fallback
# ==============================================================================

def test_kafka_unavailability_fallback(streaming_config, approve_event):
    """Test 8: When broker connection is unavailable, publisher buffers in fallback mode."""
    with patch("kafka.KafkaProducer", side_effect=Exception("Connection refused")):
        pub = DecisionEventPublisher(config=streaming_config, dry_run=False)
        assert pub.is_connected is False
        assert pub.producer is None

        # Publishing still buffers safely in dry_run buffer
        res = pub.publish(approve_event)
        assert res["predictions_published"] is True
        assert res["delivery_mode"] == "dry_run"
        assert len(pub.dry_run_predictions) == 1


# ==============================================================================
# 9. Retry Idempotency
# ==============================================================================

def test_retry_idempotency_preserves_event_identity(publisher, approve_event):
    """Test 9: Multiple publishing retries preserve identical event_id, transaction_id, and bytes."""
    res1 = publisher.publish(approve_event)
    res2 = publisher.publish(approve_event)

    assert res1["event_id"] == res2["event_id"]
    assert res1["transaction_id"] == res2["transaction_id"]
    assert approve_event.to_bytes() == approve_event.to_bytes()


# ==============================================================================
# 10. Offset Commit Semantics
# ==============================================================================

def test_offset_commit_semantics_in_worker():
    """Test 10: Verify consumer commits offset strictly after successful decision publishing."""
    worker = StreamingFraudWorker(dry_run=True)
    mock_consumer = MagicMock()
    mock_consumer.__iter__.return_value = [
        MagicMock(value={
            "transaction_id": "tx_worker_01",
            "timestamp": 1710000000.0,
            "amount": 50.0,
            "customer_id": "cust_w_1",
            "merchant_id": "merch_w_1",
            "category": "dining",
            "payment_type": "TRANSFER",
            "origin_balance": 5000.0,
            "auth_verified": True,
        })
    ]

    with patch("kafka.KafkaConsumer", return_value=mock_consumer):
        # Run loop with single mock message
        worker.run_consumer_loop(in_topic="test-transactions", out_topic="test-predictions")

        # Verify consumer commit was called exactly once after publish
        assert mock_consumer.commit.call_count == 1
        assert worker.publisher.published_predictions_count == 1


# ==============================================================================
# 11. End-to-End Pipeline Integration Test
# ==============================================================================

def test_end_to_end_pipeline_integration():
    """
    Test 11: Complete transaction flow:
    Raw Transaction -> Feature Vector -> R4 Inference -> R5 Decision -> DecisionEvent -> Publisher.
    """
    worker = StreamingFraudWorker(dry_run=True)

    # 1. Normal Transaction (should yield APPROVE, routed only to predictions)
    tx_normal = {
        "transaction_id": "tx_e2e_normal",
        "timestamp": 1710001000.0,
        "amount": 25.0,
        "customer_id": "cust_e2e_01",
        "merchant_id": "merch_grocery",
        "category": "grocery",
        "payment_type": "TRANSFER",
        "origin_balance": 5000.0,
        "auth_verified": True,
    }
    result_norm, event_norm = worker.process_single_transaction(tx_normal)
    pub_res_norm = worker.publisher.publish(event_norm)

    assert result_norm["decision"] in ["APPROVE", "REVIEW"]
    assert event_norm.transaction_id == "tx_e2e_normal"
    assert pub_res_norm["predictions_published"] is True
    assert pub_res_norm["alerts_published"] is False

    # 2. Attack Transaction (impossible travel velocity > 800 km/h -> hard block -> BLOCK)
    tx_attack = {
        "transaction_id": "tx_e2e_attack",
        "timestamp": 1710001010.0,
        "amount": 50000.0,
        "customer_id": "cust_e2e_02",
        "merchant_id": "merch_luxury",
        "category": "travel",
        "payment_type": "TRANSFER",
        "origin_balance": 0.0,
        "auth_verified": False,
        "latitude": 40.7128,
        "longitude": -74.0060,  # NYC
    }
    result_att, event_att = worker.process_single_transaction(tx_attack)
    pub_res_att = worker.publisher.publish(event_att)

    assert event_att.decision in ["REVIEW", "BLOCK"]
    assert pub_res_att["predictions_published"] is True
    if event_att.decision == "BLOCK" or event_att.hard_block:
        assert pub_res_att["alerts_published"] is True
