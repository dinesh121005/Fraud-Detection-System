"""R5.4 Dedicated Contract Test Suite — DecisionEvent v1.0.

Validates the 17 mandatory R5.4 contract requirements:
1. Valid DecisionEvent creation from completed R5 evaluation.
2. Deterministic event_id generation.
3. Strict schema_version "1.0" and event_type "finpulse.fraud_decision".
4. Deterministic JSON serialization roundtrip (to_json -> from_json).
5. UTF-8 byte serialization roundtrip (to_bytes -> from_bytes).
6. Strict exclusion of 32-feature vector (no raw feature leakage).
7. Bit-for-bit idempotency across repeated serializations.
8. Validation: Rejection of empty/whitespace transaction_id.
9. Validation: Rejection of empty/whitespace customer_id.
10. Validation: Rejection of out-of-range / NaN / Inf calibrated_probability.
11. Validation: Rejection of out-of-range / NaN / Inf risk_score.
12. Preservation of R4 ML decision (ml_decision) vs R5 hybrid decision (decision).
13. Preservation of R5.2 diagnostics structure.
14. Preservation of R5.3 matched rules and hard-block attribution.
15. Hard-block state consistency (hard_block=True -> BLOCK, score >= 85.0).
16. Bounded signals dictionary (5 canonical normalized signals).
17. Deserialization error handling on malformed JSON or invalid schema version.
"""

import os
import sys
import math
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.decision_event import DecisionEvent
from src.risk_engine.hybrid import HybridRiskEngine, HybridRiskResult
from src.risk_engine.rules import BusinessRuleEngine


@pytest.fixture
def hybrid_engine():
    return HybridRiskEngine(enable_compounding=True)


@pytest.fixture
def rule_engine():
    return BusinessRuleEngine()


@pytest.fixture
def sample_event():
    return DecisionEvent(
        transaction_id="tx_12345",
        customer_id="cust_9999",
        decision="REVIEW",
        risk_score=59.0,
        risk_level="MEDIUM",
        ml_decision="BLOCK",
        calibrated_probability=0.75,
        signals={
            "calibrated_probability": 0.75,
            "velocity_signal": 0.40,
            "behavioral_signal": 0.30,
            "rules_signal": 0.20,
            "anomaly_signal": 0.10,
        },
        diagnostics={
            "weighted_score": 48.4,
            "escalation_boost": 24.0,
            "trigger_count": 2,
            "active_triggers": ["ml", "velocity"],
            "contributions": {"ml": 31.5, "velocity": 8.7},
            "hard_block": False,
        },
        reasons=["High calibrated ML probability", "Rapid velocity spike"],
        timestamp=1700000000.0,
        model_version="finpulse-v3",
        feature_schema_version="2.0",
        matched_rules=[{"rule_id": "VEL_5M_BURST", "severity": "MEDIUM"}],
        hard_block=False,
        hard_block_rules=[],
        latency_ms=12.5,
    )


# ==============================================================================
# 1. Valid Instantiation from R5 Pipeline
# ==============================================================================

def test_valid_decision_event_creation(hybrid_engine, rule_engine):
    """Test 1: Verify DecisionEvent constructs cleanly from completed R5 HybridRiskResult."""
    tx = {"amount": 4900.0, "origin_balance": 5000.0, "auth_verified": 0}
    feat = {"tx_count_1m": 4, "tx_count_5m": 6}
    rule_res = rule_engine.evaluate(tx, feat)

    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.65,
        velocity=0.50,
        behavioral=0.30,
        anomaly=0.20,
        rule_result=rule_res,
        transaction_id="tx_pipeline_01"
    )

    event = DecisionEvent.from_hybrid_result(
        result=hybrid_res,
        customer_id="cust_101",
        timestamp=1710000000.0,
        latency_ms=8.4
    )

    assert event.transaction_id == "tx_pipeline_01"
    assert event.customer_id == "cust_101"
    assert event.decision in ["APPROVE", "REVIEW", "BLOCK"]
    assert event.risk_score == hybrid_res.risk_score
    assert event.calibrated_probability == 0.65
    assert event.model_version == "finpulse-v3"
    assert event.latency_ms == 8.4


# ==============================================================================
# 2. Deterministic event_id Generation
# ==============================================================================

def test_deterministic_event_id_stability():
    """Test 2: Verify event_id is deterministic and stable for identical event attributes."""
    e1 = DecisionEvent(
        transaction_id="tx_stable",
        customer_id="cust_1",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.10,
        signals={"calibrated_probability": 0.10},
        diagnostics={},
        reasons=[],
        timestamp=1700000000.0,
        model_version="finpulse-v3"
    )

    e2 = DecisionEvent(
        transaction_id="tx_stable",
        customer_id="cust_1",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.10,
        signals={"calibrated_probability": 0.10},
        diagnostics={},
        reasons=[],
        timestamp=1700000000.0,
        model_version="finpulse-v3"
    )

    assert e1.event_id != ""
    assert e1.event_id == e2.event_id


# ==============================================================================
# 3. Schema Version and Event Type Contract
# ==============================================================================

def test_schema_version_and_event_type_contract(sample_event):
    """Test 3: Verify schema_version is strictly '1.0' and event_type is 'finpulse.fraud_decision'."""
    assert sample_event.schema_version == "1.0"
    assert sample_event.event_type == "finpulse.fraud_decision"


# ==============================================================================
# 4. JSON Serialization Roundtrip
# ==============================================================================

def test_json_serialization_roundtrip(sample_event):
    """Test 4: Verify to_json -> from_json restores complete object without loss."""
    json_str = sample_event.to_json()
    restored = DecisionEvent.from_json(json_str)

    assert restored.event_id == sample_event.event_id
    assert restored.transaction_id == sample_event.transaction_id
    assert restored.customer_id == sample_event.customer_id
    assert restored.decision == sample_event.decision
    assert restored.risk_score == sample_event.risk_score
    assert restored.signals == sample_event.signals
    assert restored.diagnostics == sample_event.diagnostics
    assert restored.reasons == sample_event.reasons


# ==============================================================================
# 5. UTF-8 Byte Serialization Roundtrip
# ==============================================================================

def test_bytes_serialization_roundtrip(sample_event):
    """Test 5: Verify to_bytes -> from_bytes restores complete object."""
    raw_bytes = sample_event.to_bytes()
    assert isinstance(raw_bytes, bytes)
    restored = DecisionEvent.from_bytes(raw_bytes)
    assert restored == sample_event


# ==============================================================================
# 6. Strict Exclusion of 32-Feature Vector
# ==============================================================================

def test_exclusion_of_32_feature_vector(hybrid_engine):
    """Test 6: Verify 32-feature vector keys are strictly forbidden and sanitized."""
    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.20,
        velocity=0.10,
        behavioral=0.10,
        rules=0.10,
        anomaly=0.10,
        transaction_id="tx_safe"
    )

    # Factory sanitizes raw features
    event = DecisionEvent.from_hybrid_result(
        result=hybrid_res,
        customer_id="cust_safe",
        metadata={"features": [1.0] * 32, "feature_vector": [2.0] * 32, "safe_note": "ok"}
    )
    assert "features" not in event.metadata
    assert "feature_vector" not in event.metadata
    assert event.metadata["safe_note"] == "ok"

    # Direct instantiation with forbidden key must raise ValueError
    with pytest.raises(ValueError, match="Forbidden feature vector key"):
        DecisionEvent(
            transaction_id="tx_fail",
            customer_id="cust_fail",
            decision="APPROVE",
            risk_score=10.0,
            risk_level="LOW",
            ml_decision="APPROVE",
            calibrated_probability=0.10,
            signals={},
            diagnostics={},
            reasons=[],
            metadata={"features": [1.0] * 32}
        )


# ==============================================================================
# 7. Bit-for-Bit Idempotency Across Repeated Serializations
# ==============================================================================

def test_idempotency_across_repeated_serializations(sample_event):
    """Test 7: Verify 100 repeated serializations yield bit-for-bit identical byte outputs."""
    first_bytes = sample_event.to_bytes()
    for _ in range(100):
        assert sample_event.to_bytes() == first_bytes


# ==============================================================================
# 8. Validation: Empty Transaction ID Rejection
# ==============================================================================

def test_validation_empty_transaction_id():
    """Test 8: Empty or whitespace transaction_id must raise ValueError."""
    with pytest.raises(ValueError, match="transaction_id"):
        DecisionEvent(
            transaction_id="   ",
            customer_id="cust_valid",
            decision="APPROVE",
            risk_score=10.0,
            risk_level="LOW",
            ml_decision="APPROVE",
            calibrated_probability=0.10,
            signals={},
            diagnostics={},
            reasons=[]
        )


# ==============================================================================
# 9. Validation: Empty Customer ID Rejection
# ==============================================================================

def test_validation_empty_customer_id():
    """Test 9: Empty or whitespace customer_id must raise ValueError."""
    with pytest.raises(ValueError, match="customer_id"):
        DecisionEvent(
            transaction_id="tx_valid",
            customer_id="",
            decision="APPROVE",
            risk_score=10.0,
            risk_level="LOW",
            ml_decision="APPROVE",
            calibrated_probability=0.10,
            signals={},
            diagnostics={},
            reasons=[]
        )


# ==============================================================================
# 10. Validation: Calibrated Probability Range
# ==============================================================================

def test_validation_calibrated_probability_range():
    """Test 10: calibrated_probability outside [0.0, 1.0] or NaN/Inf must raise ValueError."""
    for invalid_p in (-0.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="calibrated_probability"):
            DecisionEvent(
                transaction_id="tx_valid",
                customer_id="cust_valid",
                decision="APPROVE",
                risk_score=10.0,
                risk_level="LOW",
                ml_decision="APPROVE",
                calibrated_probability=invalid_p,
                signals={},
                diagnostics={},
                reasons=[]
            )


# ==============================================================================
# 11. Validation: Risk Score Range
# ==============================================================================

def test_validation_risk_score_range():
    """Test 11: risk_score outside [0.0, 100.0] or NaN/Inf must raise ValueError."""
    for invalid_s in (-0.1, 100.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="risk_score"):
            DecisionEvent(
                transaction_id="tx_valid",
                customer_id="cust_valid",
                decision="APPROVE",
                risk_score=invalid_s,
                risk_level="LOW",
                ml_decision="APPROVE",
                calibrated_probability=0.5,
                signals={},
                diagnostics={},
                reasons=[]
            )


# ==============================================================================
# 12. Separation of ML Decision and Hybrid Decision
# ==============================================================================

def test_preservation_of_ml_decision_vs_hybrid_decision():
    """Test 12: Verify R4 ml_decision (e.g. BLOCK) and R5 decision (e.g. REVIEW) remain distinct."""
    event = DecisionEvent(
        transaction_id="tx_indep",
        customer_id="cust_indep",
        decision="REVIEW",
        risk_score=36.0,
        risk_level="MEDIUM",
        ml_decision="BLOCK",  # Independent R4 decision
        calibrated_probability=0.80,
        signals={},
        diagnostics={},
        reasons=[]
    )
    assert event.ml_decision == "BLOCK"
    assert event.decision == "REVIEW"


# ==============================================================================
# 13. Preservation of Diagnostics Structure
# ==============================================================================

def test_preservation_of_r5_2_diagnostics_structure(sample_event):
    """Test 13: Verify R5.2 diagnostics payload is fully preserved in event."""
    diag = sample_event.diagnostics
    assert "weighted_score" in diag
    assert "escalation_boost" in diag
    assert "trigger_count" in diag
    assert "active_triggers" in diag
    assert diag["trigger_count"] == 2


# ==============================================================================
# 14. Preservation of Rules and Hard-Block Attribution
# ==============================================================================

def test_preservation_of_r5_3_rules_and_hard_block():
    """Test 14: Verify R5.3 matched_rules and hard_block_rules are preserved in event."""
    event = DecisionEvent(
        transaction_id="tx_rules",
        customer_id="cust_rules",
        decision="BLOCK",
        risk_score=85.0,
        risk_level="HIGH",
        ml_decision="APPROVE",
        calibrated_probability=0.10,
        signals={},
        diagnostics={},
        reasons=["CRITICAL: Impossible travel velocity (>800 km/h)"],
        matched_rules=[{"rule_id": "BLOCK_IMPOSSIBLE_TRAVEL", "severity": "CRITICAL"}],
        hard_block=True,
        hard_block_rules=["BLOCK_IMPOSSIBLE_TRAVEL"]
    )
    assert event.hard_block is True
    assert event.hard_block_rules == ["BLOCK_IMPOSSIBLE_TRAVEL"]
    assert len(event.matched_rules) == 1


# ==============================================================================
# 15. Hard-Block Consistency Guarantee
# ==============================================================================

def test_hard_block_consistency(hybrid_engine, rule_engine):
    """Test 15: When hard block triggers, decision must be BLOCK and risk_score >= 85.0."""
    tx = {"amount": 10.0, "origin_balance": 5000.0, "auth_verified": 1}
    feat = {"speed_kmh_from_prev_tx": 900.0}
    rule_res = rule_engine.evaluate(tx, feat)

    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.01,
        velocity=0.01,
        behavioral=0.01,
        anomaly=0.01,
        rule_result=rule_res,
        transaction_id="tx_hb_test"
    )

    event = DecisionEvent.from_hybrid_result(hybrid_res, customer_id="cust_hb")
    assert event.hard_block is True
    assert event.decision == "BLOCK"
    assert event.risk_score >= 85.0


# ==============================================================================
# 16. Bounded Signals Dictionary
# ==============================================================================

def test_bounded_signals_dictionary(sample_event):
    """Test 16: Verify signals dictionary contains bounded floats in [0.0, 1.0]."""
    sigs = sample_event.signals
    for k, v in sigs.items():
        assert 0.0 <= v <= 1.0


# ==============================================================================
# 17. Deserialization Error Handling on Malformed Input
# ==============================================================================

def test_deserialization_error_handling_malformed_json():
    """Test 17: Verify proper ValueError exceptions on malformed JSON or unsupported versions."""
    with pytest.raises(ValueError, match="Malformed JSON"):
        DecisionEvent.from_json("not valid json {")

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        DecisionEvent.from_dict({
            "schema_version": "9.9",
            "event_type": "finpulse.fraud_decision",
            "transaction_id": "tx_1",
            "customer_id": "cust_1",
            "decision": "APPROVE",
            "risk_score": 10.0,
            "risk_level": "LOW",
            "ml_decision": "APPROVE",
            "calibrated_probability": 0.1
        })

    with pytest.raises(ValueError, match="Invalid event_type"):
        DecisionEvent.from_dict({
            "schema_version": "1.0",
            "event_type": "wrong.event.type",
            "transaction_id": "tx_1",
            "customer_id": "cust_1",
            "decision": "APPROVE",
            "risk_score": 10.0,
            "risk_level": "LOW",
            "ml_decision": "APPROVE",
            "calibrated_probability": 0.1
        })
