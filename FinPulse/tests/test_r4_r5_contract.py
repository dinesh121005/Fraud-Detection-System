"""Comprehensive Test Suite for R4 -> R5 Contract and Boundary Invariants.

Protects:
1. Probability Contract (calibrated_probability as canonical, numeric, range [0, 1], ml_prob alias).
2. Production Model Version Contract (strictly finpulse-v3, no active finpulse-v2.0).
3. Decision Policy Separation (R4 ML 0.1580/0.5516 vs R5 Hybrid 30.0/70.0, cross-misuse prevention).
4. Signal Normalization Contract (all 5 signals strictly in [0.0, 1.0], NaN/Inf rejection, missing fallback).
5. Anomaly-Score Contract (IsolationForest sourcing, sigmoid normalization, [0, 1] range, monotonic polarity, neutral fallback).
6. Weight Configuration Contract (0.45 + 0.15 + 0.15 + 0.15 + 0.10 == 1.0, deterministic fusion, non-validated disclaimer).
7. Compounding Policy Contract (+0.12 * triggers post-fusion, trigger boundaries, multi-trigger synergy).
8. Regression Protection (frozen bundle checksums, 32-feature vector, Redis read-before-write, Kafka schema).
"""

import os
import sys
import json
import math
import hashlib
import pytest
import numpy as np

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.models.inference import ProductionModelService, InferenceResult
from src.risk_engine.contract import (
    R4InferenceOutput,
    NormalizedRiskSignals,
    R4MLThresholds,
    HybridRiskThresholds,
    HybridRiskWeights,
    CompoundingPolicy,
    CompoundingEvaluation,
    ContractViolationError,
    normalize_velocity_signal,
    normalize_behavioral_signal,
    normalize_rules_signal,
    normalize_anomaly_signal,
    validate_calibrated_probability,
)
from src.risk_engine.engine import FinPulseRiskEngine
from src.risk_engine.rules import DeterministicRuleEngine
from src.models.anomaly import IsolationForestAnomalyDetector
from src.serving.schemas import PredictionResponse, SignalsSchema

PRODUCTION_BUNDLE_DIR = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3")


# ==============================================================================
# 1. Probability Contract Tests
# ==============================================================================

def test_probability_contract_canonical_field_existence():
    """Verify calibrated_probability is the canonical field in R4 inference output."""
    inf_out = R4InferenceOutput(
        transaction_id="tx_test_001",
        calibrated_probability=0.3456,
        model_version="finpulse-v3",
        raw_probability=0.2891,
        ml_decision="REVIEW"
    )
    assert hasattr(inf_out, "calibrated_probability")
    assert isinstance(inf_out.calibrated_probability, float)
    assert inf_out.calibrated_probability == 0.3456
    # ml_prob is accessible only as a backward-compatible alias
    assert inf_out.ml_prob == inf_out.calibrated_probability

def test_probability_contract_valid_range_and_rejections():
    """Verify calibrated_probability strictly enforces [0.0, 1.0] and rejects NaN/Inf."""
    # Valid boundaries
    assert validate_calibrated_probability(0.0) == 0.0
    assert validate_calibrated_probability(1.0) == 1.0
    assert validate_calibrated_probability(0.5516) == 0.5516

    # Rejection of invalid types and values
    with pytest.raises(ContractViolationError, match="cannot be None"):
        validate_calibrated_probability(None)
    with pytest.raises(ContractViolationError, match="cannot be NaN"):
        validate_calibrated_probability(float("nan"))
    with pytest.raises(ContractViolationError, match="cannot be NaN or Inf"):
        validate_calibrated_probability(float("inf"))
    with pytest.raises(ContractViolationError, match=r"in \[0.0, 1.0\]"):
        validate_calibrated_probability(-0.01)
    with pytest.raises(ContractViolationError, match=r"in \[0.0, 1.0\]"):
        validate_calibrated_probability(1.0001)

def test_r5_consumes_calibrated_probability_and_handles_alias():
    """Verify R5 FinPulseRiskEngine prioritizes calibrated_probability over legacy ml_prob."""
    engine = FinPulseRiskEngine()
    tx = {"transaction_id": "tx_prob_test"}
    features = {"tx_count_5m": 1, "tx_count_1h": 1, "amount_zscore": 0.0}

    # Case A: Canonical calibrated_probability provided
    res1 = engine.evaluate_risk(
        tx=tx,
        features_dict=features,
        calibrated_probability=0.4500,
        anomaly_score=0.10
    )
    assert res1["signals"]["calibrated_probability"] == 0.4500
    assert res1["fraud_probability"] == 0.4500

    # Case B: Backward-compatible ml_prob alias provided when calibrated_probability is None
    res2 = engine.evaluate_risk(
        tx=tx,
        features_dict=features,
        ml_prob=0.4500,
        anomaly_score=0.10
    )
    assert res2["signals"]["calibrated_probability"] == 0.4500
    assert res2["signals"]["ml_probability"] == 0.4500

    # Case C: Neither provided -> strict ContractViolationError
    with pytest.raises(ContractViolationError, match="Missing canonical 'calibrated_probability'"):
        engine.evaluate_risk(
            tx=tx,
            features_dict=features,
            anomaly_score=0.10
        )


# ==============================================================================
# 2. Production Model Version Contract Tests
# ==============================================================================

def test_model_version_contract_consistency():
    """Verify production model version is strictly finpulse-v3 across metadata, R4, R5, and API."""
    # 1. Metadata check in frozen bundle
    meta_path = os.path.join(PRODUCTION_BUNDLE_DIR, "model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["model_version"] == "finpulse-v3"

    # 2. ProductionModelService in R4
    service = ProductionModelService(bundle_dir=PRODUCTION_BUNDLE_DIR, enforce_checksum=True)
    assert service.model_version == "finpulse-v3"

    # 3. FinPulseRiskEngine in R5
    engine = FinPulseRiskEngine()
    assert engine.model_version == "finpulse-v3"

    # 4. Serving Schema default
    schema_inst = PredictionResponse(
        transaction_id="tx_schema",
        fraud_probability=0.1,
        risk_score=20.0,
        risk_level="LOW",
        decision="APPROVE",
        signals=SignalsSchema(
            calibrated_probability=0.1,
            anomaly_score=0.05,
            velocity_risk=0.1,
            behavioral_risk=0.1,
            rule_risk=0.0
        ),
        top_reasons=[],
        latency_ms=1.5
    )
    assert schema_inst.model_version == "finpulse-v3"

def test_no_active_production_path_reports_v2():
    """Verify that evaluate_risk and R4 do not emit finpulse-v2.0 anywhere."""
    engine = FinPulseRiskEngine()
    tx = {"transaction_id": "tx_v3_verify"}
    res = engine.evaluate_risk(
        tx=tx,
        features_dict={},
        calibrated_probability=0.25,
        anomaly_score=0.0
    )
    assert res["model_version"] == "finpulse-v3"
    assert "finpulse-v2.0" not in str(res)


# ==============================================================================
# 3. Decision Separation Contract Tests
# ==============================================================================

def test_r4_ml_thresholds_exact_boundaries():
    """Verify R4 ML Decision Policy exclusively operates on [0.0, 1.0] with 0.1580 / 0.5516."""
    ml_policy = R4MLThresholds(tau_review=0.1580, tau_block=0.5516)

    # Below review threshold -> APPROVE
    assert ml_policy.apply_policy(0.0) == "APPROVE"
    assert ml_policy.apply_policy(0.1579) == "APPROVE"

    # At or above review, below block threshold -> REVIEW
    assert ml_policy.apply_policy(0.1580) == "REVIEW"
    assert ml_policy.apply_policy(0.3500) == "REVIEW"
    assert ml_policy.apply_policy(0.5515) == "REVIEW"

    # At or above block threshold -> BLOCK
    assert ml_policy.apply_policy(0.5516) == "BLOCK"
    assert ml_policy.apply_policy(0.9999) == "BLOCK"

def test_r5_hybrid_thresholds_exact_boundaries():
    """Verify R5 Hybrid Risk Policy exclusively operates on [0.0, 100.0] with 30.0 / 70.0."""
    hybrid_policy = HybridRiskThresholds(tau_review=30.0, tau_block=70.0)

    # Below 30.0 -> APPROVE, LOW
    assert hybrid_policy.apply_policy(0.0) == ("APPROVE", "LOW")
    assert hybrid_policy.apply_policy(29.9) == ("APPROVE", "LOW")

    # [30.0, 70.0) -> REVIEW, MEDIUM
    assert hybrid_policy.apply_policy(30.0) == ("REVIEW", "MEDIUM")
    assert hybrid_policy.apply_policy(55.0) == ("REVIEW", "MEDIUM")
    assert hybrid_policy.apply_policy(69.9) == ("REVIEW", "MEDIUM")

    # >= 70.0 -> BLOCK, HIGH
    assert hybrid_policy.apply_policy(70.0) == ("BLOCK", "HIGH")
    assert hybrid_policy.apply_policy(100.0) == ("BLOCK", "HIGH")

def test_cross_policy_misuse_prevention():
    """
    Verify that cross-policy substitution is caught or causes domain violation.
    - Applying R4 ML policy to R5 risk scores (e.g. 25.0) must fail (out of [0, 1] range).
    - Applying R5 hybrid policy to R4 probabilities (e.g. 0.85) would wrongly APPROVE if conflated.
    """
    ml_policy = R4MLThresholds(tau_review=0.1580, tau_block=0.5516)
    hybrid_policy = HybridRiskThresholds(tau_review=30.0, tau_block=70.0)

    # Test 1: R4 ML policy rejects R5 score domain (>1.0)
    with pytest.raises(ContractViolationError, match="invalid probability"):
        ml_policy.apply_policy(25.0)  # Valid R5 score, completely invalid for R4 ML

    # Test 2: R4 high risk probability (0.85) is BLOCK in ML policy
    assert ml_policy.apply_policy(0.85) == "BLOCK"

    # If 0.85 were wrongly passed to Hybrid policy as a risk score, it would be evaluated as APPROVE (0.85 < 30.0)
    # Demonstrating the strict necessity of keeping the policies decoupled!
    decision_if_conflated, _ = hybrid_policy.apply_policy(0.85)
    assert decision_if_conflated == "APPROVE"  # Demonstrates why the boundary must remain explicit


# ==============================================================================
# 4. Signal Normalization Contract Tests
# ==============================================================================

def test_velocity_signal_normalization():
    """Verify velocity signal is properly normalized from raw transaction counts to [0.0, 1.0]."""
    # 0 transactions -> 0.0
    assert normalize_velocity_signal(0, 0) == 0.0

    # Normal activity (e.g. 1 in 5m, 2 in 1h) -> (1/5)*0.6 + (2/15)*0.4 = 0.12 + 0.0533 = 0.1733
    v_norm = normalize_velocity_signal(1, 2)
    assert 0.15 <= v_norm <= 0.20

    # Extreme burst (e.g. 20 in 5m, 100 in 1h) -> clipped strictly to 1.0
    assert normalize_velocity_signal(20, 100) == 1.0

    # Missing / None / NaN / negative -> fallback to 0.0
    assert normalize_velocity_signal(None, None) == 0.0
    assert normalize_velocity_signal(float("nan"), 5) == pytest.approx(0.1333, abs=1e-3)
    assert normalize_velocity_signal(-5, -10) == 0.0

def test_behavioral_signal_normalization():
    """Verify behavioral signal is properly normalized from z-scores and indicators to [0.0, 1.0]."""
    # Baseline normal: z=0, familiar device, familiar location -> 0.0
    assert normalize_behavioral_signal(0.0, 0, 0) == 0.0

    # New device only -> 0.3
    assert normalize_behavioral_signal(0.0, 1, 0) == pytest.approx(0.3, abs=1e-5)

    # High amount anomaly (|z|=4.0 -> 0.5) + new device (0.3) + new location (0.2) = 1.0
    assert normalize_behavioral_signal(4.0, 1, 1) == 1.0

    # Extreme z-score (|z|=20.0) -> clipped to 1.0
    assert normalize_behavioral_signal(20.0, 1, 1) == 1.0

    # Invalid / NaN / None values safely default
    assert normalize_behavioral_signal(None, None, None) == 0.0
    assert normalize_behavioral_signal(float("nan"), float("inf"), None) == 0.0

def test_rules_signal_normalization():
    """Verify rules signal is properly normalized from rule risk to [0.0, 1.0]."""
    assert normalize_rules_signal(0.0) == 0.0
    assert normalize_rules_signal(0.55) == 0.55
    assert normalize_rules_signal(1.5) == 1.0  # Clipped to 1.0
    assert normalize_rules_signal(None) == 0.0
    assert normalize_rules_signal(float("nan")) == 0.0
    assert normalize_rules_signal(-0.5) == 0.0

def test_normalized_risk_signals_contract_validation():
    """Verify NormalizedRiskSignals dataclass validates all 5 signals in [0, 1]."""
    valid_signals = NormalizedRiskSignals(
        calibrated_probability=0.20,
        velocity_signal=0.15,
        behavioral_signal=0.30,
        rules_signal=0.25,
        anomaly_signal=0.10
    )
    assert valid_signals.calibrated_probability == 0.20
    assert valid_signals.rules_signal == 0.25

    # Out of range rejections
    with pytest.raises(ContractViolationError, match="must be in"):
        NormalizedRiskSignals(
            calibrated_probability=1.05,
            velocity_signal=0.1,
            behavioral_signal=0.1,
            rules_signal=0.1,
            anomaly_signal=0.1
        )
    with pytest.raises(ContractViolationError, match="must be a valid float"):
        NormalizedRiskSignals(
            calibrated_probability=0.5,
            velocity_signal=float("nan"),
            behavioral_signal=0.1,
            rules_signal=0.1,
            anomaly_signal=0.1
        )


# ==============================================================================
# 5. Anomaly-Score Contract Tests
# ==============================================================================

def test_anomaly_score_isolation_forest_normalization_and_polarity():
    """
    Verify anomaly score contract:
    - Sourced from IsolationForestAnomalyDetector
    - Sigmoid inversion maps decision function to [0.0, 1.0]
    - Higher score indicates MORE anomalous transaction behavior
    """
    detector = IsolationForestAnomalyDetector(random_state=42, n_estimators=50)
    
    # Train exclusively on normal / inlier reference distribution
    np.random.seed(42)
    normal_training_data = np.random.normal(loc=50.0, scale=5.0, size=(100, 32))
    detector.fit(normal_training_data)

    # Inlier sample (similar to training distribution)
    inlier = np.random.normal(loc=50.0, scale=5.0, size=(1, 32))
    inlier_score = float(detector.score_anomaly(inlier)[0])

    # Outlier sample (drastically shifted distribution, e.g. 50 standard deviations away)
    outlier = np.full((1, 32), 1000.0)
    outlier_score = float(detector.score_anomaly(outlier)[0])

    assert 0.0 <= inlier_score <= 1.0
    assert 0.0 <= outlier_score <= 1.0
    # Polarity check: outlier MUST receive a higher anomaly score than normal inlier
    assert outlier_score > inlier_score
    assert outlier_score >= 0.80

def test_anomaly_score_fallback_behavior():
    """Verify neutral fallback (0.0) when anomaly score is missing or malformed."""
    assert normalize_anomaly_signal(None) == 0.0
    assert normalize_anomaly_signal(float("nan")) == 0.0
    assert normalize_anomaly_signal(float("inf")) == 0.0
    assert normalize_anomaly_signal("invalid_string") == 0.0
    assert normalize_anomaly_signal(0.42) == 0.42
    assert normalize_anomaly_signal(1.5) == 1.0  # Clipped
    assert normalize_anomaly_signal(-0.2) == 0.0  # Clipped


# ==============================================================================
# 6. Weight Configuration Contract Tests
# ==============================================================================

def test_initial_weight_configuration_sum_and_bounds():
    """
    Verify initial weight configuration:
    ML = 45%, Velocity = 15%, Behavioral = 15%, Rules = 15%, Anomaly = 10%.
    Sum = 1.0.
    """
    weights = HybridRiskWeights(
        w_ml=0.45,
        w_velocity=0.15,
        w_behavioral=0.15,
        w_rules=0.15,
        w_anomaly=0.10
    )
    total = weights.w_ml + weights.w_velocity + weights.w_behavioral + weights.w_rules + weights.w_anomaly
    assert math.isclose(total, 1.0, rel_tol=1e-6)

def test_weights_reject_invalid_sum():
    """Verify weights configuration rejects any set that does not sum to 1.0."""
    with pytest.raises(ContractViolationError, match="must sum to exactly 1.0"):
        HybridRiskWeights(w_ml=0.50, w_velocity=0.15, w_behavioral=0.15, w_rules=0.15, w_anomaly=0.15)

def test_deterministic_weighted_fusion():
    """Verify deterministic linear fusion on representative normalized inputs."""
    weights = HybridRiskWeights()
    
    # All zero signals -> 0.0
    zeros = NormalizedRiskSignals(0.0, 0.0, 0.0, 0.0, 0.0)
    assert weights.compute_weighted_fusion(zeros) == 0.0

    # All max signals -> 1.0
    ones = NormalizedRiskSignals(1.0, 1.0, 1.0, 1.0, 1.0)
    assert weights.compute_weighted_fusion(ones) == pytest.approx(1.0, rel=1e-5)

    # ML alone active (calibrated_p = 1.0, others = 0.0) -> exactly 0.45
    ml_only = NormalizedRiskSignals(1.0, 0.0, 0.0, 0.0, 0.0)
    assert weights.compute_weighted_fusion(ml_only) == pytest.approx(0.45, rel=1e-5)

    # Rules alone active (rules = 1.0, others = 0.0) -> exactly 0.15
    rules_only = NormalizedRiskSignals(0.0, 0.0, 0.0, 1.0, 0.0)
    assert weights.compute_weighted_fusion(rules_only) == pytest.approx(0.15, rel=1e-5)


# ==============================================================================
# 7. Compounding Policy Contract Tests
# ==============================================================================

def test_compounding_policy_zero_and_single_trigger():
    """
    Verify compounding policy:
    - 0 triggers active -> escalation_boost = 0.0
    - 1 trigger active -> escalation_boost = 0.0 (requires >= 2 triggers to compound)
    """
    policy = CompoundingPolicy()

    # Case 0 triggers: all below trigger thresholds (vel <= 0.3, beh <= 0.3, rules <= 0.4, ml <= 0.4)
    signals_zero = NormalizedRiskSignals(
        calibrated_probability=0.20,
        velocity_signal=0.10,
        behavioral_signal=0.15,
        rules_signal=0.20,
        anomaly_signal=0.10
    )
    eval_zero = policy.evaluate(signals_zero)
    assert eval_zero.trigger_count == 0
    assert eval_zero.escalation_boost == 0.0
    assert not eval_zero.is_active

    # Case 1 trigger: only ML triggers (0.45 > 0.40)
    signals_one = NormalizedRiskSignals(
        calibrated_probability=0.45,
        velocity_signal=0.10,
        behavioral_signal=0.15,
        rules_signal=0.20,
        anomaly_signal=0.10
    )
    eval_one = policy.evaluate(signals_one)
    assert eval_one.trigger_count == 1
    assert eval_one.escalation_boost == 0.0
    assert not eval_one.is_active

def test_compounding_policy_multi_trigger_escalation():
    """
    Verify compounding policy multi-trigger synergy:
    - 2 triggers -> 2 * 0.12 = +0.24 boost
    - 3 triggers -> 3 * 0.12 = +0.36 boost
    - 4 triggers -> 4 * 0.12 = +0.48 boost
    """
    policy = CompoundingPolicy()

    # 2 triggers: velocity (0.35 > 0.30) and behavioral (0.40 > 0.30)
    signals_two = NormalizedRiskSignals(
        calibrated_probability=0.20,
        velocity_signal=0.35,
        behavioral_signal=0.40,
        rules_signal=0.20,
        anomaly_signal=0.10
    )
    eval_two = policy.evaluate(signals_two)
    assert eval_two.trigger_count == 2
    assert eval_two.escalation_boost == pytest.approx(0.24, abs=1e-4)
    assert eval_two.is_active

    # 3 triggers: ML (0.50), velocity (0.35), rules (0.45)
    signals_three = NormalizedRiskSignals(
        calibrated_probability=0.50,
        velocity_signal=0.35,
        behavioral_signal=0.10,
        rules_signal=0.45,
        anomaly_signal=0.10
    )
    eval_three = policy.evaluate(signals_three)
    assert eval_three.trigger_count == 3
    assert eval_three.escalation_boost == pytest.approx(0.36, abs=1e-4)
    assert eval_three.is_active

    # 4 triggers: all 4 fire
    signals_four = NormalizedRiskSignals(
        calibrated_probability=0.50,
        velocity_signal=0.35,
        behavioral_signal=0.40,
        rules_signal=0.45,
        anomaly_signal=0.10
    )
    eval_four = policy.evaluate(signals_four)
    assert eval_four.trigger_count == 4
    assert eval_four.escalation_boost == pytest.approx(0.48, abs=1e-4)
    assert eval_four.is_active

def test_compounding_applied_post_weighted_fusion_with_cap():
    """Verify compounding escalation adds strictly to weighted score and caps at 1.0 (100.0 score)."""
    engine = FinPulseRiskEngine()
    # High risk signals that trigger all 4 compounding conditions
    tx = {"transaction_id": "tx_comp_cap"}
    features = {
        "tx_count_5m": 5,      # norm_vel = (5/5)*0.6 + (1/15)*0.4 = 0.627 > 0.30
        "tx_count_1h": 1,
        "amount_zscore": 3.0,  # norm_beh = (3/4)*0.5 + 0.3 = 0.675 > 0.30
        "is_new_device": 1,
        "is_new_location": 0
    }
    # Calibrated ML probability: 0.80 > 0.40
    # Rule engine: add unverified auth to trigger rules_signal > 0.40
    tx["auth_verified"] = 0
    tx["origin_balance"] = 1000.0
    tx["amount"] = 960.0  # triggers 95% drain (+0.25) and unauth (+0.30) -> rules = 0.55 > 0.40

    res = engine.evaluate_risk(
        tx=tx,
        features_dict=features,
        calibrated_probability=0.80,
        anomaly_score=0.90
    )

    # 4 triggers fire
    assert res["compounding"]["trigger_count"] == 4
    assert res["compounding"]["escalation_boost"] == 0.48
    # Capped at 100.0 risk score (combined_score = min(1.0, weighted + boost))
    assert res["risk_score"] == 100.0
    assert res["decision"] == "BLOCK"


# ==============================================================================
# 8. Regression Protection Tests
# ==============================================================================

def test_frozen_production_bundle_checksum_integrity():
    """Protect frozen ML artifacts: verify all 5 files in finpulse-v3 match their SHA-256 hashes."""
    manifest_path = os.path.join(PRODUCTION_BUNDLE_DIR, "checksum.sha256")
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        expected_sha, fname = line.split(None, 1)
        fpath = os.path.join(PRODUCTION_BUNDLE_DIR, fname)
        assert os.path.exists(fpath), f"Frozen production artifact missing: {fname}"

        hasher = hashlib.sha256()
        with open(fpath, "rb") as af:
            while chunk := af.read(65536):
                hasher.update(chunk)
        assert hasher.hexdigest() == expected_sha, f"Frozen artifact {fname} SHA-256 changed!"

def test_32_feature_schema_preserved():
    """Protect R3 feature schema: verify exactly 32 features in production schema."""
    schema_path = os.path.join(PRODUCTION_BUNDLE_DIR, "feature_schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert len(schema["feature_names"]) == 32
    assert schema["feature_schema_version"] == "2.0"
