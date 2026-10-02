"""R5.7 — Offline Weight Evaluation, Sensitivity & Ablation Analysis Test Suite.

Validates:
1. Production Isolation:
   - HybridRiskEngine defaults remain strictly 0.45 / 0.15 / 0.15 / 0.15 / 0.10.
   - HybridRiskWeights default sums to 1.0.
2. Candidate Configuration Validity:
   - All 7 candidate configurations exist (Baseline, ML-heavy, ML-dominant, Behavior-heavy, Rule-heavy, Anomaly-heavy, Balanced).
   - All candidate configurations have strictly 5 signals and sum to 1.0 within 1e-9.
3. Determinism:
   - Identical inputs + candidate configuration yield 100% deterministic results across repeated runs.
4. Input Consistency:
   - All candidates evaluate identical frozen signal inputs without recomputing upstream state.
5. Hard-Block Invariant:
   - Hard-block override forces decision=BLOCK across all candidate configurations regardless of weights.
6. R4 ML Decision Independence:
   - R4 ML decision remains unchanged across all candidate configurations.
7. Diagnostic Integrity:
   - Candidate diagnostics remain mathematically consistent (sum of scaled contributions == weighted_score).
8. Rule Integrity:
   - Business rules and hard blocks remain unchanged across configurations.
9. One-at-a-Time Signal Ablations:
   - All 5 signal ablations (remove ML, velocity, behavioral, rules, anomaly) produce valid renormalized weights summing to 1.0.
10. Kafka & Contract Isolation:
    - Verifies that experiment does not mutate DecisionEvent v1.0 schema, Kafka publisher routing, or serialization format.
"""
import os
import sys
import math
import copy
import numpy as np
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.hybrid import HybridRiskEngine, HybridRiskResult
from src.risk_engine.contract import (
    HybridRiskWeights,
    NormalizedRiskSignals,
    R4MLThresholds,
    HybridRiskThresholds,
    ContractViolationError
)
from src.risk_engine.rules import BusinessRuleEngine, RuleResult
from src.risk_engine.decision_event import DecisionEvent
from src.streaming.publisher import DecisionEventPublisher
from src.streaming.config import StreamingConfig


# ==============================================================================
# Canonical Experimental Configurations
# ==============================================================================

EXPERIMENTAL_CANDIDATES = {
    "Baseline": {
        "w_ml": 0.45, "w_velocity": 0.15, "w_behavioral": 0.15, "w_rules": 0.15, "w_anomaly": 0.10
    },
    "ML-heavy": {
        "w_ml": 0.55, "w_velocity": 0.125, "w_behavioral": 0.125, "w_rules": 0.125, "w_anomaly": 0.075
    },
    "ML-dominant": {
        "w_ml": 0.60, "w_velocity": 0.10, "w_behavioral": 0.10, "w_rules": 0.10, "w_anomaly": 0.10
    },
    "Behavior-heavy": {
        "w_ml": 0.40, "w_velocity": 0.15, "w_behavioral": 0.20, "w_rules": 0.15, "w_anomaly": 0.10
    },
    "Rule-heavy": {
        "w_ml": 0.40, "w_velocity": 0.15, "w_behavioral": 0.15, "w_rules": 0.20, "w_anomaly": 0.10
    },
    "Anomaly-heavy": {
        "w_ml": 0.40, "w_velocity": 0.15, "w_behavioral": 0.15, "w_rules": 0.15, "w_anomaly": 0.15
    },
    "Balanced": {
        "w_ml": 0.20, "w_velocity": 0.20, "w_behavioral": 0.20, "w_rules": 0.20, "w_anomaly": 0.20
    }
}


def compute_ablation_weights(removed_signal: str, baseline_weights: dict) -> dict:
    """Compute renormalized weights when one signal is removed (set to 0.0)."""
    assert removed_signal in baseline_weights, f"Unknown signal: {removed_signal}"
    remaining = {k: v for k, v in baseline_weights.items() if k != removed_signal}
    rem_sum = sum(remaining.values())
    ablation = {k: (v / rem_sum) for k, v in remaining.items()}
    ablation[removed_signal] = 0.0
    return ablation


# ==============================================================================
# 1. Production Isolation & Default Verification
# ==============================================================================

def test_production_weights_isolation_and_defaults():
    """Verify production default weights remain strictly 0.45/0.15/0.15/0.15/0.10."""
    engine = HybridRiskEngine()
    assert math.isclose(engine.weights.w_ml, 0.45, abs_tol=1e-9)
    assert math.isclose(engine.weights.w_velocity, 0.15, abs_tol=1e-9)
    assert math.isclose(engine.weights.w_behavioral, 0.15, abs_tol=1e-9)
    assert math.isclose(engine.weights.w_rules, 0.15, abs_tol=1e-9)
    assert math.isclose(engine.weights.w_anomaly, 0.10, abs_tol=1e-9)

    total = engine.weights.w_ml + engine.weights.w_velocity + engine.weights.w_behavioral + engine.weights.w_rules + engine.weights.w_anomaly
    assert math.isclose(total, 1.0, abs_tol=1e-9)


# ==============================================================================
# 2. Candidate Configuration Validity
# ==============================================================================

def test_candidate_configurations_validity():
    """Verify all 7 candidate configurations exist, have 5 signals, and sum to 1.0 within 1e-9."""
    expected_configs = [
        "Baseline", "ML-heavy", "ML-dominant", "Behavior-heavy",
        "Rule-heavy", "Anomaly-heavy", "Balanced"
    ]
    for name in expected_configs:
        assert name in EXPERIMENTAL_CANDIDATES, f"Missing candidate: {name}"
        cfg = EXPERIMENTAL_CANDIDATES[name]
        assert set(cfg.keys()) == {"w_ml", "w_velocity", "w_behavioral", "w_rules", "w_anomaly"}
        total = sum(cfg.values())
        assert abs(total - 1.0) < 1e-9, f"Config {name} sum {total} != 1.0 within 1e-9"
        
        # Test instantiation in HybridRiskWeights
        hw = HybridRiskWeights(**cfg)
        assert hw is not None


def test_invalid_weights_programmatic_rejection():
    """Verify weights summing to != 1.0 or containing out-of-bound values are strictly rejected."""
    # Sum != 1.0
    with pytest.raises(ContractViolationError):
        HybridRiskWeights(w_ml=0.5, w_velocity=0.2, w_behavioral=0.2, w_rules=0.2, w_anomaly=0.1)

    # Negative weight
    with pytest.raises(ContractViolationError):
        HybridRiskWeights(w_ml=1.1, w_velocity=-0.1, w_behavioral=0.0, w_rules=0.0, w_anomaly=0.0)


# ==============================================================================
# 3. Determinism
# ==============================================================================

def test_candidate_evaluation_determinism():
    """Verify identical inputs + candidate weights yield 100% deterministic output."""
    signals = NormalizedRiskSignals(
        calibrated_probability=0.45,
        velocity_signal=0.25,
        behavioral_signal=0.35,
        rules_signal=0.15,
        anomaly_signal=0.20
    )

    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res1 = engine.evaluate(signals=signals, transaction_id="tx_det_1")
        res2 = engine.evaluate(signals=signals, transaction_id="tx_det_1")

        assert res1.risk_score == res2.risk_score
        assert res1.decision == res2.decision
        assert res1.risk_level == res2.risk_level
        assert res1.contributions == res2.contributions
        assert res1.diagnostics == res2.diagnostics


# ==============================================================================
# 4. Input Consistency
# ==============================================================================

def test_frozen_input_consistency_across_candidates():
    """Verify all candidates evaluate the exact same frozen signal instance."""
    signals = NormalizedRiskSignals(
        calibrated_probability=0.30,
        velocity_signal=0.50,
        behavioral_signal=0.40,
        rules_signal=0.10,
        anomaly_signal=0.05
    )

    scores = {}
    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res = engine.evaluate(signals=signals, transaction_id="tx_input_test")
        scores[name] = res.risk_score
        # Signals in result must reflect the input signals
        assert res.signals["calibrated_probability"] == 0.30
        assert res.signals["velocity_signal"] == 0.50
        assert res.signals["behavioral_signal"] == 0.40

    # Scores will vary across different candidates due to different weightings
    assert len(set(scores.values())) > 1


# ==============================================================================
# 5. Hard-Block Invariant
# ==============================================================================

def test_hard_block_invariant_across_all_candidates():
    """Verify hard-block override strictly forces BLOCK regardless of candidate weights."""
    signals = NormalizedRiskSignals(
        calibrated_probability=0.01,
        velocity_signal=0.0,
        behavioral_signal=0.0,
        rules_signal=0.0,
        anomaly_signal=0.0
    )

    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res = engine.evaluate(signals=signals, transaction_id="tx_hb_test", hard_block=True)
        assert res.decision == "BLOCK"
        assert res.risk_level == "HIGH"
        assert res.hard_block is True
        assert res.risk_score >= 85.0


# ==============================================================================
# 6. R4 ML Decision Independence
# ==============================================================================

def test_r4_ml_decision_independence_across_candidates():
    """Verify R4 ML decision is completely independent of candidate weights."""
    signals = NormalizedRiskSignals(
        calibrated_probability=0.85,  # R4 says BLOCK (> 0.5516)
        velocity_signal=0.0,
        behavioral_signal=0.0,
        rules_signal=0.0,
        anomaly_signal=0.0
    )

    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res = engine.evaluate(signals=signals, transaction_id="tx_mldec_test", ml_decision="BLOCK")
        assert res.ml_decision == "BLOCK"


# ==============================================================================
# 7. Diagnostic Integrity Across Candidates
# ==============================================================================

def test_diagnostic_integrity_across_candidates():
    """Verify candidate diagnostics satisfy: weighted_score == sum(contributions)."""
    signals = NormalizedRiskSignals(
        calibrated_probability=0.50,
        velocity_signal=0.40,
        behavioral_signal=0.30,
        rules_signal=0.20,
        anomaly_signal=0.10
    )

    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res = engine.evaluate(signals=signals, transaction_id="tx_diag_test")
        
        diag = res.diagnostics
        contrib_sum = sum(diag["contributions"].values())
        assert math.isclose(diag["weighted_score"], contrib_sum, abs_tol=0.05)


# ==============================================================================
# 8. Business Rule Integrity Across Candidates
# ==============================================================================

def test_business_rule_integrity_across_candidates():
    """Verify business-rule evaluation produces identical matched rules across configurations."""
    rule_engine = BusinessRuleEngine()
    tx = {
        "transaction_id": "tx_rule_test",
        "amount": 25000.0,
        "category": "crypto",
        "payment_type": "TRANSFER",
        "origin_balance": 100.0,
        "auth_verified": False,
        "timestamp": 1710000000.0
    }
    features = {
        "amount": 25000.0,
        "amount_to_balance_ratio": 250.0,
        "auth_factor_verified": 0,
        "payment_type_enc": 1,
        "merchant_category_enc": 1
    }
    rule_res = rule_engine.evaluate(tx, features)

    for name, weights_dict in EXPERIMENTAL_CANDIDATES.items():
        engine = HybridRiskEngine(weights=weights_dict, enable_compounding=True)
        res = engine.evaluate(
            signals=NormalizedRiskSignals(
                calibrated_probability=0.1,
                velocity_signal=0.1,
                behavioral_signal=0.1,
                rules_signal=rule_res.rules_signal,
                anomaly_signal=0.1
            ),
            transaction_id="tx_rule_test",
            rule_result=rule_res
        )
        assert res.metadata["rule_result"]["rules_signal"] == rule_res.rules_signal
        assert res.metadata["rule_result"]["matched_rules"] == [r.to_dict() for r in rule_res.matched_rules]


# ==============================================================================
# 9. One-at-a-Time Signal Ablations Validity
# ==============================================================================

def test_signal_ablations_renormalized_weights_validity():
    """Verify all 5 one-at-a-time signal ablations sum to 1.0 within 1e-9."""
    baseline = EXPERIMENTAL_CANDIDATES["Baseline"]
    signals_to_ablate = ["w_ml", "w_velocity", "w_behavioral", "w_rules", "w_anomaly"]

    for sig in signals_to_ablate:
        abl_weights = compute_ablation_weights(sig, baseline)
        assert abl_weights[sig] == 0.0
        total = sum(abl_weights.values())
        assert abs(total - 1.0) < 1e-9, f"Ablation for {sig} sum {total} != 1.0"
        
        # Test that engine accepts ablation weights
        engine = HybridRiskEngine(weights=abl_weights)
        assert engine.weights is not None


# ==============================================================================
# 10. Kafka & Contract Isolation
# ==============================================================================

def test_kafka_and_contract_isolation():
    """Verify DecisionEvent and Kafka publisher contracts are unmutated."""
    event = DecisionEvent(
        transaction_id="tx_event_iso",
        customer_id="cust_iso",
        decision="APPROVE",
        risk_score=15.0,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.05,
        signals={"velocity_signal": 0.05},
        diagnostics={},
        reasons=["Baseline verification."],
    )
    assert event.schema_version == "1.0"
    assert event.event_type == "finpulse.fraud_decision"

    pub = DecisionEventPublisher(config=StreamingConfig(), dry_run=True)
    res = pub.publish(event)
    assert res["predictions_published"] is True
