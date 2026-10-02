"""Dedicated Test Suite for R5.1 — Production-Quality Hybrid Risk Engine Core.

Validates:
1. Deterministic Weighted Fusion (0.45 ML, 0.15 Velocity, 0.15 Behavioral, 0.15 Rules, 0.10 Anomaly).
2. Diagnostic Contributions (individual weighted contribution breakdown).
3. Exact Policy Decision Boundaries (29.999... -> APPROVE, 30.0 -> REVIEW, 69.999... -> REVIEW, 70.0 -> BLOCK).
4. Exact Risk Level Boundaries (<30.0 -> LOW, 30.0–69.999... -> MEDIUM, >=70.0 -> HIGH).
5. Comprehensive Input Validation (rejection of missing, None, non-numeric, negative, >1.0, NaN, Inf).
6. Mathematical Determinism (identical inputs yield bit-for-bit identical outputs across repeated runs).
7. Monotonicity (increasing any single risk signal while holding others constant cannot decrease score).
8. R4 ML vs R5 Hybrid Policy Independence (separate policies, no conflation of 0.158/0.5516 with 30/70).
9. Deterministic Hard-Block Overrides (cannot be downgraded by low signal fusion).
10. Model Metadata Preservation (authoritative finpulse-v3 propagated from R4 contract).
"""

import os
import sys
import math
import pytest
import numpy as np

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.hybrid import HybridRiskEngine, HybridRiskResult
from src.risk_engine.contract import (
    HybridRiskWeights,
    HybridRiskThresholds,
    R4MLThresholds,
    R4InferenceOutput,
    NormalizedRiskSignals,
    ContractViolationError,
)


@pytest.fixture
def engine():
    """Provides a fresh default instance of HybridRiskEngine."""
    return HybridRiskEngine()


# ==============================================================================
# 1. Deterministic Weighted Fusion Tests
# ==============================================================================

def test_weighted_fusion_all_zeros(engine):
    """Verify that when all 5 signals are 0.0, risk score is 0.0 and contributions are 0.0."""
    res = engine.evaluate(
        calibrated_probability=0.0,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0,
        transaction_id="tx_zero"
    )
    assert res.risk_score == 0.0
    assert res.decision == "APPROVE"
    assert res.risk_level == "LOW"
    assert res.ml_decision == "APPROVE"

    for k in ["ml_contribution", "velocity_contribution", "behavioral_contribution", "rules_contribution", "anomaly_contribution"]:
        assert res.contributions[k] == 0.0

def test_weighted_fusion_all_ones(engine):
    """Verify that when all 5 signals are 1.0, risk score is 100.0 and contributions match baseline weights."""
    res = engine.evaluate(
        calibrated_probability=1.0,
        velocity=1.0,
        behavioral=1.0,
        rules=1.0,
        anomaly=1.0,
        transaction_id="tx_ones"
    )
    assert res.risk_score == pytest.approx(100.0, abs=1e-4)
    assert res.decision == "BLOCK"
    assert res.risk_level == "HIGH"
    assert res.ml_decision == "BLOCK"

    assert res.contributions["ml_contribution"] == pytest.approx(0.45, abs=1e-5)
    assert res.contributions["velocity_contribution"] == pytest.approx(0.15, abs=1e-5)
    assert res.contributions["behavioral_contribution"] == pytest.approx(0.15, abs=1e-5)
    assert res.contributions["rules_contribution"] == pytest.approx(0.15, abs=1e-5)
    assert res.contributions["anomaly_contribution"] == pytest.approx(0.10, abs=1e-5)

def test_weighted_fusion_mixed_signals(engine):
    """
    Verify exact linear weighted fusion calculation for representative mixed values:
    ml=0.80, velocity=0.60, behavioral=0.40, rules=0.20, anomaly=0.50
    Expected normalized score:
      0.45*0.80 + 0.15*0.60 + 0.15*0.40 + 0.15*0.20 + 0.10*0.50
      = 0.360 + 0.090 + 0.060 + 0.030 + 0.050 = 0.590
    Expected risk_score: 59.0
    """
    res = engine.evaluate(
        calibrated_probability=0.80,
        velocity=0.60,
        behavioral=0.40,
        rules=0.20,
        anomaly=0.50,
        transaction_id="tx_mixed"
    )
    assert res.risk_score == pytest.approx(59.0, abs=1e-4)
    assert res.decision == "REVIEW"
    assert res.risk_level == "MEDIUM"
    assert res.ml_decision == "BLOCK"  # R4 policy: 0.80 >= 0.5516

    # Verify individual contributions sum to normalized score (0.59)
    sum_contributions = sum(res.contributions.values())
    assert sum_contributions == pytest.approx(0.59, abs=1e-4)
    assert res.ml_contribution == pytest.approx(0.36, abs=1e-5)
    assert res.velocity_contribution == pytest.approx(0.09, abs=1e-5)
    assert res.behavioral_contribution == pytest.approx(0.06, abs=1e-5)
    assert res.rules_contribution == pytest.approx(0.03, abs=1e-5)
    assert res.anomaly_contribution == pytest.approx(0.05, abs=1e-5)


# ==============================================================================
# 2. Decision & Risk-Level Policy Boundary Tests
# ==============================================================================

def test_exact_boundary_approaching_30(engine):
    """
    Explicitly test threshold boundary just below 30.0:
    29.999... -> APPROVE, LOW
    """
    # Create signals that sum to 0.29999
    # e.g., ml = 0.29999 / 0.45 = 0.6666444... and others 0.0
    ml_val = 29.999 / 45.0  # ~0.666644
    res = engine.evaluate(
        calibrated_probability=ml_val,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0
    )
    assert res.risk_score < 30.0
    assert res.decision == "APPROVE"
    assert res.risk_level == "LOW"

def test_exact_boundary_at_30(engine):
    """
    Explicitly test exact threshold boundary at 30.0:
    30.0 -> REVIEW, MEDIUM
    """
    # ml = 30.0 / 45.0 = 2/3
    ml_val = 30.0 / 45.0
    res = engine.evaluate(
        calibrated_probability=ml_val,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0
    )
    assert res.risk_score == pytest.approx(30.0, abs=1e-4)
    assert res.decision == "REVIEW"
    assert res.risk_level == "MEDIUM"

def test_exact_boundary_approaching_70(engine):
    """
    Explicitly test threshold boundary just below 70.0:
    69.999... -> REVIEW, MEDIUM
    """
    # Construct signals yielding exactly 69.999
    # ml=1.0 (45.0), vel=1.0 (15.0), beh=(9.999/15.0), rules=0, anomaly=0 -> total = 69.999
    beh_val = 9.999 / 15.0
    res = engine.evaluate(
        calibrated_probability=1.0,
        velocity=1.0,
        behavioral=beh_val,
        rules=0.0,
        anomaly=0.0
    )
    assert res.risk_score < 70.0
    assert res.decision == "REVIEW"
    assert res.risk_level == "MEDIUM"

def test_exact_boundary_at_70(engine):
    """
    Explicitly test exact threshold boundary at 70.0:
    70.0 -> BLOCK, HIGH
    """
    # ml=1.0 (45.0), vel=1.0 (15.0), beh=(10.0/15.0) (10.0) -> total = 70.0
    beh_val = 10.0 / 15.0
    res = engine.evaluate(
        calibrated_probability=1.0,
        velocity=1.0,
        behavioral=beh_val,
        rules=0.0,
        anomaly=0.0
    )
    assert res.risk_score == pytest.approx(70.0, abs=1e-4)
    assert res.decision == "BLOCK"
    assert res.risk_level == "HIGH"


# ==============================================================================
# 3. Input Validation Tests
# ==============================================================================

def test_validation_missing_signals(engine):
    """Verify that omitting any of the 5 signals raises ContractViolationError."""
    # Missing ML / calibrated_probability
    with pytest.raises(ContractViolationError, match="calibrated_probability.*cannot be None"):
        engine.evaluate(velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    # Missing velocity
    with pytest.raises(ContractViolationError, match="velocity.*cannot be None"):
        engine.evaluate(calibrated_probability=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    # Missing behavioral
    with pytest.raises(ContractViolationError, match="behavioral.*cannot be None"):
        engine.evaluate(calibrated_probability=0.5, velocity=0.5, rules=0.5, anomaly=0.5)

    # Missing rules
    with pytest.raises(ContractViolationError, match="rules.*cannot be None"):
        engine.evaluate(calibrated_probability=0.5, velocity=0.5, behavioral=0.5, anomaly=0.5)

    # Missing anomaly
    with pytest.raises(ContractViolationError, match="anomaly.*cannot be None"):
        engine.evaluate(calibrated_probability=0.5, velocity=0.5, behavioral=0.5, rules=0.5)

def test_validation_null_none_values(engine):
    """Verify that passing None for any signal raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="cannot be None"):
        engine.evaluate(calibrated_probability=None, velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match="cannot be None"):
        engine.evaluate(calibrated_probability=0.5, velocity=None, behavioral=0.5, rules=0.5, anomaly=0.5)

def test_validation_non_numeric_types(engine):
    """Verify that passing non-numeric types (strings, lists, booleans) raises ContractViolationError."""
    with pytest.raises(ContractViolationError, match="must be numeric"):
        engine.evaluate(calibrated_probability="high", velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match="must be a numeric float, got boolean"):
        engine.evaluate(calibrated_probability=True, velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match="must be numeric"):
        engine.evaluate(calibrated_probability=0.5, velocity=[0.5], behavioral=0.5, rules=0.5, anomaly=0.5)

def test_validation_negative_values(engine):
    """Verify that negative signal values (< 0.0) raise ContractViolationError."""
    with pytest.raises(ContractViolationError, match=r"strictly in \[0.0, 1.0\]"):
        engine.evaluate(calibrated_probability=-0.001, velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match=r"strictly in \[0.0, 1.0\]"):
        engine.evaluate(calibrated_probability=0.5, velocity=-0.1, behavioral=0.5, rules=0.5, anomaly=0.5)

def test_validation_above_max_range(engine):
    """Verify that signal values > 1.0 raise ContractViolationError."""
    with pytest.raises(ContractViolationError, match=r"strictly in \[0.0, 1.0\]"):
        engine.evaluate(calibrated_probability=1.0001, velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match=r"strictly in \[0.0, 1.0\]"):
        engine.evaluate(calibrated_probability=0.5, velocity=1.5, behavioral=0.5, rules=0.5, anomaly=0.5)

def test_validation_nan_and_infinity(engine):
    """Verify that NaN and Infinite signal values raise ContractViolationError."""
    with pytest.raises(ContractViolationError, match="cannot be NaN or Infinite"):
        engine.evaluate(calibrated_probability=float("nan"), velocity=0.5, behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match="cannot be NaN or Infinite"):
        engine.evaluate(calibrated_probability=0.5, velocity=float("inf"), behavioral=0.5, rules=0.5, anomaly=0.5)

    with pytest.raises(ContractViolationError, match="cannot be NaN or Infinite"):
        engine.evaluate(calibrated_probability=0.5, velocity=0.5, behavioral=float("-inf"), rules=0.5, anomaly=0.5)


# ==============================================================================
# 4. Mathematical Determinism Tests
# ==============================================================================

def test_evaluation_determinism(engine):
    """Verify that 100 repeated executions with identical inputs produce identical bit-for-bit results."""
    inputs = {
        "calibrated_probability": 0.4215,
        "velocity": 0.6500,
        "behavioral": 0.3333,
        "rules": 0.7000,
        "anomaly": 0.1250,
        "transaction_id": "tx_repeat_test"
    }

    first_res = engine.evaluate(**inputs)

    for _ in range(100):
        res = engine.evaluate(**inputs)
        assert res.risk_score == first_res.risk_score
        assert res.decision == first_res.decision
        assert res.risk_level == first_res.risk_level
        assert res.ml_decision == first_res.ml_decision
        assert res.contributions == first_res.contributions
        assert res.signals == first_res.signals


# ==============================================================================
# 5. Monotonicity Tests
# ==============================================================================

@pytest.mark.parametrize("signal_name", ["calibrated_probability", "velocity", "behavioral", "rules", "anomaly"])
def test_monotonicity_individual_signals(engine, signal_name):
    """
    Verify that increasing an individual risk signal while holding all other 4 constant
    cannot decrease the hybrid risk score.
    """
    base_inputs = {
        "calibrated_probability": 0.30,
        "velocity": 0.30,
        "behavioral": 0.30,
        "rules": 0.30,
        "anomaly": 0.30
    }

    prev_score = -1.0
    # Sweep signal from 0.0 to 1.0 in steps of 0.1
    for step in np.linspace(0.0, 1.0, 11):
        test_inputs = dict(base_inputs)
        test_inputs[signal_name] = float(step)
        res = engine.evaluate(**test_inputs)

        assert res.risk_score >= prev_score, (
            f"Monotonicity violation on '{signal_name}': step {step} produced score {res.risk_score} < prev {prev_score}"
        )
        prev_score = res.risk_score


# ==============================================================================
# 6. ML / R5 Policy Independence Tests
# ==============================================================================

def test_ml_and_r5_policy_independence_high_ml_low_hybrid(engine):
    """
    Case 1: High ML fraud probability (0.85 -> R4 BLOCK) but low other signals:
    Weighted score: 0.45 * 0.85 = 38.25 -> R5 REVIEW!
    Must cleanly report:
      ml_decision = "BLOCK"
      decision    = "REVIEW"
    """
    res = engine.evaluate(
        calibrated_probability=0.85,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0
    )
    assert res.risk_score == pytest.approx(38.25, abs=1e-4)
    assert res.ml_decision == "BLOCK"   # R4 threshold: 0.85 >= 0.5516
    assert res.decision == "REVIEW"     # R5 threshold: 30.0 <= 38.25 < 70.0
    assert res.risk_level == "MEDIUM"

def test_ml_and_r5_policy_independence_low_ml_high_hybrid(engine):
    """
    Case 2: Low ML probability (0.10 -> R4 APPROVE) but high other signals:
    0.45*0.10 + 0.15*1.0 + 0.15*1.0 + 0.15*1.0 + 0.10*1.0 = 0.045 + 0.55 = 0.595 -> 59.5
    Must cleanly report:
      ml_decision = "APPROVE"
      decision    = "REVIEW"
    """
    res = engine.evaluate(
        calibrated_probability=0.10,
        velocity=1.0,
        behavioral=1.0,
        rules=1.0,
        anomaly=1.0
    )
    assert res.risk_score == pytest.approx(59.5, abs=1e-4)
    assert res.ml_decision == "APPROVE"  # R4 threshold: 0.10 < 0.1580
    assert res.decision == "REVIEW"      # R5 threshold: 30.0 <= 59.5 < 70.0


# ==============================================================================
# 7. Hard Overrides Tests
# ==============================================================================

def test_hard_block_override_enforcement(engine):
    """
    Verify that when hard_block is True:
    - Decision is strictly BLOCK
    - Risk level is strictly HIGH
    - Score is at least 85.0 even if all 5 signals are 0.0
    - A hard BLOCK is NEVER downgraded by weighted fusion
    """
    # Case A: All signals are 0.0, but hard override is active
    res_zero = engine.evaluate(
        calibrated_probability=0.0,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0,
        hard_block=True
    )
    assert res_zero.decision == "BLOCK"
    assert res_zero.risk_level == "HIGH"
    assert res_zero.risk_score >= 85.0
    assert res_zero.hard_block is True

    # Case B: Naturally high score (e.g. 95.0) remains 95.0
    res_high = engine.evaluate(
        calibrated_probability=1.0,
        velocity=1.0,
        behavioral=1.0,
        rules=1.0,
        anomaly=0.5,
        hard_block=True
    )
    assert res_high.decision == "BLOCK"
    assert res_high.risk_level == "HIGH"
    assert res_high.risk_score == pytest.approx(95.0, abs=1e-4)


# ==============================================================================
# 8. Model Metadata Preservation Tests
# ==============================================================================

def test_model_version_preservation_from_r4_contract(engine):
    """Verify that R5 propagates model_version from R4InferenceOutput contract."""
    r4_out = R4InferenceOutput(
        transaction_id="tx_meta_test",
        calibrated_probability=0.25,
        model_version="finpulse-v3",
        raw_probability=0.20,
        ml_decision="APPROVE"
    )

    res = engine.evaluate(
        r4_output=r4_out,
        velocity=0.2,
        behavioral=0.2,
        rules=0.2,
        anomaly=0.2
    )

    assert res.model_version == "finpulse-v3"
    assert res.transaction_id == "tx_meta_test"
    assert res.ml_decision == "APPROVE"
    assert "finpulse-v2.0" not in str(res.to_dict())

def test_structured_result_contract_shape(engine):
    """Verify target contract shape matches R5.1 specification and supports dict indexing."""
    res = engine.evaluate(
        calibrated_probability=0.73,
        velocity=0.82,
        behavioral=0.61,
        rules=0.50,
        anomaly=0.77,
        transaction_id="tx_001"
    )

    d = res.to_dict()
    assert d["transaction_id"] == "tx_001"
    assert d["model_version"] == "finpulse-v3"
    assert d["calibrated_probability"] == pytest.approx(0.73, abs=1e-4)
    assert d["signals"]["ml"] == pytest.approx(0.73, abs=1e-4)
    assert d["signals"]["velocity"] == pytest.approx(0.82, abs=1e-4)
    assert d["signals"]["behavioral"] == pytest.approx(0.61, abs=1e-4)
    assert d["signals"]["rules"] == pytest.approx(0.50, abs=1e-4)
    assert d["signals"]["anomaly"] == pytest.approx(0.77, abs=1e-4)
    assert d["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert d["decision"] in ["APPROVE", "REVIEW", "BLOCK"]
    assert d["ml_decision"] in ["APPROVE", "REVIEW", "BLOCK"]

    # Verify subscript indexing
    assert res["transaction_id"] == "tx_001"
    assert res["risk_score"] == d["risk_score"]
    assert "risk_score" in res
