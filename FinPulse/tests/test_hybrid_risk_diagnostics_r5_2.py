"""R5.2 Dedicated Test Suite — Hybrid Risk Engine Diagnostics Attribution.

Validates the 16 mandatory R5.2 diagnostic requirements:
1. Individual Contributions: Mathematically exact contribution per signal.
2. Contribution Sum: sum(contributions.values()) == weighted_score within tolerance.
3. Trigger Identification: active_triggers matches exact compounding trigger thresholds.
4. Trigger Count: trigger_count == len(active_triggers).
5. Escalation Boost: K >= 2 -> K * 12.0; K < 2 -> 0.0.
6. Score Reconciliation: weighted_score + escalation_boost == final_score (clamped at 100.0).
7. No Trigger (K = 0): escalation_boost == 0.0, final_score == weighted_score.
8. One Trigger (K = 1): escalation does not apply when K < 2.
9. Two Triggers (K = 2): 2 * 12.0 = +24.0 escalation boost.
10. Four Triggers (K = 4): 4 * 12.0 = +48.0 escalation boost.
11. Hard Block: hard_block=True reflected in diagnostics, decision is BLOCK, score >= 85.0.
12. Zero Signals: All zero inputs produce valid, finite 0.0 values with no NaN or Inf.
13. All Signals = 1: Full weights, K=4, pre-clamp score 148.0, clamped final score 100.0.
14. Determinism: Repeated evaluations produce byte-for-byte identical serialized output.
15. R4 ML Decision Independence: ml_decision strictly preserved and separated from decision.
16. R5.1 Backward Compatibility: All existing fields and dictionary semantics intact.
"""

import json
import math
import os
import sys
import pytest

# Ensure FinPulse root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.hybrid import HybridRiskEngine, HybridRiskResult, HybridRiskDiagnostics
from src.risk_engine.contract import (
    HybridRiskWeights,
    HybridRiskThresholds,
    R4MLThresholds,
    CompoundingPolicy,
    NormalizedRiskSignals,
)


@pytest.fixture
def engine():
    """Default HybridRiskEngine with compounding enabled for diagnostics evaluation."""
    return HybridRiskEngine(enable_compounding=True)


@pytest.fixture
def pure_fusion_engine():
    """Default HybridRiskEngine without compounding (pure linear weighted fusion)."""
    return HybridRiskEngine(enable_compounding=False)


# ==============================================================================
# 1. Individual Contributions
# ==============================================================================

def test_diagnostics_individual_contributions_isolated(engine):
    """
    Verify each signal's contribution in isolation (signal=1.0, all others=0.0).
    Expected contributions on 0-100 scale:
    ML:         1.0 * 0.45 * 100 = 45.0
    Velocity:   1.0 * 0.15 * 100 = 15.0
    Behavioral: 1.0 * 0.15 * 100 = 15.0
    Rules:      1.0 * 0.15 * 100 = 15.0
    Anomaly:    1.0 * 0.10 * 100 = 10.0
    """
    signals_cases = [
        ("ml", {"calibrated_probability": 1.0, "velocity": 0.0, "behavioral": 0.0, "rules": 0.0, "anomaly": 0.0}, 45.0),
        ("velocity", {"calibrated_probability": 0.0, "velocity": 1.0, "behavioral": 0.0, "rules": 0.0, "anomaly": 0.0}, 15.0),
        ("behavioral", {"calibrated_probability": 0.0, "velocity": 0.0, "behavioral": 1.0, "rules": 0.0, "anomaly": 0.0}, 15.0),
        ("rules", {"calibrated_probability": 0.0, "velocity": 0.0, "behavioral": 0.0, "rules": 1.0, "anomaly": 0.0}, 15.0),
        ("anomaly", {"calibrated_probability": 0.0, "velocity": 0.0, "behavioral": 0.0, "rules": 0.0, "anomaly": 1.0}, 10.0),
    ]

    for target_key, sigs, expected_contrib in signals_cases:
        res = engine.evaluate(**sigs, apply_compounding=False)
        diag = res.diagnostics
        assert diag["contributions"][target_key] == pytest.approx(expected_contrib, abs=1e-2)
        assert diag["contribution_percentages"][target_key] == pytest.approx(100.0, abs=1e-2)
        # All other contributions must be 0.0
        for other_key in ["ml", "velocity", "behavioral", "rules", "anomaly"]:
            if other_key != target_key:
                assert diag["contributions"][other_key] == 0.0
                assert diag["contribution_percentages"][other_key] == 0.0


def test_diagnostics_individual_contributions_combination(engine):
    """
    Verify individual contributions for fractional mixed signals:
    ml=0.70, velocity=0.58, behavioral=0.21, rules=0.17, anomaly=0.25
    """
    res = engine.evaluate(
        calibrated_probability=0.70,
        velocity=0.58,
        behavioral=0.21,
        rules=0.17,
        anomaly=0.25,
        apply_compounding=True
    )
    diag = res.diagnostics

    assert diag["contributions"]["ml"] == pytest.approx(0.70 * 45.0, abs=1e-2)          # 31.5
    assert diag["contributions"]["velocity"] == pytest.approx(0.58 * 15.0, abs=1e-2)    # 8.7
    assert diag["contributions"]["behavioral"] == pytest.approx(0.21 * 15.0, abs=1e-2)  # 3.15 -> 3.15 or 3.2
    assert diag["contributions"]["rules"] == pytest.approx(0.17 * 15.0, abs=1e-2)       # 2.55 -> 2.55 or 2.5
    assert diag["contributions"]["anomaly"] == pytest.approx(0.25 * 10.0, abs=1e-2)     # 2.5


# ==============================================================================
# 2. Contribution Sum
# ==============================================================================

def test_diagnostics_contribution_sum_equals_weighted_score(engine):
    """Verify sum(contributions.values()) == weighted_score within numerical tolerance."""
    res = engine.evaluate(
        calibrated_probability=0.70,
        velocity=0.58,
        behavioral=0.21,
        rules=0.17,
        anomaly=0.25,
        apply_compounding=True
    )
    diag = res.diagnostics
    contrib_sum = sum(diag["contributions"].values())
    assert contrib_sum == pytest.approx(diag["weighted_score"], abs=0.05)
    assert diag["weighted_score"] == pytest.approx(48.4, abs=0.05)


def test_diagnostics_contribution_percentages_sum_to_100(engine):
    """Verify sum of contribution percentages equals 100.0% within numerical tolerance."""
    res = engine.evaluate(
        calibrated_probability=0.60,
        velocity=0.40,
        behavioral=0.30,
        rules=0.20,
        anomaly=0.10
    )
    diag = res.diagnostics
    pct_sum = sum(diag["contribution_percentages"].values())
    assert pct_sum == pytest.approx(100.0, abs=0.1)


# ==============================================================================
# 3. Trigger Identification
# ==============================================================================

def test_diagnostics_trigger_identification_boundaries(engine):
    """
    Verify active triggers strictly adhere to the established compounding thresholds:
    - ml:         > 0.40
    - velocity:   > 0.30
    - behavioral: > 0.30
    - rules:      > 0.40
    """
    # Just at or below thresholds -> no triggers
    res_below = engine.evaluate(
        calibrated_probability=0.40,
        velocity=0.30,
        behavioral=0.30,
        rules=0.40,
        anomaly=0.50
    )
    assert res_below.diagnostics["active_triggers"] == []
    assert res_below.diagnostics["trigger_count"] == 0

    # Just above thresholds -> all 4 triggers active
    res_above = engine.evaluate(
        calibrated_probability=0.4001,
        velocity=0.3001,
        behavioral=0.3001,
        rules=0.4001,
        anomaly=0.50
    )
    assert res_above.diagnostics["active_triggers"] == ["ml", "velocity", "behavioral", "rules"]
    assert res_above.diagnostics["trigger_count"] == 4


# ==============================================================================
# 4. Trigger Count
# ==============================================================================

def test_diagnostics_trigger_count_equals_active_triggers_len(engine):
    """Verify trigger_count == len(active_triggers) for arbitrary signal combinations."""
    test_cases = [
        # (ml, vel, beh, rules, expected_triggers)
        (0.10, 0.10, 0.10, 0.10, []),
        (0.50, 0.10, 0.10, 0.10, ["ml"]),
        (0.10, 0.35, 0.10, 0.10, ["velocity"]),
        (0.50, 0.35, 0.10, 0.10, ["ml", "velocity"]),
        (0.50, 0.35, 0.40, 0.10, ["ml", "velocity", "behavioral"]),
        (0.50, 0.35, 0.40, 0.45, ["ml", "velocity", "behavioral", "rules"]),
    ]

    for ml, vel, beh, rules, expected_list in test_cases:
        res = engine.evaluate(
            calibrated_probability=ml,
            velocity=vel,
            behavioral=beh,
            rules=rules,
            anomaly=0.10
        )
        diag = res.diagnostics
        assert diag["active_triggers"] == expected_list
        assert diag["trigger_count"] == len(expected_list)
        assert diag["trigger_count"] == len(diag["active_triggers"])


# ==============================================================================
# 5. Escalation Boost Policy (K >= 2 -> K * 0.12)
# ==============================================================================

def test_diagnostics_escalation_boost_policy(engine):
    """
    Verify escalation_boost:
    K=0 -> 0.0
    K=1 -> 0.0
    K=2 -> 2 * 12.0 = 24.0
    K=3 -> 3 * 12.0 = 36.0
    K=4 -> 4 * 12.0 = 48.0
    """
    # K=0
    r0 = engine.evaluate(calibrated_probability=0.2, velocity=0.2, behavioral=0.2, rules=0.2, anomaly=0.2, apply_compounding=True)
    assert r0.diagnostics["trigger_count"] == 0
    assert r0.diagnostics["escalation_boost"] == 0.0

    # K=1 (ml only)
    r1 = engine.evaluate(calibrated_probability=0.5, velocity=0.2, behavioral=0.2, rules=0.2, anomaly=0.2, apply_compounding=True)
    assert r1.diagnostics["trigger_count"] == 1
    assert r1.diagnostics["escalation_boost"] == 0.0

    # K=2 (ml + velocity)
    r2 = engine.evaluate(calibrated_probability=0.5, velocity=0.35, behavioral=0.2, rules=0.2, anomaly=0.2, apply_compounding=True)
    assert r2.diagnostics["trigger_count"] == 2
    assert r2.diagnostics["escalation_boost"] == pytest.approx(24.0, abs=1e-4)

    # K=3 (ml + velocity + behavioral)
    r3 = engine.evaluate(calibrated_probability=0.5, velocity=0.35, behavioral=0.35, rules=0.2, anomaly=0.2, apply_compounding=True)
    assert r3.diagnostics["trigger_count"] == 3
    assert r3.diagnostics["escalation_boost"] == pytest.approx(36.0, abs=1e-4)

    # K=4 (ml + velocity + behavioral + rules)
    r4 = engine.evaluate(calibrated_probability=0.5, velocity=0.35, behavioral=0.35, rules=0.45, anomaly=0.2, apply_compounding=True)
    assert r4.diagnostics["trigger_count"] == 4
    assert r4.diagnostics["escalation_boost"] == pytest.approx(48.0, abs=1e-4)


# ==============================================================================
# 6. Score Reconciliation (weighted_score + escalation_boost == final_score)
# ==============================================================================

def test_diagnostics_score_reconciliation(engine):
    """
    Verify: weighted_score + escalation_boost == pre_clamp_score
    and min(100.0, pre_clamp_score) == final_score.
    """
    res = engine.evaluate(
        calibrated_probability=0.70,
        velocity=0.58,
        behavioral=0.21,
        rules=0.17,
        anomaly=0.25,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["pre_escalation_score"] == diag["weighted_score"]
    assert diag["pre_clamp_score"] == pytest.approx(diag["weighted_score"] + diag["escalation_boost"], abs=1e-2)
    assert diag["final_score"] == pytest.approx(min(100.0, diag["pre_clamp_score"]), abs=1e-2)
    assert res.risk_score == pytest.approx(diag["final_score"], abs=1e-2)


# ==============================================================================
# 7. No Trigger (K = 0)
# ==============================================================================

def test_diagnostics_no_trigger_k0(engine):
    """Test K=0: escalation_boost == 0 and final_score remains identical to weighted_score."""
    res = engine.evaluate(
        calibrated_probability=0.20,
        velocity=0.20,
        behavioral=0.20,
        rules=0.20,
        anomaly=0.20,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["trigger_count"] == 0
    assert diag["active_triggers"] == []
    assert diag["escalation_boost"] == 0.0
    assert diag["weighted_score"] == pytest.approx(20.0, abs=1e-2)
    assert diag["final_score"] == pytest.approx(20.0, abs=1e-2)
    assert res.decision == "APPROVE"


# ==============================================================================
# 8. One Trigger (K = 1)
# ==============================================================================

def test_diagnostics_one_trigger_k1(engine):
    """Test K=1: single trigger does NOT activate escalation."""
    res = engine.evaluate(
        calibrated_probability=0.70,  # ml > 0.40 -> trigger 1
        velocity=0.20,
        behavioral=0.20,
        rules=0.20,
        anomaly=0.20,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["trigger_count"] == 1
    assert diag["active_triggers"] == ["ml"]
    assert diag["escalation_boost"] == 0.0
    expected_weighted = 0.70 * 45.0 + 0.20 * 55.0  # 31.5 + 11.0 = 42.5
    assert diag["weighted_score"] == pytest.approx(expected_weighted, abs=1e-2)
    assert diag["final_score"] == pytest.approx(expected_weighted, abs=1e-2)


# ==============================================================================
# 9. Two Triggers (K = 2)
# ==============================================================================

def test_diagnostics_two_triggers_k2_representative_case(engine):
    """
    Test K=2 representative scenario:
    ml=0.70 (trigger), velocity=0.58 (trigger), behavioral=0.21, rules=0.17, anomaly=0.25
    Weighted score ~ 48.4, escalation boost = 24.0, final score ~ 72.4.
    """
    res = engine.evaluate(
        calibrated_probability=0.70,
        velocity=0.58,
        behavioral=0.21,
        rules=0.17,
        anomaly=0.25,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["trigger_count"] == 2
    assert diag["active_triggers"] == ["ml", "velocity"]
    assert diag["escalation_boost"] == pytest.approx(24.0, abs=1e-2)
    assert diag["weighted_score"] == pytest.approx(48.4, abs=0.1)
    assert diag["final_score"] == pytest.approx(72.4, abs=0.1)
    assert res.decision == "BLOCK"  # >= 70.0


# ==============================================================================
# 10. Four Triggers (K = 4)
# ==============================================================================

def test_diagnostics_four_triggers_k4(engine):
    """
    Test K=4: all 4 compounding signals active.
    ml=0.50, velocity=0.40, behavioral=0.40, rules=0.50, anomaly=0.20.
    Weighted score = 0.5*45 + 0.4*15 + 0.4*15 + 0.5*15 + 0.2*10
                   = 22.5 + 6.0 + 6.0 + 7.5 + 2.0 = 44.0
    Escalation boost = 4 * 12.0 = 48.0
    Final score = 44.0 + 48.0 = 92.0
    """
    res = engine.evaluate(
        calibrated_probability=0.50,
        velocity=0.40,
        behavioral=0.40,
        rules=0.50,
        anomaly=0.20,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["trigger_count"] == 4
    assert diag["active_triggers"] == ["ml", "velocity", "behavioral", "rules"]
    assert diag["escalation_boost"] == pytest.approx(48.0, abs=1e-2)
    assert diag["weighted_score"] == pytest.approx(44.0, abs=1e-2)
    assert diag["final_score"] == pytest.approx(92.0, abs=1e-2)
    assert res.decision == "BLOCK"


# ==============================================================================
# 11. Hard Block Diagnostics
# ==============================================================================

def test_diagnostics_hard_block_enforcement(engine):
    """
    Verify hard_block:
    - hard_block == True in diagnostics.
    - Resulting decision is strictly BLOCK and risk_level is HIGH.
    - Score is bounded to at least hard_block_min_score (85.0).
    - Diagnostics does not alter or downgrade hard block.
    """
    # Low signals that would otherwise yield APPROVE (~10.0)
    res = engine.evaluate(
        calibrated_probability=0.10,
        velocity=0.10,
        behavioral=0.10,
        rules=0.10,
        anomaly=0.10,
        hard_block=True,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["hard_block"] is True
    assert res.hard_block is True
    assert res.decision == "BLOCK"
    assert res.risk_level == "HIGH"
    assert res.risk_score >= 85.0
    assert diag["final_score"] >= 85.0
    # Pre-escalation and weighted scores still reflect actual signal calculation
    assert diag["weighted_score"] == pytest.approx(10.0, abs=1e-2)


# ==============================================================================
# 12. Zero Signals
# ==============================================================================

def test_diagnostics_all_zero_signals(engine):
    """
    Verify that all zero signals:
    - weighted_score == 0.0
    - individual contributions == 0.0
    - contribution_percentages == 0.0 (NO NaN, NO Inf)
    - trigger_count == 0
    - escalation_boost == 0.0
    - final_score == 0.0
    - All values are finite floats.
    """
    res = engine.evaluate(
        calibrated_probability=0.0,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["weighted_score"] == 0.0
    assert diag["escalation_boost"] == 0.0
    assert diag["trigger_count"] == 0
    assert diag["active_triggers"] == []
    assert diag["hard_block"] is False
    assert diag["final_score"] == 0.0

    for k in ["ml", "velocity", "behavioral", "rules", "anomaly"]:
        contrib = diag["contributions"][k]
        pct = diag["contribution_percentages"][k]
        assert contrib == 0.0
        assert pct == 0.0
        assert not math.isnan(contrib)
        assert not math.isinf(contrib)
        assert not math.isnan(pct)
        assert not math.isinf(pct)


# ==============================================================================
# 13. All Signals = 1
# ==============================================================================

def test_diagnostics_all_one_signals_clamping(engine):
    """
    Verify that all signals = 1.0:
    - weighted_score == 100.0
    - contributions match baseline weights * 100: [45.0, 15.0, 15.0, 15.0, 10.0]
    - contribution_percentages == [45.0, 15.0, 15.0, 15.0, 10.0]
    - trigger_count == 4
    - escalation_boost == 48.0
    - pre_clamp_score == 148.0
    - final_score is clamped to 100.0
    """
    res = engine.evaluate(
        calibrated_probability=1.0,
        velocity=1.0,
        behavioral=1.0,
        rules=1.0,
        anomaly=1.0,
        apply_compounding=True
    )
    diag = res.diagnostics
    assert diag["weighted_score"] == pytest.approx(100.0, abs=1e-2)
    assert diag["trigger_count"] == 4
    assert diag["active_triggers"] == ["ml", "velocity", "behavioral", "rules"]
    assert diag["escalation_boost"] == pytest.approx(48.0, abs=1e-2)
    assert diag["pre_clamp_score"] == pytest.approx(148.0, abs=1e-2)
    assert diag["final_score"] == 100.0
    assert res.risk_score == 100.0
    assert res.decision == "BLOCK"
    assert res.risk_level == "HIGH"

    assert diag["contributions"]["ml"] == pytest.approx(45.0, abs=1e-2)
    assert diag["contributions"]["velocity"] == pytest.approx(15.0, abs=1e-2)
    assert diag["contributions"]["behavioral"] == pytest.approx(15.0, abs=1e-2)
    assert diag["contributions"]["rules"] == pytest.approx(15.0, abs=1e-2)
    assert diag["contributions"]["anomaly"] == pytest.approx(10.0, abs=1e-2)


# ==============================================================================
# 14. Determinism Across Repeated Runs
# ==============================================================================

def test_diagnostics_determinism_across_100_runs(engine):
    """Verify identical inputs produce byte-for-byte identical JSON serialization across 100 iterations."""
    inputs = {
        "calibrated_probability": 0.6543,
        "velocity": 0.4321,
        "behavioral": 0.3456,
        "rules": 0.5678,
        "anomaly": 0.1234,
        "transaction_id": "tx_deterministic_test",
        "apply_compounding": True
    }

    first_res = engine.evaluate(**inputs)
    first_json = json.dumps(first_res.to_dict(), sort_keys=True)

    for _ in range(100):
        run_res = engine.evaluate(**inputs)
        run_json = json.dumps(run_res.to_dict(), sort_keys=True)
        assert run_json == first_json


# ==============================================================================
# 15. R4 ML Decision Independence
# ==============================================================================

def test_diagnostics_r4_ml_decision_independence(engine):
    """
    Verify ml_decision (R4 ML policy: <0.158 -> APPROVE, 0.158-0.5516 -> REVIEW, >=0.5516 -> BLOCK)
    remains strictly distinct from decision (R5 hybrid policy: <30.0 -> APPROVE, 30-70 -> REVIEW, >=70 -> BLOCK).
    """
    # Case A: High ML (0.80 -> ML BLOCK), but low other signals -> R5 score 36.0 (R5 REVIEW)
    res_a = engine.evaluate(
        calibrated_probability=0.80,
        velocity=0.0,
        behavioral=0.0,
        rules=0.0,
        anomaly=0.0,
        apply_compounding=False
    )
    assert res_a.ml_decision == "BLOCK"
    assert res_a.decision == "REVIEW"
    assert res_a.risk_score == pytest.approx(36.0, abs=1e-2)

    # Case B: Low ML (0.10 -> ML APPROVE), but high other signals -> R5 score 55.0 (R5 REVIEW)
    res_b = engine.evaluate(
        calibrated_probability=0.10,
        velocity=1.0,
        behavioral=1.0,
        rules=1.0,
        anomaly=0.10,
        apply_compounding=False
    )
    assert res_b.ml_decision == "APPROVE"
    assert res_b.decision == "REVIEW"
    assert res_b.risk_score == pytest.approx(50.5, abs=1e-2)


# ==============================================================================
# 16. R5.1 Backward Compatibility
# ==============================================================================

def test_diagnostics_backward_compatibility_result_shape(engine):
    """
    Verify existing consumers relying on:
    - risk_score
    - signals
    - contributions
    - ml_decision
    - model_version
    continue to work without modification.
    """
    res = engine.evaluate(
        calibrated_probability=0.50,
        velocity=0.30,
        behavioral=0.20,
        rules=0.10,
        anomaly=0.10,
        transaction_id="tx_compat",
        model_version="finpulse-v3"
    )

    # Public attributes
    assert isinstance(res.risk_score, float)
    assert isinstance(res.signals, dict)
    assert isinstance(res.contributions, dict)
    assert isinstance(res.ml_decision, str)
    assert res.model_version == "finpulse-v3"
    assert isinstance(res.diagnostics, dict)

    # Dictionary indexing compatibility
    assert res["risk_score"] == res.risk_score
    assert res["decision"] == res.decision
    assert res["ml_decision"] == res.ml_decision
    assert res["signals"] == res.signals
    assert "diagnostics" in res
    assert res["diagnostics"]["trigger_count"] == res.diagnostics["trigger_count"]

    # Direct property access on HybridRiskResult
    assert res.weighted_score == res.diagnostics["weighted_score"]
    assert res.escalation_boost == res.diagnostics["escalation_boost"]
    assert res.trigger_count == res.diagnostics["trigger_count"]
    assert res.active_triggers == res.diagnostics["active_triggers"]
    assert res.contribution_percentages == res.diagnostics["contribution_percentages"]
