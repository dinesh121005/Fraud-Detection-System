"""R5.3 Dedicated Test Suite — Business and Rule Decisioning Engine.

Comprehensive validation of:
1. Rule Isolation: Independent testability of every business rule (condition true/false, fields).
2. Velocity Rules: Micro-spikes, 5-minute bursts, 1-hour surges.
3. Amount Rules: Balance depletion, extreme z-score, zero-balance high value.
4. Location & Device Rules: Impossible travel, multi-user devices, distant new location.
5. Authentication Rules: Missing step-up 2FA, high-value unverified.
6. Transaction-Context Rules: Nighttime transfers, unverified drain.
7. Hard-Block Rules: Deterministic enforcement, cannot be downgraded by weighted fusion.
8. Rule Signal Aggregation: Neutral 0.0 on no match, bounded in [0.0, 1.0].
9. Integration with R5.1/R5.2: Complete flow from R3 context -> RuleResult -> R5.1 -> R5.2.
10. Regression and Backward Compatibility: Tuple unpacking, contract adherence.
"""

import os
import sys
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.rules import (
    BusinessRuleEngine,
    DeterministicRuleEngine,
    BusinessRule,
    MatchedRule,
    RuleResult,
)
from src.risk_engine.hybrid import HybridRiskEngine, HybridRiskResult
from src.risk_engine.contract import normalize_rules_signal


@pytest.fixture
def rule_engine():
    """Provides a fresh instance of BusinessRuleEngine with default rules."""
    return BusinessRuleEngine()


@pytest.fixture
def hybrid_engine():
    """Provides a fresh HybridRiskEngine instance."""
    return HybridRiskEngine(enable_compounding=True)


# ==============================================================================
# 1. Rule Isolation & Registry Inspection Tests
# ==============================================================================

def test_rule_registry_contains_all_six_categories(rule_engine):
    """Verify default rules span all 6 required categories."""
    rules = rule_engine.list_rules()
    assert len(rules) >= 13

    categories = {r.category for r in rules}
    assert "velocity" in categories
    assert "amount" in categories
    assert "location_device" in categories
    assert "authentication" in categories
    assert "transaction_context" in categories

    hard_block_rules = rule_engine.get_hard_block_rules()
    assert len(hard_block_rules) >= 3
    for hbr in hard_block_rules:
        assert hbr.is_hard_block is True
        assert hbr.severity == "CRITICAL"


def test_rule_lookup_and_custom_registration(rule_engine):
    """Verify rule lookup by ID, category filtering, and custom rule registration."""
    rule = rule_engine.get_rule("VEL_MICRO_SPIKE")
    assert rule is not None
    assert rule.category == "velocity"
    assert rule.risk_contribution == 0.25

    # Filter by category
    vel_rules = rule_engine.get_rules_by_category("velocity")
    assert len(vel_rules) >= 3

    # Filter hard-block category
    hb_rules = rule_engine.get_rules_by_category("hard_block")
    assert len(hb_rules) >= 3

    # Register custom rule
    custom = BusinessRule(
        rule_id="CUSTOM_RULE_TEST",
        category="amount",
        severity="LOW",
        risk_contribution=0.10,
        is_hard_block=False,
        reason="Custom test rule",
        condition=lambda tx, feat: float(tx.get("amount", 0.0)) == 777.0,
    )
    rule_engine.register_rule(custom)
    assert rule_engine.get_rule("CUSTOM_RULE_TEST") is not None

    matched = rule_engine.evaluate_rule("CUSTOM_RULE_TEST", {"amount": 777.0}, {})
    assert matched is not None
    assert matched.rule_id == "CUSTOM_RULE_TEST"

    not_matched = rule_engine.evaluate_rule("CUSTOM_RULE_TEST", {"amount": 500.0}, {})
    assert not_matched is None


# ==============================================================================
# 2. Velocity Rules Isolation Tests
# ==============================================================================

def test_velocity_rule_micro_spike(rule_engine):
    """Verify VEL_MICRO_SPIKE triggers at >= 4 tx in 1 minute."""
    rule_id = "VEL_MICRO_SPIKE"
    # Below threshold (3 < 4)
    assert rule_engine.evaluate_rule(rule_id, {}, {"tx_count_1m": 3}) is None
    # Exact threshold (4 == 4)
    match_exact = rule_engine.evaluate_rule(rule_id, {}, {"tx_count_1m": 4})
    assert match_exact is not None
    assert match_exact.category == "velocity"
    assert match_exact.severity == "MEDIUM"
    assert match_exact.risk_contribution == 0.25
    assert match_exact.hard_block is False
    # Above threshold (5 > 4)
    assert rule_engine.evaluate_rule(rule_id, {}, {"tx_count_1m": 5}) is not None


def test_velocity_rule_5m_burst(rule_engine):
    """Verify VEL_5M_BURST triggers at >= 6 tx in 5 minutes."""
    rule_id = "VEL_5M_BURST"
    assert rule_engine.evaluate_rule(rule_id, {}, {"tx_count_5m": 5}) is None
    matched = rule_engine.evaluate_rule(rule_id, {}, {"tx_count_5m": 6})
    assert matched is not None
    assert matched.risk_contribution == 0.20


def test_velocity_rule_1h_surge(rule_engine):
    """Verify VEL_1H_SURGE triggers at >= 15 tx in 1 hour."""
    rule_id = "VEL_1H_SURGE"
    assert rule_engine.evaluate_rule(rule_id, {}, {"tx_count_1h": 14}) is None
    matched = rule_engine.evaluate_rule(rule_id, {}, {"tx_count_1h": 15})
    assert matched is not None
    assert matched.severity == "HIGH"
    assert matched.risk_contribution == 0.25


# ==============================================================================
# 3. Amount Rules Isolation Tests
# ==============================================================================

def test_amount_rule_balance_depletion(rule_engine):
    """Verify AMT_BALANCE_DEPLETION triggers when amount > 95% of origin_balance."""
    rule_id = "AMT_BALANCE_DEPLETION"
    # 900 / 1000 = 90% (below 95%)
    assert rule_engine.evaluate_rule(rule_id, {"amount": 900.0, "origin_balance": 1000.0}, {}) is None
    # 960 / 1000 = 96% (above 95%)
    matched = rule_engine.evaluate_rule(rule_id, {"amount": 960.0, "origin_balance": 1000.0}, {})
    assert matched is not None
    assert matched.category == "amount"
    assert matched.risk_contribution == 0.25


def test_amount_rule_extreme_zscore(rule_engine):
    """Verify AMT_EXTREME_ZSCORE triggers when amount_zscore >= 4.0."""
    rule_id = "AMT_EXTREME_ZSCORE"
    assert rule_engine.evaluate_rule(rule_id, {}, {"amount_zscore": 3.99}) is None
    matched = rule_engine.evaluate_rule(rule_id, {}, {"amount_zscore": 4.0})
    assert matched is not None
    assert matched.risk_contribution == 0.30


def test_amount_rule_zero_balance_high_value(rule_engine):
    """Verify AMT_ZERO_BALANCE_HIGH_VALUE triggers on zero-balance account with amount > 5000."""
    rule_id = "AMT_ZERO_BALANCE_HIGH_VALUE"
    assert rule_engine.evaluate_rule(rule_id, {"amount": 5000.0, "origin_balance": 0.0}, {}) is None
    assert rule_engine.evaluate_rule(rule_id, {"amount": 6000.0, "origin_balance": 100.0}, {}) is None
    matched = rule_engine.evaluate_rule(rule_id, {"amount": 5000.01, "origin_balance": 0.0}, {})
    assert matched is not None
    assert matched.risk_contribution == 0.30


# ==============================================================================
# 4. Location / Device Rules Isolation Tests
# ==============================================================================

def test_location_device_multi_user(rule_engine):
    """Verify DEV_SUSPICIOUS_MULTI_USER triggers when device is shared across >= 5 users in 24h."""
    rule_id = "DEV_SUSPICIOUS_MULTI_USER"
    assert rule_engine.evaluate_rule(rule_id, {}, {"device_user_count_24h": 4}) is None
    matched = rule_engine.evaluate_rule(rule_id, {}, {"device_user_count_24h": 5})
    assert matched is not None
    assert matched.category == "location_device"
    assert matched.risk_contribution == 0.30


def test_location_device_anomalous_combination(rule_engine):
    """Verify LOC_DEV_ANOMALOUS_COMBINATION triggers on new device + new location + distance > 500km."""
    rule_id = "LOC_DEV_ANOMALOUS_COMBINATION"
    # Distance <= 500 km
    assert rule_engine.evaluate_rule(rule_id, {}, {"is_new_device": 1, "is_new_location": 1, "distance_from_home_km": 500.0}) is None
    # Not new device
    assert rule_engine.evaluate_rule(rule_id, {}, {"is_new_device": 0, "is_new_location": 1, "distance_from_home_km": 800.0}) is None
    # All 3 conditions satisfied
    matched = rule_engine.evaluate_rule(rule_id, {}, {"is_new_device": 1, "is_new_location": 1, "distance_from_home_km": 500.1})
    assert matched is not None
    assert matched.risk_contribution == 0.25


# ==============================================================================
# 5. Authentication Rules Isolation Tests
# ==============================================================================

def test_authentication_rule_missing_step_up(rule_engine):
    """Verify AUTH_MISSING_STEP_UP triggers when auth_verified is False or 0."""
    rule_id = "AUTH_MISSING_STEP_UP"
    # Verified -> False
    assert rule_engine.evaluate_rule(rule_id, {"auth_verified": True}, {"auth_factor_verified": 1}) is None
    assert rule_engine.evaluate_rule(rule_id, {"auth_verified": 1}, {"auth_factor_verified": 1}) is None

    # Unverified -> True
    m1 = rule_engine.evaluate_rule(rule_id, {"auth_verified": False}, {})
    assert m1 is not None
    assert m1.category == "authentication"
    assert m1.risk_contribution == 0.30

    m2 = rule_engine.evaluate_rule(rule_id, {"auth_verified": 0}, {})
    assert m2 is not None


def test_authentication_rule_high_value_unverified(rule_engine):
    """Verify AUTH_HIGH_VALUE_UNVERIFIED triggers when amount > 2500 and unverified."""
    rule_id = "AUTH_HIGH_VALUE_UNVERIFIED"
    assert rule_engine.evaluate_rule(rule_id, {"amount": 2500.0, "auth_verified": 0}, {}) is None
    assert rule_engine.evaluate_rule(rule_id, {"amount": 3000.0, "auth_verified": 1}, {}) is None
    matched = rule_engine.evaluate_rule(rule_id, {"amount": 2500.01, "auth_verified": 0}, {})
    assert matched is not None
    assert matched.risk_contribution == 0.35


# ==============================================================================
# 6. Transaction-Context Rules Isolation Tests
# ==============================================================================

def test_context_rule_high_amount_night_transfer(rule_engine):
    """Verify CTX_HIGH_AMOUNT_NIGHT_TRANSFER triggers on night TRANSFER/CASH_OUT > 3000."""
    rule_id = "CTX_HIGH_AMOUNT_NIGHT_TRANSFER"
    # Daytime transfer
    assert rule_engine.evaluate_rule(rule_id, {"payment_type": "TRANSFER", "amount": 5000.0}, {"is_night": 0}) is None
    # Nighttime non-transfer
    assert rule_engine.evaluate_rule(rule_id, {"payment_type": "PAYMENT", "amount": 5000.0}, {"is_night": 1}) is None
    # Nighttime transfer <= 3000
    assert rule_engine.evaluate_rule(rule_id, {"payment_type": "TRANSFER", "amount": 3000.0}, {"is_night": 1}) is None

    # Nighttime transfer > 3000
    matched = rule_engine.evaluate_rule(rule_id, {"payment_type": "TRANSFER", "amount": 3500.0}, {"is_night": 1})
    assert matched is not None
    assert matched.category == "transaction_context"
    assert matched.risk_contribution == 0.20


def test_context_rule_high_drain_unverified(rule_engine):
    """Verify CTX_HIGH_DRAIN_UNVERIFIED triggers on drain > 90% without auth and tx_count_1m < 3."""
    rule_id = "CTX_HIGH_DRAIN_UNVERIFIED"
    tx_match = {"origin_balance": 1000.0, "amount": 950.0, "auth_verified": 0}
    matched = rule_engine.evaluate_rule(rule_id, tx_match, {"tx_count_1m": 2})
    assert matched is not None
    assert matched.risk_contribution == 0.85
    assert matched.hard_block is False

    # Elevated to hard-block when tx_count_1m >= 3 (handled by BLOCK_UNVERIFIED_RAPID_DRAIN)
    assert rule_engine.evaluate_rule(rule_id, tx_match, {"tx_count_1m": 3}) is None


# ==============================================================================
# 7. Hard-Block Rules Isolation Tests
# ==============================================================================

def test_hard_block_impossible_travel(rule_engine):
    """Verify BLOCK_IMPOSSIBLE_TRAVEL triggers hard block when speed > 800 km/h."""
    rule_id = "BLOCK_IMPOSSIBLE_TRAVEL"
    assert rule_engine.evaluate_rule(rule_id, {}, {"speed_kmh_from_prev_tx": 800.0}) is None
    matched = rule_engine.evaluate_rule(rule_id, {}, {"speed_kmh_from_prev_tx": 800.1})
    assert matched is not None
    assert matched.hard_block is True
    assert matched.severity == "CRITICAL"
    assert matched.risk_contribution == 1.0


def test_hard_block_zero_balance_high_drain(rule_engine):
    """Verify BLOCK_ZERO_BALANCE_HIGH_DRAIN_NO_AUTH triggers hard block."""
    rule_id = "BLOCK_ZERO_BALANCE_HIGH_DRAIN_NO_AUTH"
    # amount <= 10000
    assert rule_engine.evaluate_rule(rule_id, {"origin_balance": 0.0, "amount": 10000.0, "auth_verified": 0}, {}) is None
    # auth verified = 1
    assert rule_engine.evaluate_rule(rule_id, {"origin_balance": 0.0, "amount": 15000.0, "auth_verified": 1}, {}) is None
    # All satisfied
    matched = rule_engine.evaluate_rule(rule_id, {"origin_balance": 0.0, "amount": 10000.01, "auth_verified": 0}, {})
    assert matched is not None
    assert matched.hard_block is True


def test_hard_block_unverified_rapid_drain(rule_engine):
    """Verify BLOCK_UNVERIFIED_RAPID_DRAIN triggers hard block on unverified drain + velocity >= 3."""
    rule_id = "BLOCK_UNVERIFIED_RAPID_DRAIN"
    tx = {"origin_balance": 1000.0, "amount": 950.0, "auth_verified": 0}
    assert rule_engine.evaluate_rule(rule_id, tx, {"tx_count_1m": 2}) is None
    matched = rule_engine.evaluate_rule(rule_id, tx, {"tx_count_1m": 3})
    assert matched is not None
    assert matched.hard_block is True


# ==============================================================================
# 8. Batch Evaluation and Rule Signal Aggregation Tests
# ==============================================================================

def test_batch_evaluation_no_rules_matched(rule_engine):
    """Verify clean transaction produces neutral rules_signal=0.0 and no hard block."""
    tx = {
        "amount": 50.0,
        "origin_balance": 5000.0,
        "auth_verified": 1,
        "payment_type": "PAYMENT"
    }
    feat = {
        "tx_count_1m": 1,
        "tx_count_5m": 1,
        "tx_count_1h": 1,
        "amount_zscore": 0.1,
        "speed_kmh_from_prev_tx": 30.0,
        "is_night": 0
    }
    res = rule_engine.evaluate(tx, feat)
    assert res.rules_signal == 0.0
    assert res.matched_rules == []
    assert res.hard_block is False
    assert res.hard_block_rules == []
    assert res.decision == "APPROVE"
    assert res.reasons == []


def test_batch_evaluation_multiple_soft_rules_accumulation(rule_engine):
    """
    Verify multiple soft rules accumulate risk contribution bounded in [0.0, 1.0]:
    - VEL_MICRO_SPIKE: 0.25
    - AUTH_MISSING_STEP_UP: 0.30
    Total = 0.55 -> decision REVIEW
    """
    tx = {
        "amount": 100.0,
        "origin_balance": 5000.0,
        "auth_verified": 0,  # +0.30
        "payment_type": "PAYMENT"
    }
    feat = {
        "tx_count_1m": 4,  # +0.25
        "tx_count_5m": 4,
        "tx_count_1h": 4,
        "speed_kmh_from_prev_tx": 0.0
    }
    res = rule_engine.evaluate(tx, feat)
    assert res.rules_signal == pytest.approx(0.55, abs=1e-3)
    assert res.hard_block is False
    assert len(res.matched_rules) == 2
    assert res.decision == "REVIEW"


def test_batch_evaluation_clamping_at_1_0(rule_engine):
    """Verify rule_risk sum exceeding 1.0 is safely clamped to 1.0 without hard block."""
    tx = {
        "amount": 2600.0,
        "origin_balance": 5000.0,
        "auth_verified": 0,  # AUTH_MISSING (+0.30) + AUTH_HIGH_VALUE (+0.35) = 0.65
        "payment_type": "PAYMENT"
    }
    feat = {
        "tx_count_1m": 4,  # VEL_MICRO_SPIKE (+0.25)
        "tx_count_5m": 6,  # VEL_5M_BURST (+0.20)
        "tx_count_1h": 15, # VEL_1H_SURGE (+0.25) -> Sum = 1.35
        "speed_kmh_from_prev_tx": 0.0
    }
    res = rule_engine.evaluate(tx, feat)
    assert res.rules_signal == 1.0
    assert res.hard_block is False
    assert res.decision == "BLOCK"


def test_batch_evaluation_hard_block_overrides_decision(rule_engine):
    """Verify hard block sets rules_signal=1.0 and decision=BLOCK regardless of other signals."""
    tx = {"amount": 10.0, "origin_balance": 5000.0, "auth_verified": 1}
    feat = {"speed_kmh_from_prev_tx": 900.0}  # BLOCK_IMPOSSIBLE_TRAVEL

    res = rule_engine.evaluate(tx, feat)
    assert res.hard_block is True
    assert "BLOCK_IMPOSSIBLE_TRAVEL" in res.hard_block_rules
    assert res.rules_signal == 1.0
    assert res.decision == "BLOCK"


# ==============================================================================
# 9. Tuple Unpacking & Dict Compatibility Tests
# ==============================================================================

def test_rule_result_tuple_unpacking_backward_compatibility(rule_engine):
    """
    Verify legacy `rule_risk, rule_violations, hard_block = engine.evaluate_rules(...)`
    works seamlessly via tuple unpacking.
    """
    tx = {"origin_balance": 0.0, "amount": 15000.0, "auth_verified": 0}
    feat = {}

    rule_risk, rule_violations, hard_block = rule_engine.evaluate_rules(tx, feat)
    assert rule_risk == 1.0
    assert isinstance(rule_violations, list)
    assert len(rule_violations) >= 1
    assert hard_block is True


def test_rule_result_dict_indexing_and_serialization(rule_engine):
    """Verify RuleResult supports dictionary indexing and serializes cleanly."""
    tx = {"origin_balance": 1000.0, "amount": 980.0, "auth_verified": 1}
    res = rule_engine.evaluate(tx, {})

    assert res["rules_signal"] == res.rules_signal
    assert res["hard_block"] == res.hard_block
    assert "matched_rules" in res
    assert "hard_block_rules" in res
    assert "decision" in res

    d = res.to_dict()
    assert isinstance(d, dict)
    assert d["rules_signal"] == res.rules_signal


# ==============================================================================
# 10. End-to-End Integration with R5.1 & R5.2 Tests
# ==============================================================================

def test_integration_clean_transaction(rule_engine, hybrid_engine):
    """Verify clean transaction produces low rule contribution and APPROVE decision."""
    tx = {"amount": 25.0, "origin_balance": 5000.0, "auth_verified": 1}
    rule_res = rule_engine.evaluate(tx, {})

    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.05,
        velocity=0.05,
        behavioral=0.05,
        anomaly=0.05,
        rule_result=rule_res,
        transaction_id="tx_clean"
    )

    assert hybrid_res.signals["rules_signal"] == 0.0
    assert hybrid_res.contributions["rules_contribution"] == 0.0
    assert hybrid_res.diagnostics["contributions"]["rules"] == 0.0
    assert hybrid_res.decision == "APPROVE"
    assert hybrid_res.hard_block is False


def test_integration_rule_triggered_soft_escalation(rule_engine, hybrid_engine):
    """Verify soft rule violations correctly feed the rules signal into R5.1 and R5.2 diagnostics."""
    tx = {"amount": 100.0, "origin_balance": 5000.0, "auth_verified": 0}  # AUTH_MISSING (+0.30)
    feat = {"tx_count_1m": 4}  # VEL_MICRO_SPIKE (+0.25) -> rules_signal = 0.55
    rule_res = rule_engine.evaluate(tx, feat)

    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.20,
        velocity=0.10,
        behavioral=0.10,
        anomaly=0.10,
        rule_result=rule_res,
        transaction_id="tx_soft_rules"
    )

    # 0.55 rules signal * 15% weight * 100 = 8.25
    assert hybrid_res.signals["rules_signal"] == pytest.approx(0.55, abs=1e-3)
    assert hybrid_res.diagnostics["contributions"]["rules"] == pytest.approx(0.55 * 15.0, abs=1e-2)
    assert "Second-factor biometric" in hybrid_res.reasons[0] or "micro-velocity" in hybrid_res.reasons[0]


def test_integration_hard_block_overrides_low_hybrid_score(rule_engine, hybrid_engine):
    """
    CRITICAL: Hard block must force decision to BLOCK and score >= 85.0
    even if ML probability and other signals are all near 0.0.
    """
    tx = {"amount": 10.0, "origin_balance": 5000.0, "auth_verified": 1}
    feat = {"speed_kmh_from_prev_tx": 950.0}  # BLOCK_IMPOSSIBLE_TRAVEL
    rule_res = rule_engine.evaluate(tx, feat)
    assert rule_res.hard_block is True

    # Low other signals would normally yield risk_score ~ 5.0 (APPROVE)
    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.01,
        velocity=0.01,
        behavioral=0.01,
        anomaly=0.01,
        rule_result=rule_res,
        transaction_id="tx_hard_block_override"
    )

    assert hybrid_res.hard_block is True
    assert hybrid_res.decision == "BLOCK"
    assert hybrid_res.risk_level == "HIGH"
    assert hybrid_res.risk_score >= 85.0
    assert hybrid_res.diagnostics["hard_block"] is True


def test_integration_ml_decision_independence_preserved(rule_engine, hybrid_engine):
    """Verify R4 ml_decision remains independent from R5 hybrid decision when rule triggers."""
    tx = {"amount": 100.0, "origin_balance": 5000.0, "auth_verified": 1}
    rule_res = rule_engine.evaluate(tx, {})

    # ML is high (0.80 -> ML BLOCK), but other signals are 0.0 -> Hybrid score ~ 36.0 (REVIEW)
    hybrid_res = hybrid_engine.evaluate(
        calibrated_probability=0.80,
        velocity=0.0,
        behavioral=0.0,
        anomaly=0.0,
        rule_result=rule_res,
        apply_compounding=False
    )
    assert hybrid_res.ml_decision == "BLOCK"
    assert hybrid_res.decision == "REVIEW"


def test_deterministic_rule_engine_alias_parity(rule_engine):
    """Verify DeterministicRuleEngine class is fully compatible and produces identical outputs."""
    det_engine = DeterministicRuleEngine()
    tx = {"origin_balance": 0.0, "amount": 12000.0, "auth_verified": 0}
    feat = {"speed_kmh_from_prev_tx": 100.0}

    res1 = rule_engine.evaluate(tx, feat)
    res2 = det_engine.evaluate(tx, feat)

    assert res1.to_dict() == res2.to_dict()
