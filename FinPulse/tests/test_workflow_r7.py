"""
FinPulse R7 — Workflows Test Suite (R7-A, R7-B, R7-C, R7-D).

Validates:
1. R7-A Account security events & ATO compounding detection
2. R7-B Mandate registry, limits, frequency constraints & lifecycle
3. R7-C Hold state machine (HOLD -> RELEASE / DENY / EXPIRE)
4. R7-D Notification dispatching & background expiry worker
"""

import time
import pytest
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.workflow.account_events import ATOProtectionEngine, AccountSecurityEvent
from src.workflow.mandate import MandateRegistry, MandateProtectionError
from src.workflow.hold_workflow import HoldWorkflowEngine, HoldTransitionError
from src.workflow.notifier import NotificationService, HoldExpiryWorker


class TestWorkflowsR7:
    """Integrated test suite for R7 workflows."""

    # =========================================================================
    # R7-A: Account Security Events & ATO
    # =========================================================================
    def test_ato_normal_events_low_risk(self):
        engine = ATOProtectionEngine()
        now = time.time()
        ev = AccountSecurityEvent.create("cust_01", "login_anomaly", timestamp=now)
        assert engine.record_event(ev) is True
        # Duplicate is ignored
        assert engine.record_event(ev) is False

        assessment = engine.evaluate_ato_risk("cust_01", current_time=now)
        assert assessment.risk_level in ("LOW", "MEDIUM")
        assert assessment.action in ("ALLOW", "CHALLENGE_MFA")

    def test_ato_compounding_password_and_new_device(self):
        engine = ATOProtectionEngine()
        now = time.time()
        ev1 = AccountSecurityEvent.create("cust_02", "password_change", timestamp=now - 600)
        ev2 = AccountSecurityEvent.create("cust_02", "new_device_registration", timestamp=now - 100)
        engine.record_event(ev1)
        engine.record_event(ev2)

        assessment = engine.evaluate_ato_risk("cust_02", current_time=now)
        assert assessment.ato_risk_score >= 0.50
        assert assessment.requires_transaction_hold is True
        allowed, reason = engine.check_transaction_guardrails("cust_02", transaction_amount=1000.0)
        assert allowed is False
        assert "held" in reason.lower()

    # =========================================================================
    # R7-B: Mandate Registry & Protection
    # =========================================================================
    def test_mandate_lifecycle_and_execution(self):
        registry = MandateRegistry()
        # 1. Create
        mandate = registry.create_mandate(
            mandate_id="man_001",
            customer_id="cust_01",
            beneficiary_id="merch_electric",
            max_amount=500.0,
            frequency="MONTHLY"
        )
        assert mandate.status == "PENDING"

        # 2. Cannot execute while PENDING
        ok, reason = registry.validate_and_execute("man_001", 100.0)
        assert ok is False

        # 3. Activate
        registry.activate_mandate("man_001")
        assert mandate.status == "ACTIVE"

        # 4. Valid execution
        ok, reason = registry.validate_and_execute("man_001", 150.0, current_time=1700000000.0)
        assert ok is True
        assert mandate.execution_count == 1
        assert mandate.total_executed_amount == 150.0

        # 5. Amount ceiling violation
        ok, reason = registry.validate_and_execute("man_001", 600.0, current_time=1700000000.0 + 86400.0 * 35)
        assert ok is False
        assert "exceeds approved ceiling" in reason

        # 6. Frequency cadence violation (executing too early for monthly mandate)
        ok, reason = registry.validate_and_execute("man_001", 150.0, current_time=1700000000.0 + 100.0)
        assert ok is False
        assert "too early" in reason

    def test_mandate_pause_and_cancel(self):
        registry = MandateRegistry()
        registry.create_mandate("man_002", "cust_02", "merch_gym", 50.0, "MONTHLY")
        registry.activate_mandate("man_002")
        registry.pause_mandate("man_002")

        ok, reason = registry.validate_and_execute("man_002", 50.0)
        assert ok is False
        assert "not ACTIVE" in reason

        registry.cancel_mandate("man_002")
        with pytest.raises(MandateProtectionError):
            registry.activate_mandate("man_002")

    # =========================================================================
    # R7-C: Hold State Machine
    # =========================================================================
    def test_hold_confirm_release_path(self):
        engine = HoldWorkflowEngine()
        case, token = engine.create_hold("tx_h_001", "cust_h1", 850.0, timeout_seconds=60.0)
        assert case.status == "HOLD"

        # Duplicate create returns same case idempotently
        case_dup, token_dup = engine.create_hold("tx_h_001", "cust_h1", 850.0)
        assert case_dup.hold_id == case.hold_id

        # Confirm with correct token
        confirmed = engine.confirm_hold(case.hold_id, token)
        assert confirmed.status == "RELEASED"

        # Subsequent confirm is idempotent
        confirmed_again = engine.confirm_hold(case.hold_id, token)
        assert confirmed_again.status == "RELEASED"

    def test_hold_deny_path(self):
        engine = HoldWorkflowEngine()
        case, token = engine.create_hold("tx_h_002", "cust_h2", 3000.0)
        denied = engine.deny_hold(case.hold_id, token, reason="Unrecognized fraud")
        assert denied.status == "DENIED"

        # Cannot confirm after denial
        with pytest.raises(HoldTransitionError):
            engine.confirm_hold(case.hold_id, token)

    def test_hold_expiry_rejection(self):
        engine = HoldWorkflowEngine()
        case, token = engine.create_hold("tx_h_003", "cust_h3", 1000.0, timeout_seconds=10.0, current_time=100.0)

        # Attempt to confirm after timeout (current_time=200.0)
        with pytest.raises(HoldTransitionError):
            engine.confirm_hold(case.hold_id, token, current_time=200.0)

    # =========================================================================
    # R7-D: Notifier & Expiry Worker
    # =========================================================================
    def test_notifier_dispatch(self):
        notifier = NotificationService()
        captured = []
        notifier.register_listener(lambda p: captured.append(p))

        payload = notifier.send_hold_alert("cust_notif", "tx_99", 500.0, "hold_1", "tok_abc")
        assert payload.delivered is True
        assert len(captured) == 1
        assert "500.00" in captured[0].message

    def test_hold_expiry_worker(self):
        engine = HoldWorkflowEngine()
        worker = HoldExpiryWorker(engine, poll_interval_seconds=0.1)

        case, _ = engine.create_hold("tx_h_exp", "cust_exp", 750.0, timeout_seconds=1.0, current_time=1000.0)
        # Scan at current_time=1005.0 -> should expire
        expired = worker.run_once(current_time=1005.0)
        assert len(expired) == 1
        assert expired[0].hold_id == case.hold_id
        assert expired[0].status == "EXPIRED"
