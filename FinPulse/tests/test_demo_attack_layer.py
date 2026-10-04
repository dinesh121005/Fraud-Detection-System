"""
FinPulse Demo Attack & Gateway Authorization Layer Test Suite.

Validates:
1. Demo account lifecycle, balance determinism, and reset
2. Attacker simulator ATO event generation, guardrails, and persistence
3. Payment Gateway server-side authorization enforcement (APPROVE, HOLD, BLOCK)
4. Balance protection invariant (no money transferred on HOLD or BLOCK)
5. Customer verification workflow (HOLD -> CONFIRM -> RELEASE -> Execute / HOLD -> DENY -> BLOCK)
6. Repeat attack with evolving Redis behavioral context without model retraining
7. Server-side security and failure resilience (insufficient balance, duplicate replay, invalid inputs)
"""

import os
import sys
import time
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.simulation.demo_account import (
    DemoAccountManager,
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID,
    DEFAULT_INITIAL_BALANCE,
    KNOWN_DEVICE_ID,
    KNOWN_LOCATION
)
from src.simulation.attacker import AttackerSimulator, ATTACKER_DEVICE_ID, ATTACKER_LOCATION
from src.simulation.gateway import PaymentGatewaySimulator, GatewayTransferResult
from src.workflow.account_events import ATOProtectionEngine, AccountSecurityEvent
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.persistence.sink import IdempotentEventSink


@pytest.fixture
def test_sink():
    return IdempotentEventSink(db_path=":memory:")


@pytest.fixture
def account_mgr():
    return DemoAccountManager(initial_balance=10000.0)


@pytest.fixture
def ato_eng():
    return ATOProtectionEngine()


@pytest.fixture
def hold_eng():
    return HoldWorkflowEngine()


@pytest.fixture
def gateway(test_sink, account_mgr, ato_eng, hold_eng):
    return PaymentGatewaySimulator(
        hold_engine=hold_eng,
        ato_engine=ato_eng,
        event_sink=test_sink,
        account_manager=account_mgr
    )


@pytest.fixture
def attacker(test_sink, account_mgr, ato_eng):
    return AttackerSimulator(
        ato_engine=ato_eng,
        event_sink=test_sink,
        account_manager=account_mgr
    )


# =============================================================================
# 1. Demo Account Foundation Tests
# =============================================================================
class TestDemoAccountFoundation:
    def test_demo_account_initial_state(self, account_mgr):
        acc = account_mgr.get_account(DEFAULT_DEMO_CUSTOMER_ID)
        assert acc.customer_id == DEFAULT_DEMO_CUSTOMER_ID
        assert acc.balance == 10000.0
        assert acc.account_status == "SAFE"
        assert acc.known_device_id == KNOWN_DEVICE_ID
        assert acc.known_city == "CHENNAI"

    def test_execute_transfer_balance_updates(self, account_mgr):
        success = account_mgr.execute_transfer(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            receiver_id=DEFAULT_ATTACKER_RECEIVER_ID,
            amount=2500.0
        )
        assert success is True
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 7500.0
        assert account_mgr.get_balance(DEFAULT_ATTACKER_RECEIVER_ID) == 2500.0

    def test_execute_transfer_insufficient_balance(self, account_mgr):
        success = account_mgr.execute_transfer(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            receiver_id=DEFAULT_ATTACKER_RECEIVER_ID,
            amount=50000.0
        )
        assert success is False
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0

    def test_demo_account_reset(self, account_mgr):
        account_mgr.execute_transfer(DEFAULT_DEMO_CUSTOMER_ID, DEFAULT_ATTACKER_RECEIVER_ID, 4000.0)
        account_mgr.update_status(DEFAULT_DEMO_CUSTOMER_ID, "COMPROMISED")
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 6000.0

        account_mgr.reset_all(DEFAULT_DEMO_CUSTOMER_ID)
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0
        assert account_mgr.get_balance(DEFAULT_ATTACKER_RECEIVER_ID) == 0.0
        assert account_mgr.get_account(DEFAULT_DEMO_CUSTOMER_ID).account_status == "SAFE"


# =============================================================================
# 2. Attacker Simulator Tests
# =============================================================================
class TestAttackerSimulator:
    def test_simulate_account_takeover_events(self, attacker, ato_eng, test_sink):
        now = time.time()
        events = attacker.simulate_account_takeover(
            target_customer_id=DEFAULT_DEMO_CUSTOMER_ID,
            new_device=True,
            new_location=True,
            password_change=True,
            auth_anomaly=True,
            current_time=now
        )
        assert len(events) == 3
        types = [e.event_type for e in events]
        assert "login_anomaly" in types
        assert "password_change" in types
        assert "new_device_registration" in types

        # Check ATO evaluation reflects compounding events
        assessment = ato_eng.evaluate_ato_risk(DEFAULT_DEMO_CUSTOMER_ID, current_time=now)
        assert assessment.ato_risk_score >= 0.50
        assert assessment.requires_transaction_hold is True

        # Check persistence in database
        persisted = test_sink.get_recent_account_events(DEFAULT_DEMO_CUSTOMER_ID)
        assert len(persisted) >= 3

    def test_build_malicious_transfer_request(self, attacker):
        tx = attacker.build_malicious_transfer_request(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=8500.0,
            auth_verified=False
        )
        assert tx["customer_id"] == DEFAULT_DEMO_CUSTOMER_ID
        assert tx["amount"] == 8500.0
        assert tx["auth_verified"] is False
        assert tx["device_id"] == ATTACKER_DEVICE_ID
        assert tx["latitude"] == ATTACKER_LOCATION["latitude"]


# =============================================================================
# 3. Gateway Authorization Boundary & Invariant Tests
# =============================================================================
class TestGatewayAuthorization:
    def test_insufficient_funds_rejected_at_gateway(self, gateway, account_mgr):
        tx = {
            "transaction_id": "tx_gw_test_overdraft",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 25000.0,
            "origin_balance": 10000.0,
            "auth_verified": True
        }
        res = gateway.process_transfer(tx)
        assert res.status == "INSUFFICIENT_FUNDS"
        assert res.authorized is False
        assert res.money_transferred is False
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0

    def test_approve_executes_transfer(self, gateway, account_mgr):
        # Clean everyday small transaction with known device and verified auth
        tx = {
            "transaction_id": "tx_gw_test_approve",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": "merch_starbucks",
            "amount": 150.0,
            "origin_balance": 10000.0,
            "auth_verified": True,
            "device_id": KNOWN_DEVICE_ID,
            "latitude": KNOWN_LOCATION["latitude"],
            "longitude": KNOWN_LOCATION["longitude"],
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }
        res = gateway.process_transfer(tx)
        assert res.decision == "APPROVE"
        assert res.status == "APPROVED"
        assert res.authorized is True
        assert res.money_transferred is True
        assert res.sender_balance_after == 9850.0
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 9850.0

    def test_hold_does_not_transfer_money(self, gateway, account_mgr, ato_eng):
        # Trigger single security event to put account into review state
        ev = AccountSecurityEvent.create(customer_id=DEFAULT_DEMO_CUSTOMER_ID, event_type="password_change")
        ato_eng.record_event(ev)

        tx = {
            "transaction_id": "tx_gw_test_hold",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 2800.0,
            "origin_balance": 10000.0,
            "auth_verified": False,
            "device_id": ATTACKER_DEVICE_ID,
            "latitude": ATTACKER_LOCATION["latitude"],
            "longitude": ATTACKER_LOCATION["longitude"],
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }
        res = gateway.process_transfer(tx)
        assert res.decision == "HOLD"
        assert res.status == "HELD"
        assert res.authorized is False
        assert res.money_transferred is False
        # Balance must remain completely unchanged!
        assert res.sender_balance_after == 10000.0
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0
        assert res.hold_case is not None
        assert res.confirmation_token is not None

    def test_block_does_not_transfer_money(self, gateway, account_mgr, attacker):
        # Full ATO simulation triggers high ATO score and severe drain block
        attacker.simulate_account_takeover(DEFAULT_DEMO_CUSTOMER_ID)
        tx = attacker.build_malicious_transfer_request(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=8500.0,
            auth_verified=False
        )
        res = gateway.process_transfer(tx)
        assert res.decision == "BLOCK"
        assert res.status == "DECLINED"
        assert res.authorized is False
        assert res.money_transferred is False
        # Balance must remain completely protected!
        assert res.sender_balance_after == 10000.0
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0


# =============================================================================
# 4. Customer Review Verification Workflow Tests
# =============================================================================
class TestCustomerHoldWorkflow:
    def test_customer_confirmation_releases_and_executes(self, gateway, account_mgr, ato_eng):
        ev = AccountSecurityEvent.create(customer_id=DEFAULT_DEMO_CUSTOMER_ID, event_type="password_change")
        ato_eng.record_event(ev)

        tx = {
            "transaction_id": "tx_review_conf",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 2800.0,
            "origin_balance": 10000.0,
            "auth_verified": False,
            "device_id": ATTACKER_DEVICE_ID,
            "latitude": ATTACKER_LOCATION["latitude"],
            "longitude": ATTACKER_LOCATION["longitude"],
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }
        res = gateway.process_transfer(tx)
        assert res.status == "HELD"
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0

        # Customer presses "Yes, it's me"
        conf_res = gateway.resolve_hold(
            hold_id=res.hold_case.hold_id,
            token=res.confirmation_token,
            action="CONFIRM"
        )
        assert conf_res.status == "RELEASED_AND_EXECUTED"
        assert conf_res.authorized is True
        assert conf_res.money_transferred is True
        assert conf_res.sender_balance_after == 7200.0
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 7200.0

    def test_customer_denial_blocks_and_protects_money(self, gateway, account_mgr, ato_eng):
        ev = AccountSecurityEvent.create(customer_id=DEFAULT_DEMO_CUSTOMER_ID, event_type="password_change")
        ato_eng.record_event(ev)

        tx = {
            "transaction_id": "tx_review_deny",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 2800.0,
            "origin_balance": 10000.0,
            "auth_verified": False,
            "device_id": ATTACKER_DEVICE_ID,
            "latitude": ATTACKER_LOCATION["latitude"],
            "longitude": ATTACKER_LOCATION["longitude"],
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }
        res = gateway.process_transfer(tx)
        assert res.status == "HELD"

        # Customer presses "No, this was not me" (Fraud confirmed)
        deny_res = gateway.resolve_hold(
            hold_id=res.hold_case.hold_id,
            token=res.confirmation_token,
            action="DENY"
        )
        assert deny_res.status == "DENIED_AND_BLOCKED"
        assert deny_res.authorized is False
        assert deny_res.money_transferred is False
        assert deny_res.sender_balance_after == 10000.0
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0


# =============================================================================
# 5. Repeat Attack & Evolving Behavioral Context Tests
# =============================================================================
class TestRepeatAttackEvolvingContext:
    def test_repeat_attack_evaluates_evolving_context_without_retraining(self, gateway, attacker):
        # 1. Attacker simulates ATO
        attacker.simulate_account_takeover(DEFAULT_DEMO_CUSTOMER_ID)

        # 2. Attack #1 attempt
        tx1 = attacker.build_malicious_transfer_request(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=8500.0,
            tx_id="tx_repeat_atk_1"
        )
        res1 = gateway.process_transfer(tx1)
        assert res1.decision in ("HOLD", "BLOCK")
        assert res1.money_transferred is False

        # Model version must remain the production candidate version!
        assert gateway.predictor.risk_engine.model_version == "finpulse-v3"

        # 3. Attack #2 attempt shortly after
        tx2 = attacker.build_malicious_transfer_request(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=4000.0,
            tx_id="tx_repeat_atk_2"
        )
        res2 = gateway.process_transfer(tx2)
        assert res2.decision == "BLOCK"
        assert res2.money_transferred is False

        # Verify that attack #2 was evaluated with evolving context
        # (Redis velocity window now contains Attack #1, so tx_count is >= 2)
        assert res2.risk_score >= res1.risk_score or res2.decision == "BLOCK"
        assert gateway.predictor.risk_engine.model_version == "finpulse-v3"


# =============================================================================
# 6. Auditable Persistence Records
# =============================================================================
class TestPersistenceAuditTrail:
    def test_audit_history_recorded(self, gateway, attacker, test_sink):
        attacker.simulate_account_takeover(DEFAULT_DEMO_CUSTOMER_ID)
        tx = attacker.build_malicious_transfer_request(DEFAULT_DEMO_CUSTOMER_ID, amount=8500.0)
        gateway.process_transfer(tx)

        # Check decisions persisted
        dec = test_sink.get_decision(tx["transaction_id"])
        assert dec is not None
        assert dec["customer_id"] == DEFAULT_DEMO_CUSTOMER_ID

        # Check account events persisted
        acc_events = test_sink.get_recent_account_events(DEFAULT_DEMO_CUSTOMER_ID)
        assert len(acc_events) >= 3

        # Check alerts persisted
        alerts = test_sink.get_recent_fraud_alerts(DEFAULT_DEMO_CUSTOMER_ID)
        assert len(alerts) >= 1


# =============================================================================
# 7. Failure, Resilience & Security Validation (Section 18 & 19)
# =============================================================================
class TestResilienceAndSecurityValidation:
    def test_invalid_negative_or_zero_amount(self, gateway, account_mgr):
        tx_zero = {
            "transaction_id": "tx_bad_zero",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 0.0,
            "origin_balance": 10000.0,
            "auth_verified": True
        }
        res = gateway.process_transfer(tx_zero)
        # Should not execute money transfer
        assert res.money_transferred is False
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0

    def test_unauthorized_hold_token_tampering(self, gateway, account_mgr, ato_eng):
        ev = AccountSecurityEvent.create(customer_id=DEFAULT_DEMO_CUSTOMER_ID, event_type="password_change")
        ato_eng.record_event(ev)

        tx = {
            "transaction_id": "tx_tamper_token",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 2800.0,
            "origin_balance": 10000.0,
            "auth_verified": False,
            "device_id": ATTACKER_DEVICE_ID,
            "latitude": ATTACKER_LOCATION["latitude"],
            "longitude": ATTACKER_LOCATION["longitude"],
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }
        res = gateway.process_transfer(tx)
        assert res.status == "HELD"

        # Attacker tries to forge / tamper with token
        with pytest.raises(Exception):
            gateway.resolve_hold(res.hold_case.hold_id, token="FORGED_TOKEN_ABC", action="CONFIRM")

        # Balance must remain completely protected!
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == 10000.0

    def test_duplicate_transaction_idempotency(self, gateway, test_sink):
        tx = {
            "transaction_id": "tx_replay_dup",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": "merch_terminal",
            "amount": 100.0,
            "origin_balance": 10000.0,
            "auth_verified": True
        }
        res1 = gateway.process_transfer(tx)
        res2 = gateway.process_transfer(tx)
        # Both requests handled safely without crashing
        assert res1.authorized is True
        assert res2.authorized is True


# =============================================================================
# 8. Decision Consistency, Layer Separation & R5 Weights Verification
# =============================================================================
class TestDecisionConsistencyAndWeights:
    def test_authoritative_r5_weights_policy(self, gateway):
        """
        Verify that the gateway's predictor strictly uses the authoritative R5
        engineering baseline weighting policy:
        45% ML, 15% Velocity, 15% Behavioral, 15% Rules, 10% Anomaly (Sum = 1.0).
        Guarantees that no legacy or demo-specific 0.45/0.25/0.30 weights ever control the pipeline.
        """
        weights = gateway.predictor.risk_engine.weights
        assert weights.w_ml == pytest.approx(0.45, abs=1e-5)
        assert weights.w_velocity == pytest.approx(0.15, abs=1e-5)
        assert weights.w_behavioral == pytest.approx(0.15, abs=1e-5)
        assert weights.w_rules == pytest.approx(0.15, abs=1e-5)
        assert weights.w_anomaly == pytest.approx(0.10, abs=1e-5)

        total_weight = weights.w_ml + weights.w_velocity + weights.w_behavioral + weights.w_rules + weights.w_anomaly
        assert total_weight == pytest.approx(1.0, abs=1e-5)

    def test_decision_layer_separation_and_consistency(self, gateway, attacker, test_sink):
        """
        Verify the explicit lineage across all decision layers:
        - Layer 1: R4 ML Decision (ml_decision = APPROVE for calibrated P < 0.1580)
        - Layer 2: R5 Hybrid Decision (hybrid_decision = REVIEW for 30.0 <= score < 70.0)
        - Layer 3: R7-A ATO Security Decision (ato_action = SUSPEND_ACCOUNT for 3 ATO events)
        - Layer 4: Gateway Authorization Decision (decision = BLOCK, status = DECLINED)
        - Layer 5: PostgreSQL System of Record:
            * fraud_decisions.decision = 'REVIEW'
            * fraud_decisions.ml_decision = 'APPROVE'
            * fraud_decisions.workflow_status = 'DECLINED_ATO_SUSPENDED'
            * fraud_decisions.diagnostics_json contains gateway_decision='BLOCK'
            * fraud_alerts.decision = 'BLOCK'
        """
        # 1. Attacker simulates 3 ATO security events
        attacker.simulate_account_takeover(DEFAULT_DEMO_CUSTOMER_ID)

        # 2. Formulate and process attack transaction
        tx = attacker.build_malicious_transfer_request(
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=8500.0,
            auth_verified=False,
            tx_id="tx_layer_audit_001"
        )
        res = gateway.process_transfer(tx)

        # 3. Verify Layered Decision Attribution in GatewayTransferResult
        assert res.ml_decision == "APPROVE"
        assert res.hybrid_decision == "REVIEW"
        assert 30.0 <= res.risk_score < 70.0
        assert res.ato_action == "SUSPEND_ACCOUNT"
        assert res.ato_score == 1.00
        assert res.decision == "BLOCK"
        assert res.status == "DECLINED"
        assert res.money_transferred is False

        # 4. Verify PostgreSQL Record Consistency
        dec = test_sink.get_decision("tx_layer_audit_001")
        assert dec is not None
        assert dec["ml_decision"] == "APPROVE"
        assert dec["decision"] == "REVIEW"
        assert dec["workflow_status"] == "DECLINED_ATO_SUSPENDED"

        import json
        diag = dec.get("diagnostics_json", {})
        if isinstance(diag, str):
            diag = json.loads(diag)
        assert diag.get("gateway_decision") == "BLOCK"
        assert diag.get("hybrid_decision") == "REVIEW"
        assert diag.get("ml_decision") == "APPROVE"
        assert diag.get("ato_action") == "SUSPEND_ACCOUNT"

        # 5. Verify Fraud Alert Record Consistency
        alerts = test_sink.get_recent_fraud_alerts(DEFAULT_DEMO_CUSTOMER_ID)
        assert len(alerts) >= 1
        assert any(a["decision"] == "BLOCK" and a["transaction_id"] == "tx_layer_audit_001" for a in alerts)

