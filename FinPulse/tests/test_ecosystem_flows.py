"""
FinPulse Ecosystem End-to-End Test Suite (Scenarios A, B, C, D).

Validates the full authorization ecosystem:
1. Scenario A — Normal Wallet Transaction:
   - Wallet submits transfer request to Payment Gateway
   - Invariant: Balance does NOT move before authorization
   - Gateway FinPulse authorizes APPROVE
   - Balance decreases and recipient balance increases
2. Scenario B — Known High-Risk Attack:
   - ATO + new device + distant location + password reset + drain
   - FinPulse ATO Protection triggers SUSPEND_ACCOUNT
   - Gateway issues authoritative BLOCK
   - Invariant: Money transferred = False, ₹0 lost, balance untouched
3. Scenario C — Suspicious / Uncertain Transaction (HOLD):
   - Suspicious transfer -> Gateway issues HOLD
   - Invariant: Money NOT transferred while on HOLD
   - Verification CONFIRM (Yes, it's me) -> Release hold -> Gateway executes transfer -> Balance updates
   - Verification DENY (No, not me) -> Confirm fraud -> Permanent BLOCK -> Money stays safe
4. Scenario D — Novel / Missed Attack (Customer Feedback Loop):
   - Novel attack pattern evaluated
   - Customer identifies unauthorized transfer and reports fraud
   - Security Case created and persisted in PostgreSQL / SQLite
   - Ground-truth delayed fraud label (1) attached to decision
   - Record retrievable in labeled dataset for controlled retraining
   - ShadowModelRunner and ModelPromotionGate validate candidate evaluation without instant hot-patching
"""

import os
import sys
import time
import pytest
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.simulation.demo_account import (
    DemoAccountManager,
    DEFAULT_DEMO_CUSTOMER_ID,
    DEFAULT_ATTACKER_RECEIVER_ID,
    KNOWN_LOCATION,
    KNOWN_DEVICE_ID
)
from src.simulation.attacker import AttackerSimulator, ATTACKER_LOCATION, ATTACKER_DEVICE_ID
from src.simulation.gateway import PaymentGatewaySimulator, GatewayTransferResult
from src.workflow.account_events import ATOProtectionEngine, AccountSecurityEvent
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.workflow.security_case import SecurityCase
from src.persistence.sink import IdempotentEventSink
from src.challenger.shadow_scorer import ShadowModelRunner
from training.retrain import ModelPromotionGate


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
def attacker(ato_eng, test_sink, account_mgr):
    return AttackerSimulator(
        ato_engine=ato_eng,
        event_sink=test_sink,
        account_manager=account_mgr
    )


class TestEcosystemScenarios:
    """Validate Scenarios A, B, C, D and core architectural invariants."""

    def test_scenario_a_normal_wallet_transaction(self, gateway, account_mgr):
        """
        Scenario A: Normal customer payment from wallet to friend.
        - Sender: CUST_DEMO_001 (Balance: ₹10,000)
        - Recipient: CUST_FRIEND_01 (Rahul, Balance: ₹1,500)
        - Amount: ₹500
        - Expected: Gateway APPROVE -> Sender: ₹9,500, Recipient: ₹2,000.
        """
        sender_id = DEFAULT_DEMO_CUSTOMER_ID
        recipient_id = "CUST_FRIEND_01"
        amount = 500.0

        sender_before = account_mgr.get_balance(sender_id)
        recipient_before = account_mgr.get_balance(recipient_id)
        assert sender_before == 10000.0
        assert recipient_before == 1500.0

        # Normal wallet transfer request
        tx_req = {
            "transaction_id": "tx_normal_scenario_a",
            "customer_id": sender_id,
            "merchant_id": recipient_id,
            "amount": amount,
            "timestamp": time.time(),
            "payment_type": "TRANSFER",
            "category": "transfer",
            "origin_balance": sender_before,
            "dest_balance": recipient_before,
            "auth_verified": True,
            "device_id": KNOWN_DEVICE_ID,
            "latitude": KNOWN_LOCATION["latitude"],
            "longitude": KNOWN_LOCATION["longitude"],
            "location": KNOWN_LOCATION,
            "home_latitude": KNOWN_LOCATION["latitude"],
            "home_longitude": KNOWN_LOCATION["longitude"]
        }

        # Process transfer through authoritative gateway
        res = gateway.process_transfer(tx_req)

        # Invariants
        assert res.authorized is True
        assert res.decision == "APPROVE"
        assert res.status == "APPROVED"
        assert res.money_transferred is True
        assert res.amount == 500.0

        # Balance verification
        sender_after = account_mgr.get_balance(sender_id)
        recipient_after = account_mgr.get_balance(recipient_id)
        assert sender_after == 9500.0
        assert recipient_after == 2000.0
        assert res.sender_balance_after == 9500.0
        assert res.receiver_balance_after == 2000.0

    def test_scenario_b_known_high_risk_attack_blocks_before_transfer(self, gateway, attacker, account_mgr):
        """
        Scenario B: Known high-risk ATO attack attempt.
        - Attacker attempts ₹8,500 drain with new device, Mumbai location, password reset, 2FA bypass.
        - Expected: ATO triggers SUSPEND_ACCOUNT -> Gateway BLOCKS -> ₹0 transferred.
        """
        sender_id = DEFAULT_DEMO_CUSTOMER_ID
        sender_before = account_mgr.get_balance(sender_id)
        attacker_before = account_mgr.get_balance(DEFAULT_ATTACKER_RECEIVER_ID)

        events, req = attacker.build_scenario_transfer(
            scenario_type="account_takeover",
            sender_id=sender_id,
            amount=8500.0
        )
        assert len(events) >= 3

        res = gateway.process_transfer(req)

        # Invariant: Gateway is the authoritative blocker; funds MUST NOT move
        assert res.decision == "BLOCK"
        assert res.authorized is False
        assert res.money_transferred is False
        assert res.ato_action == "SUSPEND_ACCOUNT"

        # Sender balance must remain 100% intact
        sender_after = account_mgr.get_balance(sender_id)
        attacker_after = account_mgr.get_balance(DEFAULT_ATTACKER_RECEIVER_ID)
        assert sender_after == sender_before
        assert attacker_after == attacker_before

        # Timeline verification
        timeline = attacker.build_attack_timeline(events, req, res)
        assert len(timeline) >= 5
        assert any(t["status"] == "BLOCK" for t in timeline)
        assert any(t["status"] == "PROTECTED" for t in timeline)

    def test_scenario_c_hold_workflow_confirm_releases_funds(self, gateway, hold_eng, account_mgr):
        """
        Scenario C1: Suspicious transaction triggers HOLD -> Customer CONFIRMS (Yes, it's me) -> Released & Executed.
        """
        sender_id = DEFAULT_DEMO_CUSTOMER_ID
        recipient_id = "CUST_DEMO_002"
        amount = 2500.0

        sender_before = account_mgr.get_balance(sender_id)
        recipient_before = account_mgr.get_balance(recipient_id)

        # Create hold
        case, token = hold_eng.create_hold(
            transaction_id="tx_hold_c1",
            customer_id=sender_id,
            amount=amount,
            metadata={"receiver_id": recipient_id}
        )

        # Invariant: money does NOT move while on HOLD
        assert account_mgr.get_balance(sender_id) == sender_before
        assert account_mgr.get_balance(recipient_id) == recipient_before

        # Customer confirms on mobile wallet: "Yes, it's me"
        res = gateway.resolve_hold(case.hold_id, token, action="CONFIRM")

        assert res.decision == "APPROVE"
        assert res.status == "RELEASED_AND_EXECUTED"
        assert res.money_transferred is True

        # Now funds have moved
        assert account_mgr.get_balance(sender_id) == sender_before - amount
        assert account_mgr.get_balance(recipient_id) == recipient_before + amount

    def test_scenario_c_hold_workflow_deny_blocks_and_protects_funds(self, gateway, hold_eng, account_mgr):
        """
        Scenario C2: Suspicious transaction triggers HOLD -> Customer DENIES (No, not me) -> Fraud Confirmed & Blocked.
        """
        sender_id = DEFAULT_DEMO_CUSTOMER_ID
        recipient_id = DEFAULT_ATTACKER_RECEIVER_ID
        amount = 4000.0

        sender_before = account_mgr.get_balance(sender_id)
        recipient_before = account_mgr.get_balance(recipient_id)

        case, token = hold_eng.create_hold(
            transaction_id="tx_hold_c2",
            customer_id=sender_id,
            amount=amount,
            metadata={"receiver_id": recipient_id}
        )

        # Customer denies on mobile wallet: "No, this was not me"
        res = gateway.resolve_hold(case.hold_id, token, action="DENY")

        assert res.decision == "BLOCK"
        assert res.status == "DENIED_AND_BLOCKED"
        assert res.money_transferred is False

        # Invariant: Balance remains completely intact
        assert account_mgr.get_balance(sender_id) == sender_before
        assert account_mgr.get_balance(recipient_id) == recipient_before

    def test_scenario_d_novel_attack_customer_feedback_loop(self, gateway, attacker, test_sink, account_mgr):
        """
        Scenario D: Novel / Missed attack feedback lifecycle.
        1. Novel scenario executed
        2. Customer reviews transaction in wallet and reports unauthorized
        3. Security Case is created with attack_type='UNKNOWN'
        4. Delayed ground-truth fraud label is attached
        5. Labeled dataset reflects confirmed fraud for controlled ML feedback
        """
        # 1. Execute novel attack
        events, req = attacker.build_scenario_transfer(
            scenario_type="novel_attack",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=3500.0
        )
        res = gateway.process_transfer(req)
        tx_id = res.transaction_id

        # 2. Customer discovers unauthorized transfer and reports fraud
        sec_case = gateway.report_unauthorized_transaction(
            transaction_id=tx_id,
            customer_id=DEFAULT_DEMO_CUSTOMER_ID,
            customer_report="UNAUTHORIZED",
            attack_type="UNKNOWN",
            notes="Customer reported unfamiliar online transfer"
        )

        # 3. Security Case invariants
        assert isinstance(sec_case, SecurityCase)
        assert sec_case.transaction_id == tx_id
        assert sec_case.customer_id == DEFAULT_DEMO_CUSTOMER_ID
        assert sec_case.customer_report == "UNAUTHORIZED"
        assert sec_case.confirmed_label == "FRAUD"
        assert sec_case.attack_type == "UNKNOWN"
        assert sec_case.status == "CONFIRMED_FRAUD"

        # 4. Persistence verification in event sink
        stored_case = test_sink.get_security_case(sec_case.case_id)
        assert stored_case is not None
        assert stored_case["attack_type"] == "UNKNOWN"
        assert stored_case["confirmed_label"] == "FRAUD"

        cases_list = test_sink.get_security_cases(customer_id=DEFAULT_DEMO_CUSTOMER_ID)
        assert len(cases_list) >= 1
        assert any(c["case_id"] == sec_case.case_id for c in cases_list)

        # 5. Verify delayed fraud label was attached
        dec = test_sink.get_decision(tx_id)
        assert dec is not None
        assert dec["fraud_label"] == 1

        # 6. Verify retrievable in labeled dataset for controlled retraining
        labeled = test_sink.get_labeled_dataset()
        assert len(labeled) >= 1
        assert any(l["transaction_id"] == tx_id and l["fraud_label"] == 1 for l in labeled)

    def test_controlled_ml_lifecycle_promotion_gate(self):
        """
        Validate ModelPromotionGate correctly evaluates candidate model against baseline.
        """
        gate = ModelPromotionGate()

        # Passing candidate
        report_pass = gate.evaluate_candidate(
            prod_metrics={"pr_auc": 0.88, "roc_auc": 0.94, "brier_score": 0.05, "fpr_at_95_recall": 0.012},
            cand_metrics={"pr_auc": 0.89, "roc_auc": 0.95, "brier_score": 0.045, "fpr_at_95_recall": 0.011}
        )
        assert report_pass.overall_status == "APPROVED"
        assert all(c.passed for c in report_pass.checks)

        # Regressing candidate (e.g. PR-AUC drops significantly)
        report_fail = gate.evaluate_candidate(
            prod_metrics={"pr_auc": 0.88, "roc_auc": 0.94, "brier_score": 0.05, "fpr_at_95_recall": 0.012},
            cand_metrics={"pr_auc": 0.82, "roc_auc": 0.90, "brier_score": 0.08, "fpr_at_95_recall": 0.030}
        )
        assert report_fail.overall_status == "REJECTED"
        assert any(not c.passed for c in report_fail.checks)

    def test_shadow_model_isolation(self):
        """
        Validate ShadowModelRunner executes non-blocking shadow inference with fault isolation.
        """
        class FailingChallenger:
            def predict_proba(self, X):
                raise RuntimeError("Challenger failed")

        runner = ShadowModelRunner(challenger_model=FailingChallenger())
        comp = runner.evaluate_shadow(
            transaction_id="tx_shadow_test_01",
            feature_vector_32=np.zeros(32),
            production_result={"calibrated_probability": 0.02, "risk_score": 5.0}
        )
        # Fault isolation: shadow failure returns None and does not raise
        assert comp is None
