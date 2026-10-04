"""
FinPulse Dynamic Adversarial Attack Simulation & U1/U2 Unseen Attack Test Suite.

Validates the 20 minimum acceptance criteria:
1. Multiple attackers can be instantiated independently.
2. The same attacker can change device dynamically.
3. The same attacker can change location dynamically.
4. The same attacker can change recipient dynamically.
5. Attack sequence maintains persistent attacker state.
6. Attack #2 can observe the relevant history from Attack #1.
7. Attacker behavior can react to BLOCK verdict.
8. Attacker behavior can react to HOLD verdict.
9. Attacker behavior can react to APPROVE verdict.
10. U1 Context Mutation produces a pattern not in predefined attack templates.
11. U2 Compound Novel Attack produces a compound pattern distinct from predefined templates.
12. Different generated attacks produce different 32-feature vectors when contexts differ.
13. Redis state evolves correctly across sequential attacks.
14. Payment Gateway remains authoritative for the final decision.
15. Production model artifact cryptographic hash remains completely unchanged before & after.
16. DecisionEvent contract remains valid across all simulated scenarios.
17. Appropriate Kafka decision/alert events are generated when publisher is active.
18. Persistence audit records are created in the event sink.
19. Reset mechanism cleanly returns the simulation to initial state.
20. Seeded mode reproduces the exact same experiment deterministically.
"""

import os
import sys
import time
import hashlib
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
    DEFAULT_INITIAL_BALANCE,
    KNOWN_DEVICE_ID,
    KNOWN_LOCATION
)
from src.simulation.attacker import (
    AttackerSimulator,
    AttackerState,
    ATTACKER_POOL,
    ATTACK_STRATEGIES,
    ATTACK_RECIPIENTS,
    INDIAN_CITIES,
    is_predefined_template,
    get_template_classification
)
from src.simulation.gateway import PaymentGatewaySimulator, GatewayTransferResult
from src.workflow.account_events import ATOProtectionEngine, AccountSecurityEvent
from src.workflow.hold_workflow import HoldWorkflowEngine
from src.persistence.sink import IdempotentEventSink
from src.serving.predictor import ProductionPredictor


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
def predictor():
    artifacts = os.path.join(FINPULSE_DIR, "models", "artifacts")
    return ProductionPredictor(artifacts_dir=artifacts)


@pytest.fixture
def gateway(test_sink, account_mgr, ato_eng, hold_eng, predictor):
    return PaymentGatewaySimulator(
        predictor=predictor,
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
        account_manager=account_mgr,
        seed=42,
        mode="DEMO_SEEDED"
    )


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file on disk."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# 1. Multi-Attacker State & Dynamic Transitions (Criteria 1 - 6)
# =============================================================================
class TestMultiAttackerStateAndTransitions:
    def test_criterion_1_multiple_attackers_instantiated(self, attacker):
        """1. Multiple attackers can be instantiated independently with unique states."""
        prof_morpheus = attacker.set_active_profile("ATK-7F31")
        state_morpheus = attacker.get_attacker_state("ATK-7F31")
        assert state_morpheus.actor_id == "ATK-7F31"
        assert state_morpheus.alias == "Morpheus"

        prof_hydra = attacker.set_active_profile("ATK-92B4")
        state_hydra = attacker.get_attacker_state("ATK-92B4")
        assert state_hydra.actor_id == "ATK-92B4"
        assert state_hydra.alias == "Hydra"

        assert state_morpheus.actor_id != state_hydra.actor_id
        assert state_morpheus.current_device != state_hydra.current_device

    def test_criterion_2_attacker_changes_device(self, attacker):
        """2. The same attacker can dynamically change hardware device fingerprint."""
        state = attacker.get_attacker_state("ATK-7F31")
        initial_device = state.current_device

        _, req1 = attacker.build_scenario_transfer(
            scenario_type="account_takeover",
            custom_params={"device_id": "DEV_DEVICE_ROTATION_A"},
            actor_id="ATK-7F31"
        )
        assert req1["device_id"] == "DEV_DEVICE_ROTATION_A"
        assert state.current_device == "DEV_DEVICE_ROTATION_A"
        assert state.previous_device == initial_device

        _, req2 = attacker.build_scenario_transfer(
            scenario_type="account_takeover",
            custom_params={"device_id": "DEV_DEVICE_ROTATION_B"},
            actor_id="ATK-7F31"
        )
        assert req2["device_id"] == "DEV_DEVICE_ROTATION_B"
        assert state.current_device == "DEV_DEVICE_ROTATION_B"
        assert state.previous_device == "DEV_DEVICE_ROTATION_A"
        assert "DEV_DEVICE_ROTATION_B" in state.device_history

    def test_criterion_3_attacker_changes_location(self, attacker):
        """3. The same attacker can dynamically change location across Indian metro nodes."""
        state = attacker.get_attacker_state("ATK-44D9")
        
        _, req1 = attacker.build_scenario_transfer(
            scenario_type="impossible_travel",
            custom_params={"location": INDIAN_CITIES["MUMBAI"]},
            actor_id="ATK-44D9"
        )
        assert req1["location"]["city"] == "MUMBAI"
        assert state.current_location["city"] == "MUMBAI"

        _, req2 = attacker.build_scenario_transfer(
            scenario_type="impossible_travel",
            custom_params={"location": INDIAN_CITIES["DELHI"]},
            actor_id="ATK-44D9"
        )
        assert req2["location"]["city"] == "DELHI"
        assert state.current_location["city"] == "DELHI"
        assert state.previous_location["city"] == "MUMBAI"

    def test_criterion_4_attacker_changes_recipient(self, attacker):
        """4. The same attacker can dynamically change recipient accounts."""
        state = attacker.get_attacker_state("ATK-C812")

        _, req1 = attacker.build_scenario_transfer(
            scenario_type="velocity_surge",
            receiver_id="REC_MULE_01",
            actor_id="ATK-C812"
        )
        assert req1["merchant_id"] == "REC_MULE_01"
        assert state.current_recipient == "REC_MULE_01"

        _, req2 = attacker.build_scenario_transfer(
            scenario_type="velocity_surge",
            receiver_id="REC_CRYPTO_02",
            actor_id="ATK-C812"
        )
        assert req2["merchant_id"] == "REC_CRYPTO_02"
        assert state.current_recipient == "REC_CRYPTO_02"
        assert "REC_CRYPTO_02" in state.recipient_history

    def test_criterion_5_and_6_sequence_maintains_state_and_history(self, attacker, gateway):
        """5 & 6. Attack sequence maintains state, and Attack #2 observes history from Attack #1."""
        state = attacker.get_attacker_state("ATK-7F31")
        assert state.attempt_count == 0

        # Attempt 1
        _, req1 = attacker.build_scenario_transfer(
            scenario_type="account_takeover",
            amount=8500.0,
            actor_id="ATK-7F31"
        )
        res1 = gateway.process_transfer(req1)
        attacker.record_outcome_and_adapt(res1)

        assert state.attempt_count == 1
        assert state.previous_decision == res1.decision
        assert len(state.history) == 1
        assert state.history[0]["transaction_id"] == req1["transaction_id"]

        # Attempt 2
        _, req2 = attacker.build_scenario_transfer(
            scenario_type="low_and_slow",
            amount=450.0,
            actor_id="ATK-7F31"
        )
        res2 = gateway.process_transfer(req2)
        attacker.record_outcome_and_adapt(res2)

        assert state.attempt_count == 2
        assert len(state.history) == 2
        # Verify Attack #2 can observe history from Attack #1
        assert state.history[1]["transaction_id"] == req1["transaction_id"]
        assert state.history[1]["decision"] == res1.decision
        assert state.amount_history == [8500.0, 450.0]


# =============================================================================
# 2. Dynamic Adversarial Adaptation (Criteria 7 - 9)
# =============================================================================
class TestAdversarialAdaptation:
    def test_criterion_7_adaptation_to_block(self, attacker):
        """7. Attacker behavior shifts to low-and-slow stealth upon BLOCK verdict."""
        mock_block_result = GatewayTransferResult(
            transaction_id="tx_test_block",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            receiver_id="REC_MULE_01",
            amount=8500.0,
            status="DECLINED",
            authorized=False,
            money_transferred=False,
            decision="BLOCK",
            risk_score=92.0,
            risk_level="CRITICAL",
            sender_balance_before=10000.0,
            sender_balance_after=10000.0,
            receiver_balance_before=0.0,
            receiver_balance_after=0.0,
            reasons=["ATO password reset alarm"],
            matched_rules=[],
            latency_ms=1.2
        )
        adapt = attacker.record_outcome_and_adapt(mock_block_result)
        assert adapt["previous_decision"] == "BLOCK"
        assert adapt["next_recommended_strategy"] == "low_and_slow"
        assert adapt["next_recommended_amount"] <= 500.0
        assert "sub-threshold limits" in adapt["tactical_rationale"]

    def test_criterion_8_adaptation_to_hold(self, attacker):
        """8. Attacker behavior rotates hardware & shifts channel upon HOLD verdict."""
        mock_hold_result = GatewayTransferResult(
            transaction_id="tx_test_hold",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            receiver_id="REC_OFFSHORE_03",
            amount=2500.0,
            status="HELD",
            authorized=False,
            money_transferred=False,
            decision="HOLD",
            risk_score=52.0,
            risk_level="MEDIUM",
            sender_balance_before=10000.0,
            sender_balance_after=10000.0,
            receiver_balance_before=0.0,
            receiver_balance_after=0.0,
            reasons=["New device requires 2FA challenge"],
            matched_rules=[],
            latency_ms=1.4
        )
        adapt = attacker.record_outcome_and_adapt(mock_hold_result)
        assert adapt["previous_decision"] == "HOLD"
        assert adapt["recommended_device_action"] == "ROTATE_HARDWARE"
        assert "2FA verification challenge" in adapt["tactical_rationale"]

    def test_criterion_9_adaptation_to_approve(self, attacker):
        """9. Attacker behavior escalates drain amount upon APPROVE verdict."""
        mock_app_result = GatewayTransferResult(
            transaction_id="tx_test_app",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            receiver_id="REC_MULE_01",
            amount=500.0,
            status="APPROVED",
            authorized=True,
            money_transferred=True,
            decision="APPROVE",
            risk_score=15.0,
            risk_level="LOW",
            sender_balance_before=10000.0,
            sender_balance_after=9500.0,
            receiver_balance_before=0.0,
            receiver_balance_after=500.0,
            reasons=[],
            matched_rules=[],
            latency_ms=1.1
        )
        adapt = attacker.record_outcome_and_adapt(mock_app_result)
        assert adapt["previous_decision"] == "APPROVE"
        assert adapt["next_recommended_amount"] == 800.0
        assert "accelerate balance extraction" in adapt["tactical_rationale"]


# =============================================================================
# 3. U1 & U2 Unseen Attack Specifications (Criteria 10 - 11)
# =============================================================================
class TestUnseenAttackScenarios:
    def test_criterion_10_u1_context_mutation_classification_and_context(self, attacker, gateway):
        """10. U1 produces a distinct context that is not one of the predefined attack templates."""
        events, req = attacker.build_scenario_transfer(
            scenario_type="u1_context_mutation",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=2450.0
        )
        # Verify template registry classification
        assert is_predefined_template("u1_context_mutation") is False
        assert get_template_classification("u1_context_mutation") == "Not one of the predefined attack templates"
        assert req["raw_metadata"]["template_classification"] == "Not one of the predefined attack templates"

        # Verify U1 observable context characteristics
        assert 1800.0 <= req["amount"] <= 3200.0
        assert req["auth_verified"] is True
        assert req["hour_of_day"] == 23
        assert req["device_id"] == "DEV-MUT-8842"
        assert req["location"]["city"] == "HYDERABAD"
        assert req["merchant_id"] == "REC_PEER_07"
        assert len(events) == 0  # No artificial ATO password reset injected

        # Verify execution through real gateway/inference pipeline
        res = gateway.process_transfer(req)
        assert res.transaction_id == req["transaction_id"]
        assert res.decision in ["APPROVE", "HOLD", "BLOCK"]
        assert res.latency_ms > 0

    def test_criterion_11_u2_compound_novel_classification_and_signals(self, attacker, gateway):
        """11. U2 produces a compound multi-signal pattern distinct from predefined templates."""
        events, req = attacker.build_scenario_transfer(
            scenario_type="u2_compound_novel",
            sender_id=DEFAULT_DEMO_CUSTOMER_ID,
            amount=2750.0
        )
        # Verify template registry classification
        assert is_predefined_template("u2_compound_novel") is False
        assert get_template_classification("u2_compound_novel") == "Not one of the predefined attack templates"
        assert req["raw_metadata"]["template_classification"] == "Not one of the predefined attack templates"

        # Verify compound signals
        assert req["hour_of_day"] == 2  # 02:00 AM off-hours
        assert req["device_id"] == "DEV-COMPOUND-77"
        assert req["location"]["city"] == "PUNE"
        assert req["amount"] == 2750.0
        assert len(events) == 1
        assert events[0].event_type == "device_fingerprint_drift"

        # Verify execution through real gateway/inference pipeline
        res = gateway.process_transfer(req)
        assert res.transaction_id == req["transaction_id"]
        assert res.decision in ["APPROVE", "HOLD", "BLOCK"]


# =============================================================================
# 4. Pipeline & Feature Vector Variation (Criteria 12 - 14)
# =============================================================================
class TestFeatureVectorAndStateEvolution:
    def test_criterion_12_feature_vectors_differ_with_context(self, attacker, predictor):
        """12. Different generated attack contexts produce genuinely different 32-feature vectors."""
        # Clean Redis customer state to start from clean slate
        predictor.redis_window.clear_customer_state(DEFAULT_DEMO_CUSTOMER_ID)

        # Attack 1: Low & Slow (Home city Chennai, known device, ₹450)
        _, req1 = attacker.build_scenario_transfer("low_and_slow", amount=450.0)
        eval1 = predictor.predict(req1)
        feat_dict1 = eval1["features_dict"]

        # Attack 2: Impossible Travel (Delhi, distant travel speed, 03:00 AM, ₹4200)
        _, req2 = attacker.build_scenario_transfer("impossible_travel", amount=4200.0)
        eval2 = predictor.predict(req2)
        feat_dict2 = eval2["features_dict"]

        # Assert vectors are measurably different across key dimensions
        assert feat_dict1["log_amount"] != feat_dict2["log_amount"]
        assert feat_dict1["distance_from_home_km"] != feat_dict2["distance_from_home_km"]
        assert feat_dict1["cyclic_hour_sin"] != feat_dict2["cyclic_hour_sin"]
        assert eval1["risk_score"] != eval2["risk_score"]

    def test_criterion_13_redis_state_evolves_across_attacks(self, attacker, predictor):
        """13. Redis velocity and rolling counters evolve correctly with each successive attack."""
        predictor.redis_window.clear_customer_state(DEFAULT_DEMO_CUSTOMER_ID)

        # Step 1: Baseline state (empty)
        prior0 = predictor.redis_window.fetch_prior_events(DEFAULT_DEMO_CUSTOMER_ID, time.time() + 10.0)
        cnt0 = len(prior0)

        # Step 2: Ingest transaction 1
        _, req1 = attacker.build_scenario_transfer("velocity_surge", amount=500.0)
        predictor.predict(req1)
        prior1 = predictor.redis_window.fetch_prior_events(DEFAULT_DEMO_CUSTOMER_ID, time.time() + 10.0)
        cnt1 = len(prior1)
        assert cnt1 >= cnt0 + 1

        # Step 3: Ingest transaction 2
        _, req2 = attacker.build_scenario_transfer("velocity_surge", amount=700.0)
        predictor.predict(req2)
        prior2 = predictor.redis_window.fetch_prior_events(DEFAULT_DEMO_CUSTOMER_ID, time.time() + 10.0)
        cnt2 = len(prior2)
        assert cnt2 >= cnt1 + 1

    def test_criterion_14_gateway_authoritative_for_final_decision(self, gateway, account_mgr):
        """14. Payment Gateway remains authoritative: money is never moved without APPROVE."""
        # 1. Block Scenario: Balance must remain unchanged
        bal_start = account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID)
        tx_block = {
            "transaction_id": "tx_gw_test_block",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": DEFAULT_ATTACKER_RECEIVER_ID,
            "amount": 8500.0,
            "timestamp": time.time(),
            "payment_type": "TRANSFER",
            "origin_balance": bal_start,
            "dest_balance": 0.0,
            "auth_verified": False,
            "device_id": "DEV_ATTACKER_01",
            "location": INDIAN_CITIES["MUMBAI"],
            "home_location": KNOWN_LOCATION
        }
        res = gateway.process_transfer(tx_block)
        assert res.authorized is False
        assert res.money_transferred is False
        assert account_mgr.get_balance(DEFAULT_DEMO_CUSTOMER_ID) == bal_start


# =============================================================================
# 5. Production Model Frozen Integrity (Criterion 15)
# =============================================================================
class TestModelArtifactIntegrity:
    def test_criterion_15_production_model_hash_unchanged(self, attacker, gateway):
        """15. Proves that running attacks never mutates the frozen CatBoost production model artifact."""
        model_path = os.path.join(FINPULSE_DIR, "models", "artifacts", "production_candidate_model.joblib")
        cal_path = os.path.join(FINPULSE_DIR, "models", "artifacts", "production_calibrator.joblib")

        hash_model_before = compute_file_sha256(model_path)
        hash_cal_before = compute_file_sha256(cal_path)

        # Run multiple adversarial attacks (Smash & Grab, Velocity, Travel, U1, U2)
        scenarios = ["account_takeover", "velocity_surge", "impossible_travel", "u1_context_mutation", "u2_compound_novel"]
        for sc in scenarios:
            events, req = attacker.build_scenario_transfer(sc)
            gateway.process_transfer(req)

        hash_model_after = compute_file_sha256(model_path)
        hash_cal_after = compute_file_sha256(cal_path)

        assert hash_model_before == hash_model_after, "CRITICAL: Production model artifact was mutated!"
        assert hash_cal_before == hash_cal_after, "CRITICAL: Production calibrator artifact was mutated!"


# =============================================================================
# 6. Event Contracts & Persistence Audits (Criteria 16 - 18)
# =============================================================================
class TestEventContractsAndPersistence:
    def test_criterion_16_decision_event_schema_validity(self, gateway):
        """16. Validates DecisionEvent contract is generated and valid across simulated transactions."""
        tx_req = {
            "transaction_id": "tx_contract_check",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": "REC_MULE_01",
            "amount": 1200.0,
            "timestamp": time.time(),
            "payment_type": "TRANSFER",
            "origin_balance": 10000.0,
            "dest_balance": 0.0,
            "auth_verified": True,
            "device_id": KNOWN_DEVICE_ID,
            "location": KNOWN_LOCATION,
            "home_location": KNOWN_LOCATION
        }
        res = gateway.process_transfer(tx_req)
        assert res.transaction_id == "tx_contract_check"
        assert res.hybrid_decision in ["APPROVE", "REVIEW", "BLOCK"]
        assert res.ml_decision in ["APPROVE", "REVIEW", "BLOCK"]
        assert 0.0 <= res.calibrated_probability <= 1.0

    def test_criterion_17_kafka_decision_event_published(self, test_sink, account_mgr, ato_eng, hold_eng, predictor):
        """17. Verifies DecisionEventPublisher integration produces valid publishable events."""
        class MockKafkaPublisher:
            def __init__(self):
                self.published_events = []
            def publish(self, event, sync=True):
                self.published_events.append(event)
                return True

        mock_pub = MockKafkaPublisher()
        gw = PaymentGatewaySimulator(
            predictor=predictor,
            hold_engine=hold_eng,
            ato_engine=ato_eng,
            event_sink=test_sink,
            account_manager=account_mgr,
            kafka_publisher=mock_pub
        )
        tx = {
            "transaction_id": "tx_kafka_val",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": "REC_MULE_01",
            "amount": 500.0,
            "timestamp": time.time(),
            "payment_type": "TRANSFER",
            "origin_balance": 10000.0,
            "dest_balance": 0.0,
            "auth_verified": True,
            "device_id": KNOWN_DEVICE_ID,
            "location": KNOWN_LOCATION,
            "home_location": KNOWN_LOCATION
        }
        gw.process_transfer(tx)
        assert len(mock_pub.published_events) == 1
        assert mock_pub.published_events[0].transaction_id == "tx_kafka_val"

    def test_criterion_18_persistence_audit_records(self, gateway, test_sink):
        """18. Audit records for transactions, decisions, and security events are saved in the sink."""
        tx_audit = {
            "transaction_id": "tx_audit_sink_01",
            "customer_id": DEFAULT_DEMO_CUSTOMER_ID,
            "merchant_id": "REC_MULE_01",
            "amount": 1000.0,
            "timestamp": time.time(),
            "payment_type": "TRANSFER",
            "origin_balance": 10000.0,
            "dest_balance": 0.0,
            "auth_verified": True,
            "device_id": KNOWN_DEVICE_ID,
            "location": KNOWN_LOCATION,
            "home_location": KNOWN_LOCATION
        }
        gateway.process_transfer(tx_audit)
        dec = test_sink.get_decision("tx_audit_sink_01")
        assert dec is not None
        assert dec["transaction_id"] == "tx_audit_sink_01"


# =============================================================================
# 7. Clean State Reset & Reproducibility (Criteria 19 - 20)
# =============================================================================
class TestResetAndSeededReproducibility:
    def test_criterion_19_reset_returns_to_clean_state(self, attacker):
        """19. Reset clears attempt counters, adaptation logs, and returns attacker states to clean initial baseline."""
        # Execute attempts to dirty the state
        attacker.build_scenario_transfer("account_takeover", actor_id="ATK-7F31")
        attacker.build_scenario_transfer("velocity_surge", actor_id="ATK-C812")
        attacker.record_outcome_and_adapt(GatewayTransferResult(
            transaction_id="tx_dummy", sender_id="c", receiver_id="m", amount=100.0,
            status="APPROVED", authorized=True, money_transferred=True, decision="APPROVE",
            risk_score=10.0, risk_level="LOW", sender_balance_before=100.0, sender_balance_after=0.0,
            receiver_balance_before=0.0, receiver_balance_after=100.0, reasons=[], matched_rules=[], latency_ms=1.0
        ))

        assert attacker.transaction_counter > 0
        assert len(attacker.adaptation_log) > 0

        # Execute reset
        attacker.reset()

        assert attacker.transaction_counter == 0
        assert len(attacker.adaptation_log) == 0
        assert attacker.last_outcome is None
        state = attacker.get_attacker_state("ATK-7F31")
        assert state.attempt_count == 0
        assert len(state.history) == 0

    def test_criterion_20_seeded_mode_reproduces_same_experiment(self, test_sink, account_mgr, ato_eng):
        """20. Seeded mode reproduces the exact same attack parameter sequence deterministically."""
        # Experiment Run A (Seed 42)
        sim_a = AttackerSimulator(ato_engine=ato_eng, event_sink=test_sink, account_manager=account_mgr, seed=42, mode="DEMO_SEEDED")
        _, req_a1 = sim_a.build_scenario_transfer("account_takeover")
        _, req_a2 = sim_a.build_scenario_transfer("velocity_surge")
        _, req_a3 = sim_a.build_scenario_transfer("u1_context_mutation")

        # Experiment Run B (Seed 42)
        sim_b = AttackerSimulator(ato_engine=ato_eng, event_sink=test_sink, account_manager=account_mgr, seed=42, mode="DEMO_SEEDED")
        _, req_b1 = sim_b.build_scenario_transfer("account_takeover")
        _, req_b2 = sim_b.build_scenario_transfer("velocity_surge")
        _, req_b3 = sim_b.build_scenario_transfer("u1_context_mutation")

        assert req_a1["amount"] == req_b1["amount"]
        assert req_a2["amount"] == req_b2["amount"]
        assert req_a3["amount"] == req_b3["amount"]
