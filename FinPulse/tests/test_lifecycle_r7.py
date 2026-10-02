"""
FinPulse R7 — Model Lifecycle & Simulation Test Suite (R7-E, R7-H, R7-I, R7-J).

Validates:
1. R7-E Replay simulator & delayed label manager
2. R7-H Shadow / Challenger model isolation & non-interference
3. R7-I Retraining orchestrator & promotion gate logic
4. R7-J Population Stability Index (PSI) drift detection
"""

import os
import sys
import numpy as np
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.simulation.replay import TransactionReplaySimulator, DelayedLabelManager, ReplayConfig
from src.challenger.shadow_scorer import ShadowModelRunner
from src.monitoring.drift import PSIDriftMonitor
from training.retrain import ModelPromotionGate


class TestLifecycleR7:
    """Test suite for simulation, challenger scoring, promotion gate, and drift."""

    # =========================================================================
    # R7-E: Replay & Delayed Labels
    # =========================================================================
    def test_replay_simulator_generation(self):
        sim = TransactionReplaySimulator(ReplayConfig(speedup_multiplier=0.0))
        stream = sim.generate_scenario_stream("mixed", count=25)
        assert len(stream) == 25
        assert all("transaction_id" in tx for tx in stream)

        # Mock scoring callback
        def mock_score(tx):
            return {"decision": "APPROVE", "risk_score": 10.0}

        results = sim.execute_replay(stream, scoring_callback=mock_score)
        assert len(results) == 25

    def test_delayed_label_maturation(self):
        manager = DelayedLabelManager()
        manager.queue_delayed_label("tx_dl_01", original_timestamp=100.0, fraud_label=1, delay_seconds=50.0)

        # Before delay has passed (t=120)
        assert len(manager.process_maturing_labels(120.0)) == 0

        # After delay has passed (t=160)
        matured = manager.process_maturing_labels(160.0)
        assert len(matured) == 1
        assert matured[0].transaction_id == "tx_dl_01"

    # =========================================================================
    # R7-H: Shadow / Challenger Isolation
    # =========================================================================
    def test_shadow_scorer_isolation(self):
        class MockChallenger:
            def predict_proba(self, X):
                return np.array([[0.80, 0.20]])

        runner = ShadowModelRunner(challenger_model=MockChallenger(), challenger_version="finpulse-v4-test")
        dummy_vec = np.zeros(32)
        prod_res = {"decision": "APPROVE", "risk_score": 10.0, "calibrated_probability": 0.02}

        comparison = runner.evaluate_shadow("tx_s_01", dummy_vec, prod_res)
        assert comparison is not None
        assert comparison.production_decision == "APPROVE"
        assert comparison.challenger_risk_score == 20.0

    def test_shadow_scorer_exception_handled_cleanly(self):
        class BrokenChallenger:
            def predict_proba(self, X):
                raise RuntimeError("Challenger GPU OOM")

        runner = ShadowModelRunner(challenger_model=BrokenChallenger())
        prod_res = {"decision": "APPROVE", "risk_score": 10.0}
        # Must NOT raise exception; production unaffected
        comp = runner.evaluate_shadow("tx_s_err", np.zeros(32), prod_res)
        assert comp is None

    # =========================================================================
    # R7-I: Retraining Promotion Gate
    # =========================================================================
    def test_promotion_gate_approved(self):
        gate = ModelPromotionGate()
        prod = {"pr_auc": 0.88, "roc_auc": 0.960, "brier_score": 0.019, "fpr_at_target_recall": 0.013}
        cand = {"pr_auc": 0.89, "roc_auc": 0.965, "brier_score": 0.018, "fpr_at_target_recall": 0.012}

        report = gate.evaluate_candidate(prod, cand)
        assert report.overall_status == "APPROVED"
        assert all(c.passed for c in report.checks)

    def test_promotion_gate_rejected_on_pr_auc_regression(self):
        gate = ModelPromotionGate()
        prod = {"pr_auc": 0.88, "roc_auc": 0.960, "brier_score": 0.019, "fpr_at_target_recall": 0.013}
        # Candidate has inferior PR-AUC
        cand = {"pr_auc": 0.82, "roc_auc": 0.965, "brier_score": 0.018, "fpr_at_target_recall": 0.012}

        report = gate.evaluate_candidate(prod, cand)
        assert report.overall_status == "REJECTED"
        assert any(c.name == "PR-AUC Non-Regression" and not c.passed for c in report.checks)

    # =========================================================================
    # R7-J: PSI Drift Monitor
    # =========================================================================
    def test_psi_drift_stable_and_shifted(self):
        monitor = PSIDriftMonitor(num_bins=10)
        np.random.seed(42)
        ref = np.random.normal(100.0, 15.0, 1000)
        stable = np.random.normal(100.0, 15.0, 1000)
        shifted = np.random.normal(150.0, 20.0, 1000)

        rep_s = monitor.calculate_psi(ref, stable, "feat_stable")
        assert rep_s.psi_value < 0.10
        assert rep_s.severity == "STABLE"

        rep_shift = monitor.calculate_psi(ref, shifted, "feat_shifted")
        assert rep_shift.psi_value >= 0.25
        assert rep_shift.severity == "SIGNIFICANT_DRIFT"
