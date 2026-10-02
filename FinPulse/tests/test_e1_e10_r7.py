"""
FinPulse R7 — E1–E10 Evaluation Suite PyTest Integration.
"""

import os
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.evaluation.suite_e1_e10 import FullSystemEvaluationSuite


class TestE1ToE10Suite:
    """Automated PyTest wrapper for E1–E10 experimental benchmarks."""

    @pytest.fixture(scope="class")
    def suite(self):
        return FullSystemEvaluationSuite()

    def test_e1_parity_reproducibility(self, suite):
        res = suite.run_e1_parity()
        assert res.status == "PASSED"

    def test_e2_concurrency_idempotency(self, suite):
        res = suite.run_e2_concurrency()
        assert res.status == "PASSED"

    def test_e3_policy_consistency(self, suite):
        res = suite.run_e3_policy()
        assert res.status == "PASSED"

    def test_e4_hold_lifecycle(self, suite):
        res = suite.run_e4_hold_lifecycle()
        assert res.status == "PASSED"

    def test_e5_replay_delayed_labels(self, suite):
        res = suite.run_e5_replay_and_delayed_labels()
        assert res.status == "PASSED"

    def test_e6_cold_start(self, suite):
        res = suite.run_e6_cold_start()
        assert res.status == "PASSED"

    def test_e7_ato_compounding(self, suite):
        res = suite.run_e7_ato_compounding()
        assert res.status == "PASSED"

    def test_e8_load_capacity(self, suite):
        res = suite.run_e8_capacity_and_latency()
        assert res.status == "PASSED"

    def test_e9_shadow_challenger(self, suite):
        res = suite.run_e9_shadow_challenger()
        assert res.status == "PASSED"

    def test_e10_drift_promotion(self, suite):
        res = suite.run_e10_drift_and_promotion()
        assert res.status == "PASSED"
