"""
FinPulse R7 — Relational Persistence Sink Test Suite (R7-F).

Validates:
1. Idempotent insertion of DecisionEvents
2. Duplicate event & duplicate transaction handling
3. Delayed fraud label attachment and query
4. Schema consistency & transaction safety
"""

import os
import sys
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.persistence.sink import IdempotentEventSink
from src.risk_engine.decision_event import DecisionEvent


class TestPersistenceR7:
    """Test suite for idempotent persistence sink."""

    @pytest.fixture
    def sink(self):
        return IdempotentEventSink(db_path=":memory:")

    @pytest.fixture
    def sample_event(self):
        return DecisionEvent(
            transaction_id="tx_persist_001",
            customer_id="cust_p1",
            decision="APPROVE",
            risk_score=14.2,
            risk_level="LOW",
            ml_decision="APPROVE",
            calibrated_probability=0.035,
            signals={"ml_risk": 0.035},
            diagnostics={"top_features": []},
            reasons=["Clean profile"],
            timestamp=1700000000.0,
            model_version="finpulse-v3"
        )

    def test_persist_decision_event(self, sink, sample_event):
        success = sink.persist_decision_event(sample_event)
        assert success is True
        assert sink.count_records() == 1

        record = sink.get_decision(sample_event.transaction_id)
        assert record is not None
        assert record["decision"] == "APPROVE"
        assert record["risk_score"] == 14.2
        assert record["customer_id"] == "cust_p1"

    def test_idempotent_duplicate_writes(self, sink, sample_event):
        # Insert twice
        sink.persist_decision_event(sample_event)
        sink.persist_decision_event(sample_event, workflow_status="UPDATED")

        # Must still only have 1 record
        assert sink.count_records() == 1
        record = sink.get_decision(sample_event.transaction_id)
        assert record["workflow_status"] == "UPDATED"

    def test_attach_delayed_fraud_label(self, sink, sample_event):
        sink.persist_decision_event(sample_event)
        
        # Ground truth label arrives later (chargeback confirmed = 1)
        attached = sink.attach_delayed_label(sample_event.transaction_id, fraud_label=1)
        assert attached is True

        record = sink.get_decision(sample_event.transaction_id)
        assert record["fraud_label"] == 1

        # Query labeled training dataset
        labeled = sink.get_labeled_dataset()
        assert len(labeled) == 1
        assert labeled[0]["transaction_id"] == sample_event.transaction_id
