"""
FinPulse Demo Data Seeder & Presentation Environment Controller.

Provides controlled demo dataset generation:
1. Canonical APPROVE: tx_demo_approve_001 (₹2,000, CUST_001 -> CUST_002, Risk ~9)
2. Canonical HOLD:    tx_demo_hold_001    (₹50,000, CUST_001 -> CUST_892, Risk ~52, Pending Phone Confirmation)
3. Canonical BLOCK:   tx_demo_block_001   (₹90,000, CUST_999 -> CUST_892, Risk ~85+, Zero-Balance Hard Block)
"""

import os
import sys
import time
import json
import argparse

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.risk_engine.decision_event import DecisionEvent
from src.workflow.hold_workflow import HoldWorkflowEngine, HoldCase
from src.persistence.config import PostgresConfig
from src.persistence.sink import IdempotentEventSink


def seed_demo_dataset(sink: IdempotentEventSink, clean_first: bool = False):
    print("=" * 70)
    print("SEEDING CONTROLLED FINPULSE DEMO DATASET")
    print("=" * 70)

    if clean_first:
        print("Cleaning previous demo records...")
        p = "%s" if sink.backend == "postgres" else "?"
        with sink._get_cursor() as (cur, _, _):
            for tbl in ["fraud_alerts", "hold_cases", "fraud_decisions", "transactions"]:
                cur.execute(f"DELETE FROM {tbl} WHERE transaction_id LIKE 'tx_demo_%';")
        print("Previous demo records purged.")

    now = time.time()
    hold_engine = HoldWorkflowEngine()

    # 1. Canonical APPROVE
    tx_app = "tx_demo_approve_001"
    dev_app = DecisionEvent(
        transaction_id=tx_app,
        customer_id="CUST_001",
        decision="APPROVE",
        risk_score=9.2,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.0095,
        signals={"ml": 0.0095, "velocity": 0.0, "rules": 0.0},
        diagnostics={"hard_block": False},
        reasons=[],
        timestamp=now - 120.0,
        model_version="finpulse-v3",
        latency_ms=1.1
    )
    sink.persist_transaction({
        "transaction_id": tx_app, "customer_id": "CUST_001", "merchant_id": "CUST_002",
        "amount": 2000.0, "timestamp": now - 120.0, "category": "transfer",
        "payment_type": "TRANSFER", "origin_balance": 45000.0, "auth_verified": True
    })
    sink.persist_decision_event(dev_app, workflow_status="FINAL")
    print(f"  [+] Seeded Scenario A (APPROVE): {tx_app} (INR 2,000 | Risk: 9.2)")

    # 2. Canonical HOLD (Realistic ₹8,500 transfer hold on ₹10,000 balance)
    tx_hold = "tx_demo_hold_001"
    case, token = hold_engine.create_hold(
        transaction_id=tx_hold,
        customer_id="CUST_DEMO_001",
        amount=8500.0,
        timeout_seconds=300.0,
        current_time=now - 30.0,
        metadata={
            "demo_scenario": "HOLD_TO_RELEASE",
            "receiver_id": "CUST_ATTACKER",
            "device_id": "DEV_ATTACKER_01",
            "location": {"city": "MUMBAI"}
        }
    )
    dev_hold = DecisionEvent(
        transaction_id=tx_hold,
        customer_id="CUST_DEMO_001",
        decision="REVIEW",
        risk_score=58.5,
        risk_level="MEDIUM",
        ml_decision="REVIEW",
        calibrated_probability=0.28,
        signals={"ml": 0.28, "velocity": 0.35, "rules": 0.25},
        diagnostics={"hard_block": False},
        reasons=[
            "High-risk transfer attempt from unrecognized device (DEV_ATTACKER_01)",
            "Distant location anomaly (Mumbai vs home Chennai)",
            "₹8,500 transfer held: Customer ₹10,000 balance remains completely protected"
        ],
        timestamp=now - 30.0,
        model_version="finpulse-v3",
        latency_ms=1.4
    )
    sink.persist_transaction({
        "transaction_id": tx_hold, "customer_id": "CUST_DEMO_001", "merchant_id": "CUST_ATTACKER",
        "amount": 8500.0, "timestamp": now - 30.0, "category": "transfer",
        "payment_type": "TRANSFER", "origin_balance": 10000.0, "auth_verified": False
    })
    sink.persist_decision_event(dev_hold, workflow_status="HOLD")
    sink.persist_hold_case({
        "hold_id": case.hold_id, "transaction_id": case.transaction_id,
        "customer_id": case.customer_id, "amount": case.amount,
        "status": "HOLD", "token_hash": case.token_hash, "timeout_seconds": case.timeout_seconds,
        "created_at": case.created_at, "expires_at": case.expires_at,
        "customer_channel": "sms_push", "metadata": case.metadata
    })
    print(f"  [+] Seeded Scenario B (HOLD):    {tx_hold} (INR 8,500 | Case: {case.hold_id} | Token: {token[:8]}...)")

    # 3. Canonical BLOCK
    tx_blk = "tx_demo_block_001"
    dev_blk = DecisionEvent(
        transaction_id=tx_blk,
        customer_id="CUST_999",
        decision="BLOCK",
        risk_score=85.0,
        risk_level="CRITICAL",
        ml_decision="BLOCK",
        calibrated_probability=0.89,
        signals={"ml": 0.89, "rules": 1.0, "velocity": 0.8},
        diagnostics={"hard_block": True},
        reasons=["CRITICAL: Zero-balance account drain without authentication", "Missing multi-factor verification"],
        timestamp=now - 5.0,
        model_version="finpulse-v3",
        hard_block=True,
        latency_ms=1.8
    )
    sink.persist_transaction({
        "transaction_id": tx_blk, "customer_id": "CUST_999", "merchant_id": "CUST_892",
        "amount": 90000.0, "timestamp": now - 5.0, "category": "transfer",
        "payment_type": "TRANSFER", "origin_balance": 0.0, "auth_verified": False
    })
    sink.persist_decision_event(dev_blk, workflow_status="FINAL")
    sink.persist_fraud_alert({
        "alert_id": f"alt_{tx_blk}", "transaction_id": tx_blk, "event_id": str(dev_blk.event_id),
        "customer_id": "CUST_999", "severity": "CRITICAL", "decision": "BLOCK",
        "reasons": ["Zero-balance drain without authentication"], "notified": True,
        "status": "OPEN", "created_at": now - 5.0
    })
    print(f"  [+] Seeded Scenario C (BLOCK):   {tx_blk} (INR 90,000 | Hard Block | Alert OPEN)")
    print("=" * 70)
    print("DEMO DATASET SEEDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed Controlled FinPulse Demo Data")
    parser.add_argument("--clean", action="store_true", help="Purge previous demo records before seeding")
    args = parser.parse_args()

    cfg = PostgresConfig.from_env()
    snk = IdempotentEventSink(postgres_config=cfg, force_backend="postgres")
    seed_demo_dataset(snk, clean_first=args.clean)
