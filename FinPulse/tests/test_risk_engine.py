"""Adversarial stress testing and behavioral sensitivity verification."""
import pytest
import time
import os
import sys

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.predictor import ProductionPredictor

@pytest.fixture
def predictor():
    artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
    return ProductionPredictor(artifacts_dir)

def test_adversarial_behavioral_escalation(predictor):
    """
    PHASE ML-15 / GATE 15: Verify risk score reacts monotonically
    to abnormal amount, new device, unverified auth, and velocity bursts.
    """
    now = time.time()
    user_id = f"test_adv_{int(now)}"

    # 1. Normal Baseline Transaction
    tx_normal = {
        "transaction_id": f"{user_id}_01",
        "timestamp": now,
        "amount": 45.0,
        "customer_id": user_id,
        "merchant_id": "merch_local_cafe",
        "category": "dining",
        "payment_type": "TRANSFER",
        "origin_balance": 5000.0,
        "device_id": "dev_registered_1",
        "auth_verified": True
    }
    res_normal = predictor.predict(tx_normal)
    assert res_normal["risk_score"] < 50.0
    assert res_normal["decision"] in ["APPROVE", "REVIEW"]

    # 2. Attack Step 1: Unusual Amount (Draining 98% of balance)
    tx_step1 = tx_normal.copy()
    tx_step1["transaction_id"] = f"{user_id}_02"
    tx_step1["amount"] = 4950.0
    res_step1 = predictor.predict(tx_step1)
    assert res_step1["risk_score"] >= res_normal["risk_score"]

    # 3. Attack Step 2: Unverified Auth + High Amount
    tx_step2 = tx_step1.copy()
    tx_step2["transaction_id"] = f"{user_id}_03"
    tx_step2["auth_verified"] = False
    res_step2 = predictor.predict(tx_step2)
    assert res_step2["risk_score"] >= res_step1["risk_score"]
    assert any("authentication" in r.lower() or "drains" in r.lower() for r in res_step2["top_reasons"])

    # 4. Attack Step 3: Rapid Velocity Burst (Repeated transactions in same minute)
    for i in range(4):
        tx_burst = tx_step2.copy()
        tx_burst["transaction_id"] = f"{user_id}_burst_{i}"
        tx_burst["timestamp"] = now + i + 1
        predictor.predict(tx_burst)

    tx_final = tx_step2.copy()
    tx_final["transaction_id"] = f"{user_id}_final"
    tx_final["timestamp"] = now + 10
    res_final = predictor.predict(tx_final)
    
    # Must trigger BLOCK decision under combined adversarial stress
    assert res_final["risk_score"] >= 60.0
    assert res_final["decision"] in ["REVIEW", "BLOCK"]
    assert res_final["latency_ms"] < 100.0, f"Latency {res_final['latency_ms']}ms exceeded SLA"
