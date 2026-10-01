"""Tests for FastAPI serving layer endpoints."""
import pytest
from fastapi.testclient import TestClient
import os
import sys

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.api import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Verify /health returns 200 and model status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"

def test_predict_endpoint(client):
    """Verify /predict returns valid schema with decisions and reasons."""
    payload = {
        "transaction_id": "api_test_001",
        "timestamp": 1704067200.0,
        "amount": 1250.0,
        "customer_id": "cust_api_1",
        "merchant_id": "merch_api_1",
        "category": "retail",
        "payment_type": "TRANSFER",
        "origin_balance": 3000.0,
        "latitude": 37.77,
        "longitude": -122.41,
        "device_id": "dev_phone_1",
        "auth_verified": True
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["transaction_id"] == "api_test_001"
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert data["decision"] in ["APPROVE", "REVIEW", "BLOCK"]
    assert "signals" in data
    assert len(data["top_reasons"]) > 0
    assert data["latency_ms"] > 0
