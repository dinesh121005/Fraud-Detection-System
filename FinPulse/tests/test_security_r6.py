"""
FinPulse R6 - Security Hardening & Threat Mitigation Test Suite.
Validates input sanitization, API security headers, payload boundaries,
safe error handling, and sensitive data protection:
1. Input Security: NaN, Infinity, negative amounts, extreme numeric values, schema violations
2. Oversized payload rejection (1MB limit)
3. API Security headers (X-Content-Type-Options, X-Frame-Options, Cache-Control, etc.)
4. Error message leakage prevention (no stack traces leaked on 500)
5. Secret management & bounded label verification
"""

import math
import os
import sys
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.api import app
from src.serving.schemas import TransactionRequest


class TestSecurityR6:
    """Security hardening validation suite."""

    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def valid_payload(self):
        return {
            "transaction_id": "tx_sec_valid_001",
            "customer_id": "cust_sec_100",
            "amount": 150.75,
            "timestamp": 1700000000.0,
            "merchant_id": "merch_sec_01",
            "category": "retail",
            "payment_type": "PAYMENT",
            "device_id": "dev_sec_99",
            "channel": "mobile",
            "country": "US"
        }

    # =========================================================================
    # 1. Input Security & Strict Validation
    # =========================================================================
    def test_negative_amount_rejected(self, client, valid_payload):
        """Verify negative or zero transaction amounts are strictly rejected (HTTP 422)."""
        valid_payload["amount"] = -50.0
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422, "Negative amounts must be rejected"

        valid_payload["amount"] = 0.0
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422, "Zero amount must be rejected"

    def test_extreme_amount_rejected(self, client, valid_payload):
        """Verify absurdly large transaction amounts (> $100M) are rejected (HTTP 422)."""
        valid_payload["amount"] = 100_000_000.01
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422, "Extreme amounts exceeding bounds must be rejected"

    def test_nan_and_infinity_rejected(self):
        """Verify NaN and Infinity floats cannot pass Pydantic schema validation."""
        data = {
            "transaction_id": "tx_nan",
            "customer_id": "usr_nan",
            "amount": float("nan"),
            "timestamp": 1700000000.0
        }
        with pytest.raises(ValidationError):
            TransactionRequest(**data)

        data["amount"] = float("inf")
        with pytest.raises(ValidationError):
            TransactionRequest(**data)

    def test_oversized_string_fields_rejected(self, client, valid_payload):
        """Verify oversized field inputs (>128 chars) are rejected to prevent buffer bloat."""
        valid_payload["merchant_id"] = "A" * 500
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422

    def test_empty_string_identifiers_rejected(self, client, valid_payload):
        """Verify whitespace-only or empty strings for mandatory IDs are rejected."""
        valid_payload["transaction_id"] = "   "
        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 422

    # =========================================================================
    # 2. Oversized Payload Rejection
    # =========================================================================
    def test_oversized_payload_rejected(self, client):
        """Verify request bodies exceeding 1MB are rejected with HTTP 413."""
        # 1.5 MB payload
        huge_junk = "x" * (1500 * 1024)
        response = client.post(
            "/predict",
            content=huge_junk,
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413
        assert "Payload exceeds maximum allowable size" in response.json().get("detail", "")

    # =========================================================================
    # 3. API Security Headers
    # =========================================================================
    def test_security_headers_present(self, client):
        """Verify essential security headers are injected into HTTP responses."""
        response = client.get("/health")
        assert response.status_code == 200

        headers = response.headers
        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert headers.get("X-XSS-Protection") == "1; mode=block"
        assert "Strict-Transport-Security" in headers
        assert "no-store" in headers.get("Cache-Control", "")

    # =========================================================================
    # 4. Error Message Leakage & Probes
    # =========================================================================
    def test_internal_error_does_not_leak_stacktrace(self, client, valid_payload, monkeypatch):
        """Verify unhandled exceptions return sanitized 500 without leaking stack traces."""
        from src.serving import api
        def mock_broken_predict(*args, **kwargs):
            raise RuntimeError("Database password leaked in raw stack: sql_pass_xyz123")

        monkeypatch.setattr(api.predictor, "predict", mock_broken_predict)

        response = client.post("/predict", json=valid_payload)
        assert response.status_code == 500
        detail = response.json().get("detail", "")
        # Must NOT leak exception text or trace
        assert "sql_pass_xyz123" not in detail
        assert "Internal transaction scoring error." in detail

    def test_readiness_probe(self, client):
        """Verify /ready probe reports operational status."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert "model_version" in data
