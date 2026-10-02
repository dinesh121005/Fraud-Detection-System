"""
FinPulse R6 - Deployment & Infrastructure Validation Test Suite.
Validates:
1. Docker Compose specification & schema integrity
2. Service dependencies, restart policies, and healthcheck configurations
3. API readiness & liveness probe contracts
4. Prometheus scraping endpoint availability
5. Model artifact existence and loading readiness
"""

import os
import sys
import yaml
import pytest
from fastapi.testclient import TestClient

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.api import app
from src.serving.predictor import ProductionPredictor


class TestDeploymentR6:
    """Deployment configuration and infrastructure integrity tests."""

    @pytest.fixture
    def client(self):
        with TestClient(app) as client:
            yield client

    @pytest.fixture
    def compose_data(self):
        compose_path = os.path.join(FINPULSE_DIR, "docker-compose.yml")
        assert os.path.exists(compose_path), "docker-compose.yml must exist"
        with open(compose_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # =========================================================================
    # 1. Docker Compose Schema & Configuration
    # =========================================================================
    def test_compose_services_defined(self, compose_data):
        """Verify essential microservices are defined in docker-compose.yml."""
        services = compose_data.get("services", {})
        expected_services = ["kafka", "redis", "finpulse-api", "streaming-worker", "prometheus"]
        for svc in expected_services:
            assert svc in services, f"Service {svc} must be defined in docker-compose.yml"

    def test_compose_healthchecks_and_restart_policy(self, compose_data):
        """Verify restart policies and healthchecks exist for core infrastructure."""
        services = compose_data.get("services", {})
        
        # Redis healthcheck & restart policy
        redis_svc = services.get("redis", {})
        assert redis_svc.get("restart") in ["always", "unless-stopped"]
        assert "healthcheck" in redis_svc

        # Finpulse API healthcheck & restart policy
        api_svc = services.get("finpulse-api", {})
        assert api_svc.get("restart") in ["always", "unless-stopped"]
        assert "healthcheck" in api_svc

        # Streaming Worker restart policy & dependencies
        worker_svc = services.get("streaming-worker", {})
        assert worker_svc.get("restart") in ["always", "unless-stopped"]
        assert "depends_on" in worker_svc

    def test_compose_named_volumes(self, compose_data):
        """Verify persistent volumes are declared for stateful storage."""
        volumes = compose_data.get("volumes", {})
        assert "redis_data" in volumes or "redis-data" in volumes, "Persistent volume for Redis must be declared"

    # =========================================================================
    # 2. Probes & Endpoints
    # =========================================================================
    def test_health_probe_contract(self, client):
        """Verify /health returns HTTP 200 with operational status."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "healthy"
        assert "version" in data

    def test_ready_probe_contract(self, client):
        """Verify /ready returns HTTP 200 and confirms predictor readiness."""
        resp = client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("status") == "ready"
        assert "model_version" in data

    def test_prometheus_metrics_scrape_endpoint(self, client):
        """Verify /metrics returns Prometheus formatted plaintext."""
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("Content-Type", "")
        body = resp.text
        # Assert key FinPulse metrics exist
        assert "finpulse_transactions_total" in body or "process_cpu_seconds_total" in body

    # =========================================================================
    # 3. Model Artifact Availability
    # =========================================================================
    def test_production_model_artifact_loadable(self):
        """Verify production model artifacts exist and initialize correctly."""
        artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
        predictor = ProductionPredictor(artifacts_dir=artifacts_dir)
        assert predictor.model is not None
        assert predictor.calibrator is not None
        assert predictor.pipeline is not None
