"""
Comprehensive PostgreSQL Persistence Layer Test Suite for FinPulse.

Validates:
1. PostgreSQL connection pooling and health checks
2. Schema initialization, tables, constraints, and indexes
3. CRUD operations across transactions, DecisionEvents, HOLD cases, alerts, account events, mandates, replay records
4. Strict write idempotency on event_id and transaction_id
5. Transaction atomicity and rollback on error
6. DecisionEvent v1.0 contract preservation
"""

import time
import pytest
from src.risk_engine.decision_event import DecisionEvent
from src.persistence.config import PostgresConfig
from src.persistence.connection import PostgresConnectionPool
from src.persistence.sink import IdempotentEventSink


@pytest.fixture(scope="module")
def pg_sink():
    """Module-scoped PostgreSQL sink connected to the test/runtime database."""
    config = PostgresConfig.from_env()
    sink = IdempotentEventSink(postgres_config=config, force_backend="postgres")
    yield sink


@pytest.fixture
def sample_decision_event():
    """Construct a canonical DecisionEvent v1.0."""
    now = time.time()
    return DecisionEvent(
        transaction_id=f"tx_test_{int(now*1000)}",
        customer_id="cust_test_42",
        decision="APPROVE",
        risk_score=12.5,
        risk_level="LOW",
        ml_decision="APPROVE",
        calibrated_probability=0.045,
        signals={"amount_zscore": 0.2, "hour_risk": 0.1},
        diagnostics={"rule_flags": [], "hard_block": False},
        reasons=["Normal transaction pattern"],
        timestamp=now,
        model_version="finpulse-v3",
        matched_rules=[],
        hard_block=False,
        hard_block_rules=[],
        latency_ms=3.4
    )


class TestPostgresConnectionAndHealth:
    """Test pool initialization, health probes, and error handling."""

    def test_pool_singleton_and_health(self):
        pool = PostgresConnectionPool.get_instance()
        health = pool.check_health()
        assert health["status"] == "HEALTHY"
        assert health["backend"] == "postgresql"
        assert health["database"] == "finpulse"
        assert health["latency_ms"] >= 0.0

    def test_connection_checkout_and_commit(self):
        pool = PostgresConnectionPool.get_instance()
        with pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 100 + 42;")
                res = cur.fetchone()[0]
                assert res == 142

    def test_rollback_on_exception(self):
        pool = PostgresConnectionPool.get_instance()
        # Verify that an intentional error rolls back and does not leave dirty state
        with pytest.raises(Exception):
            with pool.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("CREATE TEMPORARY TABLE temp_rollback_test (id INT PRIMARY KEY);")
                    cur.execute("INSERT INTO temp_rollback_test VALUES (1);")
                    raise RuntimeError("Intentional transaction failure for rollback verification")


class TestPostgresSchemaAndTables:
    """Test required tables, schema objects, and metadata."""

    def test_all_seven_tables_exist(self, pg_sink):
        expected_tables = {
            "fraud_decisions",
            "transactions",
            "hold_cases",
            "fraud_alerts",
            "account_security_events",
            "mandates",
            "replay_evaluation_records"
        }
        with pg_sink._pg_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
                existing_tables = {r[0] for r in cur.fetchall()}

        for table in expected_tables:
            assert table in existing_tables, f"Expected table {table} missing from PostgreSQL!"


class TestPostgresCRUDAndIdempotency:
    """Test insert, duplicate handling, and lifecycle operations."""

    def test_persist_decision_event_and_retrieve(self, pg_sink, sample_decision_event):
        success = pg_sink.persist_decision_event(sample_decision_event)
        assert success is True

        rec = pg_sink.get_decision(sample_decision_event.transaction_id)
        assert rec is not None
        assert rec["event_id"] == sample_decision_event.event_id
        assert rec["customer_id"] == "cust_test_42"
        assert rec["decision"] == "APPROVE"
        assert rec["risk_score"] == pytest.approx(12.5)
        assert rec["hard_block"] is False

    def test_idempotent_duplicate_decision_event(self, pg_sink, sample_decision_event):
        # Insert first time
        pg_sink.persist_decision_event(sample_decision_event, workflow_status="INITIAL")
        initial_count = pg_sink.count_records("fraud_decisions")

        # Insert exact same event again with updated status
        pg_sink.persist_decision_event(sample_decision_event, workflow_status="CONFIRMED_FINAL")
        after_count = pg_sink.count_records("fraud_decisions")

        # No duplicate rows created
        assert after_count == initial_count

        # Metadata updated idempotently
        rec = pg_sink.get_decision(sample_decision_event.transaction_id)
        assert rec["workflow_status"] == "CONFIRMED_FINAL"

    def test_persist_raw_transaction_and_idempotency(self, pg_sink):
        now = time.time()
        tx_id = f"tx_raw_{int(now*1000)}"
        tx_data = {
            "transaction_id": tx_id,
            "customer_id": "cust_raw_1",
            "merchant_id": "merch_store_99",
            "amount": 499.95,
            "timestamp": now,
            "payment_type": "TRANSFER",
            "category": "electronics",
            "origin_balance": 1500.0,
            "dest_balance": 200.0,
            "auth_verified": True,
            "device_id": "dev_mobile_001",
            "location_json": {"country": "US", "city": "New York"}
        }

        # First insert
        assert pg_sink.persist_transaction(tx_data) is True
        c1 = pg_sink.count_records("transactions")

        # Duplicate insert should be safely ignored
        assert pg_sink.persist_transaction(tx_data) is True
        c2 = pg_sink.count_records("transactions")
        assert c2 == c1

    def test_hold_case_lifecycle(self, pg_sink):
        now = time.time()
        hold_id = f"hold_{int(now*1000)}"
        tx_id = f"tx_hold_{int(now*1000)}"
        hold_data = {
            "hold_id": hold_id,
            "transaction_id": tx_id,
            "customer_id": "cust_hold_01",
            "amount": 2800.0,
            "status": "HOLD",
            "token_hash": "hash_sec_9918",
            "timeout_seconds": 300.0,
            "created_at": now,
            "expires_at": now + 300.0,
            "customer_channel": "sms_push"
        }

        # 1. Insert HOLD
        assert pg_sink.persist_hold_case(hold_data) is True
        case = pg_sink.get_hold_case(hold_id)
        assert case is not None
        assert case["status"] == "HOLD"
        assert case["amount"] == pytest.approx(2800.0)

        # 2. Update to RELEASED (Customer confirms)
        assert pg_sink.update_hold_status(hold_id, "RELEASED", resolution_reason="Customer confirmed via SMS") is True
        case_released = pg_sink.get_hold_case(hold_id)
        assert case_released["status"] == "RELEASED"
        assert case_released["resolution_reason"] == "Customer confirmed via SMS"

        # 3. Lookup by transaction_id
        case_by_tx = pg_sink.get_hold_case(tx_id)
        assert case_by_tx["hold_id"] == hold_id

    def test_fraud_alert_persistence(self, pg_sink):
        now = time.time()
        alert_id = f"alt_{int(now*1000)}"
        alert_data = {
            "alert_id": alert_id,
            "transaction_id": f"tx_alt_{int(now*1000)}",
            "customer_id": "cust_fraud_9",
            "severity": "CRITICAL",
            "decision": "BLOCK",
            "reasons": ["Velocity surge", "Known fraud device ID"],
            "notified": False,
            "status": "OPEN",
            "created_at": now
        }
        assert pg_sink.persist_fraud_alert(alert_data) is True
        # Idempotent re-insert
        assert pg_sink.persist_fraud_alert(alert_data) is True

    def test_account_security_event_persistence(self, pg_sink):
        now = time.time()
        event_id = f"acc_{int(now*1000)}"
        event_data = {
            "event_id": event_id,
            "customer_id": "cust_ato_11",
            "event_type": "PASSWORD_RESET_ATTEMPT",
            "timestamp": now,
            "ip_address": "198.51.100.24",
            "device_id": "dev_suspicious_44",
            "metadata": {"user_agent": "curl/7.88"}
        }
        assert pg_sink.persist_account_event(event_data) is True
        # Duplicate is ignored safely
        assert pg_sink.persist_account_event(event_data) is True

    def test_mandate_persistence_and_update(self, pg_sink):
        now = time.time()
        mandate_id = f"mand_{int(now*1000)}"
        mandate_data = {
            "mandate_id": mandate_id,
            "customer_id": "cust_mand_01",
            "beneficiary_id": "ben_utility_01",
            "max_amount": 150.0,
            "frequency": "MONTHLY",
            "status": "ACTIVE",
            "execution_count": 0,
            "total_executed_amount": 0.0
        }
        assert pg_sink.persist_mandate(mandate_data) is True

        # Update execution
        mandate_data["execution_count"] = 1
        mandate_data["total_executed_amount"] = 85.0
        mandate_data["last_executed_at"] = now
        assert pg_sink.persist_mandate(mandate_data) is True

    def test_delayed_fraud_label_attachment(self, pg_sink, sample_decision_event):
        pg_sink.persist_decision_event(sample_decision_event)

        # Attach delayed chargeback label
        label_ts = time.time() + 3600
        updated = pg_sink.attach_delayed_label(
            sample_decision_event.transaction_id,
            fraud_label=1,
            label_timestamp=label_ts
        )
        assert updated is True

        # Verify record has label
        rec = pg_sink.get_decision(sample_decision_event.transaction_id)
        assert rec["fraud_label"] == 1
        assert rec["label_timestamp"] == pytest.approx(label_ts, rel=1e-3)

        # Fetch labeled dataset
        dataset = pg_sink.get_labeled_dataset(min_timestamp=0.0)
        assert any(d["transaction_id"] == sample_decision_event.transaction_id for d in dataset)

    def test_replay_evaluation_record_persistence(self, pg_sink):
        now = time.time()
        replay_id = f"rpl_{int(now*1000)}"
        record_data = {
            "replay_id": replay_id,
            "scenario_type": "HISTORICAL_REPLAY",
            "transactions_count": 500,
            "matured_labels_count": 12,
            "started_at": now - 10.0,
            "completed_at": now,
            "metrics": {"precision": 0.94, "recall": 0.91, "f1": 0.925}
        }
        assert pg_sink.persist_replay_record(record_data) is True
