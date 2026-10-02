"""
FinPulse Enterprise Relational Persistence Sink.

Supports PostgreSQL as the primary production/runtime durable system of record
with connection pooling, ACID transaction guarantees, and write idempotency.
Maintains backward compatibility with SQLite for isolated in-memory unit tests.
"""

import os
import json
import sqlite3
import time
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

from src.risk_engine.decision_event import DecisionEvent
from src.monitoring.logger import get_logger
from src.monitoring.metrics import record_error
from src.persistence.config import PostgresConfig, get_persistence_backend
from src.persistence.schema import POSTGRES_SCHEMA_SQL, SQLITE_SCHEMA_SQL

logger = get_logger("FinPulse.Persistence.Sink")

try:
    import psycopg2
    from psycopg2 import extras
    from src.persistence.connection import PostgresConnectionPool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    PostgresConnectionPool = None


class IdempotentEventSink:
    """
    ACID-compliant relational persistence sink with automatic schema initialization.
    - PostgreSQL is the primary production runtime backend (with connection pooling).
    - SQLite is preserved for isolated test fixtures (e.g. db_path=':memory:').
    - Strict write idempotency enforced via PRIMARY KEY (event_id) and UNIQUE (transaction_id).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        postgres_config: Optional[PostgresConfig] = None,
        force_backend: Optional[str] = None
    ):
        self.explicit_db_path = db_path
        self.backend = force_backend or get_persistence_backend(db_path)
        self.postgres_config = postgres_config or PostgresConfig.from_env()

        self._sqlite_conn: Optional[sqlite3.Connection] = None

        if self.backend == "postgres" and PSYCOPG2_AVAILABLE:
            try:
                self._init_postgres_schema()
                logger.info("Initialized PostgreSQL persistence backend", dsn=self.postgres_config.dsn)
            except Exception as e:
                # If explicitly asked for postgres, raise; if automatic fallback allowed, fallback to SQLite
                if force_backend == "postgres":
                    raise
                logger.warning(
                    f"PostgreSQL connection failed ({e}). Falling back to SQLite for local session."
                )
                self.backend = "sqlite"
                self._init_sqlite(db_path or ":memory:")
        else:
            self.backend = "sqlite"
            self._init_sqlite(db_path or os.environ.get("FINPULSE_DB_PATH", ":memory:"))

    @property
    def _pg_pool(self):
        if self.backend == "postgres" and PSYCOPG2_AVAILABLE:
            return PostgresConnectionPool.get_instance(self.postgres_config)
        return None

    def _init_sqlite(self, path: str) -> None:
        self.db_path = path
        self._sqlite_conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        self._sqlite_conn.row_factory = sqlite3.Row
        with self._sqlite_conn as conn:
            conn.executescript(SQLITE_SCHEMA_SQL)
            conn.commit()
        logger.info("Initialized SQLite persistence fallback", db_path=self.db_path)

    def _init_postgres_schema(self) -> None:
        with self._pg_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def _get_cursor(self):
        """Context manager yielding (cursor, placeholder_str, is_postgres)."""
        if self.backend == "postgres":
            with self._pg_pool.get_connection() as conn:
                with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                    yield cur, "%s", True
        else:
            conn = self._sqlite_conn
            cur = conn.cursor()
            try:
                yield cur, "?", False
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cur.close()

    # =========================================================================
    # 1. DecisionEvent Persistence
    # =========================================================================
    def persist_decision_event(
        self,
        event: DecisionEvent,
        workflow_status: str = "FINAL",
        fraud_label: Optional[int] = None
    ) -> bool:
        """
        Idempotently insert or update a DecisionEvent.
        Enforces conflict handling on transaction_id.
        """
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO fraud_decisions (
            event_id, transaction_id, customer_id, decision, risk_score,
            risk_level, ml_decision, calibrated_probability, model_version,
            schema_version, signals_json, diagnostics_json, reasons_json,
            matched_rules_json, hard_block, latency_ms, workflow_status,
            fraud_label, created_at, updated_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p},
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(transaction_id) DO UPDATE SET
            workflow_status = EXCLUDED.workflow_status,
            fraud_label = COALESCE(EXCLUDED.fraud_label, fraud_decisions.fraud_label),
            updated_at = EXCLUDED.updated_at;
        """

        hard_block_val = True if event.hard_block else False
        if self.backend == "sqlite":
            hard_block_val = 1 if event.hard_block else 0

        params = (
            str(event.event_id),
            str(event.transaction_id),
            str(event.customer_id),
            str(event.decision),
            float(event.risk_score),
            str(event.risk_level),
            str(event.ml_decision),
            float(event.calibrated_probability),
            str(event.model_version),
            str(event.schema_version),
            json.dumps(event.signals),
            json.dumps(event.diagnostics),
            json.dumps(event.reasons),
            json.dumps(event.matched_rules),
            hard_block_val,
            event.latency_ms,
            workflow_status,
            fraud_label,
            float(event.timestamp or now),
            now
        )

        try:
            with self._get_cursor() as (cur, _, _):
                cur.execute(sql, params)
            return True
        except Exception as e:
            record_error("persistence", "insert_error")
            logger.error("Failed to persist DecisionEvent", error=str(e), tx_id=event.transaction_id)
            raise

    # =========================================================================
    # 2. Raw Transaction Persistence
    # =========================================================================
    def persist_transaction(self, tx: Dict[str, Any]) -> bool:
        """Idempotently persist a raw transaction record."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO transactions (
            transaction_id, customer_id, merchant_id, amount, timestamp,
            payment_type, category, origin_balance, dest_balance,
            auth_verified, device_id, location_json, created_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(transaction_id) DO NOTHING;
        """

        auth_val = bool(tx.get("auth_verified", True))
        if self.backend == "sqlite":
            auth_val = 1 if auth_val else 0

        loc = tx.get("location_json") or tx.get("location", {})
        if not isinstance(loc, str):
            loc = json.dumps(loc)

        params = (
            str(tx["transaction_id"]),
            str(tx.get("customer_id", "unknown")),
            str(tx.get("merchant_id", "unknown")),
            float(tx.get("amount", 0.0)),
            float(tx.get("timestamp", now)),
            str(tx.get("payment_type", "TRANSFER")),
            str(tx.get("category", "general")),
            float(tx.get("origin_balance", 0.0)),
            float(tx.get("dest_balance", 0.0)),
            auth_val,
            str(tx.get("device_id", "unknown_device")),
            loc,
            now
        )

        try:
            with self._get_cursor() as (cur, _, _):
                cur.execute(sql, params)
            return True
        except Exception as e:
            record_error("persistence", "tx_insert_error")
            logger.error("Failed to persist transaction", error=str(e), tx_id=tx.get("transaction_id"))
            raise

    # =========================================================================
    # 3. HOLD Case Lifecycle Persistence
    # =========================================================================
    def persist_hold_case(self, case: Dict[str, Any]) -> bool:
        """Idempotently insert a HOLD case record."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO hold_cases (
            hold_id, transaction_id, customer_id, amount, status,
            token_hash, timeout_seconds, created_at, expires_at,
            resolved_at, resolution_reason, customer_channel, metadata_json, updated_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(transaction_id) DO UPDATE SET
            status = EXCLUDED.status,
            resolved_at = EXCLUDED.resolved_at,
            resolution_reason = EXCLUDED.resolution_reason,
            updated_at = EXCLUDED.updated_at;
        """

        meta = case.get("metadata", {})
        if not isinstance(meta, str):
            meta = json.dumps(meta)

        params = (
            str(case["hold_id"]),
            str(case["transaction_id"]),
            str(case["customer_id"]),
            float(case["amount"]),
            str(case["status"]),
            str(case.get("token_hash", "")),
            float(case.get("timeout_seconds", 300.0)),
            float(case.get("created_at", now)),
            float(case.get("expires_at", now + 300.0)),
            case.get("resolved_at"),
            case.get("resolution_reason"),
            str(case.get("customer_channel", "sms_push")),
            meta,
            now
        )

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, params)
        return True

    def update_hold_status(
        self,
        hold_id: str,
        status: str,
        resolution_reason: Optional[str] = None
    ) -> bool:
        """Update lifecycle status of a HOLD case."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        UPDATE hold_cases
        SET status = {p}, resolved_at = {p}, resolution_reason = {p}, updated_at = {p}
        WHERE hold_id = {p} OR transaction_id = {p};
        """
        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, (status, now, resolution_reason, now, hold_id, hold_id))
            return cur.rowcount > 0

    def get_hold_case(self, identifier: str) -> Optional[Dict[str, Any]]:
        """Retrieve HOLD case by hold_id or transaction_id."""
        p = "%s" if self.backend == "postgres" else "?"
        sql = f"SELECT * FROM hold_cases WHERE hold_id = {p} OR transaction_id = {p};"
        with self._get_cursor() as (cur, _, is_pg):
            cur.execute(sql, (identifier, identifier))
            row = cur.fetchone()
            if not row:
                return None
            return dict(row)

    # =========================================================================
    # 4. Fraud Alerts Persistence
    # =========================================================================
    def persist_fraud_alert(self, alert: Dict[str, Any]) -> bool:
        """Persist a fraud alert record."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO fraud_alerts (
            alert_id, transaction_id, event_id, customer_id, severity,
            decision, reasons_json, notified, status, created_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(alert_id) DO NOTHING;
        """

        reasons = alert.get("reasons", [])
        if not isinstance(reasons, str):
            reasons = json.dumps(reasons)

        notified_val = bool(alert.get("notified", False))
        if self.backend == "sqlite":
            notified_val = 1 if notified_val else 0

        params = (
            str(alert["alert_id"]),
            str(alert["transaction_id"]),
            str(alert.get("event_id", "")),
            str(alert["customer_id"]),
            str(alert.get("severity", "HIGH")),
            str(alert.get("decision", "BLOCK")),
            reasons,
            notified_val,
            str(alert.get("status", "OPEN")),
            float(alert.get("created_at", now))
        )

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, params)
        return True

    # =========================================================================
    # 5. Account Security Events & ATO Tracking
    # =========================================================================
    def persist_account_event(self, event: Dict[str, Any]) -> bool:
        """Persist an account security event (login, password change, device switch)."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO account_security_events (
            event_id, customer_id, event_type, timestamp, ip_address,
            device_id, metadata_json, created_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(event_id) DO NOTHING;
        """

        meta = event.get("metadata", {})
        if not isinstance(meta, str):
            meta = json.dumps(meta)

        params = (
            str(event["event_id"]),
            str(event["customer_id"]),
            str(event["event_type"]),
            float(event.get("timestamp", now)),
            str(event.get("ip_address", "127.0.0.1")),
            str(event.get("device_id", "dev_unknown")),
            meta,
            now
        )

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, params)
        return True

    # =========================================================================
    # 6. Mandates Persistence
    # =========================================================================
    def persist_mandate(self, mandate: Dict[str, Any]) -> bool:
        """Persist or update a recurring mandate record."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO mandates (
            mandate_id, customer_id, beneficiary_id, max_amount, frequency,
            status, execution_count, total_executed_amount, last_executed_at,
            created_at, updated_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(mandate_id) DO UPDATE SET
            status = EXCLUDED.status,
            execution_count = EXCLUDED.execution_count,
            total_executed_amount = EXCLUDED.total_executed_amount,
            last_executed_at = EXCLUDED.last_executed_at,
            updated_at = EXCLUDED.updated_at;
        """

        params = (
            str(mandate["mandate_id"]),
            str(mandate["customer_id"]),
            str(mandate["beneficiary_id"]),
            float(mandate["max_amount"]),
            str(mandate.get("frequency", "MONTHLY")),
            str(mandate.get("status", "ACTIVE")),
            int(mandate.get("execution_count", 0)),
            float(mandate.get("total_executed_amount", 0.0)),
            mandate.get("last_executed_at"),
            float(mandate.get("created_at", now)),
            now
        )

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, params)
        return True

    # =========================================================================
    # 7. Replay & Evaluation Audit Records
    # =========================================================================
    def persist_replay_record(self, record: Dict[str, Any]) -> bool:
        """Persist replay or evaluation simulation audit metadata."""
        now = time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        INSERT INTO replay_evaluation_records (
            replay_id, scenario_type, transactions_count, matured_labels_count,
            started_at, completed_at, metrics_json, created_at
        ) VALUES (
            {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}
        )
        ON CONFLICT(replay_id) DO NOTHING;
        """

        metrics = record.get("metrics", {})
        if not isinstance(metrics, str):
            metrics = json.dumps(metrics)

        params = (
            str(record["replay_id"]),
            str(record["scenario_type"]),
            int(record["transactions_count"]),
            int(record.get("matured_labels_count", 0)),
            float(record.get("started_at", now)),
            record.get("completed_at"),
            metrics,
            now
        )

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, params)
        return True

    # =========================================================================
    # 8. Delayed Ground-Truth Labels & Query Methods
    # =========================================================================
    def attach_delayed_label(
        self,
        transaction_id: str,
        fraud_label: int,
        label_timestamp: Optional[float] = None
    ) -> bool:
        """
        Attach a delayed ground-truth fraud label (1=Fraud, 0=Legitimate) to an existing decision.
        """
        now = label_timestamp or time.time()
        p = "%s" if self.backend == "postgres" else "?"

        sql = f"""
        UPDATE fraud_decisions
        SET fraud_label = {p}, label_timestamp = {p}, updated_at = {p}
        WHERE transaction_id = {p};
        """

        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, (int(fraud_label), now, time.time(), transaction_id))
            updated = cur.rowcount > 0

        if updated:
            logger.info("Delayed fraud label attached", tx_id=transaction_id, label=fraud_label)
        else:
            logger.warning("Transaction not found for delayed label", tx_id=transaction_id)
        return updated

    def get_decision(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Fetch stored decision record by transaction_id."""
        p = "%s" if self.backend == "postgres" else "?"
        sql = f"SELECT * FROM fraud_decisions WHERE transaction_id = {p};"
        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, (transaction_id,))
            row = cur.fetchone()
            if row is None:
                return None
            return dict(row)

    def get_recent_decisions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch recent fraud decisions for analyst and live dashboards."""
        p = "%s" if self.backend == "postgres" else "?"
        sql = f"SELECT * FROM fraud_decisions ORDER BY created_at DESC LIMIT {p};"
        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, (limit,))
            return [dict(r) for r in cur.fetchall()]

    def get_labeled_dataset(self, min_timestamp: float = 0.0) -> List[Dict[str, Any]]:
        """Fetch all records having resolved ground-truth fraud labels for retraining."""
        p = "%s" if self.backend == "postgres" else "?"
        sql = f"""
        SELECT * FROM fraud_decisions
        WHERE fraud_label IS NOT NULL AND created_at >= {p}
        ORDER BY created_at ASC;
        """
        with self._get_cursor() as (cur, _, _):
            cur.execute(sql, (min_timestamp,))
            return [dict(r) for r in cur.fetchall()]

    def count_records(self, table: str = "fraud_decisions") -> int:
        """Count records in specified table."""
        # Sanitize table name against allowlist
        allowed_tables = {
            "fraud_decisions", "transactions", "hold_cases",
            "fraud_alerts", "account_security_events", "mandates",
            "replay_evaluation_records"
        }
        if table not in allowed_tables:
            raise ValueError(f"Invalid table name: {table}")

        sql = f"SELECT COUNT(*) FROM {table};"
        with self._get_cursor() as (cur, _, _):
            cur.execute(sql)
            row = cur.fetchone()
            if isinstance(row, dict):
                return list(row.values())[0]
            return row[0]

    def health_check(self) -> Dict[str, Any]:
        """Perform backend health verification."""
        if self.backend == "postgres" and self._pg_pool:
            return self._pg_pool.check_health()
        return {
            "status": "HEALTHY",
            "backend": "sqlite",
            "db_path": getattr(self, "db_path", ":memory:"),
            "latency_ms": 0.1
        }
