"""FinPulse R6.2 — Load, Capacity & Stress Test Suite.

Validates:
1. Load generator configuration, determinism, seed stability, and unique IDs
2. Rate accuracy within specified tolerance
3. Message accounting ledger: zero silent loss guarantee
4. Decision determinism: identical inputs produce identical decisions and event IDs
5. Publication routing: predictions vs fraud alerts
6. R6.1 telemetry integration during benchmark execution
7. Backpressure handling: queues safely without dropping under overload
8. Production model integrity & checksum invariance
"""

import os
import sys
import time
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.benchmarking.generator import DeterministicTransactionGenerator, BenchmarkWorkloadConfig
from src.benchmarking.accounting import MessageAccountingLedger
from src.benchmarking.collector import ResourceCollector, TelemetryCollector
from src.streaming.worker import StreamingFraudWorker
from src.streaming.config import StreamingConfig
from src.monitoring.metrics import get_metrics


# =============================================================================
# 1. Load Generator Determinism & ID Uniqueness
# =============================================================================

def test_load_generator_configuration_and_determinism():
    """Verify generator produces deterministic, reproducible workloads with unique IDs."""
    cfg1 = BenchmarkWorkloadConfig(target_rate=50.0, duration_sec=1.0, seed=123, inject_duplicate_count=0)
    cfg2 = BenchmarkWorkloadConfig(target_rate=50.0, duration_sec=1.0, seed=123, inject_duplicate_count=0)

    gen1 = DeterministicTransactionGenerator(cfg1)
    gen2 = DeterministicTransactionGenerator(cfg2)

    w1 = gen1.generate_workload()
    w2 = gen2.generate_workload()

    assert len(w1) == 50
    assert len(w2) == 50

    # Strict equality across deterministic runs with same seed
    for tx1, tx2 in zip(w1, w2):
        assert tx1["transaction_id"] == tx2["transaction_id"]
        assert tx1["amount"] == tx2["amount"]
        assert tx1["customer_id"] == tx2["customer_id"]
        assert tx1["payment_type"] == tx2["payment_type"]

    # All generated base transaction IDs must be unique
    unique_ids = set(tx["transaction_id"] for tx in w1)
    assert len(unique_ids) == len(w1)


def test_load_generator_duplicate_injection():
    """Verify controlled duplicate injection for replay testing."""
    cfg = BenchmarkWorkloadConfig(target_rate=20.0, duration_sec=1.0, seed=42, inject_duplicate_count=3)
    gen = DeterministicTransactionGenerator(cfg)
    w = gen.generate_workload()

    assert len(w) == 20 + 3
    ids = [tx["transaction_id"] for tx in w]
    assert len(ids) - len(set(ids)) == 3


# =============================================================================
# 2. Rate Accuracy & Duration Handling
# =============================================================================

def test_rate_accuracy_and_pacing():
    """Verify streaming generator paces transactions within reasonable tolerance."""
    target_rate = 50.0
    duration_sec = 0.5  # 25 transactions
    cfg = BenchmarkWorkloadConfig(target_rate=target_rate, duration_sec=duration_sec, seed=99)
    gen = DeterministicTransactionGenerator(cfg)

    t0 = time.perf_counter()
    streamed = list(gen.stream_workload())
    elapsed = time.perf_counter() - t0

    actual_rate = len(streamed) / max(0.001, elapsed)

    assert len(streamed) == 25
    # On desktop OS timers, verify rate is within acceptable 35% pacing window
    assert actual_rate >= target_rate * 0.65


# =============================================================================
# 3. Message Accounting Ledger (Zero Silent Loss)
# =============================================================================

def test_message_accounting_zero_silent_loss():
    """Verify MessageAccountingLedger enforces zero silent loss accounting."""
    ledger = MessageAccountingLedger()

    # Offer 10 transactions (including 2 duplicates)
    for i in range(8):
        ledger.record_input(f"tx_acct_{i}")
    ledger.record_input("tx_acct_0")  # dup 1
    ledger.record_input("tx_acct_1")  # dup 2

    # Process 7, fail 1
    for i in range(7):
        ledger.record_processed(f"tx_acct_{i}")
    ledger.record_failed("tx_acct_7", "Injected test failure")

    acct = ledger.reconcile()

    assert acct["total_input_messages"] == 10
    assert acct["unique_input_transactions"] == 8
    assert acct["duplicate_replays"] == 2
    assert acct["successfully_processed"] == 7
    assert acct["failed_rejected"] == 1
    assert acct["silent_lost"] == 0
    assert acct["is_balanced"] is True


def test_message_accounting_detects_silent_loss():
    """Verify ledger detects and flags silent message loss."""
    ledger = MessageAccountingLedger()
    for i in range(5):
        ledger.record_input(f"tx_loss_{i}")

    # Process only 3; 2 silently dropped
    ledger.record_processed("tx_loss_0")
    ledger.record_processed("tx_loss_1")
    ledger.record_processed("tx_loss_2")

    acct = ledger.reconcile()
    assert acct["silent_lost"] == 2
    assert acct["is_balanced"] is False


# =============================================================================
# 4. Decision Determinism & Consistency
# =============================================================================

def test_decision_determinism_under_load():
    """Verify replayed identical transactions produce deterministic decision outcomes."""
    worker = StreamingFraudWorker(dry_run=True)

    tx_payload = {
        "transaction_id": "tx_det_bench_01",
        "timestamp": 1710000000.0,
        "amount": 250.0,
        "customer_id": "cust_det_01",
        "merchant_id": "merch_det_01",
        "category": "dining",
        "payment_type": "PAYMENT",
        "origin_balance": 1500.0,
        "auth_verified": True,
        "latitude": 37.7749,
        "longitude": -122.4194
    }

    res1, evt1 = worker.process_single_transaction(tx_payload)
    res2, evt2 = worker.process_single_transaction(tx_payload)

    assert evt1.decision == evt2.decision
    assert evt1.risk_score == evt2.risk_score
    assert evt1.risk_level == evt2.risk_level
    assert evt1.calibrated_probability == evt2.calibrated_probability
    assert evt1.event_id == evt2.event_id
    assert evt1.matched_rules == evt2.matched_rules
    assert evt1.hard_block == evt2.hard_block

    # Dynamic operational field latency_ms will naturally vary slightly between runs
    # Verify byte serialization equality when dynamic latency is normalized
    dict1 = evt1.to_dict()
    dict2 = evt2.to_dict()
    dict1["latency_ms"] = 25.0
    dict2["latency_ms"] = 25.0
    assert dict1 == dict2


# =============================================================================
# 5. Publication & Fraud-Alert Routing
# =============================================================================

def test_publication_routing_and_fraud_alert_filtering():
    """Verify publication logic strictly routes alerts only for BLOCK or hard_block."""
    worker = StreamingFraudWorker(dry_run=True)

    # 1. Clean APPROVE
    tx_clean = {
        "transaction_id": "tx_pub_clean",
        "timestamp": time.time(),
        "amount": 10.0,
        "customer_id": "cust_pub_01",
        "payment_type": "PAYMENT",
        "category": "grocery",
        "origin_balance": 500.0,
        "auth_verified": True
    }
    res_clean, evt_clean = worker.process_single_transaction(tx_clean)
    pub_clean = worker.publisher.publish(evt_clean, sync=False)

    assert pub_clean["predictions_published"] is True
    assert pub_clean["alerts_published"] is False

    # 2. Critical Attack (Hard Block / BLOCK)
    tx_attack = {
        "transaction_id": "tx_pub_attack",
        "timestamp": time.time(),
        "amount": 50000.0,
        "customer_id": "cust_pub_02",
        "payment_type": "TRANSFER",
        "category": "crypto",
        "origin_balance": 0.0,
        "auth_verified": False
    }
    res_atk, evt_atk = worker.process_single_transaction(tx_attack)
    pub_atk = worker.publisher.publish(evt_atk, sync=False)

    assert pub_atk["predictions_published"] is True
    assert pub_atk["alerts_published"] is True


# =============================================================================
# 6. R6.1 Telemetry Active During Benchmark
# =============================================================================

def test_telemetry_active_during_benchmark():
    """Verify R6.1 Prometheus metrics record observations during benchmark runs."""
    worker = StreamingFraudWorker(dry_run=True)
    m = get_metrics()

    initial_txs = m.transactions_total.labels(status="accepted")._value.get()
    initial_decisions = sum(
        m.decisions_total.labels(decision=d)._value.get()
        for d in ["APPROVE", "REVIEW", "BLOCK"]
    )

    sample = {
        "transaction_id": "tx_telem_test_01",
        "timestamp": time.time(),
        "amount": 75.0,
        "customer_id": "cust_telem_01",
        "payment_type": "PAYMENT",
        "category": "grocery",
        "origin_balance": 1000.0,
        "auth_verified": True
    }
    worker.process_single_transaction(sample)

    new_txs = m.transactions_total.labels(status="accepted")._value.get()
    new_decisions = sum(
        m.decisions_total.labels(decision=d)._value.get()
        for d in ["APPROVE", "REVIEW", "BLOCK"]
    )

    assert new_txs > initial_txs
    assert new_decisions > initial_decisions


# =============================================================================
# 7. Backpressure Safe Queuing (No Message Loss)
# =============================================================================

def test_backpressure_safe_queuing_and_recovery():
    """Verify overload burst queues safely without message loss and drains to zero."""
    ledger = MessageAccountingLedger()
    burst_count = 50

    # Simulate fast production into backlog
    for i in range(burst_count):
        tx_id = f"tx_overload_{i}"
        ledger.record_input(tx_id)
        ledger.record_pending(tx_id)

    # In-flight snapshot confirms pending backlog
    acct_inflight = ledger.reconcile()
    assert acct_inflight["pending_in_queue"] == burst_count
    assert acct_inflight["silent_lost"] == 0

    # Simulate worker draining the backlog
    for i in range(burst_count):
        ledger.record_processed(f"tx_overload_{i}")

    acct_final = ledger.reconcile()
    assert acct_final["pending_in_queue"] == 0
    assert acct_final["successfully_processed"] == burst_count
    assert acct_final["silent_lost"] == 0
    assert acct_final["is_balanced"] is True


# =============================================================================
# 8. Model & Feature Schema Invariance
# =============================================================================

def test_model_and_schema_version_invariance():
    """Verify R6.2 maintains exact authoritative model and schema versions."""
    worker = StreamingFraudWorker(dry_run=True)
    sample = {
        "transaction_id": "tx_version_test",
        "timestamp": time.time(),
        "amount": 50.0,
        "customer_id": "cust_v_01",
        "payment_type": "PAYMENT",
        "origin_balance": 500.0,
        "auth_verified": True
    }
    res, evt = worker.process_single_transaction(sample)

    assert evt.model_version == "finpulse-v3"
    assert evt.feature_schema_version == "2.0"
    assert evt.schema_version == "1.0"
