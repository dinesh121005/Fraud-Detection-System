"""Comprehensive Test Suite for R2 Redis State Management Foundation.

Validates:
1. Redis Connection & Health Check (connect, health_check, close)
2. Customer Velocity (1m, 5m, 15m, 1h sliding windows via Sorted Sets)
3. Velocity Window Eviction (cleanup_expired)
4. 30-Day Behavioral State (count, sum, mean, sample std dev, cold start)
5. Geospatial Location State (first-tx handling, coordinate & timestamp updates)
6. Device Associations (bidirectional Customer <-> Device graph, shared devices, deduplication)
7. Customer Isolation (strict separation of state across customers)
8. Restart / Persistence Usability (state survives manager re-instantiation)
9. Critical Invariant (Read-Before-Write, historical_state_before_T != state_after_T, zero lookahead)
10. Kafka Consumer Integration (end-to-end Read -> Process -> Write ordering)
"""
import os
import sys
import time
import math
import pytest

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.state.manager import RedisStateManager, CustomerHistoricalContext
from src.state.redis_client import InMemoryRedisMock
from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.consumer import TransactionConsumer
from src.streaming.config import StreamingConfig

@pytest.fixture
def mock_redis():
    """Provides a fresh, isolated InMemoryRedisMock instance."""
    client = InMemoryRedisMock()
    yield client
    client.flushall()

@pytest.fixture
def state_manager(mock_redis):
    """Provides a RedisStateManager backed by an isolated in-memory mock."""
    return RedisStateManager(redis_client=mock_redis)

# ==============================================================================
# R2.1 — Redis Connection & Health Check
# ==============================================================================

def test_redis_connection_and_health_check(state_manager):
    """Validate Python -> Redis -> PING -> PONG connection lifecycle."""
    assert state_manager.connect() is True
    assert state_manager.health_check() is True
    # Verify close does not raise
    state_manager.close()

# ==============================================================================
# R2.2 — Customer Velocity Windows (1m, 5m, 15m, 1h)
# ==============================================================================

def test_velocity_windows_calculation(state_manager):
    """
    Validate sliding window transaction velocity calculation.
    Controlled timestamps:
      T_now = 10,000.0s
      Tx1: 30s ago   ($100.00) -> in 1m, 5m, 15m, 1h
      Tx2: 2m ago    ($200.00) -> in 5m, 15m, 1h
      Tx3: 8m ago    ($300.00) -> in 15m, 1h
      Tx4: 30m ago   ($400.00) -> in 1h
      Tx5: 90m ago   ($500.00) -> outside 1h
    """
    t_now = 10000.0
    cust = "cust_vel_001"

    state_manager.record_velocity(cust, tx_id="tx_1", amount=100.0, timestamp=t_now - 30.0)
    state_manager.record_velocity(cust, tx_id="tx_2", amount=200.0, timestamp=t_now - 120.0)
    state_manager.record_velocity(cust, tx_id="tx_3", amount=300.0, timestamp=t_now - 480.0)
    state_manager.record_velocity(cust, tx_id="tx_4", amount=400.0, timestamp=t_now - 1800.0)
    state_manager.record_velocity(cust, tx_id="tx_5", amount=500.0, timestamp=t_now - 5400.0)

    vel = state_manager.get_velocity(cust, before_timestamp=t_now)

    # 1m window (t_now - 60s to t_now)
    assert vel["tx_count_1m"] == 1

    # 5m window (t_now - 300s to t_now)
    assert vel["tx_count_5m"] == 2
    assert vel["amount_sum_5m"] == 300.0  # 100 + 200

    # 15m window (t_now - 900s to t_now)
    assert vel["tx_count_15m"] == 3
    assert vel["amount_sum_15m"] == 600.0  # 100 + 200 + 300

    # 1h window (t_now - 3600s to t_now)
    assert vel["tx_count_1h"] == 4
    assert vel["amount_sum_1h"] == 1000.0  # 100 + 200 + 300 + 400

def test_velocity_cleanup_expired(state_manager, mock_redis):
    """Validate cleanup_expired trims records older than 1 hour."""
    t_now = 10000.0
    cust = "cust_vel_cleanup"

    state_manager.record_velocity(cust, "tx_recent", amount=50.0, timestamp=t_now - 100.0)
    state_manager.record_velocity(cust, "tx_old", amount=500.0, timestamp=t_now - 5000.0)

    # Before cleanup: 2 records in Sorted Set
    key = state_manager.key_velocity(cust)
    assert mock_redis.zcard(key) == 2

    # Run cleanup
    evicted = state_manager.cleanup_expired(cust, current_timestamp=t_now, lookback_seconds=3600.0)
    assert evicted == 1
    assert mock_redis.zcard(key) == 1

    # Only recent record remains
    vel = state_manager.get_velocity(cust, before_timestamp=t_now)
    assert vel["tx_count_1h"] == 1
    assert vel["amount_sum_1h"] == 50.0

# ==============================================================================
# R2.3 — 30-Day Customer Behavioral State
# ==============================================================================

def test_customer_behavior_cold_start_empty_state(state_manager):
    """Validate empty-state behavior when no history exists for customer."""
    beh = state_manager.get_customer_behavior("cust_unknown", before_timestamp=1000.0)
    assert beh["count"] == 0
    assert beh["sum"] == 0.0
    assert beh["mean"] == 0.0
    assert beh["std"] == 0.0
    assert beh["last_timestamp"] is None

def test_customer_behavior_statistics_calculation(state_manager):
    """
    Validate count, sum, mean, and sample standard deviation (N-1).
    Amounts: 100.0, 200.0, 300.0
      N = 3
      Sum = 600.0
      Mean = 200.0
      Std = sqrt(((100-200)^2 + (200-200)^2 + (300-200)^2) / 2) = sqrt(20000/2) = 100.0
    """
    cust = "cust_beh_001"
    t_base = 2000000.0

    state_manager.update_customer_behavior(cust, amount=100.0, timestamp=t_base + 10.0, tx_id="tx_1")
    state_manager.update_customer_behavior(cust, amount=200.0, timestamp=t_base + 20.0, tx_id="tx_2")
    stats = state_manager.update_customer_behavior(cust, amount=300.0, timestamp=t_base + 30.0, tx_id="tx_3")

    assert stats["count"] == 3
    assert stats["sum"] == 600.0
    assert stats["mean"] == 200.0
    assert math.isclose(stats["std"], 100.0, abs_tol=1e-4)
    assert stats["last_timestamp"] == t_base + 30.0

    # Retrieve strictly before t_base + 35.0
    queried = state_manager.get_customer_behavior(cust, before_timestamp=t_base + 35.0)
    assert queried["count"] == 3
    assert queried["sum"] == 600.0
    assert queried["mean"] == 200.0
    assert math.isclose(queried["std"], 100.0, abs_tol=1e-4)

def test_customer_behavior_30_day_eviction(state_manager):
    """Validate transactions older than 30 days are excluded from behavioral stats."""
    cust = "cust_beh_30d"
    t_now = 3000000.0
    day_in_sec = 86400.0

    # Transaction 35 days ago (outside 30-day window)
    state_manager.update_customer_behavior(cust, amount=1000.0, timestamp=t_now - (35 * day_in_sec), tx_id="tx_old")
    # Transaction 10 days ago (inside window)
    state_manager.update_customer_behavior(cust, amount=200.0, timestamp=t_now - (10 * day_in_sec), tx_id="tx_recent")

    queried = state_manager.get_customer_behavior(cust, before_timestamp=t_now)
    assert queried["count"] == 1
    assert queried["sum"] == 200.0
    assert queried["mean"] == 200.0
    assert queried["std"] == 0.0

# ==============================================================================
# R2.4 — Location and Device State
# ==============================================================================

def test_location_state_tracking(state_manager):
    """Validate tracking of previous latitude, longitude, and timestamp."""
    cust = "cust_loc_001"

    # 1. First transaction: no previous location exists
    assert state_manager.get_previous_location(cust) is None

    # 2. First location recorded (e.g. Bangalore)
    state_manager.update_location(cust, latitude=12.9716, longitude=77.5946, timestamp=1000.0)

    loc1 = state_manager.get_previous_location(cust)
    assert loc1 is not None
    assert math.isclose(loc1["latitude"], 12.9716, abs_tol=1e-4)
    assert math.isclose(loc1["longitude"], 77.5946, abs_tol=1e-4)
    assert loc1["timestamp"] == 1000.0

    # 3. Location update (e.g. Chennai)
    state_manager.update_location(cust, latitude=13.0827, longitude=80.2707, timestamp=2000.0)

    loc2 = state_manager.get_previous_location(cust)
    assert loc2 is not None
    assert math.isclose(loc2["latitude"], 13.0827, abs_tol=1e-4)
    assert math.isclose(loc2["longitude"], 80.2707, abs_tol=1e-4)
    assert loc2["timestamp"] == 2000.0

def test_device_associations_and_shared_device(state_manager):
    """
    Validate bidirectional associations:
      Customer -> Devices
      Device -> Customers
    Including multiple devices per customer, shared devices, and deduplication.
    """
    c1 = "C001"
    c2 = "C002"
    d1 = "D001"
    d2 = "D002"

    # C1 uses D1 and D2
    state_manager.add_customer_device(c1, d1)
    state_manager.add_device_customer(d1, c1)
    state_manager.add_customer_device(c1, d2)
    state_manager.add_device_customer(d2, c1)

    # C2 also uses D2 (shared device)
    state_manager.add_customer_device(c2, d2)
    state_manager.add_device_customer(d2, c2)

    # Re-adding existing device must be a deduplicated no-op
    state_manager.add_customer_device(c1, d1)

    # Verify C1 devices
    assert state_manager.get_customer_devices(c1) == ["D001", "D002"]

    # Verify C2 devices
    assert state_manager.get_customer_devices(c2) == ["D002"]

    # Verify D1 customers
    assert state_manager.get_device_customers(d1) == ["C001"]

    # Verify D2 customers (shared device detected!)
    assert state_manager.get_device_customers(d2) == ["C001", "C002"]

# ==============================================================================
# R2.5 — Customer State Isolation
# ==============================================================================

def test_customer_state_isolation(state_manager):
    """Verify Customer A cannot read or pollute Customer B's state."""
    ca = "cust_alpha"
    cb = "cust_beta"
    t_now = 5000.0

    state_manager.record_velocity(ca, "tx_a1", amount=1500.0, timestamp=t_now - 10.0)
    state_manager.update_customer_behavior(ca, amount=1500.0, timestamp=t_now - 10.0)
    state_manager.update_location(ca, latitude=40.7128, longitude=-74.0060, timestamp=t_now - 10.0)
    state_manager.add_customer_device(ca, "dev_a")

    # Customer B must have completely empty state
    vel_b = state_manager.get_velocity(cb, before_timestamp=t_now)
    assert vel_b["tx_count_1h"] == 0
    assert vel_b["amount_sum_1h"] == 0.0

    beh_b = state_manager.get_customer_behavior(cb, before_timestamp=t_now)
    assert beh_b["count"] == 0
    assert beh_b["sum"] == 0.0

    assert state_manager.get_previous_location(cb) is None
    assert state_manager.get_customer_devices(cb) == []

# ==============================================================================
# R2.6 — State Usability Across Re-instantiation (Restart Safety)
# ==============================================================================

def test_state_reinstantiation_survival(mock_redis):
    """Verify Redis state remains usable if manager/consumer is re-instantiated."""
    mgr1 = RedisStateManager(redis_client=mock_redis)
    mgr1.record_velocity("cust_restart", "tx_01", amount=250.0, timestamp=1000.0)
    mgr1.update_location("cust_restart", latitude=37.7749, longitude=-122.4194, timestamp=1000.0)

    # Re-instantiate pointing to the same backing Redis client
    mgr2 = RedisStateManager(redis_client=mock_redis)
    vel = mgr2.get_velocity("cust_restart", before_timestamp=1050.0)
    assert vel["tx_count_1h"] == 1
    assert vel["amount_sum_1h"] == 250.0

    loc = mgr2.get_previous_location("cust_restart")
    assert loc is not None
    assert math.isclose(loc["latitude"], 37.7749, abs_tol=1e-4)

# ==============================================================================
# R2.7 — Critical Invariant: Read-Before-Write Zero-Lookahead Proof
# ==============================================================================

def test_critical_invariant_historical_state_before_T_differs_from_after_T(state_manager):
    """
    Mandatory Architectural Requirement (Section 10):
    Prove:
      historical_state_before_T != state_after_T

    Scenario:
      Existing:
        10:00:01 (t=1001.0) -> $500
        10:00:20 (t=1020.0) -> $700
      Incoming:
        10:00:35 (t=1035.0) -> $900

    Before writing T_incoming:
      tx_count_1m = 2
      amount_sum_5m = $1,200.00
    After writing T_incoming:
      tx_count_1m = 3
      amount_sum_5m = $2,100.00

    The context retrieved for T_incoming must NEVER expose $2,100 or count=3.
    """
    cust = "cust_invariant_001"

    # Pre-existing transactions
    tx1 = create_sample_transaction(transaction_id="tx_01", customer_id=cust, amount=500.0, timestamp=1001.0)
    tx2 = create_sample_transaction(transaction_id="tx_02", customer_id=cust, amount=700.0, timestamp=1020.0)
    state_manager.record_transaction(tx1)
    state_manager.record_transaction(tx2)

    # Incoming transaction
    t_incoming = create_sample_transaction(transaction_id="tx_03", customer_id=cust, amount=900.0, timestamp=1035.0)

    # Step 1: READ historical state strictly BEFORE t_incoming is recorded
    context_before = state_manager.get_historical_context(t_incoming)

    # Verify zero-lookahead: tx_03 is NOT in context_before
    assert context_before.velocity["tx_count_1m"] == 2
    assert context_before.velocity["amount_sum_5m"] == 1200.0
    assert context_before.behavior["count"] == 2
    assert context_before.behavior["sum"] == 1200.0
    assert context_before.behavior["mean"] == 600.0

    # Step 2: WRITE t_incoming into Redis state
    state_manager.record_transaction(t_incoming)

    # Step 3: READ state AFTER t_incoming has been recorded (at future time t=1036.0)
    state_after = state_manager.get_velocity(cust, before_timestamp=1036.0)
    behavior_after = state_manager.get_customer_behavior(cust, before_timestamp=1036.0)

    assert state_after["tx_count_1m"] == 3
    assert state_after["amount_sum_5m"] == 2100.0
    assert behavior_after["count"] == 3
    assert behavior_after["sum"] == 2100.0

    # Formal Proof: historical_state_before_T != state_after_T
    assert context_before.velocity["tx_count_1m"] != state_after["tx_count_1m"]
    assert context_before.velocity["amount_sum_5m"] != state_after["amount_sum_5m"]
    assert context_before.behavior["sum"] != behavior_after["sum"]

# ==============================================================================
# R2.8 — Kafka Consumer -> Redis State Integration Pipeline
# ==============================================================================

def test_consumer_redis_pipeline_read_process_write_order(state_manager):
    """
    Validate TransactionConsumer + RedisStateManager pipeline:
      Kafka
        |
      TransactionConsumer
        |
      READ Redis historical state
        |
      process / expose historical context (Handler)
        |
      WRITE current transaction into Redis
    """
    received_contexts = []

    def recording_handler(event: TransactionEvent, context: CustomerHistoricalContext):
        # Capture context exactly as seen by the handler during transaction processing
        received_contexts.append((event.transaction_id, context.to_dict()))

    consumer = TransactionConsumer(
        config=StreamingConfig(),
        handler=recording_handler,
        state_manager=state_manager
    )

    t1 = create_sample_transaction(transaction_id="tx_flow_1", customer_id="cust_flow", amount=100.0, timestamp=1000.0, latitude=12.0, longitude=77.0)
    t2 = create_sample_transaction(transaction_id="tx_flow_2", customer_id="cust_flow", amount=200.0, timestamp=1030.0, latitude=13.0, longitude=78.0)
    t3 = create_sample_transaction(transaction_id="tx_flow_3", customer_id="cust_flow", amount=300.0, timestamp=1060.0, latitude=14.0, longitude=79.0)

    # Process T1
    consumer.process_event_with_state(t1)
    # Process T2
    consumer.process_event_with_state(t2)
    # Process T3
    consumer.process_event_with_state(t3)

    assert len(received_contexts) == 3

    # Verification of T1: Saw 0 previous transactions and None previous location
    tx1_id, ctx1 = received_contexts[0]
    assert tx1_id == "tx_flow_1"
    assert ctx1["velocity"]["tx_count_1h"] == 0
    assert ctx1["behavior"]["count"] == 0
    assert ctx1["previous_location"] is None

    # Verification of T2: Saw ONLY T1 (count=1, amount=100.0, location of T1)
    tx2_id, ctx2 = received_contexts[1]
    assert tx2_id == "tx_flow_2"
    assert ctx2["velocity"]["tx_count_1h"] == 1
    assert ctx2["velocity"]["amount_sum_1h"] == 100.0
    assert ctx2["behavior"]["count"] == 1
    assert ctx2["behavior"]["mean"] == 100.0
    assert ctx2["previous_location"] == {"latitude": 12.0, "longitude": 77.0, "timestamp": 1000.0}

    # Verification of T3: Saw ONLY T1 + T2 (count=2, amount=300.0, location of T2)
    tx3_id, ctx3 = received_contexts[2]
    assert tx3_id == "tx_flow_3"
    assert ctx3["velocity"]["tx_count_1h"] == 2
    assert ctx3["velocity"]["amount_sum_1h"] == 300.0
    assert ctx3["behavior"]["count"] == 2
    assert ctx3["behavior"]["mean"] == 150.0
    assert ctx3["previous_location"] == {"latitude": 13.0, "longitude": 78.0, "timestamp": 1030.0}
