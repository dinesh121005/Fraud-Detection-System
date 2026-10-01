"""Comprehensive Test Suite for R3 Real-Time Feature Engine.

Validates:
1. Authoritative 32-Feature Schema & Order Parity against Frozen CatBoost Production Model.
2. Numerical Invariants (Strictly 32 features, 0 NaNs, 0 Infs, finite float32).
3. Rejection of Invalid Vectors (31 features, 33 features, NaN, Inf).
4. Transaction Normalization (Kafka TransactionEvent <-> CommonTransactionSchema <-> dict).
5. Deterministic Cold-Start Behavior (Brand-new customer, new device, new location, zero history).
6. Zero Look-Ahead Temporal Invariant (Feature vector for T excludes T and future events).
7. Offline vs. Online Feature Parity (max_absolute_difference < 1e-4 across transaction sequences).
8. End-to-End Kafka Consumer + Redis State + Feature Engine Integration Pipeline.
9. Real-Time Latency and Throughput Performance Benchmarks.
"""
import os
import sys
import json
import time
import math
import pytest
import numpy as np
import pandas as pd

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.features.schema import (
    CommonTransactionSchema,
    FinPulseFeatureVector,
    FEATURE_SPECIFICATIONS,
    validate_feature_vector,
    FeatureExtractionResult
)
from src.features.engine import FinPulseFeatureEngine, normalize_transaction_to_dict
from src.state.manager import RedisStateManager, CustomerHistoricalContext
from src.state.redis_client import InMemoryRedisMock
from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.consumer import TransactionConsumer
from src.streaming.config import StreamingConfig

@pytest.fixture
def mock_redis():
    """Provides a fresh isolated InMemoryRedisMock."""
    client = InMemoryRedisMock()
    yield client
    client.flushall()

@pytest.fixture
def state_manager(mock_redis):
    """Provides a RedisStateManager backed by mock_redis."""
    return RedisStateManager(redis_client=mock_redis)

@pytest.fixture
def feature_engine(state_manager):
    """Provides a FinPulseFeatureEngine wired to state_manager."""
    return FinPulseFeatureEngine(state_manager=state_manager)

# ==============================================================================
# R3.1 — Schema & Production CatBoost Feature Ordering Parity
# ==============================================================================

def test_production_feature_schema_exact_order():
    """
    Hard Requirement: Verify that FinPulseFeatureVector.FEATURE_NAMES and
    FEATURE_SPECIFICATIONS exactly match the frozen CatBoost production schema
    found in models/production/finpulse-v3/feature_schema.json index-for-index.
    """
    prod_schema_path = os.path.join(FINPULSE_DIR, "models", "production", "finpulse-v3", "feature_schema.json")
    assert os.path.exists(prod_schema_path), f"Production feature schema missing: {prod_schema_path}"

    with open(prod_schema_path, "r") as f:
        prod_schema = json.load(f)

    expected_feature_names = prod_schema["feature_names"]
    assert len(expected_feature_names) == 32
    assert len(FinPulseFeatureVector.FEATURE_NAMES) == 32
    assert len(FEATURE_SPECIFICATIONS) == 32

    # Verify exact naming and position
    for idx, (expected_name, actual_name, spec) in enumerate(
        zip(expected_feature_names, FinPulseFeatureVector.FEATURE_NAMES, FEATURE_SPECIFICATIONS)
    ):
        assert actual_name == expected_name, f"Feature index {idx} mismatch: expected {expected_name}, got {actual_name}"
        assert spec.name == expected_name, f"Spec index {idx} mismatch: expected {expected_name}, got {spec.name}"
        assert spec.index == idx, f"Spec index {idx} has internal index {spec.index}"

# ==============================================================================
# R3.2 — Numerical Validation: Strict 32 Dims, Zero NaNs, Zero Infs
# ==============================================================================

def test_validate_feature_vector_valid():
    """Verify validate_feature_vector accepts valid 32-element vectors."""
    valid_32 = [float(i) for i in range(32)]
    arr = validate_feature_vector(valid_32)
    assert arr.shape == (32,)
    assert arr.dtype == np.float32

def test_validate_feature_vector_rejects_dimension_mismatch():
    """Verify validation strictly rejects 31 or 33 features."""
    vec_31 = [1.0] * 31
    with pytest.raises(ValueError, match="Invalid feature dimension"):
        validate_feature_vector(vec_31)

    vec_33 = [1.0] * 33
    with pytest.raises(ValueError, match="Invalid feature dimension"):
        validate_feature_vector(vec_33)

def test_validate_feature_vector_rejects_nan():
    """Verify validation strictly rejects NaN values."""
    vec_nan = [1.0] * 32
    vec_nan[5] = float("nan")
    with pytest.raises(ValueError, match="contains NaN"):
        validate_feature_vector(vec_nan)

def test_validate_feature_vector_rejects_inf():
    """Verify validation strictly rejects Infinite values."""
    vec_inf = [1.0] * 32
    vec_inf[10] = float("inf")
    with pytest.raises(ValueError, match="contains Infinite"):
        validate_feature_vector(vec_inf)

    vec_neginf = [1.0] * 32
    vec_neginf[20] = float("-inf")
    with pytest.raises(ValueError, match="contains Infinite"):
        validate_feature_vector(vec_neginf)

# ==============================================================================
# R3.3 — Transaction Normalization & Adapter
# ==============================================================================

def test_normalize_transaction_to_dict():
    """Verify normalization of TransactionEvent and CommonTransactionSchema into uniform dictionary."""
    event = create_sample_transaction(
        transaction_id="tx_norm_1",
        amount=199.95,
        customer_id="cust_norm",
        timestamp=1700000000.0,
        latitude=37.77,
        longitude=-122.41,
        device_id="dev_mac"
    )

    d = normalize_transaction_to_dict(event)
    assert d["transaction_id"] == "tx_norm_1"
    assert d["amount"] == 199.95
    assert d["customer_id"] == "cust_norm"
    assert d["timestamp"] == 1700000000.0
    assert d["location"]["latitude"] == 37.77
    assert d["location"]["longitude"] == -122.41
    assert d["device_id"] == "dev_mac"

    # From CommonTransactionSchema
    common = event.to_common_schema()
    d_common = normalize_transaction_to_dict(common)
    assert d_common["transaction_id"] == "tx_norm_1"
    assert d_common["amount"] == 199.95

# ==============================================================================
# R3.4 — Deterministic Cold-Start Behavior
# ==============================================================================

def test_cold_start_new_customer(feature_engine):
    """
    Validate cold-start handling for a brand-new customer with 0 history.
    Must produce exactly 32 valid finite numbers, 0 NaNs, 0 Infs, with documented defaults.
    """
    new_event = create_sample_transaction(
        transaction_id="tx_cold_01",
        amount=250.0,
        customer_id="cust_brand_new",
        timestamp=1704067200.0,  # 2024-01-01 00:00:00 UTC (Monday)
        latitude=12.9716,
        longitude=77.5946,
        device_id="dev_new_phone"
    )

    result = feature_engine.extract_features(new_event)
    assert isinstance(result, FeatureExtractionResult)
    assert len(result.features) == 32
    assert result.transaction_id == "tx_cold_01"

    vec_dict = dict(zip(result.feature_names, result.features))

    # Invariants
    assert not np.isnan(result.features).any()
    assert not np.isinf(result.features).any()

    # Documented cold-start defaults:
    assert vec_dict["amount"] == 250.0
    assert vec_dict["tx_count_1m"] == 0.0
    assert vec_dict["tx_count_5m"] == 0.0
    assert vec_dict["tx_count_15m"] == 0.0
    assert vec_dict["tx_count_1h"] == 0.0
    assert vec_dict["amount_sum_5m"] == 0.0
    assert vec_dict["amount_sum_1h"] == 0.0
    # Customer baseline equals current amount on first tx
    assert vec_dict["user_avg_amount_30d"] == 250.0
    assert vec_dict["user_std_amount_30d"] == 0.0
    assert vec_dict["amount_zscore"] == 0.0
    assert vec_dict["user_category_frequency"] == 1.0
    assert vec_dict["user_hourly_tx_deviation"] == 0.0
    assert vec_dict["distance_from_prev_loc_km"] == 0.0
    assert vec_dict["speed_kmh_from_prev_tx"] == 0.0
    assert vec_dict["is_new_device"] == 0.0  # Cold start: first device is baseline, not anomaly
    assert vec_dict["is_new_location"] == 0.0  # Cold start: first location is baseline
    assert vec_dict["device_user_count_24h"] == 1.0

# ==============================================================================
# R3.5 — Zero Look-Ahead Invariant Proof
# ==============================================================================

def test_zero_lookahead_temporal_guarantee(feature_engine, state_manager):
    """
    Mandatory Invariant: Transaction T100 must NEVER see itself or future transactions.
    Scenario:
      T1 at t=1000s: $100.00
      T2 at t=1030s: $200.00
      T3 at t=1060s: $300.00

    When evaluating T2:
      Velocity tx_count_1m MUST be 1 (only T1), NOT 2.
      Behavior count MUST be 1, NOT 2.
    """
    cust = "cust_zero_lookahead"

    t1 = create_sample_transaction(transaction_id="tx_1", customer_id=cust, amount=100.0, timestamp=1000.0)
    t2 = create_sample_transaction(transaction_id="tx_2", customer_id=cust, amount=200.0, timestamp=1030.0)
    t3 = create_sample_transaction(transaction_id="tx_3", customer_id=cust, amount=300.0, timestamp=1060.0)

    # 1. Evaluate T1
    ctx1 = state_manager.get_historical_context(t1)
    res1 = feature_engine.extract_features(t1, context=ctx1)
    state_manager.record_transaction(t1)

    f1 = dict(zip(res1.feature_names, res1.features))
    assert f1["tx_count_1m"] == 0.0
    assert f1["amount_sum_5m"] == 0.0

    # 2. Evaluate T2 (Strictly prior state: only sees T1)
    ctx2 = state_manager.get_historical_context(t2)
    res2 = feature_engine.extract_features(t2, context=ctx2)
    state_manager.record_transaction(t2)

    f2 = dict(zip(res2.feature_names, res2.features))
    assert f2["tx_count_1m"] == 1.0  # Exactly T1
    assert f2["amount_sum_5m"] == 100.0  # Exactly T1
    assert f2["user_avg_amount_30d"] == 100.0  # Exactly T1

    # 3. Evaluate T3 (Strictly prior state: sees T1 and T2)
    ctx3 = state_manager.get_historical_context(t3)
    res3 = feature_engine.extract_features(t3, context=ctx3)
    state_manager.record_transaction(t3)

    f3 = dict(zip(res3.feature_names, res3.features))
    assert f3["tx_count_1m"] == 2.0  # T1 and T2
    assert f3["amount_sum_5m"] == 300.0  # 100 + 200
    assert f3["user_avg_amount_30d"] == 150.0  # (100 + 200) / 2

# ==============================================================================
# R3.6 — Offline vs Online Feature Parity (Most Critical R3 Validation)
# ==============================================================================

def test_offline_vs_online_exact_parity(mock_redis):
    """
    CRITICAL PROOF: The live online path (Kafka Transaction -> Redis State -> Feature Engine)
    must produce IDENTICAL 32-feature vectors to the offline batch path (DataFrame -> Feature Engine).

    Metrics:
      max_absolute_difference < 1e-4
      mean_absolute_difference < 1e-5
    """
    state_mgr = RedisStateManager(redis_client=mock_redis)
    feature_eng = FinPulseFeatureEngine(state_manager=state_mgr)

    base_time = 1710000000.0
    raw_tx_records = [
        {
            "transaction_id": "parity_tx_01",
            "timestamp": base_time,
            "amount": 120.0,
            "customer_id": "parity_cust",
            "merchant_id": "merch_alpha",
            "category": "grocery",
            "payment_type": "TRANSFER",
            "origin_balance": 5000.0,
            "latitude": 37.7749,
            "longitude": -122.4194,
            "device_id": "dev_phone_1",
            "auth_verified": True
        },
        {
            "transaction_id": "parity_tx_02",
            "timestamp": base_time + 45.0,  # 45s later (within 1m window)
            "amount": 250.0,
            "customer_id": "parity_cust",
            "merchant_id": "merch_beta",
            "category": "grocery",
            "payment_type": "TRANSFER",
            "origin_balance": 4880.0,
            "latitude": 37.7760,
            "longitude": -122.4180,
            "device_id": "dev_phone_1",
            "auth_verified": True
        },
        {
            "transaction_id": "parity_tx_03",
            "timestamp": base_time + 180.0,  # 3m later (within 5m window)
            "amount": 350.0,
            "customer_id": "parity_cust",
            "merchant_id": "merch_gamma",
            "category": "retail",
            "payment_type": "CREDIT_CARD",
            "origin_balance": 4630.0,
            "latitude": 37.7850,
            "longitude": -122.4100,
            "device_id": "dev_laptop_2",  # New device!
            "auth_verified": True
        },
        {
            "transaction_id": "parity_tx_04",
            "timestamp": base_time + 720.0,  # 12m later (within 15m window)
            "amount": 500.0,
            "customer_id": "parity_cust",
            "merchant_id": "merch_delta",
            "category": "electronics",
            "payment_type": "TRANSFER",
            "origin_balance": 4280.0,
            "latitude": 37.7950,
            "longitude": -122.4000,
            "device_id": "dev_laptop_2",
            "auth_verified": False
        }
    ]

    # --- 1. Compute Offline Feature Matrix ---
    df = pd.DataFrame(raw_tx_records)
    df["is_fraud"] = 0
    X_offline, _, offline_names = feature_eng.compute_offline_features(df, is_sorted=True)
    assert X_offline.shape == (4, 32)

    # --- 2. Compute Online Feature Vectors (Streaming Read -> Compute -> Write) ---
    X_online_list = []
    for rec in raw_tx_records:
        common_tx = CommonTransactionSchema(**rec)
        # Online flow
        feat_vec = feature_eng.compute_online_features(common_tx)
        ordered_vec = feat_vec.to_ordered_vector()
        assert len(ordered_vec) == 32
        X_online_list.append(ordered_vec)

    X_online = np.array(X_online_list, dtype=np.float32)
    assert X_online.shape == (4, 32)

    # --- 3. Rigorous Parity Comparison ---
    diff = np.abs(X_offline - X_online)
    max_abs_diff = float(np.max(diff))
    mean_abs_diff = float(np.mean(diff))

    print(f"\n[Parity Check] Max absolute diff: {max_abs_diff:.6e} | Mean absolute diff: {mean_abs_diff:.6e}")

    # Check each feature position
    for j, name in enumerate(offline_names):
        feature_diff = float(np.max(np.abs(X_offline[:, j] - X_online[:, j])))
        assert feature_diff < 1e-4, f"Feature '{name}' (index {j}) parity violation: diff={feature_diff:.6e}"

    assert max_abs_diff < 1e-4, f"Offline/Online parity violated: max diff={max_abs_diff}"
    assert mean_abs_diff < 1e-5, f"Offline/Online parity violated: mean diff={mean_abs_diff}"

# ==============================================================================
# R3.7 — Kafka Consumer -> Redis State -> Feature Engine Integration Pipeline
# ==============================================================================

def test_kafka_consumer_feature_engine_integration_pipeline(mock_redis):
    """
    Validate complete end-to-end integration:
      Kafka Transaction
            ↓
      TransactionConsumer
            ↓ [Step 1: Read Redis Historical Context]
      FinPulseFeatureEngine
            ↓ [Step 2: Generate 32 Features (Zero Look-Ahead)]
      Downstream Handler (Captured)
            ↓ [Step 3: Commit Current Transaction into Redis]
      Redis State Update
    """
    state_mgr = RedisStateManager(redis_client=mock_redis)
    feature_eng = FinPulseFeatureEngine(state_manager=state_mgr)

    dispatched_payloads = []

    def downstream_worker(event: TransactionEvent, context: CustomerHistoricalContext, feat_result: FeatureExtractionResult):
        # Capture exactly what downstream inference / risk scoring receives
        assert isinstance(feat_result, FeatureExtractionResult)
        assert len(feat_result.features) == 32
        dispatched_payloads.append({
            "tx_id": event.transaction_id,
            "features": feat_result.features,
            "feature_dict": feat_result.to_dict()
        })

    consumer = TransactionConsumer(
        config=StreamingConfig(),
        handler=downstream_worker,
        state_manager=state_mgr,
        feature_engine=feature_eng
    )

    t1 = create_sample_transaction(transaction_id="flow_tx_1", customer_id="cust_pipe", amount=150.0, timestamp=1000.0)
    t2 = create_sample_transaction(transaction_id="flow_tx_2", customer_id="cust_pipe", amount=250.0, timestamp=1020.0)

    # Process T1
    consumer.process_event_with_state(t1)
    # Process T2
    consumer.process_event_with_state(t2)

    assert len(dispatched_payloads) == 2

    # Check payload 1
    p1 = dispatched_payloads[0]
    assert p1["tx_id"] == "flow_tx_1"
    assert p1["features"][0] == 150.0  # amount
    assert p1["features"][11] == 0.0   # tx_count_1m (T1 sees 0 prior tx)

    # Check payload 2
    p2 = dispatched_payloads[1]
    assert p2["tx_id"] == "flow_tx_2"
    assert p2["features"][0] == 250.0  # amount
    assert p2["features"][11] == 1.0   # tx_count_1m (T2 sees T1)
    assert p2["features"][18] == 150.0 # user_avg_amount_30d (sees T1's $150)

# ==============================================================================
# R3.8 — Performance Latency and Throughput Benchmarks
# ==============================================================================

def test_feature_engine_performance_benchmarks(feature_engine, state_manager):
    """
    Measure actual latency and throughput for R3 Feature Engine:
    1. Feature computation latency (ms per transaction)
    2. Redis historical read latency (ms)
    3. Total transaction-to-feature latency (ms)
    4. End-to-end throughput (transactions/sec)
    """
    cust_id = "benchmark_cust"
    n_transactions = 200

    # Seed 10 historical transactions
    t_start = 1700000000.0
    for k in range(10):
        ev = create_sample_transaction(
            transaction_id=f"seed_tx_{k}",
            customer_id=cust_id,
            amount=50.0 + k * 10,
            timestamp=t_start + k * 60.0
        )
        state_manager.record_transaction(ev)

    current_t = t_start + 700.0
    benchmark_events = [
        create_sample_transaction(
            transaction_id=f"bench_tx_{i}",
            customer_id=cust_id,
            amount=100.0 + (i % 20),
            timestamp=current_t + i * 5.0
        )
        for i in range(n_transactions)
    ]

    redis_latencies = []
    comp_latencies = []
    total_latencies = []

    for ev in benchmark_events:
        t0 = time.perf_counter()

        # Step 1: Redis Read
        r0 = time.perf_counter()
        ctx = state_manager.get_historical_context(ev)
        r1 = time.perf_counter()
        redis_latencies.append((r1 - r0) * 1000.0)

        # Step 2: Feature Engine Computation
        c0 = time.perf_counter()
        res = feature_engine.extract_features(ev, context=ctx)
        c1 = time.perf_counter()
        comp_latencies.append((c1 - c0) * 1000.0)

        # Step 3: State Commit
        state_manager.record_transaction(ev)

        t1 = time.perf_counter()
        total_latencies.append((t1 - t0) * 1000.0)
        assert len(res.features) == 32

    avg_redis_ms = float(np.mean(redis_latencies))
    p95_redis_ms = float(np.percentile(redis_latencies, 95))
    avg_comp_ms = float(np.mean(comp_latencies))
    p95_comp_ms = float(np.percentile(comp_latencies, 95))
    avg_total_ms = float(np.mean(total_latencies))
    p95_total_ms = float(np.percentile(total_latencies, 95))
    throughput_tps = float(n_transactions / (sum(total_latencies) / 1000.0))

    print(f"\n=================== R3 PERFORMANCE BENCHMARK ===================")
    print(f"Transactions evaluated:  {n_transactions}")
    print(f"Redis Read Latency:      Mean = {avg_redis_ms:.3f} ms | P95 = {p95_redis_ms:.3f} ms")
    print(f"Feature Compute Latency: Mean = {avg_comp_ms:.3f} ms | P95 = {p95_comp_ms:.3f} ms")
    print(f"Total E2E State+Feature: Mean = {avg_total_ms:.3f} ms | P95 = {p95_total_ms:.3f} ms")
    print(f"Throughput:              {throughput_tps:.1f} tx/sec")
    print(f"Error Rate:              0.0%")
    print(f"=================================================================")

    # Hard performance assertions: real-time streaming requires sub-10ms per transaction
    assert avg_comp_ms < 5.0, f"Feature computation mean latency too high: {avg_comp_ms:.2f} ms"
    assert avg_total_ms < 10.0, f"Total latency mean too high: {avg_total_ms:.2f} ms"
