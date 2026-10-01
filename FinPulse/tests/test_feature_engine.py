"""Comprehensive Unit & Parity Tests for FinPulse 32-Feature Engine."""
import pytest
import numpy as np
import pandas as pd
import time
import os
import sys

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.features.schema import CommonTransactionSchema, FinPulseFeatureVector
from src.features.transformations import (
    compute_log_amount,
    compute_amount_to_balance_ratio,
    compute_amount_zscore,
    compute_cyclic_time,
    haversine_distance_km,
    compute_velocity_speed_kmh,
    compute_velocity_metrics,
    compute_behavioral_metrics,
    compute_spatial_and_device_metrics
)
from src.features.engine import FinPulseFeatureEngine, compute_features_from_history
from src.state.sliding_window import RedisSlidingWindowEngine
from src.state.redis_client import InMemoryRedisMock

def test_amount_and_ratio_transformations():
    """Verify log1p and amount-to-balance ratio calculations."""
    assert compute_log_amount(0.0) == 0.0
    assert abs(compute_log_amount(99.0) - np.log1p(99.0)) < 1e-6

    # Normal ratio
    ratio = compute_amount_to_balance_ratio(500.0, 1000.0)
    assert abs(ratio - 0.5) < 1e-4

    # Zero origin balance protection
    ratio_zero = compute_amount_to_balance_ratio(500.0, 0.0)
    assert ratio_zero > 0.0 and not np.isnan(ratio_zero) and not np.isinf(ratio_zero)

def test_temporal_cyclic_encoding():
    """Verify cyclic hour sin/cos continuous representation."""
    sin_0, cos_0 = compute_cyclic_time(0)
    assert abs(sin_0 - 0.0) < 1e-5
    assert abs(cos_0 - 1.0) < 1e-5

    sin_6, cos_6 = compute_cyclic_time(6)
    assert abs(sin_6 - 1.0) < 1e-5
    assert abs(cos_6 - 0.0) < 1e-5

    sin_12, cos_12 = compute_cyclic_time(12)
    assert abs(sin_12 - 0.0) < 1e-5
    assert abs(cos_12 - (-1.0)) < 1e-5

def test_velocity_windows_and_sums():
    """Verify rolling counts and amount sums across 1m, 5m, 15m, 1h windows."""
    t_now = 10000.0
    prior_events = [
        {"timestamp": t_now - 30.0, "amount": 100.0},   # inside 1m, 5m, 15m, 1h
        {"timestamp": t_now - 120.0, "amount": 200.0},  # inside 5m, 15m, 1h
        {"timestamp": t_now - 600.0, "amount": 300.0},  # inside 15m, 1h
        {"timestamp": t_now - 1800.0, "amount": 400.0}, # inside 1h
        {"timestamp": t_now - 7200.0, "amount": 500.0}  # older than 1h (excluded)
    ]

    vel = compute_velocity_metrics(prior_events, t_now)
    assert vel["tx_count_1m"] == 1
    assert vel["tx_count_5m"] == 2
    assert vel["tx_count_15m"] == 3
    assert vel["tx_count_1h"] == 4

    assert abs(vel["amount_sum_5m"] - 300.0) < 1e-4
    assert abs(vel["amount_sum_15m"] - 600.0) < 1e-4
    assert abs(vel["amount_sum_1h"] - 1000.0) < 1e-4

def test_behavioral_mean_std_and_zscore():
    """Verify 30-day mean, sample std, and z-score deviation."""
    t_now = 100000.0
    # Customer historical transactions with amounts: 100, 200, 300 -> mean = 200, std = 100
    prior_events = [
        {"timestamp": t_now - 5000.0, "amount": 100.0, "category": "grocery", "hour_of_day": 10},
        {"timestamp": t_now - 3000.0, "amount": 200.0, "category": "grocery", "hour_of_day": 12},
        {"timestamp": t_now - 1000.0, "amount": 300.0, "category": "dining", "hour_of_day": 14}
    ]

    # Current tx: amount = 400.0 (zscore = (400 - 200)/100 = 2.0)
    beh = compute_behavioral_metrics(prior_events, current_amount=400.0, current_category="grocery", current_hour=11, current_timestamp=t_now)
    assert abs(beh["user_avg_amount_30d"] - 200.0) < 1e-2
    assert abs(beh["user_std_amount_30d"] - 100.0) < 1e-2
    assert abs(beh["amount_zscore"] - 2.0) < 1e-2
    # Category frequency: 2 out of 3 prior were grocery
    assert abs(beh["user_category_frequency"] - (2.0 / 3.0)) < 1e-3

def test_first_transaction_cold_start():
    """Verify graceful handling of first transaction (no prior history)."""
    t_now = 50000.0
    prior_events = []

    vel = compute_velocity_metrics(prior_events, t_now)
    assert vel["tx_count_1m"] == 0
    assert vel["tx_count_5m"] == 0
    assert vel["tx_count_15m"] == 0
    assert vel["tx_count_1h"] == 0
    assert vel["amount_sum_5m"] == 0.0

    beh = compute_behavioral_metrics(prior_events, current_amount=150.0, current_category="retail", current_hour=10, current_timestamp=t_now)
    assert beh["user_avg_amount_30d"] == 150.0
    assert beh["user_std_amount_30d"] == 0.0
    assert beh["amount_zscore"] == 0.0
    assert beh["user_category_frequency"] == 1.0

    spatial = compute_spatial_and_device_metrics(
        prior_events=prior_events,
        current_lat=37.77,
        current_lon=-122.41,
        home_lat=37.77,
        home_lon=-122.41,
        current_device_id="dev_first",
        current_timestamp=t_now
    )
    assert spatial["distance_from_home_km"] == 0.0
    assert spatial["distance_from_prev_loc_km"] == 0.0
    assert spatial["speed_kmh_from_prev_tx"] == 0.0
    assert spatial["is_new_device"] == 0
    assert spatial["is_new_location"] == 0

def test_spatial_and_speed_calculation():
    """Verify Haversine distance, impossible travel speed, and new location flag."""
    t_now = 10000.0
    # San Francisco to San Jose (~70 km) in 10 minutes (600s) -> Speed ~ 420 km/h
    sf_lat, sf_lon = 37.7749, -122.4194
    sj_lat, sj_lon = 37.3382, -121.8863

    prior_events = [
        {"timestamp": t_now - 600.0, "latitude": sf_lat, "longitude": sf_lon, "device_id": "dev_01"}
    ]

    spatial = compute_spatial_and_device_metrics(
        prior_events=prior_events,
        current_lat=sj_lat,
        current_lon=sj_lon,
        home_lat=sf_lat,
        home_lon=sf_lon,
        current_device_id="dev_02", # New device!
        current_timestamp=t_now
    )

    assert spatial["distance_from_home_km"] > 60.0
    assert spatial["distance_from_prev_loc_km"] > 60.0
    assert spatial["speed_kmh_from_prev_tx"] > 350.0 # High velocity travel
    assert spatial["is_new_device"] == 1
    assert spatial["is_new_location"] == 1 # > 50km from previous location

def test_feature_vector_dimension_and_schema():
    """Verify that every feature vector contains all 32 features in exact order."""
    tx = {
        "transaction_id": "tx_schema_test",
        "timestamp": 1704067200.0,
        "amount": 250.0,
        "customer_id": "cust_test_1",
        "merchant_id": "merch_test_1",
        "category": "electronics",
        "payment_type": "TRANSFER",
        "origin_balance": 1000.0,
        "latitude": 40.71,
        "longitude": -74.00,
        "device_id": "dev_phone_1",
        "auth_verified": True
    }
    vec = compute_features_from_history(tx, prior_user_events=[])
    ordered = vec.to_ordered_vector()
    assert len(ordered) == 32
    assert len(FinPulseFeatureVector.FEATURE_NAMES) == 32

    # Check mapping
    f_dict = vec.to_dict()
    assert f_dict["amount"] == 250.0
    assert f_dict["is_weekend"] == int(vec.day_of_week >= 5)

def test_offline_online_feature_parity():
    """
    CRITICAL: Verify that offline batch processing and online Redis state
    produce IDENTICAL 32-feature vectors for the exact same transaction history.
    """
    mock_redis = InMemoryRedisMock()
    redis_engine = RedisSlidingWindowEngine(redis_client=mock_redis)
    feature_engine = FinPulseFeatureEngine(redis_engine=redis_engine)

    # 4 sequential transactions for customer 'user_parity_test'
    base_t = 1704067200.0
    tx_list = [
        {
            "transaction_id": "tx_p_01",
            "timestamp": base_t,
            "amount": 100.0,
            "customer_id": "user_parity_test",
            "merchant_id": "merch_1",
            "category": "grocery",
            "payment_type": "TRANSFER",
            "origin_balance": 5000.0,
            "latitude": 37.77,
            "longitude": -122.41,
            "device_id": "dev_apple_1",
            "auth_verified": True
        },
        {
            "transaction_id": "tx_p_02",
            "timestamp": base_t + 120.0, # 2 minutes later
            "amount": 200.0,
            "customer_id": "user_parity_test",
            "merchant_id": "merch_2",
            "category": "grocery",
            "payment_type": "TRANSFER",
            "origin_balance": 4900.0,
            "latitude": 37.78,
            "longitude": -122.40,
            "device_id": "dev_apple_1",
            "auth_verified": True
        },
        {
            "transaction_id": "tx_p_03",
            "timestamp": base_t + 240.0, # 4 minutes later
            "amount": 300.0,
            "customer_id": "user_parity_test",
            "merchant_id": "merch_3",
            "category": "retail",
            "payment_type": "TRANSFER",
            "origin_balance": 4700.0,
            "latitude": 37.79,
            "longitude": -122.39,
            "device_id": "dev_apple_2", # New device!
            "auth_verified": True
        },
        {
            "transaction_id": "tx_p_04",
            "timestamp": base_t + 600.0, # 10 minutes later
            "amount": 400.0,
            "customer_id": "user_parity_test",
            "merchant_id": "merch_4",
            "category": "dining",
            "payment_type": "TRANSFER",
            "origin_balance": 4400.0,
            "latitude": 37.80,
            "longitude": -122.38,
            "device_id": "dev_apple_2",
            "auth_verified": False
        }
    ]

    # 1. Compute Offline Matrix
    df_tx = pd.DataFrame(tx_list)
    df_tx["is_fraud"] = 0
    X_offline, _, _ = feature_engine.compute_offline_features(df_tx, is_sorted=True)

    # 2. Compute Online Vectors (One by one through Redis state)
    X_online = []
    for item in tx_list:
        schema_obj = CommonTransactionSchema(**item)
        vec_online = feature_engine.compute_online_features(schema_obj)
        X_online.append(vec_online.to_ordered_vector())

    X_online = np.array(X_online, dtype=np.float32)

    # Compare Offline vs Online
    diff = np.abs(X_offline - X_online)
    max_diff = np.max(diff)
    print(f"Maximum discrepancy between offline and online features: {max_diff}")
    assert max_diff < 1e-4, f"Offline/Online feature parity violated! Discrepancy: {max_diff}"
