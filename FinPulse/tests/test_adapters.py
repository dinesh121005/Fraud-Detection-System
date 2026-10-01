"""Tests for Dataset Adapters, Leakage Elimination, and Schema Compliance."""
import pytest
import os
import yaml
import pandas as pd
import numpy as np
import sys

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.data.paysim_adapter import PaySimAdapter
from src.data.sparkov_adapter import SparkovAdapter
from src.data.ieee_adapter import IeeeCisAdapter
from src.data.splitters import temporal_train_val_test_split
from src.features.pipeline import FinPulseFeaturePipeline
from src.features.schema import CommonTransactionSchema

@pytest.fixture
def config():
    config_path = os.path.join(FINPULSE_DIR, "configs", "datasets.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["datasets"]

def test_paysim_leakage_elimination(config):
    """GATE 2: PaySim must not expose post-transaction balances or simulation flags."""
    adapter = PaySimAdapter(config["paysim"])
    df_raw = adapter.load_raw(sample_size=100)
    assert "newbalanceOrig" in df_raw.columns, "Raw dataset must contain post-tx balance to test stripping"
    
    df_clean = adapter.clean_and_filter_leakage(df_raw)
    assert "newbalanceOrig" not in df_clean.columns
    assert "oldbalanceDest" not in df_clean.columns
    assert "newbalanceDest" not in df_clean.columns
    assert "isFlaggedFraud" not in df_clean.columns

    df_common = adapter.transform_to_common_schema(df_clean)
    assert "is_fraud" in df_common.columns
    assert "amount" in df_common.columns
    assert "origin_balance" in df_common.columns

def test_sparkov_sanitization(config):
    """GATE 2: Sparkov must strip generator token 'fraud_' from merchant names."""
    adapter = SparkovAdapter(config["sparkov"])
    df_raw = adapter.load_raw(split="train", sample_size=100)
    df_clean = adapter.clean_and_filter_leakage(df_raw)
    
    assert "trans_num" not in df_clean.columns
    assert not df_clean["merchant"].str.startswith("fraud_").any(), "All 'fraud_' prefixes must be sanitized"

    df_common = adapter.transform_to_common_schema(df_clean)
    assert "distance_from_home_km" in df_common.columns
    assert (df_common["distance_from_home_km"] >= 0).all()

def test_temporal_train_val_test_split():
    """GATE 6: Temporal split must have zero chronological overlap."""
    timestamps = np.arange(1000, 2000, 10)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "amount": np.random.uniform(10, 500, len(timestamps)),
        "is_fraud": np.random.choice([0, 1], len(timestamps), p=[0.9, 0.1])
    })

    train_df, val_df, test_df = temporal_train_val_test_split(df, time_col="timestamp", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()

def test_feature_pipeline_vector_dimension():
    """GATE 3: Feature vector must strictly match 32 dimensions."""
    pipeline = FinPulseFeaturePipeline()
    tx = {
        "transaction_id": "tx_test_01",
        "timestamp": 1704067200,
        "amount": 150.0,
        "customer_id": "cust_123",
        "merchant_id": "merch_456",
        "category": "grocery",
        "payment_type": "TRANSFER",
        "origin_balance": 1000.0,
        "location": {"latitude": 37.77, "longitude": -122.41},
        "device_id": "dev_test",
        "auth_verified": True
    }
    state = {
        "user_avg_amount_30d": 120.0,
        "user_std_amount_30d": 40.0,
        "tx_count_1m": 1,
        "tx_count_5m": 2,
        "tx_count_15m": 3,
        "tx_count_1h": 4
    }

    vec = pipeline.transform_transaction_dict(tx, state)
    ordered_vec = vec.to_ordered_vector()
    assert len(ordered_vec) == 32, f"Expected 32 features, got {len(ordered_vec)}"
