"""Sparkov Credit Card Dataset Adapter with Generator Sanitization."""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_adapter import BaseDatasetAdapter

class SparkovAdapter(BaseDatasetAdapter):
    """
    Adapter for Sparkov synthetic credit card fraud datasets.
    Sanitizes synthetic generator leaks (strips 'fraud_' merchant prefix).
    Maps cardholder GPS (home) and merchant POS GPS (transaction).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.train_path = config.get("train_raw_path", "d:/Fraud-Detection-System/FinPulse/data/Sparkov/fraudTrain.csv")
        self.test_path = config.get("test_raw_path", "d:/Fraud-Detection-System/FinPulse/data/Sparkov/fraudTest.csv")

    def load_raw(self, split: str = "train", sample_size: Optional[int] = None) -> pd.DataFrame:
        target_path = self.train_path if split == "train" else self.test_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Sparkov raw dataset not found at {target_path}")

        if sample_size:
            df = pd.read_csv(target_path, nrows=sample_size)
        else:
            df = pd.read_csv(target_path)
        return df

    def clean_and_filter_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce decision-time boundaries:
        - Drops trans_num (post-authorization system token).
        - Strips 'fraud_' prefix from merchant names to prevent generator-token leakage.
        """
        df_clean = df.copy()
        if "trans_num" in df_clean.columns:
            df_clean = df_clean.drop(columns=["trans_num"])

        # Sanitize merchant names
        if "merchant" in df_clean.columns:
            df_clean["merchant"] = df_clean["merchant"].astype(str).str.replace(r"^fraud_", "", regex=True)

        df_clean = df_clean.dropna(subset=["is_fraud", "amt", "unix_time"])
        return df_clean

    def transform_to_common_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Sparkov columns to canonical FinPulse format with explicit spatial mappings."""
        trans_dt = pd.to_datetime(df["trans_date_trans_time"])

        # Vectorized Haversine distance from customer home (lat, long) to merchant (merch_lat, merch_long)
        lat1 = np.radians(df["lat"].values.astype(float))
        lon1 = np.radians(df["long"].values.astype(float))
        lat2 = np.radians(df["merch_lat"].values.astype(float))
        lon2 = np.radians(df["merch_long"].values.astype(float))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
        dist_home_km = np.round(6371.0 * c, 2)

        common_df = pd.DataFrame({
            "transaction_id": "sparkov_" + df.index.astype(str),
            "timestamp": df["unix_time"].astype(float),
            "amount": df["amt"].astype(float),
            "customer_id": "cc_" + df["cc_num"].astype(str),
            "merchant_id": df["merchant"].astype(str),
            "category": df["category"].astype(str),
            "payment_type": "CREDIT_CARD",
            "origin_balance": 5000.0, # Baseline card limit proxy
            "latitude": df["merch_lat"].astype(float),
            "longitude": df["merch_long"].astype(float),
            "home_latitude": df["lat"].astype(float),
            "home_longitude": df["long"].astype(float),
            "distance_from_home_km": dist_home_km,
            "device_id": "pos_card_" + df["cc_num"].astype(str),
            "auth_verified": 1,
            "hour_of_day": trans_dt.dt.hour.astype(int),
            "day_of_week": trans_dt.dt.dayofweek.astype(int),
            "is_fraud": df["is_fraud"].astype(int),
            "dataset_source": "sparkov"
        })

        return common_df
