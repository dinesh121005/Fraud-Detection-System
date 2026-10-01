"""IEEE-CIS Fraud Dataset Adapter with Identity Table Merging and Device Profiling."""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_adapter import BaseDatasetAdapter

class IEEECISAdapter(BaseDatasetAdapter):
    """
    Adapter for IEEE-CIS transaction and identity datasets.
    Merges transaction and identity tables on TransactionID.
    Extracts device information, product categories, and temporal cycle signals.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.tx_path = config.get("train_transaction_path", "d:/Fraud-Detection-System/FinPulse/data/IEE-CIS/train_transaction.csv")
        self.id_path = config.get("train_identity_path", "d:/Fraud-Detection-System/FinPulse/data/IEE-CIS/train_identity.csv")

    def load_raw(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        if not os.path.exists(self.tx_path):
            raise FileNotFoundError(f"IEEE-CIS transaction dataset not found at {self.tx_path}")

        core_cols = [
            "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
            "ProductCD", "card1", "card2", "card4", "card6",
            "P_emaildomain", "R_emaildomain", "dist1"
        ]
        
        if sample_size:
            df_tx = pd.read_csv(self.tx_path, nrows=sample_size, usecols=lambda c: c in core_cols or c.startswith("C") or c.startswith("D"))
        else:
            df_tx = pd.read_csv(self.tx_path, usecols=lambda c: c in core_cols or c.startswith("C") or c.startswith("D"))

        if os.path.exists(self.id_path):
            id_cols = ["TransactionID", "DeviceType", "DeviceInfo"]
            df_id = pd.read_csv(self.id_path, usecols=id_cols)
            df_merged = pd.merge(df_tx, df_id, on="TransactionID", how="left")
        else:
            df_merged = df_tx
            df_merged["DeviceType"] = "unknown"
            df_merged["DeviceInfo"] = "unknown"

        return df_merged

    def clean_and_filter_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop null targets and impute core identifying columns."""
        df_clean = df.dropna(subset=["isFraud", "TransactionAmt", "TransactionDT"]).copy()
        df_clean["DeviceType"] = df_clean["DeviceType"].fillna("desktop")
        df_clean["DeviceInfo"] = df_clean["DeviceInfo"].fillna("unknown")
        df_clean["P_emaildomain"] = df_clean["P_emaildomain"].fillna("anonymous.com")
        df_clean["card4"] = df_clean["card4"].fillna("visa")
        df_clean["card6"] = df_clean["card6"].fillna("credit")
        return df_clean

    def transform_to_common_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert IEEE-CIS columns to canonical FinPulse format with explicit missing-field representation."""
        base_timestamp = 1512086400.0
        timestamps = base_timestamp + df["TransactionDT"].astype(float)
        
        hours = ((df["TransactionDT"] // 3600) % 24).astype(int)
        days = ((df["TransactionDT"] // 86400) % 7).astype(int)

        # Build cardholder ID proxy from card1 + card2 + P_emaildomain
        customer_ids = "ieee_cust_" + df["card1"].astype(str) + "_" + df["card2"].fillna(0).astype(int).astype(str)

        common_df = pd.DataFrame({
            "transaction_id": "ieee_" + df["TransactionID"].astype(str),
            "timestamp": timestamps,
            "amount": df["TransactionAmt"].astype(float),
            "customer_id": customer_ids,
            "merchant_id": "prod_" + df["ProductCD"].astype(str),
            "category": df["ProductCD"].astype(str),
            "payment_type": df["card6"].astype(str).str.upper(),
            "origin_balance": 10000.0, # Baseline card limit proxy
            "latitude": 0.0,
            "longitude": 0.0,
            "home_latitude": 0.0,
            "home_longitude": 0.0,
            "device_id": df["DeviceType"].astype(str) + "_" + df["DeviceInfo"].astype(str),
            "auth_verified": 1,
            "hour_of_day": hours,
            "day_of_week": days,
            "is_fraud": df["isFraud"].astype(int),
            "dataset_source": "ieee_cis"
        })

        return common_df

# Alias for backwards compatibility
IeeeCisAdapter = IEEECISAdapter
