"""PaySim Mobile Money Dataset Adapter with Strict Leakage Elimination."""
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_adapter import BaseDatasetAdapter

class PaySimAdapter(BaseDatasetAdapter):
    """
    Adapter for PaySim mobile money financial fraud simulation dataset.
    Strictly removes post-transaction balance clearing columns:
    'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', and 'isFlaggedFraud'.
    Explicitly accounts for missing GPS and hardware device concepts.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.raw_path = config.get("raw_path", "d:/Fraud-Detection-System/FinPulse/data/paysim.csv")

    def load_raw(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        if not os.path.exists(self.raw_path):
            raise FileNotFoundError(f"PaySim raw dataset not found at {self.raw_path}")
        
        if sample_size:
            df = pd.read_csv(self.raw_path, nrows=sample_size)
        else:
            df = pd.read_csv(self.raw_path)
        return df

    def clean_and_filter_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce decision-time boundaries:
        - Drops newbalanceOrig: Criminal transfers drain full balance, revealing target.
        - Drops oldbalanceDest & newbalanceDest: Clearing balances unavailable at gateway auth time.
        - Drops isFlaggedFraud: Static simulation rule artifact.
        """
        leakage_cols = ["newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFlaggedFraud"]
        df_clean = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors="ignore")
        df_clean = df_clean.dropna(subset=["isFraud", "amount", "oldbalanceOrg", "step"])
        return df_clean

    def transform_to_common_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PaySim columns to canonical FinPulse format with explicit missing-field representation."""
        # 1 step = 1 hour. Set reference timestamp = 1704067200 (2024-01-01 00:00:00 UTC)
        base_timestamp = 1704067200.0
        timestamps = base_timestamp + (df["step"].astype(float) * 3600.0)

        common_df = pd.DataFrame({
            "transaction_id": "paysim_" + df.index.astype(str),
            "timestamp": timestamps,
            "amount": df["amount"].astype(float),
            "customer_id": df["nameOrig"].astype(str),
            "merchant_id": df["nameDest"].astype(str),
            "category": np.where(df["nameDest"].str.startswith("M"), "merchant", "p2p_transfer"),
            "payment_type": df["type"].astype(str).str.upper(),
            "origin_balance": df["oldbalanceOrg"].astype(float),
            # Explicitly represented missing concepts in mobile money simulator:
            "latitude": 0.0,
            "longitude": 0.0,
            "home_latitude": 0.0,
            "home_longitude": 0.0,
            "device_id": "unknown_device",
            "auth_verified": 1,
            "hour_of_day": (df["step"] % 24).astype(int),
            "day_of_week": ((df["step"] // 24) % 7).astype(int),
            "is_fraud": df["isFraud"].astype(int),
            "dataset_source": "paysim"
        })

        return common_df
