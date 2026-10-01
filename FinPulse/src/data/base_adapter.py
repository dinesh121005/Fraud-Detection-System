"""Abstract Base Dataset Adapter for FinPulse."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

class BaseDatasetAdapter(ABC):
    """
    Standard interface for dataset-specific ingestion, profiling,
    leakage prevention, and mapping into the common FinPulse representation.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Load raw dataset records into a DataFrame."""
        pass

    @abstractmethod
    def clean_and_filter_leakage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce strict decision-time isolation:
        Remove post-transaction settlement balances, clearing feedback,
        or synthetic generator tokens.
        """
        pass

    @abstractmethod
    def transform_to_common_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataset-specific fields into the canonical FinPulse schema:
        [transaction_id, timestamp, amount, customer_id, merchant_id,
         category, payment_type, origin_balance, is_fraud]
        """
        pass

    def process(self) -> pd.DataFrame:
        """Execute full ingestion, leakage elimination, and canonical transformation pipeline."""
        df_raw = self.load_raw()
        df_clean = self.clean_and_filter_leakage(df_raw)
        df_common = self.transform_to_common_schema(df_clean)
        return df_common
