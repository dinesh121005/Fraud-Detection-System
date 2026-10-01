"""Categorical encodings for transaction types and merchant categories."""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

PAYMENT_TYPE_MAP = {
    "TRANSFER": 0,
    "CASH_OUT": 1,
    "PAYMENT": 2,
    "DEBIT": 3,
    "CASH_IN": 4,
    "CREDIT_CARD": 5
}

class FastFrequencyEncoder:
    """Frequency encoder for medium-to-high cardinality categorical fields."""

    def __init__(self, default_val: float = 0.001):
        self.frequencies: Dict[str, float] = {}
        self.default_val = default_val

    def fit(self, series: pd.Series):
        counts = series.astype(str).value_counts(normalize=True)
        self.frequencies = counts.to_dict()
        return self

    def transform_single(self, category: str) -> float:
        return self.frequencies.get(str(category), self.default_val)

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.astype(str).map(self.frequencies).fillna(self.default_val)
