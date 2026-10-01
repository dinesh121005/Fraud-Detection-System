"""Chronological Temporal Splitter for Fraud Detection Validation."""
from typing import Tuple
import pandas as pd
import numpy as np

def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_col: str = "timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame strictly based on chronological order to eliminate
    temporal lookahead leakage.

    Splits:
    - Train Split (earliest 70% of time)
    - Validation Split (middle 15% of time)
    - Test Split (latest 15% of time)

    Parameters:
    - df: Input DataFrame
    - time_col: Column containing timestamp or continuous time step
    - train_ratio: Proportion of observations for training (default: 0.70)
    - val_ratio: Proportion of observations for validation (default: 0.15)
    - test_ratio: Proportion of observations for testing (default: 0.15)

    Returns:
    - (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Sort strictly by timestamp
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    # Integrity verification
    max_train_time = train_df[time_col].max()
    min_val_time = val_df[time_col].min()
    max_val_time = val_df[time_col].max()
    min_test_time = test_df[time_col].min()

    assert max_train_time <= min_val_time, f"Temporal leak: max_train ({max_train_time}) > min_val ({min_val_time})"
    assert max_val_time <= min_test_time, f"Temporal leak: max_val ({max_val_time}) > min_test ({min_test_time})"

    return train_df, val_df, test_df
