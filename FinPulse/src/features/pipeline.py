"""Feature Engineering Pipeline for FinPulse with Real Feature Engine Delegation."""
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional
from .schema import CommonTransactionSchema, FinPulseFeatureVector
from .engine import FinPulseFeatureEngine, compute_features_from_history
from .encoders import FastFrequencyEncoder

class FinPulseFeaturePipeline:
    """Unified pipeline delegating to FinPulseFeatureEngine for zero placeholder calculation."""

    def __init__(self, feature_engine: Optional[FinPulseFeatureEngine] = None):
        self.engine = feature_engine or FinPulseFeatureEngine()
        self.category_encoder = self.engine.category_encoder
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit encoders on training dataset."""
        self.engine.fit(df)
        self.is_fitted = True
        return self

    def transform_transaction_dict(
        self,
        tx: Dict[str, Any],
        historical_state: Optional[Dict[str, Any]] = None
    ) -> FinPulseFeatureVector:
        """
        Transform a single transaction dict into the 32-feature FinPulse vector.
        If historical_state is provided, reconstructs prior event context.
        Otherwise, uses online Redis engine state.
        """
        prior_events = []
        prior_dev_events = []
        if historical_state:
            # If explicit historical events list passed
            if "prior_events" in historical_state:
                prior_events = historical_state["prior_events"]
            if "prior_device_events" in historical_state:
                prior_dev_events = historical_state["prior_device_events"]

        feat_vec = compute_features_from_history(
            tx_dict=tx,
            prior_user_events=prior_events,
            prior_device_events=prior_dev_events,
            category_encoder=self.category_encoder
        )

        if historical_state:
            # Directly override any velocity / behavioral / spatial metrics already calculated in historical_state
            for k, v in historical_state.items():
                if hasattr(feat_vec, k) and k not in ("prior_events", "prior_device_events"):
                    setattr(feat_vec, k, v)

        return feat_vec
