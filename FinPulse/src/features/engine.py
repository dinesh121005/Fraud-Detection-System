"""Unified FinPulse 32-Feature Engine with Strict Offline/Online Parity."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Dict, Any, List, Tuple, Optional, Union
from collections import defaultdict, deque

from .schema import (
    CommonTransactionSchema,
    FinPulseFeatureVector,
    validate_feature_vector,
    FeatureExtractionResult
)
from .transformations import (
    compute_log_amount,
    compute_amount_to_balance_ratio,
    compute_cyclic_time,
    compute_velocity_metrics,
    compute_behavioral_metrics,
    compute_spatial_and_device_metrics
)
from .encoders import PAYMENT_TYPE_MAP, FastFrequencyEncoder
from src.state.sliding_window import RedisSlidingWindowEngine
from src.state.manager import RedisStateManager, CustomerHistoricalContext

if TYPE_CHECKING:
    from src.streaming.schema import TransactionEvent

def normalize_transaction_to_dict(
    tx: Union[CommonTransactionSchema, TransactionEvent, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    R3.1 Adapter: Normalize Kafka TransactionEvent, CommonTransactionSchema, or dict
    into canonical dictionary format for the authoritative feature engine.
    """
    if hasattr(tx, "to_common_schema"):  # TransactionEvent
        common = tx.to_common_schema()
        d = common.model_dump()
    elif isinstance(tx, CommonTransactionSchema):
        d = tx.model_dump()
    elif isinstance(tx, dict):
        d = tx.copy()
    elif hasattr(tx, "model_dump"):
        d = tx.model_dump()
    elif hasattr(tx, "__dict__"):
        d = tx.__dict__.copy()
    else:
        raise TypeError(f"Unsupported transaction input type: {type(tx)}")

    # Ensure location dictionary
    loc = d.get("location")
    if not isinstance(loc, dict):
        lat = float(d.get("latitude", getattr(loc, "latitude", 0.0)))
        lon = float(d.get("longitude", getattr(loc, "longitude", 0.0)))
        d["location"] = {"latitude": lat, "longitude": lon}
    else:
        d["location"] = {
            "latitude": float(loc.get("latitude", d.get("latitude", 0.0))),
            "longitude": float(loc.get("longitude", d.get("longitude", 0.0)))
        }

    # Ensure home_location dictionary
    hloc = d.get("home_location")
    if not isinstance(hloc, dict):
        hlat = d.get("home_latitude", getattr(hloc, "latitude", None))
        hlon = d.get("home_longitude", getattr(hloc, "longitude", None))
        if hlat is not None and hlon is not None:
            d["home_location"] = {"latitude": float(hlat), "longitude": float(hlon)}
        else:
            d["home_location"] = None
    else:
        d["home_location"] = {
            "latitude": float(hloc.get("latitude", d.get("home_latitude", 0.0))),
            "longitude": float(hloc.get("longitude", d.get("home_longitude", 0.0)))
        }

    return d

def compute_rule_violations_count(
    amount: float,
    origin_balance: float,
    auth_verified: int,
    speed_kmh: float,
    tx_count_1m: int
) -> int:
    """Evaluate deterministic business rules and return count of triggered violations."""
    count = 0
    if origin_balance == 0.0 and amount > 5000.0 and auth_verified == 0:
        count += 1
    if origin_balance > 0.0 and (amount / (origin_balance + 1e-5)) > 0.90:
        count += 1
    if speed_kmh > 800.0:
        count += 1
    if tx_count_1m >= 3:
        count += 1
    if auth_verified == 0 and amount > 2500.0:
        count += 1
    return count

def compute_features_from_history(
    tx_dict: Dict[str, Any],
    prior_user_events: List[Dict[str, Any]],
    prior_device_events: Optional[List[Dict[str, Any]]] = None,
    category_encoder: Optional[FastFrequencyEncoder] = None,
    isolation_forest_model: Optional[Any] = None
) -> FinPulseFeatureVector:
    """
    Core mathematical feature calculation for a single transaction.
    This pure function is shared by both offline batch processing and online Redis serving.
    Guarantees 100% offline/online feature parity.
    """
    # 1. Transaction base attributes
    amount = float(tx_dict.get("amount", 0.0))
    origin_balance = float(tx_dict.get("origin_balance", 0.0))
    ts = float(tx_dict.get("timestamp", time.time()))

    # Temporal base attributes
    if "hour_of_day" in tx_dict:
        hour = int(tx_dict["hour_of_day"])
    else:
        hour = int((ts % 86400) // 3600)

    if "day_of_week" in tx_dict:
        day = int(tx_dict["day_of_week"])
    else:
        day = int(((ts // 86400) + 3) % 7) # Unix epoch (1970-01-01) was Thursday (3)

    is_weekend = int(day >= 5)
    is_night = int(hour < 6 or hour > 22)
    sin_hour, cos_hour = compute_cyclic_time(hour)

    # Categorical encodings
    raw_payment = str(tx_dict.get("payment_type", "TRANSFER")).upper()
    payment_type_enc = PAYMENT_TYPE_MAP.get(raw_payment, PAYMENT_TYPE_MAP.get("OTHER", 6))
    raw_cat = str(tx_dict.get("category", "general"))
    if category_encoder is not None:
        cat_enc = int(category_encoder.transform_single(raw_cat) * 1000)
    else:
        cat_enc = int((abs(hash(raw_cat)) % 1000))

    # Location & Device inputs
    loc = tx_dict.get("location", {})
    if isinstance(loc, dict):
        lat = float(loc.get("latitude", tx_dict.get("latitude", 0.0)))
        lon = float(loc.get("longitude", tx_dict.get("longitude", 0.0)))
    else:
        lat = float(tx_dict.get("latitude", 0.0))
        lon = float(tx_dict.get("longitude", 0.0))

    home_loc = tx_dict.get("home_location", {})
    if isinstance(home_loc, dict):
        home_lat = float(home_loc.get("latitude", tx_dict.get("home_latitude", lat)))
        home_lon = float(home_loc.get("longitude", tx_dict.get("home_longitude", lon)))
    else:
        home_lat = float(tx_dict.get("home_latitude", lat))
        home_lon = float(tx_dict.get("home_longitude", lon))

    device_id = str(tx_dict.get("device_id", "unknown_device"))
    auth_verified = 1 if tx_dict.get("auth_verified", True) else 0

    # 2. Velocity features (from prior history strictly before timestamp)
    vel_metrics = compute_velocity_metrics(prior_user_events, ts)

    # 3. Behavioral features (up to 30 days history)
    beh_metrics = compute_behavioral_metrics(
        prior_events=prior_user_events,
        current_amount=amount,
        current_category=raw_cat,
        current_hour=hour,
        current_timestamp=ts
    )

    # 4. Contextual & location features
    spatial_metrics = compute_spatial_and_device_metrics(
        prior_events=prior_user_events,
        current_lat=lat,
        current_lon=lon,
        home_lat=home_lat,
        home_lon=home_lon,
        current_device_id=device_id,
        current_timestamp=ts,
        device_user_events=prior_device_events
    )

    # 5. Anomaly & Rule features
    rule_count = compute_rule_violations_count(
        amount=amount,
        origin_balance=origin_balance,
        auth_verified=auth_verified,
        speed_kmh=spatial_metrics["speed_kmh_from_prev_tx"],
        tx_count_1m=vel_metrics["tx_count_1m"]
    )

    # Isolation Forest score
    if_score = float(tx_dict.get("isolation_forest_score", 0.10))

    return FinPulseFeatureVector(
        # Group A
        amount=amount,
        log_amount=compute_log_amount(amount),
        payment_type_enc=payment_type_enc,
        merchant_category_enc=cat_enc,
        amount_to_balance_ratio=compute_amount_to_balance_ratio(amount, origin_balance),
        # Group B
        hour_of_day=hour,
        day_of_week=day,
        is_weekend=is_weekend,
        is_night=is_night,
        cyclic_hour_sin=round(sin_hour, 5),
        cyclic_hour_cos=round(cos_hour, 5),
        # Group C
        tx_count_1m=vel_metrics["tx_count_1m"],
        tx_count_5m=vel_metrics["tx_count_5m"],
        tx_count_15m=vel_metrics["tx_count_15m"],
        tx_count_1h=vel_metrics["tx_count_1h"],
        amount_sum_5m=round(vel_metrics["amount_sum_5m"], 2),
        amount_sum_15m=round(vel_metrics["amount_sum_15m"], 2),
        amount_sum_1h=round(vel_metrics["amount_sum_1h"], 2),
        # Group D
        user_avg_amount_30d=beh_metrics["user_avg_amount_30d"],
        user_std_amount_30d=beh_metrics["user_std_amount_30d"],
        amount_zscore=beh_metrics["amount_zscore"],
        user_category_frequency=beh_metrics["user_category_frequency"],
        user_hourly_tx_deviation=beh_metrics["user_hourly_tx_deviation"],
        # Group E
        distance_from_home_km=spatial_metrics["distance_from_home_km"],
        distance_from_prev_loc_km=spatial_metrics["distance_from_prev_loc_km"],
        speed_kmh_from_prev_tx=spatial_metrics["speed_kmh_from_prev_tx"],
        is_new_device=spatial_metrics["is_new_device"],
        is_new_location=spatial_metrics["is_new_location"],
        device_user_count_24h=spatial_metrics["device_user_count_24h"],
        # Group F
        auth_factor_verified=auth_verified,
        deterministic_rule_count=rule_count,
        isolation_forest_score=round(if_score, 4)
    )

class FinPulseFeatureEngine:
    """
    Production-grade Feature Engine supporting both:
    1. Online real-time feature extraction via RedisStateManager historical context.
    2. Offline batch training feature matrix extraction with strict temporal leakage prevention.
    Guarantees 100% offline/online feature parity.
    """

    def __init__(
        self,
        state_manager: Optional[Any] = None,
        redis_engine: Optional[Any] = None,
        category_encoder: Optional[FastFrequencyEncoder] = None
    ):
        # Prefer explicit state_manager, then redis_engine, then default RedisStateManager
        self.state_manager = state_manager or redis_engine or RedisStateManager()
        self.redis_engine = self.state_manager
        self.category_encoder = category_encoder or FastFrequencyEncoder()
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """Fit encoders on training dataset."""
        if "category" in df.columns:
            self.category_encoder.fit(df["category"])
        self.is_fitted = True
        return self

    def compute_features(
        self,
        tx: Union[CommonTransactionSchema, TransactionEvent, Dict[str, Any]],
        context: Optional[CustomerHistoricalContext] = None
    ) -> FinPulseFeatureVector:
        """
        Authoritative 32-feature extraction for a single transaction.
        Enforces zero look-ahead: historical features are derived strictly prior to tx.timestamp.
        Validates the output vector to guarantee strictly 32 dimensions, 0 NaNs, and 0 Infs.
        """
        tx_dict = normalize_transaction_to_dict(tx)
        cust_id = str(tx_dict.get("customer_id", "default_cust"))
        ts = float(tx_dict.get("timestamp", time.time()))
        dev_id = str(tx_dict.get("device_id", "unknown_device"))

        # 1. Retrieve or use historical context
        if context is not None:
            prior_user_events = getattr(context, "prior_events", [])
            prior_dev_events = getattr(context, "device_events", [])
        elif self.state_manager is not None:
            if hasattr(self.state_manager, "get_prior_events"):
                prior_user_events = self.state_manager.get_prior_events(cust_id, before_timestamp=ts)
                prior_dev_events = self.state_manager.get_device_events(dev_id, before_timestamp=ts)
            elif hasattr(self.state_manager, "fetch_prior_events"):
                prior_user_events = self.state_manager.fetch_prior_events(cust_id, ts)
                prior_dev_events = self.state_manager.fetch_device_events(dev_id, ts)
            else:
                prior_user_events = []
                prior_dev_events = []
        else:
            prior_user_events = []
            prior_dev_events = []

        # 2. Compute 32 features using authoritative shared pure function
        feature_vec = compute_features_from_history(
            tx_dict=tx_dict,
            prior_user_events=prior_user_events,
            prior_device_events=prior_dev_events,
            category_encoder=self.category_encoder
        )

        # 3. Validate feature vector (strictly 32 dimensions, 0 NaNs, 0 Infs)
        validate_feature_vector(feature_vec.to_ordered_vector(), expected_dim=32)

        return feature_vec

    def extract_features(
        self,
        event: Union[TransactionEvent, CommonTransactionSchema, Dict[str, Any]],
        context: Optional[CustomerHistoricalContext] = None
    ) -> FeatureExtractionResult:
        """
        Real-time pipeline extraction hook for R3 Kafka/Consumer integration.
        Returns a validated FeatureExtractionResult with exactly 32 ordered features.
        """
        tx_dict = normalize_transaction_to_dict(event)
        feature_vec = self.compute_features(tx_dict, context=context)
        ordered_vec = feature_vec.to_ordered_vector()

        return FeatureExtractionResult(
            transaction_id=str(tx_dict.get("transaction_id", "")),
            customer_id=str(tx_dict.get("customer_id", "")),
            timestamp=float(tx_dict.get("timestamp", 0.0)),
            features=ordered_vec,
            feature_names=list(FinPulseFeatureVector.FEATURE_NAMES),
            metadata={"source": "R3_real_time_feature_engine"}
        )

    def compute_online_features(
        self,
        tx: CommonTransactionSchema
    ) -> FinPulseFeatureVector:
        """
        Online inference feature generation:
        1. Queries Redis for customer's prior events strictly before tx.timestamp.
        2. Computes the real 32 features.
        3. Commits current transaction to Redis for future queries.
        """
        tx_dict = normalize_transaction_to_dict(tx)
        cust_id = tx.customer_id
        ts = tx.timestamp
        dev_id = tx.device_id

        # 1. Fetch prior history from state manager
        if hasattr(self.state_manager, "get_prior_events"):
            prior_user_events = self.state_manager.get_prior_events(cust_id, before_timestamp=ts)
            prior_dev_events = self.state_manager.get_device_events(dev_id, before_timestamp=ts)
        elif hasattr(self.state_manager, "fetch_prior_events"):
            prior_user_events = self.state_manager.fetch_prior_events(cust_id, ts)
            prior_dev_events = self.state_manager.fetch_device_events(dev_id, ts)
        else:
            prior_user_events = []
            prior_dev_events = []

        # 2. Compute 32 features using shared pure function
        feature_vec = compute_features_from_history(
            tx_dict=tx_dict,
            prior_user_events=prior_user_events,
            prior_device_events=prior_dev_events,
            category_encoder=self.category_encoder
        )
        validate_feature_vector(feature_vec.to_ordered_vector(), expected_dim=32)

        # 3. Commit new transaction to state
        if hasattr(self.state_manager, "record_transaction"):
            import inspect
            sig = inspect.signature(self.state_manager.record_transaction)
            if len(sig.parameters) == 1:
                if isinstance(tx, CommonTransactionSchema):
                    from src.streaming.schema import TransactionEvent
                    event = TransactionEvent.from_common_schema(tx)
                else:
                    event = tx
                self.state_manager.record_transaction(event)
            else:
                loc = tx_dict.get("location", {})
                lat = loc.get("latitude", 0.0) if isinstance(loc, dict) else 0.0
                lon = loc.get("longitude", 0.0) if isinstance(loc, dict) else 0.0
                home_loc = tx_dict.get("home_location", {})
                home_lat = home_loc.get("latitude", lat) if isinstance(home_loc, dict) else lat
                home_lon = home_loc.get("longitude", lon) if isinstance(home_loc, dict) else lon
                self.state_manager.record_transaction(
                    customer_id=cust_id,
                    tx_id=tx.transaction_id,
                    timestamp=ts,
                    amount=tx.amount,
                    category=tx.category,
                    hour=feature_vec.hour_of_day,
                    lat=lat,
                    lon=lon,
                    device_id=dev_id,
                    home_lat=home_lat,
                    home_lon=home_lon
                )

        return feature_vec

    def compute_offline_features(
        self,
        df: pd.DataFrame,
        is_sorted: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Offline batch feature generation.
        Processes transactions strictly chronologically.
        Maintains customer and device histories in-memory, computing the exact
        same 32 features without temporal lookahead leakage.

        Returns:
            X: float32 numpy array of shape (N, 32)
            y: int numpy array of shape (N,)
            feature_names: List of 32 feature names
        """
        if not is_sorted:
            df = df.sort_values(by="timestamp").reset_index(drop=True)

        n = len(df)
        feature_names = FinPulseFeatureVector.FEATURE_NAMES
        X = np.zeros((n, 32), dtype=np.float32)
        y = df["is_fraud"].values.astype(int) if "is_fraud" in df.columns else np.zeros(n, dtype=int)

        # In-memory customer state tracking: customer_id -> deque of event dicts
        user_histories: Dict[str, deque] = defaultdict(deque)
        # Device state tracking: device_id -> deque of (timestamp, customer_id)
        device_histories: Dict[str, deque] = defaultdict(deque)

        t_lookback_30d = 30.0 * 86400.0
        t_lookback_24h = 86400.0

        records = df.to_dict(orient="records")

        for i, row in enumerate(records):
            cust_id = str(row.get("customer_id", "default_cust"))
            dev_id = str(row.get("device_id", "unknown_device"))
            ts = float(row.get("timestamp", 0.0))

            # Fetch customer past events strictly before ts
            user_deque = user_histories[cust_id]
            # Prune events older than 30 days
            min_user_time = ts - t_lookback_30d
            while user_deque and user_deque[0]["timestamp"] < min_user_time:
                user_deque.popleft()

            # Device events
            dev_deque = device_histories[dev_id] if dev_id != "unknown_device" else None
            if dev_deque is not None:
                min_dev_time = ts - t_lookback_24h
                while dev_deque and dev_deque[0]["timestamp"] < min_dev_time:
                    dev_deque.popleft()

            # 1. Compute 32 features using shared pure function
            prior_user_list = list(user_deque)
            prior_dev_list = list(dev_deque) if dev_deque is not None else None

            feat_vec = compute_features_from_history(
                tx_dict=row,
                prior_user_events=prior_user_list,
                prior_device_events=prior_dev_list,
                category_encoder=self.category_encoder
            )

            X[i] = feat_vec.to_ordered_vector()

            # 2. Append current event to customer history for FUTURE events
            loc = row.get("location", {})
            lat = float(loc.get("latitude", row.get("latitude", 0.0))) if isinstance(loc, dict) else float(row.get("latitude", 0.0))
            lon = float(loc.get("longitude", row.get("longitude", 0.0))) if isinstance(loc, dict) else float(row.get("longitude", 0.0))

            user_deque.append({
                "transaction_id": row.get("transaction_id", f"tx_{i}"),
                "timestamp": ts,
                "amount": float(row.get("amount", 0.0)),
                "category": str(row.get("category", "general")),
                "hour_of_day": feat_vec.hour_of_day,
                "latitude": lat,
                "longitude": lon,
                "device_id": dev_id
            })

            if dev_deque is not None:
                dev_deque.append({
                    "timestamp": ts,
                    "customer_id": cust_id
                })

        validate_feature_vector(X, expected_dim=32)
        return X, y, feature_names
