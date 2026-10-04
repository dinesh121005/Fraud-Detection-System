"""Redis-backed real-time sliding window and customer state engine."""
import time
import json
from typing import Dict, Any, List, Optional
from .redis_client import get_redis_client
from src.features.transformations import (
    compute_velocity_metrics,
    compute_behavioral_metrics,
    compute_spatial_and_device_metrics
)

class RedisSlidingWindowEngine:
    """
    Maintains real-time historical state using Redis Sorted Sets and Hashes.
    Key structures:
      - user:{customer_id}:txs     -> ZSET (score: timestamp, member: JSON of transaction)
      - user:{customer_id}:devices -> SET (members: device_ids)
      - user:{customer_id}:profile -> HASH (home_lat, home_lon, first_seen)
      - device:{device_id}:users   -> ZSET (score: timestamp, member: customer_id)
    """

    def __init__(self, redis_client=None, default_lookback_seconds: float = 30 * 86400.0):
        self.r = redis_client or get_redis_client()
        self.default_lookback = default_lookback_seconds

    def fetch_prior_events(
        self,
        customer_id: str,
        current_timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Fetch customer events strictly PRIOR to current_timestamp (t < current_timestamp).
        Evicts events older than lookback window.
        """
        key = f"user:{customer_id}:txs"
        min_time = current_timestamp - self.default_lookback

        # Evict very old events (> 30 days)
        try:
            self.r.zremrangebyscore(key, -float("inf"), min_time)
        except Exception:
            pass

        # Query events up to (exclusive) current timestamp
        try:
            # Redis zrangebyscore with '(' indicates exclusive boundary
            raw_members = self.r.zrangebyscore(key, min_time, current_timestamp)
        except Exception:
            raw_members = []

        events: List[Dict[str, Any]] = []
        for item in raw_members:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                ev = json.loads(item)
                # Strict check: only include events strictly before current_timestamp
                if float(ev.get("timestamp", 0.0)) < current_timestamp:
                    events.append(ev)
            except Exception:
                continue

        # Sort chronologically
        events.sort(key=lambda x: float(x.get("timestamp", 0.0)))
        return events

    def fetch_device_events(
        self,
        device_id: str,
        current_timestamp: float
    ) -> List[Dict[str, Any]]:
        """Fetch transactions across all users associated with this device in the last 24h."""
        if not device_id or device_id == "unknown_device":
            return []

        key = f"device:{device_id}:users"
        min_time = current_timestamp - 86400.0

        try:
            self.r.zremrangebyscore(key, -float("inf"), min_time)
            raw_members = self.r.zrangebyscore(key, min_time, current_timestamp)
        except Exception:
            raw_members = []

        dev_events = []
        for item in raw_members:
            try:
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                parts = item.split(":", 1)
                dev_events.append({
                    "timestamp": float(parts[0]),
                    "customer_id": parts[1] if len(parts) > 1 else ""
                })
            except Exception:
                continue

        return dev_events

    def get_online_state_features(
        self,
        customer_id: str,
        current_timestamp: float,
        current_amount: float,
        current_category: str,
        current_hour: int,
        current_lat: float = 0.0,
        current_lon: float = 0.0,
        home_lat: float = 0.0,
        home_lon: float = 0.0,
        device_id: str = "unknown_device"
    ) -> Dict[str, Any]:
        """
        Compute velocity, behavioral, and spatial features for incoming transaction
        using Redis state strictly BEFORE recording current transaction.
        """
        prior_events = self.fetch_prior_events(customer_id, current_timestamp)
        device_events = self.fetch_device_events(device_id, current_timestamp)

        # 1. Rolling velocity metrics (1m, 5m, 15m, 1h)
        vel_dict = compute_velocity_metrics(prior_events, current_timestamp)

        # 2. Behavioral 30-day metrics
        beh_dict = compute_behavioral_metrics(
            prior_events=prior_events,
            current_amount=current_amount,
            current_category=current_category,
            current_hour=current_hour,
            current_timestamp=current_timestamp
        )

        # If home_lat/home_lon not explicitly passed, attempt to load from Redis profile
        if home_lat == 0.0 and home_lon == 0.0:
            try:
                prof = self.r.hgetall(f"user:{customer_id}:profile")
                if prof:
                    hlat = prof.get(b"home_lat", prof.get("home_lat"))
                    hlon = prof.get(b"home_lon", prof.get("home_lon"))
                    if hlat is not None and hlon is not None:
                        home_lat = float(hlat)
                        home_lon = float(hlon)
            except Exception:
                pass

        # 3. Spatial and device metrics
        spatial_dict = compute_spatial_and_device_metrics(
            prior_events=prior_events,
            current_lat=current_lat,
            current_lon=current_lon,
            home_lat=home_lat,
            home_lon=home_lon,
            current_device_id=device_id,
            current_timestamp=current_timestamp,
            device_user_events=device_events
        )

        # Merge state features
        state_features = {}
        state_features.update(vel_dict)
        state_features.update(beh_dict)
        state_features.update(spatial_dict)
        return state_features

    def record_and_fetch_velocity(
        self,
        customer_id: str,
        tx_id: str,
        timestamp: float,
        amount: float,
        category: str = "general",
        hour: int = 12,
        lat: float = 0.0,
        lon: float = 0.0,
        device_id: str = "unknown_device",
        home_lat: float = 0.0,
        home_lon: float = 0.0
    ) -> Dict[str, Any]:
        """
        Backward-compatible helper: extracts decision-time state features strictly
        BEFORE committing the current transaction into Redis.
        """
        prior_events = self.fetch_prior_events(customer_id, timestamp)
        device_events = self.fetch_device_events(device_id, timestamp)
        state = self.get_online_state_features(
            customer_id=customer_id,
            current_timestamp=timestamp,
            current_amount=amount,
            current_category=category,
            current_hour=hour,
            current_lat=lat,
            current_lon=lon,
            home_lat=home_lat,
            home_lon=home_lon,
            device_id=device_id
        )
        state["prior_events"] = prior_events
        state["prior_device_events"] = device_events

        self.record_transaction(
            customer_id=customer_id,
            tx_id=tx_id,
            timestamp=timestamp,
            amount=amount,
            category=category,
            hour=hour,
            lat=lat,
            lon=lon,
            device_id=device_id,
            home_lat=home_lat,
            home_lon=home_lon
        )
        return state

    def record_transaction(
        self,
        customer_id: str,
        tx_id: str,
        timestamp: float,
        amount: float,
        category: str = "general",
        hour: int = 12,
        lat: float = 0.0,
        lon: float = 0.0,
        device_id: str = "unknown_device",
        home_lat: float = 0.0,
        home_lon: float = 0.0
    ):
        """
        Commit transaction to Redis state after features have been evaluated.
        """
        # 1. Add to customer sorted set
        tx_key = f"user:{customer_id}:txs"
        event_payload = {
            "transaction_id": tx_id,
            "timestamp": float(timestamp),
            "amount": float(amount),
            "category": str(category),
            "hour_of_day": int(hour),
            "latitude": float(lat),
            "longitude": float(lon),
            "device_id": str(device_id)
        }
        try:
            self.r.zadd(tx_key, {json.dumps(event_payload): float(timestamp)})
        except Exception:
            pass

        # 2. Add to device sorted set
        if device_id and device_id != "unknown_device":
            dev_key = f"device:{device_id}:users"
            dev_member = f"{float(timestamp)}:{customer_id}"
            try:
                self.r.zadd(dev_key, {dev_member: float(timestamp)})
            except Exception:
                pass

        # 3. Save profile home coordinates if available
        if home_lat != 0.0 or home_lon != 0.0:
            prof_key = f"user:{customer_id}:profile"
            try:
                self.r.hset(prof_key, mapping={"home_lat": home_lat, "home_lon": home_lon})
            except Exception:
                pass

    def clear_customer_state(self, customer_id: str) -> bool:
        """Cleanly evict rolling transaction state, profile, and device associations for a customer."""
        try:
            self.r.delete(f"user:{customer_id}:txs")
            self.r.delete(f"user:{customer_id}:profile")
            self.r.delete(f"user:{customer_id}:devices")
            return True
        except Exception:
            return False

    def remove_transaction(self, customer_id: str, tx_id: str) -> bool:
        """
        Evict a blocked/declined fraudulent attempt so it does not contaminate valid customer spending volume.
        """
        tx_key = f"user:{customer_id}:txs"
        try:
            members = self.r.zrange(tx_key, 0, -1)
            for m in members:
                m_str = m.decode("utf-8") if isinstance(m, bytes) else str(m)
                if f'"{tx_id}"' in m_str:
                    self.r.zrem(tx_key, m)
                    return True
            return False
        except Exception:
            return False
