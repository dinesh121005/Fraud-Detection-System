"""Redis State Manager for Real-Time Fraud Detection (R2 Foundation).

Maintains low-latency historical context per customer and device:
- Velocity Sliding Windows (1m, 5m, 15m, 1h) via Sorted Sets
- 30-Day Customer Behavioral Statistics (count, sum, mean, std)
- Previous Geospatial Coordinates & Timestamp
- Bidirectional Customer <-> Device Graph
- Strict Read -> Process -> Write Order Invariant Enforcement
"""
import os
import sys
import json
import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Set, Tuple

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.state.redis_client import get_redis_client, InMemoryRedisMock
from src.streaming.schema import TransactionEvent

logger = logging.getLogger("FinPulse.RedisStateManager")

@dataclass
class CustomerHistoricalContext:
    """Historical context snapshot retrieved strictly BEFORE current transaction is recorded."""
    customer_id: str
    query_timestamp: float
    velocity: Dict[str, Any] = field(default_factory=dict)
    behavior: Dict[str, Any] = field(default_factory=dict)
    previous_location: Optional[Dict[str, float]] = None
    customer_devices: List[str] = field(default_factory=list)
    device_customers: List[str] = field(default_factory=list)
    prior_events: List[Dict[str, Any]] = field(default_factory=list)
    device_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class RedisStateManager:
    """
    R2 Redis Historical State Manager.
    Guarantees:
    - Zero-lookahead state isolation: historical context excludes current transaction.
    - Customer and device namespace isolation.
    - Deterministic numerical calculations with empty-state safeguards.
    """

    LOOKBACK_30_DAYS = 30.0 * 86400.0  # 2,592,000 seconds
    VELOCITY_WINDOWS = {
        "1m": 60.0,
        "5m": 300.0,
        "15m": 900.0,
        "1h": 3600.0
    }

    def __init__(
        self,
        redis_client=None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        force_mock: bool = False
    ):
        self.r = redis_client or get_redis_client(
            host=host,
            port=port,
            db=db,
            password=password,
            force_mock=force_mock
        )

    # --- Connection & Health ---

    def connect(self) -> bool:
        """Establish or verify Redis connection."""
        return self.health_check()

    def health_check(self) -> bool:
        """Verify Redis connection health with PING."""
        try:
            return bool(self.r.ping())
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    def close(self):
        """Close connection cleanly."""
        try:
            if hasattr(self.r, "close"):
                self.r.close()
        except Exception:
            pass

    # --- Key Helper Functions ---

    @staticmethod
    def key_velocity(customer_id: str) -> str:
        return f"customer:{customer_id}:velocity"

    @staticmethod
    def key_behavior(customer_id: str) -> str:
        return f"customer:{customer_id}:behavior"

    @staticmethod
    def key_behavior_txs(customer_id: str) -> str:
        return f"customer:{customer_id}:behavior_txs"

    @staticmethod
    def key_location(customer_id: str) -> str:
        return f"customer:{customer_id}:location"

    @staticmethod
    def key_customer_devices(customer_id: str) -> str:
        return f"customer:{customer_id}:devices"

    @staticmethod
    def key_device_customers(device_id: str) -> str:
        return f"device:{device_id}:customers"

    @staticmethod
    def key_device_events(device_id: str) -> str:
        return f"device:{device_id}:users"

    # --- A. Transaction Velocity ---

    def record_velocity(
        self,
        customer_id: str,
        tx_id: str,
        amount: float,
        timestamp: float
    ) -> bool:
        """
        Record transaction into customer velocity sorted set.
        Member encodes transaction details uniquely; score is event timestamp.
        """
        key = self.key_velocity(customer_id)
        member = json.dumps({
            "tx_id": str(tx_id),
            "amount": float(amount),
            "timestamp": float(timestamp)
        })
        try:
            self.r.zadd(key, {member: float(timestamp)})
            return True
        except Exception as e:
            logger.error(f"Failed to record velocity for {customer_id}: {e}")
            return False

    def get_velocity(
        self,
        customer_id: str,
        before_timestamp: float
    ) -> Dict[str, Any]:
        """
        Retrieve velocity metrics strictly BEFORE before_timestamp (t < before_timestamp).
        Calculates counts and sums for 1m, 5m, 15m, 1h windows.
        """
        key = self.key_velocity(customer_id)
        t_now = float(before_timestamp)
        min_time_1h = t_now - self.VELOCITY_WINDOWS["1h"]

        # Fetch records up to (exclusive) before_timestamp
        try:
            raw_members = self.r.zrangebyscore(key, min_time_1h, f"({t_now}")
        except Exception:
            try:
                # Fallback if exclusive syntax is not directly supported by client wrapper
                all_in_range = self.r.zrangebyscore(key, min_time_1h, t_now)
                raw_members = []
                for m in all_in_range:
                    decoded = m.decode("utf-8") if isinstance(m, bytes) else m
                    ev = json.loads(decoded)
                    if float(ev.get("timestamp", 0.0)) < t_now:
                        raw_members.append(m)
            except Exception as e:
                logger.warning(f"Error querying velocity for {customer_id}: {e}")
                raw_members = []

        # Parse valid events
        events: List[Tuple[float, float]] = []  # list of (timestamp, amount)
        for item in raw_members:
            try:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else item
                ev = json.loads(decoded)
                ts = float(ev["timestamp"])
                if ts < t_now:
                    events.append((ts, float(ev["amount"])))
            except Exception:
                continue

        # Compute counts and sums for each window
        t_1m = t_now - self.VELOCITY_WINDOWS["1m"]
        t_5m = t_now - self.VELOCITY_WINDOWS["5m"]
        t_15m = t_now - self.VELOCITY_WINDOWS["15m"]

        tx_count_1m = sum(1 for ts, _ in events if ts >= t_1m)
        tx_count_5m = sum(1 for ts, _ in events if ts >= t_5m)
        tx_count_15m = sum(1 for ts, _ in events if ts >= t_15m)
        tx_count_1h = len(events)

        amount_sum_5m = sum(amt for ts, amt in events if ts >= t_5m)
        amount_sum_15m = sum(amt for ts, amt in events if ts >= t_15m)
        amount_sum_1h = sum(amt for _, amt in events)

        return {
            "tx_count_1m": int(tx_count_1m),
            "tx_count_5m": int(tx_count_5m),
            "tx_count_15m": int(tx_count_15m),
            "tx_count_1h": int(tx_count_1h),
            "amount_sum_5m": round(float(amount_sum_5m), 2),
            "amount_sum_15m": round(float(amount_sum_15m), 2),
            "amount_sum_1h": round(float(amount_sum_1h), 2)
        }

    def cleanup_expired(
        self,
        customer_id: str,
        current_timestamp: float,
        lookback_seconds: float = 3600.0
    ) -> int:
        """Remove velocity events older than the 1h window."""
        key = self.key_velocity(customer_id)
        expiry_boundary = current_timestamp - lookback_seconds
        try:
            return self.r.zremrangebyscore(key, "-inf", expiry_boundary)
        except Exception:
            return 0

    # --- B. 30-Day Customer Behavioral State ---

    def update_customer_behavior(
        self,
        customer_id: str,
        amount: float,
        timestamp: float,
        tx_id: str = "",
        category: str = "general",
        hour: Optional[int] = None,
        latitude: float = 0.0,
        longitude: float = 0.0,
        device_id: str = "unknown_device"
    ) -> Dict[str, Any]:
        """
        Record transaction in the customer's 30-day behavioral state.
        Evicts transactions older than 30 days and maintains exact count, sum, mean, std.
        """
        z_key = self.key_behavior_txs(customer_id)
        h_key = self.key_behavior(customer_id)
        t_now = float(timestamp)
        min_time_30d = t_now - self.LOOKBACK_30_DAYS

        # 1. Add to 30-day ZSET
        hour_val = int(hour if hour is not None else (t_now % 86400) // 3600)
        payload = {
            "tx_id": str(tx_id),
            "transaction_id": str(tx_id),
            "amount": float(amount),
            "timestamp": t_now,
            "category": str(category),
            "hour_of_day": hour_val,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "device_id": str(device_id)
        }
        member = json.dumps(payload)
        try:
            self.r.zadd(z_key, {member: t_now})
            # 2. Prune records older than 30 days
            self.r.zremrangebyscore(z_key, "-inf", min_time_30d)
        except Exception as e:
            logger.warning(f"Error updating behavior ZSET for {customer_id}: {e}")

        # 3. Compute exact 30-day statistics across all transactions in window
        try:
            raw_members = self.r.zrangebyscore(z_key, min_time_30d, "+inf")
        except Exception:
            raw_members = []

        amounts: List[float] = []
        for item in raw_members:
            try:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else item
                ev = json.loads(decoded)
                amounts.append(float(ev["amount"]))
            except Exception:
                continue

        n = len(amounts)
        total_sum = sum(amounts)
        mean_amt = (total_sum / n) if n > 0 else 0.0

        if n >= 2:
            variance = sum((x - mean_amt) ** 2 for x in amounts) / (n - 1)
            std_amt = math.sqrt(variance)
        else:
            std_amt = 0.0

        stats = {
            "count": int(n),
            "sum": round(float(total_sum), 2),
            "mean": round(float(mean_amt), 4),
            "std": round(float(std_amt), 4),
            "last_timestamp": t_now
        }

        # 4. Cache summary in Hash
        try:
            self.r.hset(h_key, mapping=stats)
        except Exception as e:
            logger.warning(f"Error caching behavior hash for {customer_id}: {e}")

        return stats

    def get_customer_behavior(
        self,
        customer_id: str,
        before_timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieve 30-day behavioral metrics strictly BEFORE before_timestamp.
        If before_timestamp is provided, derives metrics dynamically from the 30-day ZSET
        strictly filtering t < before_timestamp to prevent lookahead contamination.
        """
        z_key = self.key_behavior_txs(customer_id)
        h_key = self.key_behavior(customer_id)

        # Empty state defaults
        empty_state = {
            "count": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "last_timestamp": None
        }

        if before_timestamp is None:
            # Return cached Hash summary
            try:
                h_data = self.r.hgetall(h_key)
                if not h_data:
                    return empty_state
                decoded = {
                    (k.decode("utf-8") if isinstance(k, bytes) else str(k)): (v.decode("utf-8") if isinstance(v, bytes) else str(v))
                    for k, v in h_data.items()
                }
                return {
                    "count": int(decoded.get("count", 0)),
                    "sum": float(decoded.get("sum", 0.0)),
                    "mean": float(decoded.get("mean", 0.0)),
                    "std": float(decoded.get("std", 0.0)),
                    "last_timestamp": float(decoded["last_timestamp"]) if "last_timestamp" in decoded else None
                }
            except Exception:
                return empty_state

        # Strict zero-lookahead query: compute dynamically for t < before_timestamp in 30d window
        t_now = float(before_timestamp)
        min_time_30d = t_now - self.LOOKBACK_30_DAYS

        try:
            raw_members = self.r.zrangebyscore(z_key, min_time_30d, f"({t_now}")
        except Exception:
            try:
                all_members = self.r.zrangebyscore(z_key, min_time_30d, t_now)
                raw_members = []
                for m in all_members:
                    decoded = m.decode("utf-8") if isinstance(m, bytes) else m
                    ev = json.loads(decoded)
                    if float(ev.get("timestamp", 0.0)) < t_now:
                        raw_members.append(m)
            except Exception:
                raw_members = []

        amounts: List[float] = []
        last_t: Optional[float] = None
        for item in raw_members:
            try:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else item
                ev = json.loads(decoded)
                ts = float(ev["timestamp"])
                if ts < t_now:
                    amounts.append(float(ev["amount"]))
                    if last_t is None or ts > last_t:
                        last_t = ts
            except Exception:
                continue

        n = len(amounts)
        if n == 0:
            return empty_state

        total_sum = sum(amounts)
        mean_amt = total_sum / n
        std_amt = math.sqrt(sum((x - mean_amt) ** 2 for x in amounts) / (n - 1)) if n >= 2 else 0.0

        return {
            "count": int(n),
            "sum": round(float(total_sum), 2),
            "mean": round(float(mean_amt), 4),
            "std": round(float(std_amt), 4),
            "last_timestamp": last_t
        }

    def get_prior_events(
        self,
        customer_id: str,
        before_timestamp: float,
        lookback_seconds: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch customer events strictly PRIOR to before_timestamp (t < before_timestamp).
        Evicts events older than lookback window (default 30 days).
        Returns chronologically sorted list of transaction dictionaries.
        """
        z_key = self.key_behavior_txs(customer_id)
        t_now = float(before_timestamp)
        lookback = lookback_seconds or self.LOOKBACK_30_DAYS
        min_time = t_now - lookback

        try:
            raw_members = self.r.zrangebyscore(z_key, min_time, f"({t_now}")
        except Exception:
            try:
                all_members = self.r.zrangebyscore(z_key, min_time, t_now)
                raw_members = []
                for m in all_members:
                    decoded = m.decode("utf-8") if isinstance(m, bytes) else m
                    ev = json.loads(decoded)
                    if float(ev.get("timestamp", 0.0)) < t_now:
                        raw_members.append(m)
            except Exception:
                raw_members = []

        events: List[Dict[str, Any]] = []
        for item in raw_members:
            try:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else item
                ev = json.loads(decoded)
                ts = float(ev.get("timestamp", 0.0))
                if ts < t_now:
                    events.append({
                        "transaction_id": str(ev.get("transaction_id", ev.get("tx_id", ""))),
                        "timestamp": ts,
                        "amount": float(ev.get("amount", 0.0)),
                        "category": str(ev.get("category", "general")),
                        "hour_of_day": int(ev.get("hour_of_day", (ts % 86400) // 3600)),
                        "latitude": float(ev.get("latitude", 0.0)),
                        "longitude": float(ev.get("longitude", 0.0)),
                        "device_id": str(ev.get("device_id", "unknown_device"))
                    })
            except Exception:
                continue

        events.sort(key=lambda x: x["timestamp"])
        return events

    # --- C. Location State ---

    def update_location(
        self,
        customer_id: str,
        latitude: float,
        longitude: float,
        timestamp: float
    ) -> bool:
        """Store the customer's latest transaction location."""
        key = self.key_location(customer_id)
        try:
            self.r.hset(key, mapping={
                "latitude": float(latitude),
                "longitude": float(longitude),
                "timestamp": float(timestamp)
            })
            return True
        except Exception as e:
            logger.warning(f"Error updating location for {customer_id}: {e}")
            return False

    def get_previous_location(self, customer_id: str) -> Optional[Dict[str, float]]:
        """Retrieve previous location and timestamp; returns None if first transaction."""
        key = self.key_location(customer_id)
        try:
            data = self.r.hgetall(key)
            if not data:
                return None
            decoded = {
                (k.decode("utf-8") if isinstance(k, bytes) else str(k)): float(v.decode("utf-8") if isinstance(v, bytes) else v)
                for k, v in data.items()
            }
            if "latitude" in decoded and "longitude" in decoded:
                return {
                    "latitude": decoded["latitude"],
                    "longitude": decoded["longitude"],
                    "timestamp": decoded.get("timestamp", 0.0)
                }
            return None
        except Exception:
            return None

    # --- D. Device Associations ---

    def add_customer_device(self, customer_id: str, device_id: str) -> bool:
        """Associate a device with a customer."""
        if not device_id or device_id == "unknown_device":
            return False
        key = self.key_customer_devices(customer_id)
        try:
            self.r.sadd(key, device_id)
            return True
        except Exception as e:
            logger.warning(f"Error adding customer device: {e}")
            return False

    def get_customer_devices(self, customer_id: str) -> List[str]:
        """Retrieve all unique devices linked to customer."""
        key = self.key_customer_devices(customer_id)
        try:
            members = self.r.smembers(key)
            return sorted([m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members])
        except Exception:
            return []

    def add_device_customer(self, device_id: str, customer_id: str, timestamp: Optional[float] = None) -> bool:
        """Associate a customer with a device, updating both unique set and 24h activity log."""
        if not device_id or device_id == "unknown_device":
            return False
        set_key = self.key_device_customers(device_id)
        z_key = self.key_device_events(device_id)
        try:
            self.r.sadd(set_key, customer_id)
            if timestamp is not None:
                t_now = float(timestamp)
                member = f"{t_now}:{customer_id}"
                self.r.zadd(z_key, {member: t_now})
                self.r.zremrangebyscore(z_key, "-inf", t_now - 86400.0)
            return True
        except Exception as e:
            logger.warning(f"Error adding device customer: {e}")
            return False

    def get_device_customers(self, device_id: str) -> List[str]:
        """Retrieve all customers who have used this device."""
        if not device_id or device_id == "unknown_device":
            return []
        key = self.key_device_customers(device_id)
        try:
            members = self.r.smembers(key)
            return sorted([m.decode("utf-8") if isinstance(m, bytes) else str(m) for m in members])
        except Exception:
            return []

    def get_device_events(self, device_id: str, before_timestamp: float) -> List[Dict[str, Any]]:
        """Fetch transactions across all users associated with this device in the prior 24h."""
        if not device_id or device_id == "unknown_device":
            return []
        key = self.key_device_events(device_id)
        t_now = float(before_timestamp)
        min_time = t_now - 86400.0

        try:
            self.r.zremrangebyscore(key, "-inf", min_time)
            raw_members = self.r.zrangebyscore(key, min_time, f"({t_now}")
        except Exception:
            raw_members = []

        dev_events = []
        for item in raw_members:
            try:
                decoded = item.decode("utf-8") if isinstance(item, bytes) else str(item)
                parts = decoded.split(":", 1)
                dev_events.append({
                    "timestamp": float(parts[0]),
                    "customer_id": parts[1] if len(parts) > 1 else ""
                })
            except Exception:
                continue

        if not dev_events:
            # Fallback to known device customers set if 24h event log is not populated
            for c in self.get_device_customers(device_id):
                dev_events.append({"timestamp": t_now - 1.0, "customer_id": c})

        return dev_events

    # --- E. High-Level Invariant: Read Historical State -> Process -> Write State ---

    def get_historical_context(self, event: TransactionEvent) -> CustomerHistoricalContext:
        """
        MANDATORY READ STEP:
        Reads customer velocity, behavior, previous location, device relationships,
        and prior events list strictly prior to event.timestamp.
        The current transaction is NOT in the retrieved context.
        """
        cust = event.customer_id
        ts = event.timestamp
        dev = event.device_id

        # 1. Read velocity
        vel = self.get_velocity(cust, before_timestamp=ts)

        # 2. Read 30-day behavior
        beh = self.get_customer_behavior(cust, before_timestamp=ts)

        # 3. Read previous location
        loc = self.get_previous_location(cust)

        # 4. Read devices and shared users
        c_devices = self.get_customer_devices(cust)
        d_customers = self.get_device_customers(dev) if dev != "unknown_device" else []

        # 5. Read prior events list & device events list for feature engine
        prior_ev = self.get_prior_events(cust, before_timestamp=ts)
        dev_ev = self.get_device_events(dev, before_timestamp=ts) if dev != "unknown_device" else []

        return CustomerHistoricalContext(
            customer_id=cust,
            query_timestamp=ts,
            velocity=vel,
            behavior=beh,
            previous_location=loc,
            customer_devices=c_devices,
            device_customers=d_customers,
            prior_events=prior_ev,
            device_events=dev_ev
        )

    def record_transaction(self, event: TransactionEvent):
        """
        MANDATORY WRITE STEP:
        Commits the transaction to Redis state AFTER evaluation.
        """
        cust = event.customer_id
        ts = event.timestamp
        amt = event.amount
        tx_id = event.transaction_id
        dev = event.device_id
        cat = event.category
        hour = int((ts % 86400) // 3600)

        # 1. Update velocity sorted set & clean expired
        self.record_velocity(cust, tx_id=tx_id, amount=amt, timestamp=ts)
        self.cleanup_expired(cust, current_timestamp=ts)

        # 2. Update 30-day behavior with full event attributes
        self.update_customer_behavior(
            customer_id=cust,
            amount=amt,
            timestamp=ts,
            tx_id=tx_id,
            category=cat,
            hour=hour,
            latitude=event.latitude,
            longitude=event.longitude,
            device_id=dev
        )

        # 3. Update previous location (only if lat/lon provided)
        if event.latitude != 0.0 or event.longitude != 0.0:
            self.update_location(cust, latitude=event.latitude, longitude=event.longitude, timestamp=ts)

        # 4. Update bidirectional device associations
        if dev and dev != "unknown_device":
            self.add_customer_device(cust, dev)
            self.add_device_customer(dev, cust, timestamp=ts)

    def record_and_get_historical_context(self, event: TransactionEvent) -> CustomerHistoricalContext:
        """
        Atomic orchestration enforcing READ-BEFORE-WRITE order:
        1. READ historical state before T
        2. WRITE T into state
        3. Return historical state snapshot
        """
        context = self.get_historical_context(event)
        self.record_transaction(event)
        return context
