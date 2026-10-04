"""
FinPulse Demo Account Foundation (D1).

Provides an isolated, deterministic demonstration customer account model and state manager.
Guarantees:
- Strict isolation from real financial accounts
- Deterministic initial balances and credentials
- Authoritative balance updates only when explicit gateway authorization permits
- Clean, repeatable state resetting with baseline Redis state seeding
"""

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

from src.monitoring.logger import get_logger
from src.state.redis_client import get_redis_client

logger = get_logger("FinPulse.Simulation.DemoAccount")

DEFAULT_DEMO_CUSTOMER_ID = "CUST_DEMO_001"
DEFAULT_ATTACKER_RECEIVER_ID = "CUST_ATTACKER"
DEFAULT_INITIAL_BALANCE = 10000.0  # ₹10,000

KNOWN_DEVICE_ID = "DEV_KNOWN_01"
KNOWN_LOCATION = {
    "city": "CHENNAI",
    "latitude": 13.0827,
    "longitude": 80.2707
}

RECIPIENTS_REGISTRY = {
    "CUST_FRIEND_01": {"name": "Rahul Sharma (Friend)", "city": "CHENNAI", "type": "PEER"},
    "CUST_DEMO_002": {"name": "Priya Patel (Family)", "city": "CHENNAI", "type": "PEER"},
    "CUST_MERCH_01": {"name": "FreshMart Groceries", "city": "CHENNAI", "type": "MERCHANT"},
    "CUST_ATTACKER": {"name": "CUST_ATTACKER (Untrusted / Attack Destination)", "city": "MUMBAI", "type": "UNTRUSTED"}
}


@dataclass
class DemoAccount:
    """Controlled demonstration account."""
    customer_id: str
    balance: float
    account_status: str  # "SAFE", "UNDER_ATTACK", "COMPROMISED", "SUSPENDED"
    known_device_id: str
    home_latitude: float
    home_longitude: float
    known_city: str
    created_at: float
    updated_at: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DemoAccountManager:
    """
    Manages state, balance modifications, and deterministic resets for demo accounts.
    Thread-safe in-memory store with optional Redis synchronization.
    """

    def __init__(self, redis_client=None, initial_balance: float = DEFAULT_INITIAL_BALANCE):
        self.r = redis_client or get_redis_client()
        self.initial_balance = float(initial_balance)
        self._accounts: Dict[str, DemoAccount] = {}
        self.reset_all()

    def get_account(self, customer_id: str = DEFAULT_DEMO_CUSTOMER_ID) -> DemoAccount:
        """Fetch demo account record, creating default if not yet initialized."""
        if customer_id not in self._accounts:
            now = time.time()
            self._accounts[customer_id] = DemoAccount(
                customer_id=customer_id,
                balance=self.initial_balance if customer_id == DEFAULT_DEMO_CUSTOMER_ID else 0.0,
                account_status="SAFE",
                known_device_id=KNOWN_DEVICE_ID,
                home_latitude=KNOWN_LOCATION["latitude"],
                home_longitude=KNOWN_LOCATION["longitude"],
                known_city=KNOWN_LOCATION["city"],
                created_at=now,
                updated_at=now,
                metadata={"is_demo": True}
            )
        return self._accounts[customer_id]

    def get_balance(self, customer_id: str = DEFAULT_DEMO_CUSTOMER_ID) -> float:
        """Get current balance for account."""
        acc = self.get_account(customer_id)
        return acc.balance

    def update_status(self, customer_id: str, new_status: str) -> DemoAccount:
        """Update account security/lifecycle status."""
        acc = self.get_account(customer_id)
        acc.account_status = new_status
        acc.updated_at = time.time()
        logger.info("Demo account status updated", customer_id=customer_id, status=new_status)
        return acc

    def execute_transfer(
        self,
        sender_id: str,
        receiver_id: str,
        amount: float
    ) -> bool:
        """
        Execute balance deduction and credit strictly when gateway authorization allows.
        Returns True if successful, False if insufficient balance.
        """
        sender = self.get_account(sender_id)
        receiver = self.get_account(receiver_id)

        if sender.balance < amount:
            logger.warning("Transfer execution rejected: insufficient balance", sender=sender_id, balance=sender.balance, amount=amount)
            return False

        sender.balance = round(sender.balance - amount, 2)
        sender.updated_at = time.time()

        receiver.balance = round(receiver.balance + amount, 2)
        receiver.updated_at = time.time()

        logger.info(
            "Transfer executed successfully by gateway",
            sender=sender_id,
            receiver=receiver_id,
            amount=amount,
            sender_new_balance=sender.balance,
            receiver_new_balance=receiver.balance
        )
        return True

    def seed_redis_baseline(self, customer_id: str = DEFAULT_DEMO_CUSTOMER_ID) -> None:
        """
        Seed Redis behavioral baseline for customer:
        - Sets customer profile home coordinates to Chennai
        - Records a prior legitimate transaction from DEV_KNOWN_01 at Chennai
        - Registers DEV_KNOWN_01 in user device set
        """
        now = time.time()
        # 1. Profile home location
        try:
            prof_key = f"user:{customer_id}:profile"
            self.r.hset(prof_key, mapping={
                "home_lat": KNOWN_LOCATION["latitude"],
                "home_lon": KNOWN_LOCATION["longitude"],
                "known_device": KNOWN_DEVICE_ID,
                "first_seen": now - 86400.0 * 7
            })
        except Exception:
            pass

        # 2. Known device set
        try:
            dev_key = f"user:{customer_id}:devices"
            self.r.sadd(dev_key, KNOWN_DEVICE_ID)
        except Exception:
            pass

        # 3. Baseline historical transaction (2 hours ago, normal ₹500 grocery in Chennai)
        try:
            tx_key = f"user:{customer_id}:txs"
            base_payload = {
                "transaction_id": f"tx_base_{customer_id.lower()}_01",
                "timestamp": now - 7200.0,
                "amount": 500.0,
                "category": "groceries",
                "hour_of_day": int(((now - 7200.0) % 86400) // 3600),
                "latitude": KNOWN_LOCATION["latitude"],
                "longitude": KNOWN_LOCATION["longitude"],
                "device_id": KNOWN_DEVICE_ID
            }
            self.r.zadd(tx_key, {json.dumps(base_payload): now - 7200.0})
        except Exception:
            pass

        logger.info("Seeded Redis baseline for demo customer", customer_id=customer_id, device=KNOWN_DEVICE_ID)

    def reset_all(self, customer_id: str = DEFAULT_DEMO_CUSTOMER_ID) -> None:
        """
        Reset demo account and attacker balance to baseline, and restore Redis baseline.
        """
        now = time.time()
        self._accounts[customer_id] = DemoAccount(
            customer_id=customer_id,
            balance=self.initial_balance,
            account_status="SAFE",
            known_device_id=KNOWN_DEVICE_ID,
            home_latitude=KNOWN_LOCATION["latitude"],
            home_longitude=KNOWN_LOCATION["longitude"],
            known_city=KNOWN_LOCATION["city"],
            created_at=now,
            updated_at=now,
            metadata={"is_demo": True}
        )

        self._accounts[DEFAULT_ATTACKER_RECEIVER_ID] = DemoAccount(
            customer_id=DEFAULT_ATTACKER_RECEIVER_ID,
            balance=0.0,
            account_status="ACTIVE",
            known_device_id="DEV_ATTACKER_01",
            home_latitude=19.0760,
            home_longitude=72.8777,
            known_city="MUMBAI",
            created_at=now,
            updated_at=now,
            metadata={"is_demo_receiver": True}
        )

        # Standard demonstration counterparties for wallet transfers
        self._accounts["CUST_FRIEND_01"] = DemoAccount(
            customer_id="CUST_FRIEND_01",
            balance=1500.0,
            account_status="SAFE",
            known_device_id="DEV_RAHUL_01",
            home_latitude=KNOWN_LOCATION["latitude"],
            home_longitude=KNOWN_LOCATION["longitude"],
            known_city="CHENNAI",
            created_at=now,
            updated_at=now,
            metadata={"name": "Rahul Sharma (Friend)", "is_recipient": True}
        )

        self._accounts["CUST_DEMO_002"] = DemoAccount(
            customer_id="CUST_DEMO_002",
            balance=2500.0,
            account_status="SAFE",
            known_device_id="DEV_PRIYA_01",
            home_latitude=KNOWN_LOCATION["latitude"],
            home_longitude=KNOWN_LOCATION["longitude"],
            known_city="CHENNAI",
            created_at=now,
            updated_at=now,
            metadata={"name": "Priya Patel (Family)", "is_recipient": True}
        )

        self._accounts["CUST_MERCH_01"] = DemoAccount(
            customer_id="CUST_MERCH_01",
            balance=50000.0,
            account_status="SAFE",
            known_device_id="DEV_TERMINAL_01",
            home_latitude=KNOWN_LOCATION["latitude"],
            home_longitude=KNOWN_LOCATION["longitude"],
            known_city="CHENNAI",
            created_at=now,
            updated_at=now,
            metadata={"name": "FreshMart Groceries", "is_merchant": True}
        )

        # Clear Redis keys for demo customer
        try:
            self.r.delete(f"user:{customer_id}:txs")
            self.r.delete(f"user:{customer_id}:devices")
            self.r.delete(f"user:{customer_id}:profile")
            self.r.delete(f"customer:{customer_id}:velocity")
            self.r.delete(f"customer:{customer_id}:behavior")
            self.r.delete(f"customer:{customer_id}:location")
        except Exception:
            pass

        # Re-seed legitimate baseline
        self.seed_redis_baseline(customer_id)
        logger.info("Demo accounts and state reset completely", customer_id=customer_id, balance=self.initial_balance)

    def get_recipients(self) -> Dict[str, Dict[str, Any]]:
        """Return available recipient registry."""
        return RECIPIENTS_REGISTRY
