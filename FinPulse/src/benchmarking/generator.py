"""FinPulse R6.2 — Deterministic Workload & Transaction Generator.

Provides reproducible, rate-controlled transaction stream generation for
load and stress testing under varying concurrency and target throughputs.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional


@dataclass
class BenchmarkWorkloadConfig:
    """Configuration for reproducible benchmark workloads."""
    target_rate: float
    duration_sec: float
    seed: int = 42
    concurrency: int = 1
    num_customers: int = 50
    num_merchants: int = 100
    high_risk_ratio: float = 0.05
    inject_duplicate_count: int = 2
    workload_id: str = "finpulse-workload-r6-2"


class DeterministicTransactionGenerator:
    """
    Deterministic transaction generator for FinPulse benchmark scenarios.
    
    Guarantees:
    - Stable pseudo-random streams keyed by seed
    - Unique transaction IDs with prefix tracking
    - Bounded customer and merchant pools to exercise velocity & behavioral states
    - Controlled insertion of duplicate transactions to validate idempotency & replay metrics
    """

    CATEGORIES = [
        "grocery",
        "shopping_pos",
        "shopping_net",
        "travel",
        "dining",
        "entertainment",
        "crypto",
        "cash_out",
    ]

    PAYMENT_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]

    def __init__(self, config: BenchmarkWorkloadConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.expected_count = int(config.target_rate * config.duration_sec)

    def generate_single_transaction(
        self,
        index: int,
        timestamp: Optional[float] = None,
        is_replay: bool = False,
        replay_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a single deterministic transaction dict matching schema contracts."""
        ts = timestamp if timestamp is not None else (1710000000.0 + index * (1.0 / max(1.0, self.config.target_rate)))
        cust_id = f"cust_bench_{self.rng.randint(1, self.config.num_customers):04d}"
        merch_id = f"merch_bench_{self.rng.randint(1, self.config.num_merchants):04d}"
        
        is_high_risk = (self.rng.random() < self.config.high_risk_ratio)

        if is_replay and replay_id:
            tx_id = replay_id
        else:
            tx_id = f"tx_load_{self.config.target_rate:04.0f}_{index:06d}"

        if is_high_risk:
            amount = round(self.rng.uniform(2500.0, 15000.0), 2)
            payment_type = "TRANSFER"
            category = "crypto"
            origin_balance = round(self.rng.uniform(10.0, 500.0), 2)
            auth_verified = False
        else:
            amount = round(self.rng.uniform(5.0, 350.0), 2)
            payment_type = self.rng.choice(self.PAYMENT_TYPES)
            category = self.rng.choice(self.CATEGORIES)
            origin_balance = round(self.rng.uniform(1000.0, 20000.0), 2)
            auth_verified = True

        return {
            "transaction_id": tx_id,
            "timestamp": ts,
            "amount": amount,
            "customer_id": cust_id,
            "merchant_id": merch_id,
            "category": category,
            "payment_type": payment_type,
            "origin_balance": origin_balance,
            "auth_verified": auth_verified,
            "latitude": 37.7749 + (0.005 * (index % 10)),
            "longitude": -122.4194 + (0.005 * (index % 10)),
            "device_id": f"dev_{cust_id[-4:]}",
        }

    def generate_workload(self) -> List[Dict[str, Any]]:
        """Generate complete pre-materialized deterministic batch workload."""
        workload = []
        for i in range(self.expected_count):
            tx = self.generate_single_transaction(i)
            workload.append(tx)

        # Inject controlled duplicate/replay transactions
        if self.config.inject_duplicate_count > 0 and len(workload) > 5:
            for d in range(self.config.inject_duplicate_count):
                source_idx = d * 2
                duplicate_tx = dict(workload[source_idx])
                workload.append(duplicate_tx)

        return workload

    def stream_workload(self) -> Iterator[Dict[str, Any]]:
        """Stream transactions throttled to approximate target_rate."""
        interval = 1.0 / max(1.0, self.config.target_rate)
        for i in range(self.expected_count):
            t_start = time.perf_counter()
            tx = self.generate_single_transaction(i, timestamp=time.time())
            yield tx
            elapsed = time.perf_counter() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
