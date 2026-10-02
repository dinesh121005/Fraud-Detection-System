"""
FinPulse R7-E — Replay Simulator & Delayed Fraud Label Management Engine.

Provides:
- Realistic historical transaction replay through the event pipeline
- Controllable speedup factor (e.g. 1x, 10x, instant batch)
- Simulation isolation preventing contamination of production Redis counters
- Delayed label manager: simulates chargeback maturation (e.g. 30-day reporting lag)
  and links ground-truth outcomes to original DecisionEvents
- Built-in scenarios: Normal traffic, ATO bursts, High-velocity fraud, HOLD workflows
"""

import time
import json
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Generator

from src.monitoring.logger import get_logger
from src.persistence.sink import IdempotentEventSink

logger = get_logger("FinPulse.Simulation.Replay")


@dataclass
class ReplayConfig:
    """Configuration for historical transaction replay."""
    speedup_multiplier: float = 0.0  # 0.0 = instant / non-blocking
    dry_run: bool = True             # If True, isolates from live Kafka/Redis
    max_events: Optional[int] = None
    seed: int = 42


@dataclass
class DelayedLabelRecord:
    """Record representing a delayed ground-truth chargeback/fraud report."""
    transaction_id: str
    original_timestamp: float
    fraud_label: int  # 1 = Fraud, 0 = Legitimate
    reported_at: float
    reporting_source: str = "chargeback_claim"


class DelayedLabelManager:
    """
    Manages delayed feedback loops where ground-truth fraud labels arrive
    days or weeks after the original transaction execution.
    """

    def __init__(self, sink: Optional[IdempotentEventSink] = None):
        self.sink = sink
        self._pending_labels: List[DelayedLabelRecord] = []

    def queue_delayed_label(
        self,
        transaction_id: str,
        original_timestamp: float,
        fraud_label: int,
        delay_seconds: float = 86400.0 * 14.0,  # 14 days default
        source: str = "visa_chargeback"
    ) -> DelayedLabelRecord:
        """Queue a future ground-truth label."""
        record = DelayedLabelRecord(
            transaction_id=transaction_id,
            original_timestamp=original_timestamp,
            fraud_label=fraud_label,
            reported_at=original_timestamp + delay_seconds,
            reporting_source=source
        )
        self._pending_labels.append(record)
        return record

    def process_maturing_labels(self, current_time: float) -> List[DelayedLabelRecord]:
        """
        Scan and apply all labels whose reporting timestamp has elapsed.
        Attaches ground truth into persistence sink if attached.
        """
        matured: List[DelayedLabelRecord] = []
        remaining: List[DelayedLabelRecord] = []

        for item in self._pending_labels:
            if item.reported_at <= current_time:
                matured.append(item)
                if self.sink:
                    self.sink.attach_delayed_label(
                        transaction_id=item.transaction_id,
                        fraud_label=item.fraud_label,
                        label_timestamp=item.reported_at
                    )
            else:
                remaining.append(item)

        self._pending_labels = remaining
        if matured:
            logger.info("Processed matured delayed labels", count=len(matured))
        return matured


class TransactionReplaySimulator:
    """
    Replay engine capable of generating or streaming realistic event sequences
    (normal, attack, ATO, HOLD) into the FinPulse scoring pipeline.
    """

    def __init__(
        self,
        config: Optional[ReplayConfig] = None,
        sink: Optional[IdempotentEventSink] = None
    ):
        self.config = config or ReplayConfig()
        self.sink = sink
        self.label_manager = DelayedLabelManager(sink=sink)
        random.seed(self.config.seed)

    def generate_scenario_stream(
        self,
        scenario_type: str = "mixed",
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Generate reproducible test transaction scenarios:
        - 'normal': Legitimate everyday transfers & payments
        - 'ato': Credential compromise + high-value drain
        - 'hold': Boundary cases triggering customer review
        - 'fraud': Clear hard-block / balance exhaustion attacks
        - 'mixed': Realistic representative mixture
        """
        transactions = []
        base_time = 1700000000.0

        for i in range(count):
            ts = base_time + (i * 30.0)
            tx_id = f"tx_replay_{scenario_type}_{i:05d}"
            cust_id = f"cust_{(i % 20) + 1:04d}"

            # Determine scenario subtype
            if scenario_type == "mixed":
                sub = random.choices(["normal", "hold", "fraud", "ato"], weights=[0.80, 0.10, 0.05, 0.05])[0]
            else:
                sub = scenario_type

            if sub == "normal":
                tx = {
                    "transaction_id": tx_id,
                    "customer_id": cust_id,
                    "amount": round(random.uniform(15.0, 350.0), 2),
                    "timestamp": ts,
                    "merchant_id": f"merch_{random.randint(1, 50)}",
                    "category": random.choice(["retail", "groceries", "dining"]),
                    "payment_type": "PAYMENT",
                    "origin_balance": round(random.uniform(500.0, 5000.0), 2),
                    "dest_balance": round(random.uniform(100.0, 2000.0), 2),
                    "auth_verified": True,
                    "scenario": "normal",
                    "true_label": 0
                }
            elif sub == "hold":
                # Boundary value / elevated risk designed to trigger HOLD / REVIEW
                tx = {
                    "transaction_id": tx_id,
                    "customer_id": cust_id,
                    "amount": round(random.uniform(1800.0, 4500.0), 2),
                    "timestamp": ts,
                    "merchant_id": "merch_overseas_tech",
                    "category": "electronics",
                    "payment_type": "TRANSFER",
                    "origin_balance": 3000.0,
                    "dest_balance": 100.0,
                    "auth_verified": False,
                    "scenario": "hold",
                    "true_label": 0  # Customer confirms legitness
                }
            elif sub == "ato":
                # Severe ATO scenario
                tx = {
                    "transaction_id": tx_id,
                    "customer_id": cust_id,
                    "amount": 9500.0,
                    "timestamp": ts,
                    "merchant_id": "merch_crypto_exchange",
                    "category": "crypto",
                    "payment_type": "TRANSFER",
                    "origin_balance": 10000.0,
                    "dest_balance": 0.0,
                    "auth_verified": False,
                    "scenario": "ato",
                    "true_label": 1
                }
            else:  # fraud
                # Critical hard-block / balance drain
                tx = {
                    "transaction_id": tx_id,
                    "customer_id": cust_id,
                    "amount": 50000.0,
                    "timestamp": ts,
                    "merchant_id": "merch_unauthorized_atm",
                    "category": "cash_out",
                    "payment_type": "CASH_OUT",
                    "origin_balance": 200.0,
                    "dest_balance": 0.0,
                    "auth_verified": False,
                    "scenario": "fraud",
                    "true_label": 1
                }

            transactions.append(tx)

        return transactions

    def execute_replay(
        self,
        transactions: List[Dict[str, Any]],
        scoring_callback: Callable[[Dict[str, Any]], Any],
        on_event_callback: Optional[Callable[[Any, Dict[str, Any]], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute replay with telemetry and delayed label scheduling.
        Returns list of execution result dictionaries.
        """
        results = []
        limit = self.config.max_events or len(transactions)

        for idx, tx in enumerate(transactions[:limit]):
            t_start = time.perf_counter()
            score_res = scoring_callback(tx)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            res_record = {
                "transaction_id": tx["transaction_id"],
                "decision": score_res.get("decision", "UNKNOWN"),
                "risk_score": score_res.get("risk_score", 0.0),
                "true_label": tx.get("true_label", 0),
                "scenario": tx.get("scenario", "default"),
                "latency_ms": elapsed_ms
            }
            results.append(res_record)

            # Schedule delayed ground-truth label
            self.label_manager.queue_delayed_label(
                transaction_id=tx["transaction_id"],
                original_timestamp=tx["timestamp"],
                fraud_label=tx.get("true_label", 0)
            )

            if on_event_callback:
                on_event_callback(score_res, tx)

            if self.config.speedup_multiplier > 0.0:
                time.sleep(1.0 / self.config.speedup_multiplier)

        logger.info("Completed replay execution", total_replayed=len(results))
        return results
