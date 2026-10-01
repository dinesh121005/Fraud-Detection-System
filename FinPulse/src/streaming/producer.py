"""Kafka Transaction Producer and Simulator.

Provides resilient publishing with partitioned customer routing,
exponential backoff retry, and realistic transaction event simulation.
"""
import os
import sys
import time
import math
import random
import logging
from typing import Optional, List, Dict, Any, Callable

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.config import StreamingConfig, load_streaming_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Producer] %(message)s")
logger = logging.getLogger("FinPulse.KafkaProducer")

class TransactionProducer:
    """
    Resilient Kafka Producer for publishing canonical TransactionEvents.
    Features:
    - Partition key routing by customer_id for ordered consumer streams.
    - Retry and transient error handling with exponential backoff.
    - Offline fallback / dry-run buffer when broker is unreachable.
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        dry_run: bool = False
    ):
        self.config = config or load_streaming_config()
        self.dry_run = dry_run
        self.producer = None
        self.is_connected = False
        self.published_count = 0
        self.failed_count = 0
        self.dry_run_buffer: List[TransactionEvent] = []

        if not self.dry_run:
            self._connect()

    def _connect(self, max_retries: int = 3, initial_delay: float = 1.0):
        """Attempt connection to Kafka broker with exponential backoff."""
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                from kafka import KafkaProducer
                from kafka.errors import NoBrokersAvailable

                logger.info(f"Connecting to Kafka brokers at '{self.config.bootstrap_servers}' (attempt {attempt}/{max_retries})...")
                self.producer = KafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    acks=self.config.producer.acks,
                    retries=self.config.producer.retries,
                    compression_type=self.config.producer.compression_type,
                    linger_ms=self.config.producer.linger_ms,
                    request_timeout_ms=self.config.producer.request_timeout_ms
                )
                self.is_connected = True
                logger.info("Successfully connected to Kafka brokers.")
                return
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("All connection attempts failed. Operating in dry-run buffer fallback mode.")
                    self.is_connected = False

    def publish_transaction(
        self,
        event: TransactionEvent,
        topic: Optional[str] = None,
        sync: bool = False
    ) -> bool:
        """
        Publish a validated TransactionEvent to Kafka.
        Partition key is set to customer_id to guarantee chronological in-order delivery per customer.
        """
        target_topic = topic or self.config.topics.inbound_transactions

        # If offline or dry-run, capture in buffer
        if not self.is_connected or self.producer is None:
            self.dry_run_buffer.append(event)
            self.published_count += 1
            logger.debug(f"[DRY-RUN] Buffered tx '{event.transaction_id}' for customer '{event.customer_id}' (${event.amount:.2f})")
            return True

        try:
            payload = event.to_bytes()
            key = event.customer_id.encode("utf-8")

            future = self.producer.send(target_topic, key=key, value=payload)

            if sync:
                record_metadata = future.get(timeout=self.config.producer.request_timeout_ms / 1000.0)
                logger.info(f"Published tx '{event.transaction_id}' to topic '{record_metadata.topic}' partition {record_metadata.partition} offset {record_metadata.offset}")
            else:
                def on_send_success(record_metadata):
                    logger.debug(f"Ack tx '{event.transaction_id}' at offset {record_metadata.offset}")

                def on_send_error(excp):
                    logger.error(f"Async publish failed for tx '{event.transaction_id}': {excp}")
                    self.failed_count += 1

                future.add_callback(on_send_success).add_errback(on_send_error)

            self.published_count += 1
            return True

        except Exception as e:
            logger.error(f"Failed to publish transaction '{event.transaction_id}': {e}")
            self.failed_count += 1
            return False

    def flush(self, timeout: Optional[float] = None):
        """Flush internal producer queues."""
        if self.producer:
            self.producer.flush(timeout=timeout)

    def close(self, timeout: Optional[float] = 5.0):
        """Gracefully flush and close producer."""
        if self.producer:
            try:
                logger.info("Closing Kafka producer...")
                self.producer.flush(timeout=timeout)
                self.producer.close(timeout=timeout)
                self.is_connected = False
                logger.info("Kafka producer closed cleanly.")
            except Exception as e:
                logger.warning(f"Error during producer shutdown: {e}")

class TransactionSimulator:
    """
    Realistic Transaction Event Generator.
    Produces high-fidelity stream of credit card and transfer events
    mimicking Sparkov and PaySim distributions for testing and simulation.
    """

    CATEGORIES = [
        "grocery_pos", "entertainment", "gas_transport",
        "shopping_net", "food_dining", "travel", "misc_online"
    ]
    PAYMENT_TYPES = ["CREDIT_CARD", "TRANSFER", "PAYMENT", "CASH_OUT"]
    HOME_HUBS = [
        (40.7128, -74.0060),  # New York
        (34.0522, -118.2437), # Los Angeles
        (41.8781, -87.6298),  # Chicago
        (29.7604, -95.3698),  # Houston
        (37.7749, -122.4194)  # San Francisco
    ]

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.customers = [f"cc_cust_{i:04d}" for i in range(1, 101)]
        self.merchants = [f"merch_{cat}_{j:03d}" for cat in self.CATEGORIES for j in range(1, 11)]
        self.customer_homes = {
            c: self.rng.choice(self.HOME_HUBS) for c in self.customers
        }

    def generate_event(
        self,
        tx_index: int,
        customer_id: Optional[str] = None,
        force_fraud: bool = False,
        timestamp: Optional[float] = None
    ) -> TransactionEvent:
        """Generate a single realistic transaction event."""
        cust = customer_id or self.rng.choice(self.customers)
        home_lat, home_lon = self.customer_homes[cust]
        now = timestamp if timestamp is not None else 1704067200.0 + (tx_index * 10.0)

        if force_fraud:
            # Anomaly parameters: high amount, foreign location, unverified auth
            amount = round(self.rng.uniform(2500.0, 9500.0), 2)
            lat = round(home_lat + self.rng.uniform(5.0, 15.0), 4)
            lon = round(home_lon + self.rng.uniform(5.0, 15.0), 4)
            auth = False
            device = f"unknown_pos_{self.rng.randint(9000, 9999)}"
            is_fraud = 1
        else:
            # Normal cardholder spending: small distance jitter from home
            amount = round(self.rng.lognormvariate(3.5, 0.8), 2)
            amount = max(1.50, min(amount, 1200.0))
            lat = round(home_lat + self.rng.uniform(-0.15, 0.15), 4)
            lon = round(home_lon + self.rng.uniform(-0.15, 0.15), 4)
            auth = True
            device = f"device_{cust}"
            is_fraud = 0

        return TransactionEvent(
            schema_version="1.0",
            transaction_id=f"tx_sim_{tx_index:07d}",
            timestamp=now,
            amount=amount,
            customer_id=cust,
            merchant_id=self.rng.choice(self.merchants),
            category=self.rng.choice(self.CATEGORIES),
            payment_type=self.rng.choice(self.PAYMENT_TYPES),
            origin_balance=round(self.rng.uniform(500.0, 10000.0), 2),
            latitude=lat,
            longitude=lon,
            home_latitude=home_lat,
            home_longitude=home_lon,
            device_id=device,
            auth_verified=auth,
            is_fraud=is_fraud,
            metadata={"simulator_run": True, "tx_index": tx_index}
        )

    def run_simulation(
        self,
        producer: TransactionProducer,
        rate_per_sec: float = 2.0,
        max_events: Optional[int] = None,
        fraud_ratio: float = 0.04
    ) -> int:
        """Run transaction simulation publishing events to the provided producer."""
        interval = 1.0 / max(rate_per_sec, 0.01)
        count = 0
        logger.info(f"Starting transaction simulator (rate={rate_per_sec} tx/s, max_events={max_events or 'infinite'}, fraud_ratio={fraud_ratio * 100:.1f}%)...")

        try:
            while max_events is None or count < max_events:
                force_fraud = random.random() < fraud_ratio
                event = self.generate_event(tx_index=count + 1, force_fraud=force_fraud)
                producer.publish_transaction(event)
                count += 1
                if count % 10 == 0:
                    logger.info(f"Published {count} transactions (Latest: {event.transaction_id}, ${event.amount:.2f}, FraudFlag: {event.is_fraud})")
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Simulator interrupted by user.")
        finally:
            producer.flush()
            logger.info(f"Simulator finished. Total transactions dispatched: {count}")
        return count
