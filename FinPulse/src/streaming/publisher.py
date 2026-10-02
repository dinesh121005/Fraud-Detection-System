"""R5.5 — Kafka Decision Event Publisher with R6.1 Telemetry.

Resilient publisher for emitting canonical R5.4 DecisionEvents to:
1. 'predictions' topic: All completed hybrid decisions (APPROVE, REVIEW, BLOCK)
2. 'fraud-alerts' topic: High-risk and hard-block alerts (decision == BLOCK or hard_block == True)

Invariants:
- Uses canonical UTF-8 deterministic serialization via DecisionEvent.to_bytes()
- Kafka message key is set to transaction_id for deterministic transaction routing
- Event idempotency: event_id is deterministic and preserved across retries
- Raw 32-feature vector is never emitted
- Delivery acknowledgement: Synchronous (sync=True) or asynchronous with callbacks
- Offline fallback / dry-run buffer when broker is unreachable
- Kafka failures are surfaced explicitly; business decisions are never altered
"""

import time
import logging
from typing import Dict, Any, List, Optional

from src.risk_engine.decision_event import DecisionEvent
from src.streaming.config import StreamingConfig, load_streaming_config
from src.monitoring.metrics import record_fraud_alert, record_stage_latency, record_error

logger = logging.getLogger("FinPulse.DecisionPublisher")


class KafkaDeliveryError(Exception):
    """Raised when Kafka fails to accept or confirm delivery of a DecisionEvent."""
    pass


class DecisionEventPublisher:
    """
    Kafka Publisher for R5.4 DecisionEvents with R6.1 Observability.
    
    Guarantees:
    - Deterministic UTF-8 serialization via DecisionEvent.to_bytes()
    - Idempotency key routing: Kafka message key = transaction_id
    - Multi-topic routing:
      * 'predictions': All decisions (APPROVE, REVIEW, BLOCK)
      * 'fraud-alerts': Decisions with decision == "BLOCK" or hard_block == True
    - Resilient offline fallback / dry-run buffering
    - Explicit failure propagation without altering business decision
    - Low-cardinality Prometheus telemetry recording
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
        self.published_predictions_count = 0
        self.published_alerts_count = 0
        self.failed_count = 0
        self.dry_run_predictions: List[DecisionEvent] = []
        self.dry_run_alerts: List[DecisionEvent] = []

        if not self.dry_run:
            self._connect()

    def _connect(self, max_retries: int = 3, initial_delay: float = 1.0):
        """Attempt connection to Kafka broker with exponential backoff."""
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                from kafka import KafkaProducer
                logger.info(
                    f"Connecting DecisionPublisher to Kafka at '{self.config.bootstrap_servers}' "
                    f"(attempt {attempt}/{max_retries})..."
                )
                self.producer = KafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    acks=self.config.producer.acks,
                    retries=self.config.producer.retries,
                    compression_type=self.config.producer.compression_type,
                    linger_ms=self.config.producer.linger_ms,
                    request_timeout_ms=self.config.producer.request_timeout_ms,
                )
                self.is_connected = True
                logger.info("DecisionPublisher successfully connected to Kafka.")
                return
            except Exception as e:
                logger.warning(f"Kafka connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("All connection attempts failed. Operating in dry-run buffer mode.")
                    self.is_connected = False

    def should_alert(self, event: DecisionEvent) -> bool:
        """
        Determine if an event requires routing to 'fraud-alerts'.
        Policy: decision == 'BLOCK' or hard_block is True.
        """
        return event.decision == "BLOCK" or bool(event.hard_block)

    def publish(
        self,
        event: DecisionEvent,
        sync: bool = False,
        predictions_topic: Optional[str] = None,
        alerts_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Publish a validated DecisionEvent to Kafka topics with latency telemetry.
        
        Parameters:
            event: Canonical R5.4 DecisionEvent instance.
            sync: Whether to block waiting for broker delivery acknowledgement.
            predictions_topic: Override topic for predictions (defaults to config).
            alerts_topic: Override topic for fraud alerts (defaults to config).
            
        Returns:
            Dictionary with delivery status, event metadata, and published topics.
        """
        if not isinstance(event, DecisionEvent):
            raise TypeError(f"Expected DecisionEvent instance, got {type(event).__name__}")

        p_topic = predictions_topic or self.config.topics.outbound_predictions
        a_topic = alerts_topic or self.config.topics.alerts
        is_alert = self.should_alert(event)
        published_topics: List[str] = []

        t_pub_start = time.perf_counter()

        # 1. Offline or Dry-Run Mode
        if not self.is_connected or self.producer is None:
            self.dry_run_predictions.append(event)
            self.published_predictions_count += 1
            published_topics.append(p_topic)

            if is_alert:
                self.dry_run_alerts.append(event)
                self.published_alerts_count += 1
                published_topics.append(a_topic)
                record_fraud_alert()

            pub_latency = (time.perf_counter() - t_pub_start) * 1000.0
            record_stage_latency("kafka_publish", pub_latency)

            logger.debug(
                f"[DRY-RUN] Buffered DecisionEvent '{event.event_id}' for Tx '{event.transaction_id}' "
                f"[{event.decision}] -> topics: {published_topics}"
            )
            return {
                "transaction_id": event.transaction_id,
                "event_id": event.event_id,
                "predictions_published": True,
                "alerts_published": is_alert,
                "is_alert": is_alert,
                "topics": published_topics,
                "delivery_mode": "dry_run",
            }

        # 2. Live Kafka Delivery
        try:
            payload = event.to_bytes()
            key = event.transaction_id.encode("utf-8")
            futures = []

            # 2a. Publish to 'predictions' (all decisions)
            fut_pred = self.producer.send(p_topic, key=key, value=payload)
            futures.append((p_topic, fut_pred))

            # 2b. Publish to 'fraud-alerts' (high-risk / hard-block only)
            fut_alert = None
            if is_alert:
                fut_alert = self.producer.send(a_topic, key=key, value=payload)
                futures.append((a_topic, fut_alert))

            if sync:
                timeout_sec = self.config.producer.request_timeout_ms / 1000.0
                for topic_name, fut in futures:
                    record_metadata = fut.get(timeout=timeout_sec)
                    published_topics.append(record_metadata.topic)
                    logger.debug(
                        f"Delivered event '{event.event_id}' to topic '{record_metadata.topic}' "
                        f"partition {record_metadata.partition} offset {record_metadata.offset}"
                    )
            else:
                published_topics.append(p_topic)
                if is_alert:
                    published_topics.append(a_topic)

            self.published_predictions_count += 1
            if is_alert:
                self.published_alerts_count += 1
                record_fraud_alert()

            pub_latency = (time.perf_counter() - t_pub_start) * 1000.0
            record_stage_latency("kafka_publish", pub_latency)

            return {
                "transaction_id": event.transaction_id,
                "event_id": event.event_id,
                "predictions_published": True,
                "alerts_published": is_alert,
                "is_alert": is_alert,
                "topics": published_topics,
                "delivery_mode": "sync" if sync else "async",
            }

        except Exception as e:
            self.failed_count += 1
            record_error("publisher", "connection_error")
            logger.error(
                f"Failed to publish DecisionEvent '{event.event_id}' for Tx '{event.transaction_id}': {e}"
            )
            raise KafkaDeliveryError(f"Kafka delivery failed for transaction '{event.transaction_id}': {e}") from e

    def flush(self, timeout: Optional[float] = None):
        """Flush internal producer queues."""
        if self.producer:
            self.producer.flush(timeout=timeout)

    def close(self, timeout: Optional[float] = 5.0):
        """Flush and close Kafka producer."""
        if self.producer:
            try:
                self.producer.flush(timeout=timeout)
                self.producer.close(timeout=timeout)
                self.is_connected = False
                logger.info("DecisionPublisher closed cleanly.")
            except Exception as e:
                logger.warning(f"Error during DecisionPublisher shutdown: {e}")
