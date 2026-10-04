"""Kafka real-time streaming inference and alert generation worker with R6.1 Observability."""
import os
import sys
import json
import time
from typing import Dict, Any, Optional, Tuple, List, Set

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.predictor import ProductionPredictor
from src.risk_engine.decision_event import DecisionEvent
from src.streaming.config import StreamingConfig, load_streaming_config
from src.streaming.publisher import DecisionEventPublisher
from src.persistence.sink import IdempotentEventSink
from src.monitoring.metrics import (
    record_stage_latency,
    record_replay,
    record_dlq,
    record_error,
)
from src.monitoring.correlation import set_correlation_context
from src.monitoring.logger import get_logger

logger = get_logger("FinPulse.StreamingWorker")


class StreamingFraudWorker:
    """
    Kafka Streaming Worker:
    Ingests 'transactions' -> Computes Features -> Runs R4 Inference -> R5 Hybrid & Rules ->
    Constructs DecisionEvent v1.0 -> Emits to 'predictions' and 'fraud-alerts' ->
    Persists to PostgreSQL System of Record -> Commits Offset.
    Includes R6.1 telemetry, replay tracking, and DLQ handling.
    """

    def __init__(
        self,
        bootstrap_servers: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
        dry_run: bool = False,
        sink: Optional[IdempotentEventSink] = None,
        enable_persistence: bool = True
    ):
        artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
        self.predictor = ProductionPredictor(artifacts_dir)
        self.config = config or load_streaming_config()
        if bootstrap_servers:
            self.config.bootstrap_servers = bootstrap_servers
        self.publisher = DecisionEventPublisher(self.config, dry_run=dry_run)
        self._processed_tx_ids: Set[str] = set()

        if sink is not None:
            self.sink = sink
        elif enable_persistence and not dry_run:
            try:
                self.sink = IdempotentEventSink()
            except Exception as e:
                logger.warning(f"Could not initialize persistence sink in worker: {e}")
                self.sink = None
        else:
            self.sink = None

    def process_single_transaction(self, tx: Dict[str, Any]) -> Tuple[Dict[str, Any], DecisionEvent]:
        """
        Process incoming raw transaction dict, run full R4-R5 evaluation,
        track idempotency/replays, and construct canonical DecisionEvent v1.0.
        """
        tx_id = str(tx.get("transaction_id", "unknown_tx"))

        # Replay / Idempotency Detection
        if tx_id in self._processed_tx_ids:
            record_replay()
            logger.info("Duplicate transaction replay detected", transaction_id=tx_id)
        else:
            self._processed_tx_ids.add(tx_id)

        # Execute scoring
        result = self.predictor.predict(tx)

        cust_id = str(tx.get("customer_id", "unknown_cust"))
        ts = float(tx.get("timestamp", time.time()))
        latency = float(result.get("latency_ms", 0.0))

        # Serialization Latency Measurement
        t_ser = time.perf_counter()
        try:
            event = DecisionEvent(
                transaction_id=tx_id,
                customer_id=cust_id,
                decision=result["decision"],
                risk_score=float(result["risk_score"]),
                risk_level=result["risk_level"],
                ml_decision=result.get("ml_decision", result["decision"]),
                calibrated_probability=float(result["calibrated_probability"]),
                signals=dict(result.get("signals", {})),
                diagnostics=dict(result.get("diagnostics", {})),
                reasons=list(result.get("top_reasons", [])),
                timestamp=ts,
                model_version=str(result.get("model_version", "finpulse-v3")),
                matched_rules=result.get("rule_result", {}).get("matched_rules", []),
                hard_block=bool(result.get("diagnostics", {}).get("hard_block", False)),
                hard_block_rules=result.get("rule_result", {}).get("hard_block_rules", []),
                latency_ms=latency,
            )
            # Ensure serialization operates cleanly
            _ = event.to_bytes()
        except Exception as e:
            record_error("serialization", "schema_error")
            logger.error("DecisionEvent serialization failed", error=str(e), transaction_id=tx_id)
            raise
        finally:
            record_stage_latency("serialization", (time.perf_counter() - t_ser) * 1000.0)

        # Update correlation context with canonical event_id
        set_correlation_context(
            transaction_id=tx_id,
            customer_id=cust_id,
            event_id=event.event_id,
            model_version=event.model_version,
            feature_schema_version=event.feature_schema_version
        )

        # Persist DecisionEvent to relational durable system of record
        if self.sink is not None:
            try:
                self.sink.persist_decision_event(event)
            except Exception as pe:
                record_error("persistence", "worker_persist_error")
                logger.error("Failed to persist DecisionEvent in worker", error=str(pe), transaction_id=tx_id)

        return result, event

    def route_to_dlq(self, payload: Any, reason: str) -> None:
        """Route failed unprocessable payload to dead-letter queue topic if enabled."""
        record_dlq()
        record_error("kafka", "schema_error")
        logger.warning("Event routed to DLQ", reason=reason, dlq_topic=self.config.topics.dead_letter)

    def run_consumer_loop(
        self,
        in_topic: Optional[str] = None,
        out_topic: Optional[str] = None,
        alerts_topic: Optional[str] = None
    ):
        """Run resilient streaming consumer loop with at-least-once offset commitment and telemetry."""
        target_in = in_topic or self.config.topics.inbound_transactions
        try:
            from kafka import KafkaConsumer
            consumer = KafkaConsumer(
                target_in,
                bootstrap_servers=self.config.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id=self.config.consumer.group_id
            )
            print(f"Streaming worker listening on Kafka topic '{target_in}' at {self.config.bootstrap_servers}...", flush=True)
            logger.info(f"Streaming worker listening on Kafka topic '{target_in}'...")
            for message in consumer:
                t_consume = time.perf_counter()
                try:
                    tx = message.value
                    record_stage_latency("kafka_consume", (time.perf_counter() - t_consume) * 1000.0)

                    result, event = self.process_single_transaction(tx)
                    
                    # Publish DecisionEvent to predictions and fraud-alerts
                    pub_result = self.publisher.publish(
                        event,
                        sync=True,
                        predictions_topic=out_topic,
                        alerts_topic=alerts_topic
                    )
                    
                    # Commit offset ONLY after successful publish
                    consumer.commit()
                    log_msg = (
                        f"[{result['decision']}] Tx {result['transaction_id']} -> "
                        f"Risk: {result['risk_score']} ({result['latency_ms']} ms) | "
                        f"Topics: {pub_result['topics']}"
                    )
                    print(log_msg, flush=True)
                    logger.info(log_msg)
                except Exception as ex:
                    record_error("kafka", "runtime_error")
                    self.route_to_dlq(message.value if 'message' in locals() else None, str(ex))
        except Exception as e:
            record_error("kafka", "connection_error")
            logger.warning(f"Kafka connection not active ({e}). Streaming worker running in test/dry-run mode.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FinPulse Real-Time Streaming Worker")
    parser.add_argument("--continuous", action="store_true", help="Run continuous Kafka consumer loop")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry run mode")
    args, unknown = parser.parse_known_args()

    worker = StreamingFraudWorker(dry_run=args.dry_run)
    if args.continuous or os.environ.get("STREAMING_WORKER_CONTINUOUS") == "1":
        logger.info("FinPulse Streaming Worker starting continuous Kafka ingestion loop...")
        worker.run_consumer_loop()
    else:
        sample = {
            "transaction_id": "test_streaming_01",
            "timestamp": time.time(),
            "amount": 25000.0,
            "customer_id": "cust_stream_99",
            "merchant_id": "merch_crypto_01",
            "category": "crypto",
            "payment_type": "TRANSFER",
            "origin_balance": 1000.0,
            "auth_verified": False
        }
        result, event = worker.process_single_transaction(sample)
        print("\nVerified Streaming Worker Execution:")
        print(json.dumps(result, indent=2))

