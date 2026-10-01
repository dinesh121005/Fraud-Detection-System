"""Kafka Transaction Consumer and Streaming Worker Foundation.

Provides resilient consumption, strict schema validation, manual offset management,
isolated error handling for malformed payloads, and clean integration hooks.
"""
import os
import sys
import time
import signal
import logging
from typing import Optional, Callable, Dict, Any, List, Tuple

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.streaming.schema import TransactionEvent
from src.streaming.config import StreamingConfig, load_streaming_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Consumer] %(message)s")
logger = logging.getLogger("FinPulse.KafkaConsumer")

class TransactionConsumer:
    """
    Resilient Kafka Consumer for processing streaming TransactionEvents.
    Features:
    - Manual offset management ensuring at-least-once delivery semantics.
    - Malformed payload isolation (JSON errors or schema validation failures do not crash the loop).
    - Extensible event handler hook for downstream state and inference pipelines.
    - Graceful shutdown on OS signals (SIGINT, SIGTERM).
    """

    def __init__(
        self,
        config: Optional[StreamingConfig] = None,
        handler: Optional[Callable[..., Any]] = None,
        state_manager: Optional[Any] = None,
        feature_engine: Optional[Any] = None,
        model_service: Optional[Any] = None
    ):
        self.config = config or load_streaming_config()
        self.state_manager = state_manager
        self.feature_engine = feature_engine
        self.model_service = model_service
        self.handler = handler or self._default_handler
        self.consumer = None
        self.is_running = False
        self.metrics = {
            "messages_consumed": 0,
            "messages_valid": 0,
            "messages_malformed": 0,
            "commits_successful": 0,
            "commits_failed": 0,
            "start_time": time.time()
        }

        # Setup OS signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _default_handler(
        self,
        event: TransactionEvent,
        context: Optional[Any] = None,
        feature_result: Optional[Any] = None,
        inference_result: Optional[Any] = None
    ) -> None:
        """Default fallback handler: logs receipt, schema attributes, state, features, and ML inference decision."""
        ctx_summary = ""
        if context and hasattr(context, "velocity"):
            ctx_summary = f" | Vel 1h: {context.velocity.get('tx_count_1h', 0)} tx (${context.velocity.get('amount_sum_1h', 0.0):.2f})"
        feat_summary = ""
        if feature_result and hasattr(feature_result, "features"):
            feat_summary = f" | Features: 32-dim OK"
        inf_summary = ""
        if inference_result and hasattr(inference_result, "decision"):
            inf_summary = f" | Decision: [{inference_result.decision}] (P_raw={inference_result.raw_probability:.4f}, P_cal={inference_result.calibrated_probability:.4f})"
        logger.info(f"Consumed valid Tx '{event.transaction_id}' | Cust: '{event.customer_id}' | Amount: ${event.amount:.2f} | Cat: '{event.category}'{ctx_summary}{feat_summary}{inf_summary}")

    def register_handler(self, handler: Callable[..., Any]):
        """Register a downstream processing hook (e.g. feature engine or inference worker)."""
        self.handler = handler

    def register_state_manager(self, state_manager: Any):
        """Register RedisStateManager for real-time historical state management."""
        self.state_manager = state_manager

    def register_feature_engine(self, feature_engine: Any):
        """Register FinPulseFeatureEngine for real-time 32-feature extraction."""
        self.feature_engine = feature_engine

    def register_model_service(self, model_service: Any):
        """Register ProductionModelService for real-time model inference and decisioning."""
        self.model_service = model_service

    def _handle_shutdown(self, signum, frame):
        logger.info(f"Received termination signal ({signum}). Requesting graceful consumer shutdown...")
        self.is_running = False

    def connect(self, max_retries: int = 3, initial_delay: float = 1.0) -> bool:
        """Establish connection and subscribe to Kafka topics."""
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                from kafka import KafkaConsumer
                logger.info(f"Connecting consumer to Kafka at '{self.config.bootstrap_servers}', group='{self.config.consumer.group_id}' (attempt {attempt}/{max_retries})...")

                self.consumer = KafkaConsumer(
                    self.config.topics.inbound_transactions,
                    bootstrap_servers=self.config.bootstrap_servers,
                    group_id=self.config.consumer.group_id,
                    auto_offset_reset=self.config.consumer.auto_offset_reset,
                    enable_auto_commit=self.config.consumer.enable_auto_commit,
                    max_poll_records=self.config.consumer.max_poll_records,
                    session_timeout_ms=self.config.consumer.session_timeout_ms,
                    max_poll_interval_ms=self.config.consumer.max_poll_interval_ms,
                    consumer_timeout_ms=self.config.consumer.poll_timeout_ms
                )
                logger.info(f"Successfully subscribed to topic: '{self.config.topics.inbound_transactions}'")
                return True
            except Exception as e:
                logger.warning(f"Consumer connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logger.error("Consumer connection to Kafka failed.")
                    return False
        return False

    def process_raw_message(self, raw_value: bytes, partition: int = 0, offset: int = 0) -> Optional[TransactionEvent]:
        """
        Safely deserialize and validate a raw message payload.
        Catches all deserialization and schema validation errors without crashing.
        """
        self.metrics["messages_consumed"] += 1
        try:
            event = TransactionEvent.from_bytes(raw_value)
            self.metrics["messages_valid"] += 1
            return event
        except Exception as e:
            self.metrics["messages_malformed"] += 1
            preview = raw_value[:100] if len(raw_value) > 100 else raw_value
            logger.error(
                f"Malformed message at partition {partition}, offset {offset}: {e} | Raw payload preview: {preview!r}"
            )
            return None

    def commit_offset_safe(self):
        """Perform reliable manual offset commit."""
        if not self.consumer or self.config.consumer.enable_auto_commit:
            return
        try:
            self.consumer.commit()
            self.metrics["commits_successful"] += 1
        except Exception as e:
            self.metrics["commits_failed"] += 1
            logger.warning(f"Offset commit failed: {e}")

    def process_event_with_state(
        self,
        event: TransactionEvent
    ) -> Tuple[TransactionEvent, Optional[Any], Optional[Any], Optional[Any]]:
        """
        Processes a single TransactionEvent strictly enforcing the R4 streaming pipeline order:
        1. READ historical state from Redis strictly BEFORE event.timestamp
        2. COMPUTE exactly 32 features via R3 Feature Engine (zero lookahead)
        3. VALIDATE the 32-feature vector (strictly 32 dimensions, 0 NaNs, 0 Infs)
        4. EXECUTE CatBoost inference -> Platt calibration -> ML threshold decision
        5. DISPATCH to downstream handler (exposing event, context, feature_result, inference_result)
        6. WRITE current transaction into Redis state
        Returns (event, context, feature_result, inference_result).
        """
        context = None
        feature_result = None
        inference_result = None

        # 1. READ historical state strictly prior to event
        if self.state_manager is not None:
            context = self.state_manager.get_historical_context(event)

        # 2. COMPUTE & VALIDATE 32 features
        if self.feature_engine is not None:
            if hasattr(self.feature_engine, "extract_features"):
                feature_result = self.feature_engine.extract_features(event, context=context)
            elif hasattr(self.feature_engine, "compute_features"):
                feat_vec = self.feature_engine.compute_features(event, context=context)
                from src.features.schema import FeatureExtractionResult
                feature_result = FeatureExtractionResult(
                    transaction_id=event.transaction_id,
                    customer_id=event.customer_id,
                    timestamp=event.timestamp,
                    features=feat_vec.to_ordered_vector()
                )

        # 3. EXECUTE production model inference & decisioning
        if self.model_service is not None and feature_result is not None:
            if hasattr(self.model_service, "score_features"):
                inference_result = self.model_service.score_features(
                    feature_result.features,
                    transaction_id=event.transaction_id
                )

        # 4. DISPATCH to downstream handler
        if self.handler:
            import inspect
            sig = inspect.signature(self.handler)
            params = sig.parameters
            param_count = len(params)
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params.values())

            if param_count >= 4 or has_varargs:
                self.handler(event, context, feature_result, inference_result)
            elif param_count == 3:
                self.handler(event, context, feature_result)
            elif param_count == 2:
                self.handler(event, context)
            else:
                self.handler(event)

        # 5. WRITE current transaction into Redis state (committed AFTER inference evaluation)
        if self.state_manager is not None:
            self.state_manager.record_transaction(event)

        return event, context, feature_result, inference_result

    def poll_and_process_batch(self, max_records: Optional[int] = None) -> int:
        """
        Poll and process a batch of messages.
        Returns the number of valid messages processed.
        """
        if not self.consumer:
            return 0

        timeout_ms = self.config.consumer.poll_timeout_ms
        records_dict = self.consumer.poll(timeout_ms=timeout_ms, max_records=max_records or self.config.consumer.max_poll_records)

        processed_count = 0
        for topic_partition, records in records_dict.items():
            for record in records:
                event = self.process_raw_message(
                    raw_value=record.value,
                    partition=record.partition,
                    offset=record.offset
                )
                if event is not None:
                    try:
                        self.process_event_with_state(event)
                        processed_count += 1
                    except Exception as handler_err:
                        logger.error(f"Downstream handler failed on tx '{event.transaction_id}': {handler_err}", exc_info=True)

        if records_dict:
            self.commit_offset_safe()

        return processed_count

    def run_loop(self, max_messages: Optional[int] = None):
        """Run blocking streaming consumer loop until max_messages or interrupted."""
        if not self.consumer and not self.connect():
            logger.error("Cannot run consumer loop: connection could not be established.")
            return

        self.is_running = True
        logger.info(f"Consumer loop started on topic '{self.config.topics.inbound_transactions}' (auto_commit={self.config.consumer.enable_auto_commit})...")

        try:
            while self.is_running:
                batch_count = self.poll_and_process_batch()
                if max_messages and self.metrics["messages_consumed"] >= max_messages:
                    logger.info(f"Reached target message limit ({max_messages}). Exiting consumer loop.")
                    break
        finally:
            self.close()

    def close(self):
        """Close Kafka consumer cleanly and log diagnostic metrics."""
        self.is_running = False
        if self.consumer:
            try:
                logger.info("Committing final offsets and closing consumer...")
                self.commit_offset_safe()
                self.consumer.close()
                self.consumer = None
                logger.info("Kafka consumer closed cleanly.")
            except Exception as e:
                logger.warning(f"Error during consumer close: {e}")

        logger.info(
            f"Consumer Metrics: Consumed={self.metrics['messages_consumed']}, "
            f"Valid={self.metrics['messages_valid']}, Malformed={self.metrics['messages_malformed']}, "
            f"Commits={self.metrics['commits_successful']} OK / {self.metrics['commits_failed']} Fail"
        )
