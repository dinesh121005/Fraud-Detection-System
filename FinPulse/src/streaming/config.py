"""Configuration Management for Kafka Streaming Foundation."""
import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Ensure FinPulse root discovery
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

class KafkaTopicsConfig(BaseModel):
    inbound_transactions: str = "transactions"
    outbound_predictions: str = "predictions"
    alerts: str = "fraud-alerts"
    dead_letter: str = "transactions-dlq"

class KafkaConsumerConfig(BaseModel):
    group_id: str = "finpulse-inference-workers"
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = False
    max_poll_records: int = 100
    poll_timeout_ms: int = 1000
    session_timeout_ms: int = 45000
    max_poll_interval_ms: int = 300000

class KafkaProducerConfig(BaseModel):
    acks: str = "all"
    retries: int = 3
    compression_type: Optional[str] = "gzip"
    linger_ms: int = 10
    request_timeout_ms: int = 15000

class StreamingConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    topics: KafkaTopicsConfig = Field(default_factory=KafkaTopicsConfig)
    consumer: KafkaConsumerConfig = Field(default_factory=KafkaConsumerConfig)
    producer: KafkaProducerConfig = Field(default_factory=KafkaProducerConfig)

def load_streaming_config(config_path: Optional[str] = None) -> StreamingConfig:
    """
    Load streaming configuration from YAML file and apply environment variable overrides.
    Priority: Environment Variables > YAML file > Default values.
    """
    target_path = config_path or os.path.join(FINPULSE_DIR, "configs", "streaming.yaml")
    cfg_dict: Dict[str, Any] = {}

    if os.path.exists(target_path):
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                raw_yaml = yaml.safe_load(f)
                if raw_yaml and "streaming" in raw_yaml and "kafka" in raw_yaml["streaming"]:
                    cfg_dict = raw_yaml["streaming"]["kafka"]
        except Exception as e:
            print(f"Warning: Failed to load streaming config from {target_path}: {e}")

    # Build base config
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", cfg_dict.get("bootstrap_servers", "localhost:9092"))
    
    # Topics
    topics_raw = cfg_dict.get("topics", {})
    topics = KafkaTopicsConfig(
        inbound_transactions=os.environ.get("KAFKA_TOPIC_TRANSACTIONS", topics_raw.get("inbound_transactions", "transactions")),
        outbound_predictions=os.environ.get("KAFKA_TOPIC_PREDICTIONS", topics_raw.get("outbound_predictions", "predictions")),
        alerts=os.environ.get("KAFKA_TOPIC_ALERTS", topics_raw.get("alerts", "fraud-alerts")),
        dead_letter=os.environ.get("KAFKA_TOPIC_DLQ", topics_raw.get("dead_letter", "transactions-dlq"))
    )

    # Consumer
    cons_raw = cfg_dict.get("consumer", {})
    auto_commit_env = os.environ.get("KAFKA_ENABLE_AUTO_COMMIT")
    enable_auto_commit = (auto_commit_env.lower() in ("true", "1")) if auto_commit_env is not None else cons_raw.get("enable_auto_commit", False)
    
    consumer = KafkaConsumerConfig(
        group_id=os.environ.get("KAFKA_GROUP_ID", cons_raw.get("group_id", "finpulse-inference-workers")),
        auto_offset_reset=os.environ.get("KAFKA_AUTO_OFFSET_RESET", cons_raw.get("auto_offset_reset", "latest")),
        enable_auto_commit=enable_auto_commit,
        max_poll_records=int(os.environ.get("KAFKA_MAX_POLL_RECORDS", cons_raw.get("max_poll_records", 100))),
        poll_timeout_ms=int(os.environ.get("KAFKA_POLL_TIMEOUT_MS", cons_raw.get("poll_timeout_ms", 1000))),
        session_timeout_ms=int(os.environ.get("KAFKA_SESSION_TIMEOUT_MS", cons_raw.get("session_timeout_ms", 45000))),
        max_poll_interval_ms=int(os.environ.get("KAFKA_MAX_POLL_INTERVAL_MS", cons_raw.get("max_poll_interval_ms", 300000)))
    )

    # Producer
    prod_raw = cfg_dict.get("producer", {})
    producer = KafkaProducerConfig(
        acks=os.environ.get("KAFKA_ACKS", str(prod_raw.get("acks", "all"))),
        retries=int(os.environ.get("KAFKA_RETRIES", prod_raw.get("retries", 3))),
        compression_type=os.environ.get("KAFKA_COMPRESSION_TYPE", prod_raw.get("compression_type", "gzip")),
        linger_ms=int(os.environ.get("KAFKA_LINGER_MS", prod_raw.get("linger_ms", 10))),
        request_timeout_ms=int(os.environ.get("KAFKA_REQUEST_TIMEOUT_MS", prod_raw.get("request_timeout_ms", 15000)))
    )

    return StreamingConfig(
        bootstrap_servers=bootstrap,
        topics=topics,
        consumer=consumer,
        producer=producer
    )
