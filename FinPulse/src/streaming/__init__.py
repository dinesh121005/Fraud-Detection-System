"""FinPulse Streaming Package: Kafka Real-Time Event Pipeline."""
from .schema import TransactionEvent, create_sample_transaction
from .config import StreamingConfig, load_streaming_config
from .producer import TransactionProducer, TransactionSimulator
from .consumer import TransactionConsumer
from .publisher import DecisionEventPublisher, KafkaDeliveryError

__all__ = [
    "TransactionEvent",
    "create_sample_transaction",
    "StreamingConfig",
    "load_streaming_config",
    "TransactionProducer",
    "TransactionSimulator",
    "TransactionConsumer",
    "DecisionEventPublisher",
    "KafkaDeliveryError",
]
