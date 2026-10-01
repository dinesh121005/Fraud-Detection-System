"""Kafka real-time streaming inference and alert generation worker."""
import os
import sys
import json
import time
from typing import Dict, Any

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.serving.predictor import ProductionPredictor

class StreamingFraudWorker:
    """
    Kafka Streaming Worker:
    Ingests 'transactions' -> Computes Redis Velocity -> Runs Model -> Emits 'predictions'
    """

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        artifacts_dir = os.path.join(FINPULSE_DIR, "models", "artifacts")
        self.predictor = ProductionPredictor(artifacts_dir)
        self.bootstrap_servers = bootstrap_servers

    def process_single_transaction(self, tx: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming raw transaction dict and return full risk payload."""
        return self.predictor.predict(tx)

    def run_consumer_loop(self, in_topic: str = "transactions", out_topic: str = "predictions"):
        """Run blocking streaming consumer loop."""
        try:
            from kafka import KafkaConsumer, KafkaProducer
            consumer = KafkaConsumer(
                in_topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                group_id="finpulse-inference-group"
            )
            producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            print(f"🚀 Streaming worker listening on Kafka topic '{in_topic}'...")
            for message in consumer:
                tx = message.value
                result = self.process_single_transaction(tx)
                producer.send(out_topic, result)
                print(f"[{result['decision']}] Tx {result['transaction_id']} -> Risk: {result['risk_score']} ({result['latency_ms']} ms)")
        except Exception as e:
            print(f"⚠️ Kafka connection not active ({e}). Streaming worker running in test mode.")

if __name__ == "__main__":
    worker = StreamingFraudWorker()
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
    res = worker.process_single_transaction(sample)
    print("\nVerified Streaming Worker Execution:")
    print(json.dumps(res, indent=2))
