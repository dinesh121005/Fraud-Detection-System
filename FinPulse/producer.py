# producer.py
from kafka import KafkaProducer
import json
import time
import random

# Create Kafka producer
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

print("🚀 Producer started... sending transactions")

while True:
    transaction = {
        "user_id": random.randint(1000, 9999),
        "amount": round(random.uniform(10, 5000), 2),
        "type": random.choice(["TRANSFER", "PAYMENT", "CASH_OUT"]),
    }
    producer.send("transactions", transaction)
    print(f"✅ Sent: {transaction}")
    time.sleep(2)
