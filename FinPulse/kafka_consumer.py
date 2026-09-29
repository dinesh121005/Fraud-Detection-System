from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='10.121.50.11:9092',  # IP of Laptop3
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='test-group'
)

print("Listening for messages on 'test-topic'...")

for message in consumer:
    print(f"Received: {message.value}")