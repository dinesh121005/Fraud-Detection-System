import os
import json
import joblib
import pandas as pd
from xgboost import XGBClassifier
import sys
sys.path.append(r"D:\ssn\FinPulse")

# Optional Kafka import
try:
    from kafka import KafkaConsumer, KafkaProducer
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None

# --------------------------
# 1. Load model and encoder
# --------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "../models/fraud_model.json")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "../models/label_encoder.pkl")

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError("Model or encoder not found. Please train the model first.")

model = XGBClassifier()
model.load_model(MODEL_PATH)
le_type = joblib.load(ENCODER_PATH)

# --------------------------
# 2. Import preprocessing function
# --------------------------
from utils.preprocessing import preprocess_transaction

# --------------------------
# 3. Prediction function
# --------------------------
def predict_fraud(transaction: dict) -> dict:
    # Ensure type is encoded
    if isinstance(transaction.get("type"), str):
        transaction["type"] = le_type.transform([transaction["type"]])[0]

    X = preprocess_transaction(transaction)
    prob = model.predict_proba(X)[0][1]
    is_fraud = int(prob >= 0.5)
    return {
        "isFraud": is_fraud,
        "Fraud_Prob": round(prob, 4)
    }

# --------------------------
# 4. Kafka Streaming
# --------------------------
def kafka_predict(consume_topic="transactions", produce_topic="predictions", server="localhost:9092"):
    if KafkaConsumer is None or KafkaProducer is None:
        raise ImportError("KafkaConsumer or KafkaProducer not available. Make sure 'kafka-python' is installed.")

    consumer = KafkaConsumer(
        consume_topic,
        bootstrap_servers=server,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='fraud-detector'
    )

    producer = KafkaProducer(
        bootstrap_servers=server,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

    print(f"[Kafka] Listening for transactions on topic: {consume_topic}")
    for message in consumer:
        tx = message.value
        result = predict_fraud(tx)
        tx.update(result)
        producer.send(produce_topic, tx)
        print(f"[Kafka] Processed tx_id={tx.get('step', '?')} -> isFraud={result['isFraud']}")

# --------------------------
# 5. CLI Example Usage
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud detection predictor")
    parser.add_argument("--kafka", action="store_true", help="Run in Kafka stream mode")
    args = parser.parse_args()

    if args.kafka:
        kafka_predict()
    else:
        sample_tx = {
            "step": 12,
            "type": "TRANSFER",
            "amount": 5000.0,
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 5000.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 5000.0,
            "auth_verified": 1,
            "device_known": 1,
            "geo_known": 1,
            "initiated_by": "sender",
            "receiver_risk_score": 0.35,
            "sender_hourly_tx_count": 1,
            "sender_daily_tx_count": 3
        }
        result = predict_fraud(sample_tx)
        print(f"Sample transaction fraud prediction -> isFraud: {result['isFraud']} | Probability: {result['Fraud_Prob']}")
