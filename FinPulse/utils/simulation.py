# utils/simulation.py
import random, math

def random_geo(center=(12.9716,77.5946), km_radius=50):
    lat0, lon0 = center
    r = km_radius * math.sqrt(random.random())/111
    theta = random.random() * 2 * math.pi
    return lat0 + r * math.cos(theta), lon0 + r * math.sin(theta)

def generate_transaction(simulate_attack=False):
    types = ["CASH_OUT", "PAYMENT", "TRANSFER", "DEBIT"]
    tx = {
        "step": random.randint(1, 1000),
        "type": random.choice(types),
        "amount": round(random.uniform(1, 5000), 2),
        "oldbalanceOrg": round(random.uniform(0, 10000), 2),
        "newbalanceOrig": 0,
        "oldbalanceDest": round(random.uniform(0, 10000), 2),
        "newbalanceDest": 0,
        "sender": random.randint(100000, 999999),
        "receiver": random.randint(100000, 999999),
        "sender_upi": f"user{random.randint(1000,9999)}@upi",
        "receiver_upi": f"user{random.randint(1000,9999)}@upi",
        "auth_verified": 1,
        "initiated_by": "sender",
        "device_id": f"device_{random.randint(1,50)}",
        "device_known": 1,
        "tx_channel": random.choice(["UPI_APP","WEB","API"]),
        "receiver_risk_score": round(random.random(), 3),
        "sender_hourly_tx_count": random.randint(0,5),
        "sender_daily_tx_count": random.randint(0,20),
        "geo_lat": None,
        "geo_lon": None,
        "geo_known": 1
    }
    tx["newbalanceOrig"] = max(tx["oldbalanceOrg"] - tx["amount"], 0)
    tx["newbalanceDest"] = tx["oldbalanceDest"] + tx["amount"]

    # simulate geo
    lat, lon = random_geo()
    tx["geo_lat"] = round(lat,6)
    tx["geo_lon"] = round(lon,6)

    # Simulate rare attack pattern
    if simulate_attack or random.random() < 0.03:
        tx["auth_verified"] = 0
        tx["initiated_by"] = "third_party"
        tx["device_known"] = 0
        tx["geo_known"] = 0
        tx["amount"] = round(tx["oldbalanceOrg"] + random.uniform(1, 5000), 2)
        tx["newbalanceOrig"] = 0
        tx["newbalanceDest"] = tx["oldbalanceDest"] + tx["amount"]
        tx["receiver_risk_score"] = min(1.0, tx["receiver_risk_score"] + random.uniform(0.2, 0.8))

    return tx
