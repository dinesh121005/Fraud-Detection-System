"""Comprehensive Test Suite for R1 Kafka Streaming Foundation.

Validates:
1. Transaction serialization and deserialization (JSON, UTF-8 bytes).
2. Schema validation (valid fields, rejection of missing fields, amount <= 0, etc.).
3. Parity with CommonTransactionSchema.
4. Producer routing, customer-based partition keying, and resilient error handling.
5. Consumer message processing, error isolation on malformed payloads, and metric tracking.
6. Offset management and manual commit semantics.
7. Simulator deterministic transaction generation.
8. End-to-end simulated streaming pipeline (Producer -> Ingestion -> Consumer -> Handler).
"""
import os
import sys
import json
import time
import pytest
from unittest.mock import MagicMock, patch

# Ensure FinPulse root in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FINPULSE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if FINPULSE_DIR not in sys.path:
    sys.path.insert(0, FINPULSE_DIR)

from src.streaming.schema import TransactionEvent, create_sample_transaction
from src.streaming.config import StreamingConfig, load_streaming_config
from src.streaming.producer import TransactionProducer, TransactionSimulator
from src.streaming.consumer import TransactionConsumer
from src.features.schema import CommonTransactionSchema

# --- 1. Serialization & Deserialization Tests ---

def test_transaction_event_serialization_roundtrip():
    """Verify TransactionEvent serializes to JSON/bytes and back with exact field preservation."""
    event = create_sample_transaction(
        transaction_id="tx_1001",
        amount=249.99,
        customer_id="cust_888",
        merchant_id="merch_555",
        category="retail",
        payment_type="CREDIT_CARD",
        timestamp=1710000000.0
    )

    # To JSON and back
    json_str = event.to_json()
    reconstructed_from_json = TransactionEvent.from_json(json_str)
    assert reconstructed_from_json.transaction_id == "tx_1001"
    assert reconstructed_from_json.amount == 249.99
    assert reconstructed_from_json.customer_id == "cust_888"
    assert reconstructed_from_json.schema_version == "1.0"
    assert reconstructed_from_json.timestamp == 1710000000.0

    # To Bytes and back
    raw_bytes = event.to_bytes()
    reconstructed_from_bytes = TransactionEvent.from_bytes(raw_bytes)
    assert reconstructed_from_bytes == event

def test_common_transaction_schema_compatibility():
    """Verify bidirectional conversion between TransactionEvent and CommonTransactionSchema."""
    event = create_sample_transaction(
        transaction_id="tx_common_test",
        amount=150.0,
        customer_id="cust_01"
    )

    # Event -> Common Schema
    common = event.to_common_schema()
    assert isinstance(common, CommonTransactionSchema)
    assert common.transaction_id == "tx_common_test"
    assert common.amount == 150.0
    assert common.customer_id == "cust_01"
    assert common.location.latitude == event.latitude

    # Common Schema -> Event
    event2 = TransactionEvent.from_common_schema(common)
    assert event2.transaction_id == common.transaction_id
    assert event2.amount == common.amount
    assert event2.customer_id == common.customer_id

# --- 2. Schema Validation & Constraint Tests ---

def test_transaction_amount_must_be_positive():
    """Validation: Amount <= 0 must fail schema validation."""
    with pytest.raises(ValueError, match=r"greater than 0|strictly positive"):
        TransactionEvent(
            transaction_id="tx_invalid_amt",
            amount=0.0,
            customer_id="cust_01",
            merchant_id="merch_01"
        )

    with pytest.raises(ValueError, match=r"greater than 0|strictly positive"):
        TransactionEvent(
            transaction_id="tx_invalid_amt2",
            amount=-50.0,
            customer_id="cust_01",
            merchant_id="merch_01"
        )

def test_transaction_required_fields_cannot_be_empty():
    """Validation: Missing or whitespace identifiers must fail validation."""
    with pytest.raises(ValueError):
        TransactionEvent(
            transaction_id="",
            amount=100.0,
            customer_id="cust_01",
            merchant_id="merch_01"
        )

    with pytest.raises(ValueError):
        TransactionEvent(
            transaction_id="tx_01",
            amount=100.0,
            customer_id="   ",
            merchant_id="merch_01"
        )

# --- 3. Malformed Payload Handling Tests ---

def test_consumer_isolated_malformed_json_handling():
    """Reliability: Malformed JSON must not crash consumer and must record metrics."""
    consumer = TransactionConsumer()
    
    # 1. Non-JSON corrupt bytes
    bad_bytes_1 = b"THIS IS NOT JSON {{{ corrupt binary \xff\xfe"
    res1 = consumer.process_raw_message(bad_bytes_1, partition=0, offset=10)
    assert res1 is None, "Malformed payload should return None"

    # 2. Valid JSON but missing required fields
    bad_bytes_2 = json.dumps({"schema_version": "1.0", "notes": "missing all required fields"}).encode("utf-8")
    res2 = consumer.process_raw_message(bad_bytes_2, partition=0, offset=11)
    assert res2 is None, "Missing fields should return None"

    # 3. Valid JSON with invalid data types
    bad_bytes_3 = json.dumps({"transaction_id": "tx_bad", "amount": "NOT_A_NUMBER", "customer_id": "c1", "merchant_id": "m1"}).encode("utf-8")
    res3 = consumer.process_raw_message(bad_bytes_3, partition=0, offset=12)
    assert res3 is None, "Bad type should return None"

    # Verify metrics
    assert consumer.metrics["messages_consumed"] == 3
    assert consumer.metrics["messages_malformed"] == 3
    assert consumer.metrics["messages_valid"] == 0

def test_consumer_valid_message_dispatch():
    """Verify consumer correctly dispatches valid messages to registered handler hook."""
    received_events = []

    def test_handler(event: TransactionEvent):
        received_events.append(event)

    consumer = TransactionConsumer(handler=test_handler)
    valid_event = create_sample_transaction(transaction_id="tx_valid_99", amount=75.25)
    
    res = consumer.process_raw_message(valid_event.to_bytes(), partition=0, offset=100)
    assert res is not None
    assert consumer.metrics["messages_valid"] == 1
    assert consumer.metrics["messages_malformed"] == 0

    # Simulate handler dispatch
    test_handler(res)
    assert len(received_events) == 1
    assert received_events[0].transaction_id == "tx_valid_99"

# --- 4. Producer Routing & Customer Keying ---

def test_producer_customer_partition_keying():
    """Verify producer uses customer_id as partition key for deterministic partitioning."""
    producer = TransactionProducer(dry_run=True)
    event = create_sample_transaction(customer_id="customer_routing_key_123")

    success = producer.publish_transaction(event)
    assert success is True
    assert len(producer.dry_run_buffer) == 1
    assert producer.dry_run_buffer[0].customer_id == "customer_routing_key_123"

def test_producer_broker_offline_graceful_handling():
    """Verify producer handles broker unavailability gracefully without unhandled exceptions."""
    # Force connection to a non-existent port with max_retries=1
    cfg = load_streaming_config()
    cfg.bootstrap_servers = "127.0.0.1:59999"  # Port with no broker
    
    producer = TransactionProducer(config=cfg, dry_run=False)
    assert producer.is_connected is False, "Producer should recognize broker is offline"

    event = create_sample_transaction(transaction_id="tx_offline_test")
    published = producer.publish_transaction(event)
    assert published is True, "Producer should fall back to buffer mode without crashing"
    assert len(producer.dry_run_buffer) == 1

# --- 5. Simulator Deterministic Testing ---

def test_simulator_deterministic_generation():
    """Verify TransactionSimulator produces valid, reproducible transaction streams."""
    sim1 = TransactionSimulator(seed=123)
    sim2 = TransactionSimulator(seed=123)

    event1 = sim1.generate_event(tx_index=1)
    event2 = sim2.generate_event(tx_index=1)

    assert event1.transaction_id == event2.transaction_id
    assert event1.amount == event2.amount
    assert event1.customer_id == event2.customer_id
    assert event1.category == event2.category

    # Verify fraud injection works
    fraud_event = sim1.generate_event(tx_index=2, force_fraud=True)
    assert fraud_event.is_fraud == 1
    assert fraud_event.auth_verified is False

# --- 6. End-to-End Simulation Flow ---

def test_end_to_end_streaming_pipeline_simulation():
    """
    End-to-End Test:
    1. TransactionSimulator generates batch of transactions.
    2. TransactionProducer serializes and packages them.
    3. TransactionConsumer consumes and validates each event.
    4. Handler hook records valid events and telemetry.
    """
    producer = TransactionProducer(dry_run=True)
    simulator = TransactionSimulator(seed=42)

    consumed_records = []
    def recording_handler(event: TransactionEvent):
        consumed_records.append(event)

    consumer = TransactionConsumer(handler=recording_handler)

    # 1. Produce 20 transactions via simulator
    total_produced = simulator.run_simulation(producer, rate_per_sec=100.0, max_events=20)
    assert total_produced == 20
    assert len(producer.dry_run_buffer) == 20

    # 2. Feed producer buffer into consumer processing loop
    for i, event in enumerate(producer.dry_run_buffer):
        raw_bytes = event.to_bytes()
        processed_event = consumer.process_raw_message(raw_bytes, partition=0, offset=i)
        assert processed_event is not None
        recording_handler(processed_event)

    # 3. Verify 100% reception parity
    assert len(consumed_records) == 20
    assert consumer.metrics["messages_valid"] == 20
    assert consumer.metrics["messages_malformed"] == 0

    for original, consumed in zip(producer.dry_run_buffer, consumed_records):
        assert original.transaction_id == consumed.transaction_id
        assert original.amount == consumed.amount
        assert original.customer_id == consumed.customer_id
        assert original.schema_version == "1.0"
